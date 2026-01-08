#!/usr/bin/env python3
"""
Exp-020: Per-layer attention analysis
Goal: Identify critical attention layers (complement to MLP analysis)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-020: Per-Layer Attention")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    models_config = [
        ('gpt2', 'gpt2', 'attn', 12),
        ('opt', 'facebook/opt-125m', 'self_attn', 12),
    ]

    all_results = {}

    for model_name, model_id, attn_pattern, n_layers in models_config:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()

        state = {k: v.clone() for k, v in model.state_dict().items()}

        def ppl(text):
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

        def restore():
            model.load_state_dict(state)

        def quant_except(patterns):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    if any(p in name for p in patterns):
                        continue
                    flat = param.view(-1)
                    mx = flat.abs().max()
                    if mx > 0:
                        scale = mx / 7.0
                        q = torch.round(flat / scale).clamp(-8, 7) * scale
                        param.data.copy_(q.view(param.shape))

        # Baseline
        baseline = {l: ppl(t) for l, t in TEXTS.items()}
        print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

        # Baseline disparity (no protection)
        restore()
        quant_except([])
        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp_none = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
        print(f"No protection: {disp_none:.1f}x disparity")

        # Per-layer attention
        layer_results = {}
        print(f"\n{'Layer':<8} {'Disparity':>12} {'Delta':>10}")
        print("-" * 32)

        for layer_idx in range(n_layers):
            restore()

            if model_name == 'gpt2':
                patterns = [f'h.{layer_idx}.attn']
            else:  # OPT
                patterns = [f'layers.{layer_idx}.self_attn']

            quant_except(patterns)

            q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
            deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
            disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

            delta = disp_none - disp
            layer_results[f'layer_{layer_idx}'] = {
                'disparity': disp,
                'delta': delta,
            }
            print(f"Layer {layer_idx:<3} {disp:>12.1f}x {delta:>+10.1f}")

        # Find critical layers
        sorted_layers = sorted(layer_results.items(), key=lambda x: x[1]['disparity'])
        best = sorted_layers[0]
        worst = sorted_layers[-1]

        all_results[model_name] = {
            'baseline_disparity': disp_none,
            'layer_results': layer_results,
            'best_layer': best[0],
            'worst_layer': worst[0],
            'best_disparity': best[1]['disparity'],
            'worst_disparity': worst[1]['disparity'],
        }

        print(f"\nBest attention: {best[0]} ({best[1]['disparity']:.1f}x)")
        print(f"Worst attention: {worst[0]} ({worst[1]['disparity']:.1f}x)")

        del model
        gc.collect()

    # Cross-model comparison
    print("\n" + "=" * 60)
    print("Per-Layer Attention Summary")
    print("=" * 60)

    print(f"\n{'Model':<10} {'Best Attn':<12} {'Disp':>8} {'Worst Attn':<12} {'Disp':>8}")
    print("-" * 55)
    for model_name, results in all_results.items():
        print(f"{model_name:<10} {results['best_layer']:<12} {results['best_disparity']:>7.1f}x "
              f"{results['worst_layer']:<12} {results['worst_disparity']:>7.1f}x")

    # Compare to MLP findings
    print("\n" + "=" * 60)
    print("MLP vs Attention Comparison")
    print("=" * 60)
    print("\n             MLP               Attention")
    print("Model     Best    Worst     Best    Worst")
    print("-" * 50)
    print(f"GPT-2     L0      L1        {all_results['gpt2']['best_layer']}   {all_results['gpt2']['worst_layer']}")
    print(f"OPT       L11     L7        {all_results['opt']['best_layer']}   {all_results['opt']['worst_layer']}")

    end = datetime.now()

    result = {
        "id": "Exp-020",
        "name": "Per-Layer Attention",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": all_results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp020_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
