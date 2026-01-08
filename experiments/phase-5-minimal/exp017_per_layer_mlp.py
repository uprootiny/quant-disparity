#!/usr/bin/env python3
"""
Exp-017: Per-layer MLP contribution analysis
Goal: Understand WHY MLP matters more for GPT-2 vs Attention for OPT-125M
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-017: Per-Layer MLP Analysis")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    models_config = [
        ('gpt2', 'gpt2', 'mlp', 12),
        ('opt', 'facebook/opt-125m', 'fc', 12),
    ]

    all_results = {}

    for model_name, model_id, mlp_pattern, n_layers in models_config:
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
            protected = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    if any(p in name for p in patterns):
                        protected += param.numel()
                        continue
                    flat = param.view(-1)
                    mx = flat.abs().max()
                    if mx > 0:
                        scale = mx / 7.0
                        q = torch.round(flat / scale).clamp(-8, 7) * scale
                        param.data.copy_(q.view(param.shape))
            return protected

        # Baseline
        baseline = {l: ppl(t) for l, t in TEXTS.items()}
        print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

        # Test each layer's MLP protection
        layer_results = {}

        # First: no protection baseline
        restore()
        quant_except([])
        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp_none = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
        layer_results['none'] = {'disparity': disp_none, 'layer': -1}
        print(f"\nNo protection: {disp_none:.1f}x disparity")

        print(f"\n{'Layer':<8} {'Disparity':>12} {'Delta':>10}")
        print("-" * 32)

        for layer_idx in range(n_layers):
            restore()

            # Protect only this layer's MLP
            if model_name == 'gpt2':
                patterns = [f'h.{layer_idx}.mlp']
            else:  # OPT
                patterns = [f'layers.{layer_idx}.fc1', f'layers.{layer_idx}.fc2']

            quant_except(patterns)

            q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
            deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
            disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

            delta = disp_none - disp
            layer_results[f'layer_{layer_idx}'] = {
                'disparity': disp,
                'delta': delta,
                'layer': layer_idx
            }
            print(f"Layer {layer_idx:<3} {disp:>12.1f}x {delta:>+10.1f}")

        # Find most critical layer
        layers_only = {k: v for k, v in layer_results.items() if k != 'none'}
        best_layer = min(layers_only.items(), key=lambda x: x[1]['disparity'])
        worst_layer = max(layers_only.items(), key=lambda x: x[1]['disparity'])

        print(f"\nMost critical MLP: {best_layer[0]} ({best_layer[1]['disparity']:.1f}x)")
        print(f"Least critical MLP: {worst_layer[0]} ({worst_layer[1]['disparity']:.1f}x)")

        all_results[model_name] = {
            'baseline_disparity': disp_none,
            'per_layer': layer_results,
            'most_critical': best_layer[0],
            'least_critical': worst_layer[0],
        }

        del model
        gc.collect()

    # Cross-model analysis
    print("\n" + "=" * 60)
    print("Cross-Model MLP Criticality Pattern")
    print("=" * 60)

    for model_name, results in all_results.items():
        layers = [(k, v['disparity']) for k, v in results['per_layer'].items()
                  if k != 'none']
        layers.sort(key=lambda x: x[1])

        print(f"\n{model_name}:")
        print(f"  Critical layers: {', '.join([l[0] for l in layers[:3]])}")
        print(f"  Non-critical:    {', '.join([l[0] for l in layers[-3:]])}")

    end = datetime.now()

    result = {
        "id": "Exp-017",
        "name": "Per-Layer MLP Analysis",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": all_results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp017_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
