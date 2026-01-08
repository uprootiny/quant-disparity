#!/usr/bin/env python3
"""
Exp-021: Combined critical layers strategy
Goal: Test protecting best MLP + best attention layers together
Based on Exp-017 and Exp-020 findings
"""

import json
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-021: Combined Critical Layers")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Based on previous findings:
    # GPT-2: L0 MLP (139x) + L0 Attn (54x) are critical
    # OPT: L11 MLP (92x) + L0 Attn (101x) are critical

    models_config = [
        ('gpt2', 'gpt2', {
            'best_mlp': ['h.0.mlp'],
            'best_attn': ['h.0.attn'],
            'best_combo': ['h.0.mlp', 'h.0.attn'],
            'layer0_full': ['h.0.'],
            'avoid_anti': ['h.0.', 'h.2.'],  # Avoid L1 (anti-critical)
        }),
        ('opt', 'facebook/opt-125m', {
            'best_mlp': ['layers.11.fc1', 'layers.11.fc2'],
            'best_attn': ['layers.0.self_attn'],
            'best_combo': ['layers.11.fc1', 'layers.11.fc2', 'layers.0.self_attn'],
            'layer0_full': ['layers.0.'],
            'avoid_anti': ['layers.0.', 'layers.11.'],  # Avoid L7 (anti-critical)
        }),
    ]

    all_results = {}

    for model_name, model_id, configs in models_config:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()

        total = sum(p.numel() for p in model.parameters())
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

        # Test all configurations
        test_configs = [
            ("none", []),
            ("best_mlp", configs['best_mlp']),
            ("best_attn", configs['best_attn']),
            ("best_combo", configs['best_combo']),
            ("layer0_full", configs['layer0_full']),
            ("avoid_anti", configs['avoid_anti']),
        ]

        results = {}
        print(f"\n{'Config':<15} {'Protected':>10} {'%':>6} {'Disparity':>10}")
        print("-" * 45)

        for name, patterns in test_configs:
            restore()
            protected = quant_except(patterns)

            q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
            deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
            disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

            pct = protected / total * 100
            results[name] = {
                'protected': protected,
                'pct': pct,
                'disparity': disp,
            }
            print(f"{name:<15} {protected:>10,} {pct:>5.1f}% {disp:>10.1f}x")

        # Find best strategy
        valid = [(k, v) for k, v in results.items() if k != 'none' and v['disparity'] < float('inf')]
        if valid:
            best = min(valid, key=lambda x: x[1]['disparity'])
            most_efficient = min(valid, key=lambda x: x[1]['disparity'] / x[1]['pct'] if x[1]['pct'] > 0 else float('inf'))
        else:
            best = most_efficient = (None, {})

        print(f"\nBest disparity: {best[0]} ({best[1].get('disparity', 'N/A'):.1f}x)")
        print(f"Most efficient: {most_efficient[0]}")

        all_results[model_name] = {
            'results': results,
            'best_strategy': best[0],
            'best_disparity': best[1].get('disparity'),
        }

        del model
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("Combined Strategy Summary")
    print("=" * 60)

    print(f"\n{'Model':<10} {'Best Strategy':<15} {'Disparity':>10} {'Overhead':>10}")
    print("-" * 50)
    for model_name, data in all_results.items():
        best = data['best_strategy']
        if best:
            disp = data['results'][best]['disparity']
            pct = data['results'][best]['pct']
            print(f"{model_name:<10} {best:<15} {disp:>9.1f}x {pct:>9.1f}%")

    end = datetime.now()

    result = {
        "id": "Exp-021",
        "name": "Combined Critical Layers",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": all_results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp021_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
