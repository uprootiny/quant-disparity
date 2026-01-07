#!/usr/bin/env python3
"""
Exp-012: Layer-specific protection
Goal: Test if protecting specific layers is more efficient than magnitude-based
Hypothesis: H5.2a - Layer 0 alone might be sufficient
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-012: Layer-Specific Protection")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Analyze layer structure
    print("\nLayer Structure Analysis:")
    layer_weights = {}
    total = 0
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        # Parse layer name
        if 'wte' in name or 'wpe' in name:
            layer = 'embeddings'
        elif 'h.' in name:
            layer_num = int(name.split('.')[2])
            layer = f'layer_{layer_num}'
        elif 'ln_f' in name:
            layer = 'final_ln'
        else:
            layer = 'other'

        if layer not in layer_weights:
            layer_weights[layer] = 0
        layer_weights[layer] += param.numel()
        total += param.numel()

    for layer, count in sorted(layer_weights.items()):
        pct = count / total * 100
        print(f"  {layer}: {count:,} ({pct:.1f}%)")

    print(f"  Total: {total:,}")

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_except(protect_patterns):
        """Quantize all weights except those matching patterns."""
        protected = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue

                # Check if protected
                is_protected = any(p in name for p in protect_patterns)

                if is_protected:
                    protected += param.numel()
                    continue  # Keep in original precision

                # Quantize
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))

        return protected

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"\nBaseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test configurations
    configs = [
        ("none", []),
        ("embeddings", ["wte", "wpe"]),
        ("layer_0", ["h.0."]),
        ("layer_0+embed", ["wte", "wpe", "h.0."]),
        ("attention_only", ["attn"]),
        ("mlp_only", ["mlp"]),
        ("first_3_layers", ["h.0.", "h.1.", "h.2."]),
    ]

    results = {}
    print(f"\n{'Config':<20} {'Protected':>12} {'%':>6} {'Disp':>10}")
    print("-" * 52)

    for name, patterns in configs:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        pct = protected / total * 100
        results[name] = {
            'patterns': patterns,
            'protected': protected,
            'protected_pct': pct,
            'disparity': disp,
        }

        print(f"{name:<20} {protected:>12,} {pct:>6.1f}% {disp:>10.1f}x")
        gc.collect()

    # Analysis
    print("\n" + "=" * 50)
    print("Efficiency Analysis (disparity per % protected)")
    print("=" * 50)

    base_disp = results['none']['disparity']
    for name, r in results.items():
        if name == 'none' or r['protected_pct'] == 0:
            continue
        reduction = base_disp - r['disparity']
        efficiency = reduction / r['protected_pct']
        print(f"  {name}: {efficiency:.1f} disparity reduction per %")

    # Find best
    valid = [(n, r['disparity'], r['protected_pct']) for n, r in results.items()
             if np.isfinite(r['disparity']) and r['protected_pct'] > 0]
    if valid:
        best = min(valid, key=lambda x: x[1])
        most_efficient = max(valid, key=lambda x: (results['none']['disparity'] - x[1]) / x[2])
        print(f"\n  Best disparity: {best[0]} ({best[1]:.1f}x at {best[2]:.1f}%)")
        print(f"  Most efficient: {most_efficient[0]}")

    end = datetime.now()

    result = {
        "id": "Exp-012",
        "name": "Layer-Specific Protection",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "layer_weights": layer_weights,
        "total_weights": total,
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp012_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
