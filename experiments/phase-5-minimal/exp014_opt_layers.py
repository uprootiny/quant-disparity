#!/usr/bin/env python3
"""
Exp-014: OPT-125M layer-specific validation
Goal: Confirm MLP > Attention pattern holds across models
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-014: OPT-125M Layers")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading OPT-125M...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Analyze structure
    print("\nLayer Analysis:")
    components = {'embed': 0, 'attention': 0, 'mlp': 0, 'other': 0}
    total = 0
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        n = param.numel()
        total += n
        if 'embed' in name:
            components['embed'] += n
        elif 'self_attn' in name or 'attn' in name:
            components['attention'] += n
        elif 'fc1' in name or 'fc2' in name:
            components['mlp'] += n
        else:
            components['other'] += n

    for c, n in components.items():
        print(f"  {c}: {n:,} ({n/total*100:.1f}%)")

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
    print(f"\nBaseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test configs
    configs = [
        ("none", []),
        ("embeddings", ["embed"]),
        ("attention", ["self_attn", "attn"]),
        ("mlp", ["fc1", "fc2"]),
        ("layer0", ["layers.0."]),
    ]

    results = {}
    print(f"\n{'Config':<15} {'Protected':>12} {'%':>6} {'Disp':>10}")
    print("-" * 47)

    for name, patterns in configs:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        pct = protected / total * 100
        results[name] = {'protected': protected, 'pct': pct, 'disparity': disp}
        print(f"{name:<15} {protected:>12,} {pct:>6.1f}% {disp:>10.1f}x")

        gc.collect()

    # Compare to GPT-2 pattern
    print("\n" + "=" * 50)
    print("Cross-Model Comparison")
    print("=" * 50)
    print("\nGPT-2 pattern: MLP (20x) > Layer0 (55x) > None (278x) > Attn (291x) > Embed (1216x)")
    print(f"OPT pattern:   ", end="")

    ranked = sorted([(n, r['disparity']) for n, r in results.items()], key=lambda x: x[1])
    print(" > ".join([f"{n} ({d:.0f}x)" for n, d in ranked]))

    # Check if pattern matches
    gpt2_order = ['mlp', 'layer0', 'none', 'attention', 'embeddings']
    opt_order = [n for n, d in ranked if n != 'none']

    match = all(opt_order[i] in gpt2_order[:i+2] for i in range(min(len(opt_order), len(gpt2_order)-1)))
    print(f"\nPattern matches GPT-2: {'YES' if match else 'PARTIAL'}")

    end = datetime.now()

    result = {
        "id": "Exp-014",
        "name": "OPT-125M Layers",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "model": "facebook/opt-125m",
        "results": results,
        "pattern_match": match,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp014_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
