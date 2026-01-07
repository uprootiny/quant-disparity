#!/usr/bin/env python3
"""
Exp-011: Fine-grained threshold sweep
Goal: Find the minimum preservation threshold for acceptable disparity
Hypothesis: H5.1a - <5% can achieve <50x disparity
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps.',
    'he': 'השועל החום קופץ.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-011: Threshold Sweep")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}
    total = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)
    print(f"Total weights: {total:,}")

    # Pre-compute sorted magnitudes
    all_mags = []
    for n, p in model.named_parameters():
        if 'weight' in n:
            all_mags.append(p.data.abs().view(-1))
    all_mags = torch.cat(all_mags)
    sorted_mags, _ = all_mags.sort(descending=True)
    del all_mags

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(out.loss).item()

    def restore():
        model.load_state_dict(state)

    def quant(preserve_pct):
        if preserve_pct > 0:
            n_keep = max(1, int(total * preserve_pct / 100))
            thresh = sorted_mags[n_keep - 1].item()
        else:
            thresh = float('inf')

        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'weight' not in n:
                    continue
                flat = p.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7)
                    dq = q * scale
                    mask = flat.abs() >= thresh
                    new = torch.where(mask, flat, dq)
                    p.data.copy_(new.view(p.shape))

    # Baseline
    baseline = {lang: ppl(text) for lang, text in TEXTS.items()}
    print(f"\nBaseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Fine-grained sweep: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10
    k_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]

    results = {}
    print(f"\n{'k%':>6} {'Disparity':>12} {'Weights':>12}")
    print("-" * 32)

    for k in k_values:
        restore()
        quant(k)

        q_ppl = {lang: ppl(text) for lang, text in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disparity = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        n_preserved = int(total * k / 100)
        results[k] = {
            'preserved': n_preserved,
            'disparity': disparity,
            'deg_en': deg['en'],
            'deg_he': deg['he'],
        }

        print(f"{k:>6.1f} {disparity:>12.2f}x {n_preserved:>12,}")
        gc.collect()

    # Find cliff
    print("\n" + "=" * 50)
    print("Analysis: Finding the Cliff")
    print("=" * 50)

    # Find where disparity drops most
    k_list = sorted(results.keys())
    for i in range(1, len(k_list)):
        prev_k, curr_k = k_list[i-1], k_list[i]
        prev_d, curr_d = results[prev_k]['disparity'], results[curr_k]['disparity']
        if np.isfinite(prev_d) and np.isfinite(curr_d):
            drop = (prev_d - curr_d) / prev_d * 100
            if drop > 10:
                print(f"  Cliff at {prev_k}%→{curr_k}%: {prev_d:.1f}x → {curr_d:.1f}x ({drop:.1f}% drop)")

    # Find minimum for <50x
    under_50 = [(k, r['disparity']) for k, r in results.items()
                if np.isfinite(r['disparity']) and r['disparity'] < 50]
    if under_50:
        min_k, min_d = min(under_50, key=lambda x: x[0])
        print(f"\n  Minimum for <50x disparity: {min_k}% (achieves {min_d:.1f}x)")
    else:
        print("\n  No k value achieves <50x disparity")

    # Find optimal (lowest disparity)
    finite = [(k, r['disparity']) for k, r in results.items() if np.isfinite(r['disparity'])]
    if finite:
        opt_k, opt_d = min(finite, key=lambda x: x[1])
        print(f"  Optimal: {opt_k}% (disparity = {opt_d:.1f}x)")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-011",
        "name": "Threshold Sweep",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "hypothesis": "H5.1a: <5% can achieve <50x disparity",
        "k_values_tested": k_values,
        "results": {str(k): v for k, v in results.items()},
        "optimal_k": opt_k if finite else None,
        "optimal_disparity": opt_d if finite else None,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / f"exp011_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
