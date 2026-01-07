#!/usr/bin/env python3
"""
Exp-011b: Quick threshold sweep (1%, 2%, 3%, 4%, 5%)
Simplified version for faster completion
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-011b: Quick Sweep")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}
    total = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)

    # Pre-compute magnitudes
    all_mags = torch.cat([p.data.abs().view(-1) for n, p in model.named_parameters() if 'weight' in n])
    sorted_mags, _ = all_mags.sort(descending=True)
    del all_mags

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def quant(k):
        thresh = sorted_mags[int(total * k / 100) - 1].item() if k > 0 else float('inf')
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'weight' not in n: continue
                flat = p.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    mask = flat.abs() >= thresh
                    p.data.copy_(torch.where(mask, flat, q).view(p.shape))

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Sweep
    results = {}
    print(f"\n{'k%':>4} {'Weights':>10} {'Size':>8} {'Disp':>10}")
    print("-" * 36)

    for k in [0, 1, 2, 3, 4, 5]:
        model.load_state_dict(state)
        quant(k)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        n = int(total * k / 100)
        size_mb = n * 2 / 1e6  # FP16 overhead

        results[k] = {'weights': n, 'size_mb': size_mb, 'disparity': disp}
        print(f"{k:>4}% {n:>10,} {size_mb:>7.1f}MB {disp:>10.1f}x")

        gc.collect()

    # Analysis
    print("\n" + "=" * 40)
    print("Sweet Spot Analysis")
    print("=" * 40)

    finite = [(k, r['disparity'], r['size_mb']) for k, r in results.items() if np.isfinite(r['disparity'])]
    if finite:
        # Best disparity
        best = min(finite, key=lambda x: x[1])
        print(f"Best disparity: {best[0]}% → {best[1]:.1f}x ({best[2]:.1f}MB overhead)")

        # Efficiency (disparity reduction per MB)
        base_disp = results[0]['disparity']
        efficiencies = [(k, (base_disp - d) / max(s, 0.1), d) for k, d, s in finite if k > 0]
        if efficiencies:
            most_efficient = max(efficiencies, key=lambda x: x[1])
            print(f"Most efficient: {most_efficient[0]}% → {most_efficient[2]:.1f}x")

    end = datetime.now()

    result = {
        "id": "Exp-011b",
        "name": "Quick Threshold Sweep",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": {str(k): v for k, v in results.items()},
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp011b_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
