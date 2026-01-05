#!/usr/bin/env python3
"""
Exp-010: Preservation test on full 6-language set
Goal: Confirm 5% optimal holds with full language coverage
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps.',
    'de': 'Der schnelle braune Fuchs springt.',
    'fr': 'Le renard brun rapide saute.',
    'zh': '敏捷的棕色狐狸跳跃。',
    'he': 'השועל החום קופץ.',
    'ar': 'الثعلب البني يقفز.',
}

RESOURCE = {'en': 1.0, 'de': 0.85, 'fr': 0.80, 'zh': 0.50, 'he': 0.15, 'ar': 0.25}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-010: Full Preservation")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}
    total = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)

    # Pre-compute magnitudes
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
            n_keep = int(total * preserve_pct / 100)
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
    print("\nBaseline:")
    baseline = {lang: ppl(text) for lang, text in TEXTS.items()}

    # Test k values
    results = {}
    for k in [0, 5, 10]:
        print(f"\n{k}% preservation:")
        restore()
        quant(k)

        q_ppl = {lang: ppl(text) for lang, text in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}

        hr = [l for l in TEXTS if RESOURCE[l] > 0.5]
        lr = [l for l in TEXTS if RESOURCE[l] <= 0.5]

        hr_avg = np.mean([deg[l] for l in hr])
        lr_avg = np.mean([deg[l] for l in lr])
        disparity = lr_avg / hr_avg if hr_avg > 0 else float('inf')

        results[k] = {
            'hr_avg': hr_avg,
            'lr_avg': lr_avg,
            'disparity': disparity,
        }

        print(f"  HR: {hr_avg:+.0f}%, LR: {lr_avg:+.0f}%")
        print(f"  Disparity: {disparity:.2f}x")

        gc.collect()

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  {'k%':<6} {'Disparity':>12}")
    print("-" * 20)
    for k in results:
        print(f"  {k:<6} {results[k]['disparity']:>12.2f}x")

    optimal_k = min(results.keys(), key=lambda k: results[k]['disparity'])
    print(f"\nOptimal: {optimal_k}%")
    print(f"Disparity reduction: {(results[0]['disparity'] - results[optimal_k]['disparity'])/results[0]['disparity']*100:.1f}%")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-010",
        "name": "Full Preservation",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "languages": list(TEXTS.keys()),
        "results": {str(k): v for k, v in results.items()},
        "optimal_k": optimal_k,
        "disparity_reduction_pct": (results[0]['disparity'] - results[optimal_k]['disparity'])/results[0]['disparity']*100,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp010_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
