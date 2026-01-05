#!/usr/bin/env python3
"""
Exp-007: Preservation test on OPT-125M
Goal: Verify if 5% optimal preservation holds for OPT architecture
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-007: OPT-125M Preservation")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading OPT-125M...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
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
            n_keep = int(total * preserve_pct / 100)
            thresh = sorted_mags[n_keep - 1].item()
        else:
            thresh = float('inf')

        kept = 0
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
                    kept += mask.sum().item()
        return kept

    # Baseline
    print("\nBaseline:")
    baseline = {lang: ppl(text) for lang, text in TEXTS.items()}
    for l, v in baseline.items():
        print(f"  {l}: {v:.2f}")

    # Test preservation levels
    results = {}
    for k in [0, 5, 10]:
        print(f"\n{k}% preservation:")
        restore()
        kept = quant(k)

        q_ppl = {lang: ppl(text) for lang, text in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disparity = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        results[k] = {'kept': kept, 'disparity': disparity, 'deg': deg}
        print(f"  Kept: {kept:,}")
        print(f"  Disparity: {disparity:.2f}x")

        gc.collect()

    # Find optimal
    optimal_k = min(results.keys(), key=lambda k: results[k]['disparity'])
    print("\n" + "=" * 50)
    print("Summary:")
    for k in results:
        print(f"  {k}%: {results[k]['disparity']:.2f}x")
    print(f"\nOptimal: {optimal_k}% (disparity = {results[optimal_k]['disparity']:.2f}x)")

    # Compare to GPT-2
    print("\nCross-model comparison:")
    print(f"  GPT-2 optimal: 5% (45.39x)")
    print(f"  OPT-125M optimal: {optimal_k}% ({results[optimal_k]['disparity']:.2f}x)")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-007",
        "name": "OPT-125M Preservation",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "model": "facebook/opt-125m",
        "results": {str(k): v for k, v in results.items()},
        "optimal_k": optimal_k,
        "gpt2_optimal": 5,
        "pattern_consistent": optimal_k == 5,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp007_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
