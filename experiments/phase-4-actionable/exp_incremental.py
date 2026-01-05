#!/usr/bin/env python3
"""
Incremental preservation test - simple and fast.
Tests 0% and 5% preservation to validate optimal finding.
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps.',
    'he': 'השועל החום קופץ.',
}


def main():
    print("Incremental Preservation Test")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}
    total = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)

    # Collect magnitudes once
    all_mags = []
    for n, p in model.named_parameters():
        if 'weight' in n:
            all_mags.append(p.data.abs().view(-1))
    all_mags = torch.cat(all_mags)
    sorted_mags, _ = all_mags.sort(descending=True)

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
    base = {lang: ppl(text) for lang, text in TEXTS.items()}
    for l, v in base.items():
        print(f"  {l}: {v:.2f}")

    # Test 0% and 5%
    results = {}
    for k in [0, 5]:
        print(f"\n{k}% preservation:")
        restore()
        kept = quant(k)

        q_ppl = {lang: ppl(text) for lang, text in TEXTS.items()}
        deg = {l: (q_ppl[l] - base[l]) / base[l] * 100 for l in TEXTS}

        disparity = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        results[k] = {
            'kept': kept,
            'deg_en': deg['en'],
            'deg_he': deg['he'],
            'disparity': disparity
        }

        print(f"  Kept: {kept:,}")
        print(f"  en: {deg['en']:+.0f}%, he: {deg['he']:+.0f}%")
        print(f"  Disparity: {disparity:.2f}x")

        gc.collect()

    # Summary
    print("\n" + "=" * 40)
    print("Summary:")
    for k in results:
        print(f"  {k}%: disparity = {results[k]['disparity']:.2f}x")

    improvement = (results[0]['disparity'] - results[5]['disparity']) / results[0]['disparity'] * 100
    print(f"\nImprovement 0%→5%: {improvement:.1f}%")

    # Save
    out = {
        "id": "incremental-preservation",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2",
        "results": {str(k): v for k, v in results.items()},
        "improvement_pct": improvement,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "results" / f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {out_file}")
    print("✓ Experiment completed")
    return results


if __name__ == "__main__":
    main()
