#!/usr/bin/env python3
"""Validate disparity pattern on OPT-125M."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog and runs through the forest.',
    'zh': '敏捷的棕色狐狸跳过懒狗，穿过森林寻找食物。',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ורץ ביער.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول ويجري.',
}

RESOURCE = {'en': 1.0, 'zh': 0.5, 'he': 0.15, 'ar': 0.25}


def main():
    print("OPT-125M Validation")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading OPT-125M...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    model.eval()

    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    total_weights = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)
    print(f"Total weights: {total_weights:,}")

    def compute_ppl(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()

    def restore():
        model.load_state_dict(original_state)
        model.eval()

    def quantize(preserve_pct):
        if preserve_pct > 0:
            all_mags = []
            for name, param in model.named_parameters():
                if 'weight' in name:
                    all_mags.append(param.data.abs().view(-1))
            all_mags = torch.cat(all_mags)
            n_preserve = int(len(all_mags) * preserve_pct / 100)
            sorted_mags, _ = all_mags.sort(descending=True)
            threshold = sorted_mags[n_preserve - 1].item()
            del all_mags, sorted_mags
        else:
            threshold = float('inf')

        preserved = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                flat = param.data.view(-1)
                abs_vals = flat.abs()
                abs_max = abs_vals.max()
                if abs_max > 0:
                    scale = abs_max / 7.0
                    quantized = torch.round(flat / scale)
                    quantized = torch.clamp(quantized, -8, 7)
                    dequantized = quantized * scale

                    if preserve_pct > 0:
                        mask = abs_vals >= threshold
                        new_weights = torch.where(mask, flat, dequantized)
                        preserved += mask.sum().item()
                    else:
                        new_weights = dequantized

                    param.data.copy_(new_weights.view(param.shape))

        return preserved

    # Baseline
    print("\nBaseline PPL:")
    baseline = {lang: compute_ppl(text) for lang, text in TEXTS.items()}
    for lang, ppl in baseline.items():
        print(f"  {lang}: {ppl:.2f}")

    # Test preservation levels
    results = {}
    for k in [0, 5, 10, 20]:
        print(f"\nTesting {k}% preservation...")
        restore()
        preserved = quantize(k)

        quant_ppl = {lang: compute_ppl(text) for lang, text in TEXTS.items()}
        degradation = {lang: (quant_ppl[lang] - baseline[lang]) / baseline[lang] * 100
                       for lang in TEXTS}

        hr_langs = [l for l, r in RESOURCE.items() if r > 0.5]
        lr_langs = [l for l, r in RESOURCE.items() if r <= 0.5]

        hr_avg = np.mean([degradation[l] for l in hr_langs])
        lr_avg = np.mean([degradation[l] for l in lr_langs])
        disparity = lr_avg / hr_avg if hr_avg > 0 else float('inf')

        results[k] = {
            'preserved': preserved,
            'hr_avg': hr_avg,
            'lr_avg': lr_avg,
            'disparity': disparity,
        }

        print(f"  Preserved: {preserved:,} ({preserved/total_weights*100:.1f}%)")
        print(f"  HR avg: {hr_avg:.0f}%, LR avg: {lr_avg:.0f}%")
        print(f"  Disparity: {disparity:.2f}x")

        gc.collect()

    # Summary
    print("\n" + "=" * 50)
    print("Summary - OPT-125M")
    print("=" * 50)
    print(f"\n{'k%':<6} {'Disparity':>12}")
    print("-" * 20)
    for k in sorted(results.keys()):
        print(f"{k:<6} {results[k]['disparity']:>12.2f}x")

    # Find optimal
    optimal_k = min(results.keys(), key=lambda k: results[k]['disparity'])
    print(f"\nOptimal: {optimal_k}% preservation (disparity = {results[optimal_k]['disparity']:.2f}x)")

    # Compare to GPT-2
    print("\nComparison to GPT-2:")
    print("  GPT-2 optimal: 5% (45.39x)")
    print(f"  OPT-125M optimal: {optimal_k}% ({results[optimal_k]['disparity']:.2f}x)")

    # Save
    output = {
        "experiment": "OPT-125M-validation",
        "timestamp": datetime.now().isoformat(),
        "model": "facebook/opt-125m",
        "baseline_ppl": baseline,
        "results": {str(k): v for k, v in results.items()},
        "optimal_k": optimal_k,
        "status": "SUCCESS"
    }

    output_file = Path(__file__).parent / "results" / f"opt125m_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return results


if __name__ == "__main__":
    main()
