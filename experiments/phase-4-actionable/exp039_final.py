#!/usr/bin/env python3
"""
EXP-039 Final: Memory-Efficient Intervention Study

Key insight from v3: INT4 quantization causes 52x disparity ratio.
This experiment tests if weight preservation reduces that disparity.

Memory optimization:
- Load model once
- Save weights once
- Modify in-place
- Restore from saved weights
"""

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
}

RESOURCE = {'en': 1.0, 'zh': 0.5, 'he': 0.2}


def main():
    print("=" * 60)
    print("EXP-039 Final: Intervention Study")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Save original weights
    print("2. Saving original weights...")
    original_weights = {}
    weight_magnitudes = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            original_weights[name] = param.data.clone()
            weight_magnitudes.append(param.data.abs().view(-1))

    all_magnitudes = torch.cat(weight_magnitudes)
    total_weights = len(all_magnitudes)
    print(f"   Total weights: {total_weights:,}")

    # Sort magnitudes once
    sorted_mags, _ = all_magnitudes.sort(descending=True)

    def restore_weights():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_weights:
                    param.data.copy_(original_weights[name])

    def compute_ppl(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()

    def quantize_with_preservation(preserve_pct):
        """Quantize to INT4, preserving top preserve_pct% weights."""
        if preserve_pct > 0:
            n_preserve = max(1, int(total_weights * preserve_pct / 100))
            threshold = sorted_mags[n_preserve - 1].item()
        else:
            threshold = float('inf')

        preserved = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue

                flat = param.view(-1)
                abs_vals = flat.abs()
                preserve_mask = abs_vals >= threshold

                abs_max = abs_vals.max()
                if abs_max > 0:
                    scale = abs_max / 7
                    original = flat.clone()

                    quantized = torch.round(flat / scale)
                    quantized = torch.clamp(quantized, -8, 7)
                    flat.data = quantized * scale

                    # Restore preserved
                    flat.data[preserve_mask] = original[preserve_mask]

                preserved += preserve_mask.sum().item()

        return preserved

    # Baseline
    print("\n3. Computing baseline...")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = compute_ppl(text)
        print(f"   {lang}: {baseline[lang]:.2f}")

    # Test conditions
    print("\n4. Testing preservation levels...")
    print("-" * 65)

    results = {}

    for k in [0, 1, 5, 10, 20, 50]:
        restore_weights()
        preserved = quantize_with_preservation(k)

        quant = {}
        for lang, text in TEXTS.items():
            quant[lang] = compute_ppl(text)

        # Compute metrics
        degradation = {lang: (quant[lang] - baseline[lang]) / baseline[lang]
                      for lang in TEXTS}

        hr_degs = [degradation[l] for l in RESOURCE if RESOURCE[l] > 0.5]
        lr_degs = [degradation[l] for l in RESOURCE if RESOURCE[l] <= 0.5]

        hr_avg = np.mean(hr_degs)
        lr_avg = np.mean(lr_degs)
        ratio = lr_avg / hr_avg if hr_avg > 0 else float('inf')

        results[k] = {
            'preserved': preserved,
            'preserved_pct': preserved / total_weights * 100,
            'hr_deg': hr_avg,
            'lr_deg': lr_avg,
            'disparity': ratio,
            'degradation': degradation,
        }

        print(f"k={k:3d}%: preserved={preserved/1e6:.1f}M, "
              f"HR_deg={hr_avg*100:+.0f}%, LR_deg={lr_avg*100:+.0f}%, "
              f"disparity={ratio:.2f}x")

        gc.collect()

    # Restore for cleanliness
    restore_weights()

    # Summary
    print("\n" + "=" * 60)
    print("5. Summary")
    print("=" * 60)
    print(f"\n{'k%':<8} {'Preserved%':<12} {'HR Deg':<12} {'LR Deg':<12} {'Disparity':<10}")
    print("-" * 54)

    for k in sorted(results.keys()):
        r = results[k]
        print(f"{k:<8} {r['preserved_pct']:<12.1f} "
              f"{r['hr_deg']*100:<+12.0f}% {r['lr_deg']*100:<+12.0f}% "
              f"{r['disparity']:<10.2f}x")

    # Hypothesis test
    print("\n" + "=" * 60)
    print("6. Hypothesis Evaluation")
    print("=" * 60)

    k_vals = sorted(results.keys())
    disparities = [results[k]['disparity'] for k in k_vals]

    # Handle inf values
    finite_pairs = [(k, d) for k, d in zip(k_vals, disparities) if np.isfinite(d)]

    if len(finite_pairs) >= 3:
        from scipy.stats import pearsonr
        ks, ds = zip(*finite_pairs)
        r, p = pearsonr(ks, ds)

        print(f"\nCorrelation (k vs disparity): r = {r:.4f}, p = {p:.4f}")
        print(f"Prediction: r < 0 (more preservation → less disparity)")

        supported = r < -0.3
        print(f"\nH-039: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    # Improvement analysis
    d0 = results[0]['disparity']
    d50 = results[50]['disparity']

    print(f"\nDisparity at 0% preservation:  {d0:.2f}x")
    print(f"Disparity at 50% preservation: {d50:.2f}x")

    if np.isfinite(d0) and np.isfinite(d50):
        improvement = (d0 - d50) / d0 * 100
        print(f"Improvement: {improvement:.1f}%")

    # Save results
    output = {
        "experiment_id": "EXP-039-final",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2",
        "results": {str(k): {kk: vv for kk, vv in v.items() if kk != 'degradation'}
                   for k, v in results.items()},
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"exp039_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
