#!/usr/bin/env python3
"""
EXP-039 Fixed: Intervention Study with Correct Weight Modification

The previous version had a bug where `flat.data = ...` didn't properly
persist weight changes. This version uses `copy_()` for in-place modification.

Hypothesis: Preserving top-k% of weights reduces disparity ratio.
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
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول ويجري.',
}

RESOURCE = {'en': 1.0, 'zh': 0.5, 'he': 0.2, 'ar': 0.35}


def main():
    print("=" * 60)
    print("EXP-039 Fixed: Intervention Study")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Save original weights using state_dict (guaranteed correct)
    print("2. Saving original weights via state_dict...")
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Count total weights
    total_weights = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)
    print(f"   Total weights: {total_weights:,}")

    # Collect all weight magnitudes for global thresholding
    print("3. Computing magnitude distribution...")
    all_mags = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_mags.append(param.data.abs().view(-1))
    all_mags = torch.cat(all_mags)
    sorted_mags, _ = all_mags.sort(descending=True)
    del all_mags
    gc.collect()

    def restore_model():
        """Restore model to original state."""
        model.load_state_dict(original_state)
        model.eval()

    def compute_ppl(text):
        """Compute perplexity for a single text."""
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()

    def aggressive_int4_quantize(preserve_pct):
        """
        Aggressive INT4 quantization with optional preservation.
        Uses copy_() for guaranteed in-place modification.
        """
        # Determine preservation threshold
        if preserve_pct > 0:
            n_preserve = max(1, int(total_weights * preserve_pct / 100))
            threshold = sorted_mags[n_preserve - 1].item()
        else:
            threshold = float('inf')  # Preserve nothing

        total_preserved = 0
        total_quantized = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue

                original = param.data.clone()
                flat = param.data.view(-1)
                abs_vals = flat.abs()

                # INT4 quantization: range [-8, 7]
                abs_max = abs_vals.max()
                if abs_max > 0:
                    scale = abs_max / 7.0
                    quantized = torch.round(flat / scale)
                    quantized = torch.clamp(quantized, -8, 7)
                    dequantized = quantized * scale

                    # Create preservation mask
                    preserve_mask = abs_vals >= threshold

                    # Apply quantization
                    new_weights = torch.where(preserve_mask, flat, dequantized)

                    # Use copy_() for guaranteed in-place modification
                    param.data.copy_(new_weights.view(param.shape))

                    total_preserved += preserve_mask.sum().item()
                    total_quantized += (~preserve_mask).sum().item()

        return total_preserved, total_quantized

    # Compute baseline
    print("\n4. Computing baseline (FP32)...")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = compute_ppl(text)
        print(f"   {lang}: {baseline[lang]:.2f}")

    # Test preservation levels
    print("\n5. Testing INT4 with preservation levels...")
    print("-" * 70)

    results = {}
    k_values = [0, 1, 5, 10, 20]

    for k in k_values:
        # Restore original weights
        restore_model()

        # Apply quantization
        preserved, quantized = aggressive_int4_quantize(k)

        # Compute post-quantization perplexity
        quant_ppl = {}
        for lang, text in TEXTS.items():
            quant_ppl[lang] = compute_ppl(text)

        # Compute degradation
        degradation = {}
        for lang in TEXTS:
            deg = (quant_ppl[lang] - baseline[lang]) / baseline[lang] * 100
            degradation[lang] = deg

        # Compute disparity
        hr_langs = [l for l in RESOURCE if RESOURCE[l] > 0.5]
        lr_langs = [l for l in RESOURCE if RESOURCE[l] <= 0.5]

        hr_avg = np.mean([degradation[l] for l in hr_langs])
        lr_avg = np.mean([degradation[l] for l in lr_langs])

        if hr_avg > 0:
            disparity = lr_avg / hr_avg
        else:
            disparity = float('inf')

        results[k] = {
            'preserved': preserved,
            'preserved_pct': preserved / total_weights * 100,
            'baseline': baseline.copy(),
            'quantized': quant_ppl.copy(),
            'degradation': degradation.copy(),
            'hr_avg': hr_avg,
            'lr_avg': lr_avg,
            'disparity': disparity,
        }

        print(f"k={k:3d}%: preserved={preserved/1e6:.1f}M ({preserved/total_weights*100:.1f}%)")
        print(f"       Baseline PPL: en={baseline['en']:.1f}, zh={baseline['zh']:.1f}, he={baseline['he']:.1f}, ar={baseline['ar']:.1f}")
        print(f"       Quant PPL:    en={quant_ppl['en']:.1f}, zh={quant_ppl['zh']:.1f}, he={quant_ppl['he']:.1f}, ar={quant_ppl['ar']:.1f}")
        print(f"       Degradation:  en={degradation['en']:+.0f}%, zh={degradation['zh']:+.0f}%, he={degradation['he']:+.0f}%, ar={degradation['ar']:+.0f}%")
        print(f"       HR avg={hr_avg:.0f}%, LR avg={lr_avg:.0f}%, Disparity={disparity:.2f}x")
        print()

        gc.collect()

    # Restore final
    restore_model()

    # Summary
    print("=" * 60)
    print("6. Summary Table")
    print("=" * 60)
    print(f"\n{'k%':<6} {'Preserved%':<12} {'HR Deg%':<12} {'LR Deg%':<12} {'Disparity':<10}")
    print("-" * 52)

    for k in sorted(results.keys()):
        r = results[k]
        print(f"{k:<6} {r['preserved_pct']:<12.1f} {r['hr_avg']:<+12.0f} {r['lr_avg']:<+12.0f} {r['disparity']:<10.2f}x")

    # Hypothesis evaluation
    print("\n" + "=" * 60)
    print("7. Hypothesis Evaluation")
    print("=" * 60)

    k_vals = sorted(results.keys())
    disparities = [results[k]['disparity'] for k in k_vals]

    # Filter finite values
    finite_pairs = [(k, d) for k, d in zip(k_vals, disparities) if np.isfinite(d)]

    if len(finite_pairs) >= 3:
        from scipy.stats import pearsonr
        ks, ds = zip(*finite_pairs)
        r, p = pearsonr(ks, ds)

        print(f"\nCorrelation (k% vs disparity): r = {r:.4f}, p = {p:.4f}")
        print(f"Prediction: r < 0 means more preservation reduces disparity")

        if r < -0.3:
            print("\nH-039: SUPPORTED - Preservation reduces disparity")
        elif r > 0.3:
            print("\nH-039: OPPOSITE EFFECT - Preservation increases disparity")
        else:
            print("\nH-039: INCONCLUSIVE - No clear relationship")

    # Improvement analysis
    if 0 in results and 20 in results:
        d0 = results[0]['disparity']
        d20 = results[20]['disparity']
        if np.isfinite(d0) and np.isfinite(d20):
            improvement = (d0 - d20) / d0 * 100
            print(f"\nDisparity at 0% preservation:  {d0:.2f}x")
            print(f"Disparity at 20% preservation: {d20:.2f}x")
            print(f"Improvement: {improvement:.1f}%")

    # Save results
    output = {
        "experiment_id": "EXP-039-fixed",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2",
        "hypothesis": "Preserving top-k% weights reduces low/high resource disparity",
        "results": {str(k): {
            'preserved': v['preserved'],
            'preserved_pct': v['preserved_pct'],
            'hr_avg_degradation': v['hr_avg'],
            'lr_avg_degradation': v['lr_avg'],
            'disparity': v['disparity'],
            'per_lang_degradation': v['degradation'],
        } for k, v in results.items()},
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"exp039_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
