#!/usr/bin/env python3
"""Run B-020: 20% Preservation test."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog and runs through the forest looking for food.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund und rennt durch den Wald.',
    'fr': 'Le renard brun rapide saute par-dessus le chien paresseux et court dans la forêt.',
    'zh': '敏捷的棕色狐狸跳过懒狗，穿过森林寻找食物。',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ורץ ביער בחיפוש אחר אוכל.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول ويجري عبر الغابة.',
}

RESOURCE_LEVELS = {'en': 1.0, 'de': 0.85, 'fr': 0.80, 'zh': 0.50, 'he': 0.15, 'ar': 0.25}


def main():
    print("B-020: 20% Preservation Test")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Save state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Baseline
    print("\nBaseline PPL:")
    baseline = {}
    for lang, text in TEXTS.items():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        baseline[lang] = torch.exp(outputs.loss).item()
        print(f"  {lang}: {baseline[lang]:.2f}")

    # Quantize with 20% preservation
    print("\nApplying INT4 with 20% preservation...")

    total_weights = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)

    # Compute threshold
    all_mags = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_mags.append(param.data.abs().view(-1))
    all_mags = torch.cat(all_mags)
    n_preserve = int(len(all_mags) * 0.20)
    sorted_mags, _ = all_mags.sort(descending=True)
    threshold = sorted_mags[n_preserve - 1].item()
    del all_mags, sorted_mags

    preserved_count = 0
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

                mask = abs_vals >= threshold
                new_weights = torch.where(mask, flat, dequantized)
                param.data.copy_(new_weights.view(param.shape))
                preserved_count += mask.sum().item()

    print(f"Preserved: {preserved_count:,} ({preserved_count/total_weights*100:.1f}%)")

    # Post-quantization PPL
    print("\nQuantized PPL:")
    quant_ppl = {}
    degradation = {}
    for lang, text in TEXTS.items():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        quant_ppl[lang] = torch.exp(outputs.loss).item()
        degradation[lang] = (quant_ppl[lang] - baseline[lang]) / baseline[lang] * 100
        print(f"  {lang}: {quant_ppl[lang]:.2f} ({degradation[lang]:+.0f}%)")

    # Disparity
    hr = [l for l, r in RESOURCE_LEVELS.items() if r > 0.5]
    lr = [l for l, r in RESOURCE_LEVELS.items() if r <= 0.5]

    hr_avg = np.mean([degradation[l] for l in hr])
    lr_avg = np.mean([degradation[l] for l in lr])
    disparity = lr_avg / hr_avg if hr_avg > 0 else float('inf')

    print(f"\nHR avg: {hr_avg:.0f}%, LR avg: {lr_avg:.0f}%")
    print(f"Disparity ratio: {disparity:.2f}x")

    # Save result
    result = {
        "id": "B-020",
        "name": "20% Preservation",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2",
        "preserve_pct": 20,
        "weights_preserved": preserved_count,
        "weights_preserved_pct_actual": preserved_count / total_weights * 100,
        "degradation_pct": degradation,
        "hr_avg_degradation": hr_avg,
        "lr_avg_degradation": lr_avg,
        "disparity_ratio": disparity,
        "status": "SUCCESS"
    }

    output_file = Path(__file__).parent / "results" / f"B-020_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {output_file}")
    return result


if __name__ == "__main__":
    main()
