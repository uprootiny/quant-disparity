#!/usr/bin/env python3
"""
Exp-018: BLOOM-560M validation
Goal: Test disparity on truly multilingual model (trained on 46 languages)
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
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
    'zh': '敏捷的棕色狐狸跳过了懒狗。',
    'fr': 'Le renard brun rapide saute par-dessus le chien paresseux.',
}

HR_LANGS = ['en', 'fr']
LR_LANGS = ['he', 'ar', 'zh']


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-018: BLOOM-560M Validation")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading BLOOM-560M (this may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
    model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m')
    model.eval()

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quantize_all():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))

    # Baseline
    print("\nComputing baseline perplexities...")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        print(f"  {lang}: {baseline[lang]:.1f}")

    # Quantize
    print("\nApplying INT4 quantization...")
    quantize_all()

    # Post-quantization
    print("Computing quantized perplexities...")
    quantized = {}
    for lang, text in TEXTS.items():
        quantized[lang] = ppl(text)
        print(f"  {lang}: {quantized[lang]:.1f}")

    # Calculate degradation
    degradation = {}
    for lang in TEXTS:
        deg = (quantized[lang] - baseline[lang]) / baseline[lang] * 100
        degradation[lang] = deg

    print("\n" + "=" * 60)
    print("Degradation Analysis")
    print("=" * 60)

    print(f"\n{'Language':<10} {'Baseline':>12} {'Quantized':>12} {'Degradation':>12}")
    print("-" * 50)
    for lang in TEXTS:
        print(f"{lang:<10} {baseline[lang]:>12.1f} {quantized[lang]:>12.1f} {degradation[lang]:>+11.0f}%")

    # Calculate disparity
    hr_deg = np.mean([degradation[l] for l in HR_LANGS])
    lr_deg = np.mean([degradation[l] for l in LR_LANGS])
    disparity = lr_deg / hr_deg if hr_deg > 0 else float('inf')

    print(f"\nHigh-resource mean: {hr_deg:+.0f}%")
    print(f"Low-resource mean:  {lr_deg:+.0f}%")
    print(f"Disparity ratio:    {disparity:.1f}x")

    # Compare to GPT-2/OPT
    print("\n" + "=" * 60)
    print("Cross-Model Comparison")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Disparity':>12}")
    print("-" * 30)
    print(f"{'GPT-2':<15} {'214x':>12}")
    print(f"{'OPT-125M':<15} {'153x':>12}")
    print(f"{'BLOOM-560M':<15} {f'{disparity:.1f}x':>12}")

    if disparity < 50:
        conclusion = "BLOOM shows MUCH LOWER disparity - multilingual training helps!"
    elif disparity < 100:
        conclusion = "BLOOM shows LOWER disparity than English-centric models"
    else:
        conclusion = "BLOOM shows similar disparity despite multilingual training"

    print(f"\nConclusion: {conclusion}")

    end = datetime.now()

    result = {
        "id": "Exp-018",
        "name": "BLOOM-560M Validation",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "model": "bigscience/bloom-560m",
        "baseline": baseline,
        "quantized": quantized,
        "degradation": degradation,
        "hr_mean_deg": hr_deg,
        "lr_mean_deg": lr_deg,
        "disparity": disparity,
        "conclusion": conclusion,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp018_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
