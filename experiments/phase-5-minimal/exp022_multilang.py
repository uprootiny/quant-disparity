#!/usr/bin/env python3
"""
Exp-022: Multi-language validation
Goal: Test disparity across all 6 languages with best strategies
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund.',
    'fr': 'Le renard brun rapide saute par-dessus le chien paresseux.',
    'zh': '敏捷的棕色狐狸跳过了懒狗。',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}

HR_LANGS = ['en', 'de', 'fr']
LR_LANGS = ['zh', 'ar', 'he']


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-022: Multi-Language Validation")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_except(patterns):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                if any(p in name for p in patterns):
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))

    # Baseline for all languages
    print("Computing baseline perplexities...")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        print(f"  {lang}: {baseline[lang]:.1f}")

    # Test strategies
    strategies = [
        ("none", []),
        ("layer0", ["h.0."]),
        ("layer0+layer2", ["h.0.", "h.2."]),  # Avoid L1
    ]

    all_results = {}

    for strat_name, patterns in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat_name}")
        print("=" * 60)

        restore()
        quant_except(patterns)

        quantized = {l: ppl(t) for l, t in TEXTS.items()}
        degradation = {l: (quantized[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}

        print(f"\n{'Lang':<6} {'Baseline':>10} {'Quantized':>12} {'Degradation':>12}")
        print("-" * 45)
        for lang in TEXTS:
            print(f"{lang:<6} {baseline[lang]:>10.1f} {quantized[lang]:>12.1f} {degradation[lang]:>+11.0f}%")

        hr_deg = np.mean([degradation[l] for l in HR_LANGS])
        lr_deg = np.mean([degradation[l] for l in LR_LANGS])
        disparity = lr_deg / hr_deg if hr_deg > 0 else float('inf')

        print(f"\nHigh-resource mean: {hr_deg:+.0f}%")
        print(f"Low-resource mean:  {lr_deg:+.0f}%")
        print(f"Disparity ratio:    {disparity:.1f}x")

        # Per-language disparity vs English
        print(f"\nPer-language disparity (vs English):")
        en_deg = degradation['en']
        for lang in TEXTS:
            if lang != 'en':
                lang_disp = degradation[lang] / en_deg if en_deg > 0 else float('inf')
                print(f"  {lang}: {lang_disp:.1f}x")

        all_results[strat_name] = {
            'baseline': baseline,
            'quantized': quantized,
            'degradation': degradation,
            'hr_mean': hr_deg,
            'lr_mean': lr_deg,
            'disparity': disparity,
        }

    # Summary
    print("\n" + "=" * 60)
    print("Strategy Comparison")
    print("=" * 60)
    print(f"\n{'Strategy':<15} {'HR Deg':>10} {'LR Deg':>12} {'Disparity':>10}")
    print("-" * 50)
    for strat, data in all_results.items():
        print(f"{strat:<15} {data['hr_mean']:>+9.0f}% {data['lr_mean']:>+11.0f}% {data['disparity']:>10.1f}x")

    end = datetime.now()

    result = {
        "id": "Exp-022",
        "name": "Multi-Language Validation",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "languages": list(TEXTS.keys()),
        "hr_langs": HR_LANGS,
        "lr_langs": LR_LANGS,
        "results": all_results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp022_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
