#!/usr/bin/env python3
"""
Exp-009: Full 6-language coverage
Goal: Comprehensive test across all script types and resource levels
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund.',
    'fr': 'Le renard brun rapide saute par-dessus le chien paresseux.',
    'zh': '敏捷的棕色狐狸跳过懒狗。',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
}

METADATA = {
    'en': {'script': 'Latin', 'resource': 1.0},
    'de': {'script': 'Latin', 'resource': 0.85},
    'fr': {'script': 'Latin', 'resource': 0.80},
    'zh': {'script': 'Han', 'resource': 0.50},
    'he': {'script': 'Hebrew', 'resource': 0.15},
    'ar': {'script': 'Arabic', 'resource': 0.25},
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-009: Full Coverage")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(out.loss).item()

    # Baseline
    print("\nBaseline PPL:")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        m = METADATA[lang]
        print(f"  {lang} ({m['script']}, res={m['resource']:.2f}): {baseline[lang]:.2f}")

    # Quantize
    print("\nApplying INT4...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                q = torch.round(flat / scale).clamp(-8, 7)
                param.data.copy_((q * scale).view(param.shape))

    # Quantized
    print("Quantized PPL:")
    quant = {}
    for lang, text in TEXTS.items():
        quant[lang] = ppl(text)
        print(f"  {lang}: {quant[lang]:.2f}")

    # Degradation
    deg = {lang: (quant[lang] - baseline[lang]) / baseline[lang] * 100
           for lang in TEXTS}

    # Analysis by resource level
    hr_langs = [l for l in TEXTS if METADATA[l]['resource'] > 0.5]
    lr_langs = [l for l in TEXTS if METADATA[l]['resource'] <= 0.5]

    hr_avg = np.mean([deg[l] for l in hr_langs])
    lr_avg = np.mean([deg[l] for l in lr_langs])
    disparity = lr_avg / hr_avg if hr_avg > 0 else float('inf')

    print("\n" + "=" * 50)
    print("Analysis")
    print("=" * 50)

    print("\nBy Language:")
    for lang in sorted(deg.keys(), key=lambda x: deg[x], reverse=True):
        m = METADATA[lang]
        print(f"  {lang} ({m['script']}): {deg[lang]:+.0f}%")

    print(f"\nBy Resource Level:")
    print(f"  High-resource ({', '.join(hr_langs)}): {hr_avg:+.0f}%")
    print(f"  Low-resource ({', '.join(lr_langs)}): {lr_avg:+.0f}%")
    print(f"  Disparity ratio: {disparity:.2f}x")

    print("\nBy Script:")
    scripts = {}
    for lang in TEXTS:
        s = METADATA[lang]['script']
        if s not in scripts:
            scripts[s] = []
        scripts[s].append(deg[lang])
    for script, degs in sorted(scripts.items(), key=lambda x: np.mean(x[1]), reverse=True):
        print(f"  {script}: {np.mean(degs):+.0f}%")

    # Correlation analysis
    resources = [METADATA[l]['resource'] for l in TEXTS]
    degradations = [deg[l] for l in TEXTS]
    from scipy.stats import pearsonr
    r, p = pearsonr(resources, degradations)

    print(f"\nResource-Degradation Correlation:")
    print(f"  r = {r:.4f}, p = {p:.4f}")
    print(f"  Interpretation: {'Strong negative' if r < -0.5 else 'Moderate negative' if r < 0 else 'No'} correlation")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-009",
        "name": "Full Coverage",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "languages": list(TEXTS.keys()),
        "baseline_ppl": baseline,
        "quantized_ppl": quant,
        "degradation_pct": deg,
        "hr_avg": hr_avg,
        "lr_avg": lr_avg,
        "disparity_ratio": disparity,
        "correlation": {"r": r, "p": p},
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp009_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
