#!/usr/bin/env python3
"""
Exp-008: Arabic language addition
Goal: Test another RTL script (Arabic vs Hebrew)
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
}

SCRIPTS = {'en': 'Latin', 'he': 'Hebrew', 'ar': 'Arabic'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-008: Arabic Addition")
    print("=" * 40)

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
        print(f"  {lang} ({SCRIPTS[lang]}): {baseline[lang]:.2f}")

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
        print(f"  {lang} ({SCRIPTS[lang]}): {quant[lang]:.2f}")

    # Degradation
    deg = {lang: (quant[lang] - baseline[lang]) / baseline[lang] * 100
           for lang in TEXTS}
    print("\nDegradation:")
    for lang in TEXTS:
        print(f"  {SCRIPTS[lang]}: {deg[lang]:+.0f}%")

    # Disparity vs English
    print("\nDisparity vs English:")
    disparities = {}
    for lang in ['he', 'ar']:
        d = deg[lang] / deg['en'] if deg['en'] > 0 else float('inf')
        disparities[lang] = d
        print(f"  {SCRIPTS[lang]}: {d:.2f}x")

    # RTL comparison
    print("\nRTL Script Comparison:")
    print(f"  Hebrew: {deg['he']:+.0f}%")
    print(f"  Arabic: {deg['ar']:+.0f}%")
    print(f"  Ratio (he/ar): {deg['he']/deg['ar']:.2f}x" if deg['ar'] > 0 else "  N/A")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-008",
        "name": "Arabic Addition",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "scripts_tested": list(SCRIPTS.values()),
        "baseline_ppl": baseline,
        "quantized_ppl": quant,
        "degradation_pct": deg,
        "disparity_vs_en": disparities,
        "rtl_comparison": {
            "hebrew_deg": deg['he'],
            "arabic_deg": deg['ar'],
            "hebrew_arabic_ratio": deg['he'] / deg['ar'] if deg['ar'] > 0 else None,
        },
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp008_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
