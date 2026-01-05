#!/usr/bin/env python3
"""
Exp-003: Script diversity test
Goal: Compare degradation across Latin, Hebrew, and Han scripts
Hypothesis: Non-Latin scripts suffer more degradation
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',  # Latin
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',        # Hebrew
    'zh': '敏捷的棕色狐狸跳过懒狗。',                        # Han
}

SCRIPTS = {'en': 'Latin', 'he': 'Hebrew', 'zh': 'Han'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-003: Script Diversity")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(out.loss).item()

    def quantize():
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

    # Baseline
    print("\nBaseline PPL:")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        print(f"  {lang} ({SCRIPTS[lang]}): {baseline[lang]:.2f}")

    # Quantized
    print("\nQuantized PPL:")
    quantize()
    quant = {}
    for lang, text in TEXTS.items():
        quant[lang] = ppl(text)
        print(f"  {lang} ({SCRIPTS[lang]}): {quant[lang]:.2f}")

    # Degradation
    deg = {lang: (quant[lang] - baseline[lang]) / baseline[lang] * 100
           for lang in TEXTS}
    print("\nDegradation by script:")
    for lang in TEXTS:
        print(f"  {SCRIPTS[lang]}: {deg[lang]:+.0f}%")

    # Disparity vs English
    print("\nDisparity ratios (vs English):")
    disparities = {}
    for lang in ['he', 'zh']:
        disp = deg[lang] / deg['en'] if deg['en'] > 0 else float('inf')
        disparities[lang] = disp
        print(f"  {SCRIPTS[lang]}/{SCRIPTS['en']}: {disp:.2f}x")

    # Script ranking
    print("\nScript degradation ranking:")
    ranked = sorted(deg.items(), key=lambda x: x[1], reverse=True)
    for i, (lang, d) in enumerate(ranked, 1):
        print(f"  {i}. {SCRIPTS[lang]}: {d:+.0f}%")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-003",
        "name": "Script Diversity",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "hypothesis": "Non-Latin scripts suffer more degradation",
        "scripts_tested": list(SCRIPTS.values()),
        "baseline_ppl": baseline,
        "quantized_ppl": quant,
        "degradation_pct": deg,
        "disparity_vs_en": disparities,
        "ranking": [SCRIPTS[lang] for lang, _ in ranked],
        "hypothesis_supported": deg['he'] > deg['en'] and deg['zh'] > deg['en'],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp003_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    print(f"Hypothesis supported: {result['hypothesis_supported']}")
    return result


if __name__ == "__main__":
    main()
