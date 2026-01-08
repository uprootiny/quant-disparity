#!/usr/bin/env python3
"""
Exp-023: Longer text validation
Goal: Confirm disparity patterns hold with paragraph-length texts
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Paragraph-length texts (roughly same content, ~50-100 tokens each)
TEXTS = {
    'en': """The quick brown fox jumps over the lazy dog. This is a longer passage
    that contains multiple sentences to test whether our quantization disparity
    findings hold with more substantial text inputs. The fox continues to run
    through the forest, searching for food and adventure.""",

    'he': """השועל החום המהיר קופץ מעל הכלב העצלן. זהו קטע ארוך יותר שמכיל מספר
    משפטים כדי לבדוק האם ממצאי הפערים בכימות שלנו מחזיקים מעמד עם קלטי טקסט
    משמעותיים יותר. השועל ממשיך לרוץ ביער, מחפש אוכל והרפתקאות.""",

    'ar': """الثعلب البني السريع يقفز فوق الكلب الكسول. هذا مقطع أطول يحتوي على
    عدة جمل لاختبار ما إذا كانت نتائج تباين التكميم لدينا صحيحة مع مدخلات نصية
    أكثر جوهرية. يستمر الثعلب في الجري عبر الغابة باحثاً عن الطعام والمغامرة.""",

    'zh': """敏捷的棕色狐狸跳过了懒狗。这是一段较长的文字，包含多个句子，
    用于测试我们的量化差异发现是否适用于更实质性的文本输入。狐狸继续
    在森林中奔跑，寻找食物和冒险。""",
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-023: Longer Text Validation")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Check token counts
    print("Token counts:")
    for lang, text in TEXTS.items():
        tokens = len(tokenizer.encode(text))
        print(f"  {lang}: {tokens} tokens")

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
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

    # Baseline
    print("\nBaseline perplexities:")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        print(f"  {lang}: {baseline[lang]:.1f}")

    # Test with no protection and layer0 protection
    strategies = [
        ("none", []),
        ("layer0", ["h.0."]),
    ]

    results = {}

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

        en_deg = degradation['en']
        hr_mean = degradation['en']  # Just English for comparison
        lr_langs = ['he', 'ar', 'zh']
        lr_mean = np.mean([degradation[l] for l in lr_langs])

        disparity = lr_mean / hr_mean if hr_mean > 0 else float('inf')

        print(f"\nEnglish degradation: {en_deg:+.0f}%")
        print(f"LR mean degradation: {lr_mean:+.0f}%")
        print(f"Disparity ratio: {disparity:.1f}x")

        # Per-language disparity
        print("\nPer-language disparity (vs English):")
        for lang in TEXTS:
            if lang != 'en':
                lang_disp = degradation[lang] / en_deg if en_deg > 0 else float('inf')
                print(f"  {lang}: {lang_disp:.1f}x")

        results[strat_name] = {
            'baseline': baseline,
            'quantized': quantized,
            'degradation': degradation,
            'disparity': disparity,
        }

    # Compare to short text results
    print("\n" + "=" * 60)
    print("Comparison: Short vs Long Texts")
    print("=" * 60)
    print("\nShort text baseline disparity: 79x (Exp-022)")
    print(f"Long text baseline disparity:  {results['none']['disparity']:.1f}x")
    print(f"\nShort text Layer0 disparity: 3.8x (Exp-022)")
    print(f"Long text Layer0 disparity:  {results['layer0']['disparity']:.1f}x")

    consistent = abs(results['none']['disparity'] - 79) / 79 < 0.5  # Within 50%
    print(f"\nConsistency check: {'PASS' if consistent else 'DIFFERENT PATTERN'}")

    end = datetime.now()

    result = {
        "id": "Exp-023",
        "name": "Longer Text Validation",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "consistency": consistent,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp023_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
