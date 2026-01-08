#!/usr/bin/env python3
"""
Exp-024: Arabic and Chinese focus
Goal: Deep-dive into non-Hebrew low-resource languages
"""

import json
from pathlib import Path
from datetime import datetime
import torch

# Multiple text samples for Arabic and Chinese
TEXTS = {
    'en': [
        'The quick brown fox jumps over the lazy dog.',
        'Hello world, how are you today?',
        'The weather is beautiful this morning.',
    ],
    'ar': [
        'الثعلب البني السريع يقفز فوق الكلب الكسول.',
        'مرحبا بالعالم، كيف حالك اليوم؟',
        'الطقس جميل هذا الصباح.',
    ],
    'zh': [
        '敏捷的棕色狐狸跳过了懒狗。',
        '你好世界，今天你好吗？',
        '今天早上天气很好。',
    ],
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-024: Arabic/Chinese Focus")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Analyze tokenization
    print("Tokenization analysis:")
    for lang, texts in TEXTS.items():
        total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
        avg_tokens = total_tokens / len(texts)
        print(f"  {lang}: avg {avg_tokens:.1f} tokens/sentence")

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

    # Per-sentence analysis
    print("\n" + "=" * 60)
    print("Per-Sentence Baseline Analysis")
    print("=" * 60)

    baseline = {}
    for lang, texts in TEXTS.items():
        baseline[lang] = [ppl(t) for t in texts]
        print(f"\n{lang}:")
        for i, (text, p) in enumerate(zip(texts, baseline[lang])):
            print(f"  [{i}] PPL={p:.1f}: {text[:40]}...")

    # Test strategies
    strategies = [
        ("none", []),
        ("layer0", ["h.0."]),
        ("layer0+2", ["h.0.", "h.2."]),
    ]

    results = {}

    for strat_name, patterns in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat_name}")
        print("=" * 60)

        restore()
        quant_except(patterns)

        quantized = {}
        degradation = {}

        for lang, texts in TEXTS.items():
            quantized[lang] = [ppl(t) for t in texts]
            degradation[lang] = [
                (q - b) / b * 100 if b > 0 else float('inf')
                for q, b in zip(quantized[lang], baseline[lang])
            ]

        # Average degradation per language
        avg_deg = {lang: sum(d) / len(d) for lang, d in degradation.items()}

        print(f"\n{'Lang':<6} {'Avg Baseline':>12} {'Avg Quant':>12} {'Avg Deg':>12}")
        print("-" * 45)
        for lang in TEXTS:
            avg_b = sum(baseline[lang]) / len(baseline[lang])
            avg_q = sum(quantized[lang]) / len(quantized[lang])
            print(f"{lang:<6} {avg_b:>12.1f} {avg_q:>12.1f} {avg_deg[lang]:>+11.0f}%")

        # Disparity vs English
        en_deg = avg_deg['en']
        ar_disp = avg_deg['ar'] / en_deg if en_deg > 0 else float('inf')
        zh_disp = avg_deg['zh'] / en_deg if en_deg > 0 else float('inf')

        print(f"\nDisparity vs English:")
        print(f"  Arabic:  {ar_disp:.1f}x")
        print(f"  Chinese: {zh_disp:.1f}x")

        results[strat_name] = {
            'avg_degradation': avg_deg,
            'ar_disparity': ar_disp,
            'zh_disparity': zh_disp,
        }

    # Summary
    print("\n" + "=" * 60)
    print("Strategy Comparison for Arabic/Chinese")
    print("=" * 60)
    print(f"\n{'Strategy':<12} {'Arabic Disp':>12} {'Chinese Disp':>12}")
    print("-" * 40)
    for strat, data in results.items():
        print(f"{strat:<12} {data['ar_disparity']:>11.1f}x {data['zh_disparity']:>11.1f}x")

    end = datetime.now()

    result = {
        "id": "Exp-024",
        "name": "Arabic/Chinese Focus",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp024_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
