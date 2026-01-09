#!/usr/bin/env python3
"""
Exp-034: Multi-language validation of L0+L11 synergy
Goal: Test the synergy strategy across 6 languages
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund.',
    'fr': 'Le rapide renard brun saute par-dessus le chien paresseux.',
    'zh': '敏捷的棕色狐狸跳过了懒狗。',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}

RESOURCE_LEVEL = {'en': 'high', 'de': 'high', 'fr': 'high', 'zh': 'medium', 'ar': 'low', 'he': 'low'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-034: Multi-Language Synergy Validation")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_except(patterns):
        protected = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                if any(p in name for p in patterns):
                    protected += param.numel()
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))
        return protected

    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print("Baseline PPL:")
    for l, v in baseline.items():
        print(f"  {l}: {v:.1f}")

    strategies = [
        ("none", []),
        ("layer0", ["h.0."]),
        ("layer0+11", ["h.0.", "h.11."]),
        ("even_layers", ["h.0.", "h.2.", "h.4.", "h.6.", "h.8.", "h.10."]),
    ]

    results = {}

    print(f"\n{'Strategy':<12} {'%':>6}", end="")
    for l in TEXTS:
        print(f" {l:>6}", end="")
    print(f" {'Avg LR':>8}")
    print("-" * 70)

    for name, patterns in strategies:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        if en_deg > 0:
            disp = {l: deg[l] / en_deg for l in TEXTS}
        else:
            disp = {l: float('inf') for l in TEXTS}

        # Average for low-resource only
        lr_langs = [l for l in TEXTS if RESOURCE_LEVEL[l] == 'low']
        avg_lr = sum(disp[l] for l in lr_langs) / len(lr_langs)

        pct = protected / total * 100
        results[name] = {
            'protected': protected,
            'pct': pct,
            'disparity': disp,
            'degradation': deg,
            'avg_lr_disp': avg_lr,
        }

        print(f"{name:<12} {pct:>5.1f}%", end="")
        for l in TEXTS:
            print(f" {disp[l]:>5.1f}x", end="")
        print(f" {avg_lr:>7.1f}x")

    # Analysis by resource level
    print("\n" + "=" * 70)
    print("Analysis by Resource Level")
    print("=" * 70)

    for strat in ['none', 'layer0', 'layer0+11']:
        d = results[strat]['disparity']
        high = sum(d[l] for l in ['en', 'de', 'fr']) / 3
        med = d['zh']
        low = sum(d[l] for l in ['ar', 'he']) / 2
        print(f"\n{strat}:")
        print(f"  High-resource (en/de/fr): {high:.1f}x")
        print(f"  Medium (zh):              {med:.1f}x")
        print(f"  Low-resource (ar/he):     {low:.1f}x")

    # Improvement calculation
    print("\n" + "=" * 70)
    print("Improvement from L0+L11 Strategy")
    print("=" * 70)

    baseline_strat = results['none']
    synergy_strat = results['layer0+11']

    print(f"\n{'Language':<10} {'Baseline':>10} {'L0+L11':>10} {'Improvement':>12}")
    print("-" * 45)
    for l in TEXTS:
        base = baseline_strat['disparity'][l]
        syn = synergy_strat['disparity'][l]
        imp = base / syn if syn > 0 else float('inf')
        print(f"{l:<10} {base:>9.1f}x {syn:>9.1f}x {imp:>11.1f}x better")

    end = datetime.now()

    result = {
        "id": "Exp-034",
        "name": "Multi-Language Synergy Validation",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp034_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
