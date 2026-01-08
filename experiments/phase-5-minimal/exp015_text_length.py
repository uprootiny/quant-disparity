#!/usr/bin/env python3
"""
Exp-015: Text length sensitivity
Goal: Rule out short-text artifacts in disparity measurement
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# Texts of varying lengths
TEXTS_SHORT = {
    'en': 'The fox.',
    'he': 'השועל.',
}

TEXTS_MEDIUM = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}

TEXTS_LONG = {
    'en': 'The quick brown fox jumps over the lazy dog and runs through the forest looking for food. It is a beautiful day.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ורץ ביער בחיפוש אחר אוכל. זהו יום יפה.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-015: Text Length Sensitivity")
    print("=" * 50)

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

    def quantize():
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

    text_sets = [
        ('short', TEXTS_SHORT),
        ('medium', TEXTS_MEDIUM),
        ('long', TEXTS_LONG),
    ]

    results = {}
    print(f"\n{'Length':<10} {'en_tok':<8} {'he_tok':<8} {'Disp':>10}")
    print("-" * 40)

    for name, texts in text_sets:
        restore()

        # Count tokens
        en_tok = len(tokenizer(texts['en'])['input_ids'])
        he_tok = len(tokenizer(texts['he'])['input_ids'])

        # Baseline
        baseline = {l: ppl(t) for l, t in texts.items()}

        # Quantize
        quantize()
        quant = {l: ppl(t) for l, t in texts.items()}

        # Degradation
        deg = {l: (quant[l] - baseline[l]) / baseline[l] * 100 for l in texts}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        results[name] = {
            'en_tokens': en_tok,
            'he_tokens': he_tok,
            'baseline': baseline,
            'quantized': quant,
            'degradation': deg,
            'disparity': disp,
        }

        print(f"{name:<10} {en_tok:<8} {he_tok:<8} {disp:>10.1f}x")

    # Analyze consistency
    print("\n" + "=" * 50)
    print("Consistency Analysis")
    print("=" * 50)

    disparities = [r['disparity'] for r in results.values()]
    mean_d = np.mean(disparities)
    std_d = np.std(disparities)
    cv = std_d / mean_d * 100 if mean_d > 0 else 0

    print(f"\nDisparities: {[f'{d:.1f}x' for d in disparities]}")
    print(f"Mean: {mean_d:.1f}x")
    print(f"Std:  {std_d:.1f}x")
    print(f"CV:   {cv:.1f}%")

    if cv < 20:
        conclusion = "CONSISTENT - Text length does not significantly affect disparity"
    elif cv < 50:
        conclusion = "MODERATE VARIANCE - Some sensitivity to text length"
    else:
        conclusion = "HIGH VARIANCE - Disparity measurement is text-length dependent"

    print(f"\nConclusion: {conclusion}")

    end = datetime.now()

    result = {
        "id": "Exp-015",
        "name": "Text Length Sensitivity",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "mean_disparity": mean_d,
        "std_disparity": std_d,
        "cv_pct": cv,
        "conclusion": conclusion,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp015_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
