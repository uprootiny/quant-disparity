#!/usr/bin/env python3
"""
Exp-004: Statistical validation
Goal: Run 3 trials to compute mean and std of disparity
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
}

N_TRIALS = 3


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-004: Statistical Validation")
    print(f"Running {N_TRIALS} trials")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    trials = []

    for trial in range(N_TRIALS):
        print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")

        # Fresh model load each trial
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        model.eval()

        def ppl(text):
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                out = model(**inputs, labels=inputs['input_ids'])
            return torch.exp(out.loss).item()

        # Baseline
        baseline = {lang: ppl(text) for lang, text in TEXTS.items()}
        print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

        # Quantize
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
        quant = {lang: ppl(text) for lang, text in TEXTS.items()}
        print(f"Quantized: en={quant['en']:.1f}, he={quant['he']:.1f}")

        # Degradation
        deg_en = (quant['en'] - baseline['en']) / baseline['en'] * 100
        deg_he = (quant['he'] - baseline['he']) / baseline['he'] * 100
        disparity = deg_he / deg_en if deg_en > 0 else float('inf')

        print(f"Disparity: {disparity:.2f}x")

        trials.append({
            'trial': trial + 1,
            'baseline': baseline,
            'quantized': quant,
            'deg_en': deg_en,
            'deg_he': deg_he,
            'disparity': disparity,
        })

        # Cleanup
        del model
        gc.collect()

    # Statistics
    disparities = [t['disparity'] for t in trials]
    mean_disp = np.mean(disparities)
    std_disp = np.std(disparities)

    print("\n" + "=" * 40)
    print("Statistical Summary")
    print("=" * 40)
    print(f"Trials: {N_TRIALS}")
    print(f"Disparities: {[f'{d:.1f}x' for d in disparities]}")
    print(f"Mean: {mean_disp:.2f}x")
    print(f"Std:  {std_disp:.2f}x")
    print(f"CV:   {std_disp/mean_disp*100:.1f}%" if mean_disp > 0 else "N/A")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-004",
        "name": "Statistical Validation",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "n_trials": N_TRIALS,
        "trials": trials,
        "disparity_mean": mean_disp,
        "disparity_std": std_disp,
        "disparity_cv_pct": std_disp / mean_disp * 100 if mean_disp > 0 else None,
        "conclusion": "Disparity is consistent" if std_disp / mean_disp < 0.1 else "High variance",
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp004_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
