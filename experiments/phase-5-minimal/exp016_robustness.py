#!/usr/bin/env python3
"""
Exp-016: Statistical robustness
Goal: Confidence intervals on key findings across models
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

MODELS = [
    ('gpt2', 'gpt2'),
    ('opt', 'facebook/opt-125m'),
]

N_RUNS = 3


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-016: Statistical Robustness")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {}

    for model_name, model_id in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        runs = []

        for run in range(N_RUNS):
            print(f"\n  Run {run+1}/{N_RUNS}:")

            # Fresh model load
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.eval()

            def ppl(text):
                inputs = tokenizer(text, return_tensors='pt')
                with torch.no_grad():
                    return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

            # Baseline
            baseline = {l: ppl(t) for l, t in TEXTS.items()}
            print(f"    Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

            # Quantize
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

            quant = {l: ppl(t) for l, t in TEXTS.items()}
            deg = {l: (quant[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
            disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

            print(f"    Disparity: {disp:.1f}x")
            runs.append(disp)

            del model
            gc.collect()

        # Statistics
        finite_runs = [r for r in runs if np.isfinite(r)]
        if finite_runs:
            mean_d = np.mean(finite_runs)
            std_d = np.std(finite_runs)
            cv = std_d / mean_d * 100 if mean_d > 0 else 0
        else:
            mean_d = std_d = cv = float('nan')

        all_results[model_name] = {
            'runs': runs,
            'mean': mean_d,
            'std': std_d,
            'cv': cv,
        }

        print(f"\n  {model_name} Summary:")
        print(f"    Mean: {mean_d:.1f}x ± {std_d:.1f}x")
        print(f"    CV: {cv:.1f}%")

    # Cross-model comparison
    print("\n" + "=" * 60)
    print("Cross-Model Summary")
    print("=" * 60)
    print(f"\n{'Model':<10} {'Mean':>12} {'Std':>10} {'CV':>8}")
    print("-" * 42)
    for m, r in all_results.items():
        print(f"{m:<10} {r['mean']:>12.1f}x {r['std']:>10.1f}x {r['cv']:>8.1f}%")

    # Conclusion
    print("\nConclusions:")
    for m, r in all_results.items():
        if r['cv'] < 5:
            print(f"  {m}: HIGHLY REPRODUCIBLE")
        elif r['cv'] < 20:
            print(f"  {m}: REPRODUCIBLE")
        else:
            print(f"  {m}: VARIABLE")

    end = datetime.now()

    result = {
        "id": "Exp-016",
        "name": "Statistical Robustness",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "n_runs": N_RUNS,
        "models": list(all_results.keys()),
        "results": all_results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp016_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
