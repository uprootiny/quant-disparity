#!/usr/bin/env python3
"""
Exp-028: Gradient-based weight selection
Goal: Compare magnitude vs gradient-based selection for critical weights
Hypothesis: Gradient sensitivity may identify multilingual-critical weights better
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-028: Gradient-Based Selection")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    total_params = sum(p.numel() for p in model.parameters())

    # Compute gradients on multilingual calibration set
    print("Computing gradients on multilingual calibration set...")
    model.train()  # Enable gradient computation

    # Accumulate gradients across languages
    gradient_importance = {}
    for lang, text in TEXTS.items():
        model.zero_grad()
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs, labels=inputs['input_ids'])
        outputs.loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in gradient_importance:
                    gradient_importance[name] = torch.zeros_like(param.data)
                gradient_importance[name] += param.grad.abs()

    model.eval()

    # Compute magnitude importance
    print("Computing magnitude importance...")
    magnitude_importance = {}
    for name, param in model.named_parameters():
        magnitude_importance[name] = param.data.abs()

    # Save original state
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_with_selection(importance_dict, keep_pct):
        """Quantize all weights except top keep_pct% by importance."""
        # Flatten all importances
        all_importances = []
        for name, imp in importance_dict.items():
            all_importances.append(imp.flatten())
        all_importances = torch.cat(all_importances)

        # Find threshold
        k = int(len(all_importances) * keep_pct / 100)
        if k > 0:
            threshold = torch.topk(all_importances, k).values[-1].item()
        else:
            threshold = float('inf')

        protected = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in importance_dict:
                    continue

                imp = importance_dict[name]
                mask = imp >= threshold

                # Keep high-importance weights, quantize rest
                flat = param.view(-1)
                imp_flat = imp.view(-1)

                for i in range(len(flat)):
                    if imp_flat[i] >= threshold:
                        protected += 1
                        continue  # Keep original

                    # Quantize
                    mx = flat.abs().max()
                    if mx > 0:
                        scale = mx / 7.0
                        flat[i] = torch.round(flat[i] / scale).clamp(-8, 7) * scale

                param.data.copy_(flat.view(param.shape))

        return protected

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"\nBaseline: {', '.join([f'{l}={v:.1f}' for l, v in baseline.items()])}")

    # Test both selection methods
    results = {}
    keep_pcts = [1, 3, 5]

    print(f"\n{'Method':<12} {'Keep%':>6} {'Protected':>12} {'he Disp':>10} {'ar Disp':>10}")
    print("-" * 55)

    for keep_pct in keep_pcts:
        for method_name, importance in [('magnitude', magnitude_importance),
                                         ('gradient', gradient_importance)]:
            restore()
            protected = quant_with_selection(importance, keep_pct)

            q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
            deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
            en_deg = deg['en']

            disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf')
                    for l in TEXTS if l != 'en'}

            key = f"{method_name}_{keep_pct}pct"
            results[key] = {
                'method': method_name,
                'keep_pct': keep_pct,
                'protected': protected,
                'disparity': disp,
                'avg_disp': np.mean(list(disp.values())),
            }

            print(f"{method_name:<12} {keep_pct:>5}% {protected:>12,} "
                  f"{disp['he']:>9.1f}x {disp['ar']:>9.1f}x")

    # Compare methods
    print("\n" + "=" * 60)
    print("Comparison: Magnitude vs Gradient Selection")
    print("=" * 60)

    for keep_pct in keep_pcts:
        mag_key = f"magnitude_{keep_pct}pct"
        grad_key = f"gradient_{keep_pct}pct"

        mag_disp = results[mag_key]['avg_disp']
        grad_disp = results[grad_key]['avg_disp']

        winner = "Gradient" if grad_disp < mag_disp else "Magnitude"
        diff = abs(mag_disp - grad_disp)

        print(f"\n{keep_pct}% preservation:")
        print(f"  Magnitude: {mag_disp:.1f}x avg disparity")
        print(f"  Gradient:  {grad_disp:.1f}x avg disparity")
        print(f"  Winner:    {winner} (by {diff:.1f}x)")

    end = datetime.now()

    result = {
        "id": "Exp-028",
        "name": "Gradient-Based Selection",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp028_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
