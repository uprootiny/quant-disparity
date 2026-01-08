#!/usr/bin/env python3
"""
Exp-029: Gradient-based quantization
Goal: Test if gradient-sensitive weight selection reduces disparity better than magnitude
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {'en': 'The quick brown fox jumps.', 'he': 'השועל החום המהיר קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-029: Gradient-Based Quantization")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # Compute gradients
    print("Computing gradients on multilingual data...")
    model.train()
    model.zero_grad()

    for lang, text in TEXTS.items():
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs, labels=inputs['input_ids'])
        outputs.loss.backward()

    # Store gradient importance per weight
    gradient_importance = {}
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            gradient_importance[name] = param.grad.abs()

    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_by_importance(importance_dict, keep_pct):
        """Keep top keep_pct% by importance, quantize rest."""
        # Flatten all importances
        all_imp = []
        for name, imp in importance_dict.items():
            all_imp.append(imp.flatten())
        all_imp = torch.cat(all_imp)

        k = int(len(all_imp) * keep_pct / 100)
        threshold = torch.topk(all_imp, k).values[-1].item() if k > 0 else float('inf')

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in importance_dict:
                    continue

                imp = importance_dict[name]
                flat = param.view(-1)
                imp_flat = imp.view(-1)

                # Quantize only low-importance weights
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    for i in range(len(flat)):
                        if imp_flat[i] < threshold:
                            flat[i] = torch.round(flat[i] / scale).clamp(-8, 7) * scale

                param.data.copy_(flat.view(param.shape))

    def quant_by_magnitude(keep_pct):
        """Keep top keep_pct% by magnitude, quantize rest."""
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.abs().flatten())
        all_weights = torch.cat(all_weights)

        k = int(len(all_weights) * keep_pct / 100)
        threshold = torch.topk(all_weights, k).values[-1].item() if k > 0 else float('inf')

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue

                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    for i in range(len(flat)):
                        if flat[i].abs() < threshold:
                            flat[i] = torch.round(flat[i] / scale).clamp(-8, 7) * scale

                param.data.copy_(flat.view(param.shape))

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test at different preservation levels
    results = {}
    keep_pcts = [3, 5, 10]

    print(f"\n{'Method':<12} {'Keep%':>6} {'he Disp':>10}")
    print("-" * 32)

    for keep_pct in keep_pcts:
        # Magnitude-based
        restore()
        quant_by_magnitude(keep_pct)
        mag_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        mag_deg = {l: (mag_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        mag_disp = mag_deg['he'] / mag_deg['en'] if mag_deg['en'] > 0 else float('inf')

        # Gradient-based
        restore()
        quant_by_importance(gradient_importance, keep_pct)
        grad_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        grad_deg = {l: (grad_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        grad_disp = grad_deg['he'] / grad_deg['en'] if grad_deg['en'] > 0 else float('inf')

        results[f'magnitude_{keep_pct}'] = mag_disp
        results[f'gradient_{keep_pct}'] = grad_disp

        print(f"{'magnitude':<12} {keep_pct:>5}% {mag_disp:>10.1f}x")
        print(f"{'gradient':<12} {keep_pct:>5}% {grad_disp:>10.1f}x")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Gradient vs Magnitude Selection")
    print("=" * 60)

    for keep_pct in keep_pcts:
        mag = results[f'magnitude_{keep_pct}']
        grad = results[f'gradient_{keep_pct}']
        winner = "Gradient" if grad < mag else "Magnitude"
        diff = abs(mag - grad)
        print(f"\n{keep_pct}%: {winner} wins by {diff:.1f}x")
        print(f"  Magnitude: {mag:.1f}x | Gradient: {grad:.1f}x")

    end = datetime.now()

    result = {
        "id": "Exp-029",
        "name": "Gradient-Based Quantization",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp029_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
