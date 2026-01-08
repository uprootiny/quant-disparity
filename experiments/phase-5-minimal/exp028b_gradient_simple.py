#!/usr/bin/env python3
"""
Exp-028b: Gradient-based selection (simplified)
Goal: Quick test of gradient vs magnitude selection
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-028b: Gradient Selection (Simple)")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # Compute gradients on multilingual text
    print("Computing gradients...")
    model.train()
    model.zero_grad()

    # Combined loss from both languages
    total_loss = 0
    for lang, text in TEXTS.items():
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs, labels=inputs['input_ids'])
        total_loss = total_loss + outputs.loss

    total_loss.backward()

    # Store gradient magnitudes
    gradient_mag = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_mag[name] = param.grad.abs().mean().item()

    model.eval()

    # Store magnitude
    weight_mag = {}
    for name, param in model.named_parameters():
        weight_mag[name] = param.data.abs().mean().item()

    # Compare: which layers have highest gradient vs magnitude
    print("\nTop layers by gradient sensitivity:")
    sorted_grad = sorted(gradient_mag.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, val in sorted_grad:
        print(f"  {name[:40]:<40} {val:.6f}")

    print("\nTop layers by weight magnitude:")
    sorted_mag = sorted(weight_mag.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, val in sorted_mag:
        print(f"  {name[:40]:<40} {val:.6f}")

    # Check overlap
    grad_top = set(n for n, v in sorted_grad)
    mag_top = set(n for n, v in sorted_mag)
    overlap = grad_top & mag_top

    print(f"\nOverlap in top 10: {len(overlap)}/10")
    print(f"Unique to gradient: {grad_top - mag_top}")
    print(f"Unique to magnitude: {mag_top - grad_top}")

    end = datetime.now()

    result = {
        "id": "Exp-028b",
        "name": "Gradient Selection Simple",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "top_gradient": sorted_grad[:5],
        "top_magnitude": sorted_mag[:5],
        "overlap_count": len(overlap),
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp028b_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
