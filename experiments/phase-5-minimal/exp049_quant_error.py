#!/usr/bin/env python3
"""
Exp-049: Quantization error analysis
Goal: Measure actual quantization error per layer
"""

import torch
from transformers import AutoModelForCausalLM

print("Loading...")
model = AutoModelForCausalLM.from_pretrained('gpt2')

def measure_quant_error(param):
    """Measure quantization error for a parameter."""
    flat = param.data.flatten()
    mx = flat.abs().max()
    if mx == 0:
        return 0, 0

    scale = mx / 7.0
    quantized = torch.round(flat / scale).clamp(-8, 7) * scale

    # Error metrics
    abs_error = (flat - quantized).abs()
    rel_error = abs_error / (flat.abs() + 1e-8)

    return abs_error.mean().item(), rel_error.mean().item()

print("Quantization Error by Layer")
print("=" * 60)
print(f"{'Layer':<10} {'Abs Error':>12} {'Rel Error':>12}")
print("-" * 36)

layer_errors = {}
for layer_idx in range(12):
    prefix = f"transformer.h.{layer_idx}."
    abs_errors = []
    rel_errors = []

    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            abs_e, rel_e = measure_quant_error(param)
            abs_errors.append(abs_e)
            rel_errors.append(rel_e)

    avg_abs = sum(abs_errors) / len(abs_errors) if abs_errors else 0
    avg_rel = sum(rel_errors) / len(rel_errors) if rel_errors else 0
    layer_errors[layer_idx] = (avg_abs, avg_rel)

    print(f"Layer {layer_idx:<4} {avg_abs:>12.6f} {avg_rel*100:>11.2f}%")

# Identify extremes
print("\n" + "=" * 60)
max_abs_layer = max(layer_errors.items(), key=lambda x: x[1][0])
min_abs_layer = min(layer_errors.items(), key=lambda x: x[1][0])

print(f"Highest abs error: Layer {max_abs_layer[0]} ({max_abs_layer[1][0]:.6f})")
print(f"Lowest abs error:  Layer {min_abs_layer[0]} ({min_abs_layer[1][0]:.6f})")

# Compare L0 MLP vs L11 MLP specifically
print("\n" + "=" * 60)
print("MLP Component Error Comparison")
print("=" * 60)

for component in ['mlp.c_fc.weight', 'mlp.c_proj.weight']:
    print(f"\n{component}:")
    for layer_idx in [0, 1, 10, 11]:
        name = f"transformer.h.{layer_idx}.{component}"
        param = dict(model.named_parameters())[name]
        abs_e, rel_e = measure_quant_error(param)
        print(f"  L{layer_idx}: abs={abs_e:.6f}, rel={rel_e*100:.2f}%")

print("\nDone.")
