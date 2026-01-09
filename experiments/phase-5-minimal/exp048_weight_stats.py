#!/usr/bin/env python3
"""
Exp-048: Weight statistics axis
Goal: What makes L0 and L11 special? Analyze weight distributions.
"""

import torch
from transformers import AutoModelForCausalLM

print("Loading...")
model = AutoModelForCausalLM.from_pretrained('gpt2')

def get_layer_stats(layer_idx):
    """Get statistics for a specific layer."""
    stats = {}
    prefix = f"transformer.h.{layer_idx}."

    for name, param in model.named_parameters():
        if not name.startswith(prefix) or 'weight' not in name:
            continue

        short_name = name.replace(prefix, "")
        flat = param.data.flatten()

        stats[short_name] = {
            'mean': flat.mean().item(),
            'std': flat.std().item(),
            'max': flat.abs().max().item(),
            'sparsity': (flat.abs() < 0.01).float().mean().item(),
            'outliers': (flat.abs() > flat.std() * 3).float().mean().item(),
        }

    return stats

# Compare key layers
print("Weight Statistics by Layer")
print("=" * 70)

for layer_idx in [0, 1, 5, 10, 11]:
    stats = get_layer_stats(layer_idx)
    print(f"\nLayer {layer_idx}:")

    # Aggregate stats
    total_outliers = 0
    total_sparsity = 0
    max_val = 0
    count = 0

    for name, s in stats.items():
        total_outliers += s['outliers']
        total_sparsity += s['sparsity']
        max_val = max(max_val, s['max'])
        count += 1

    avg_outliers = total_outliers / count if count > 0 else 0
    avg_sparsity = total_sparsity / count if count > 0 else 0

    print(f"  Avg outlier rate: {avg_outliers*100:.2f}%")
    print(f"  Avg sparsity:     {avg_sparsity*100:.2f}%")
    print(f"  Max abs value:    {max_val:.4f}")

# Specific comparison: L0 vs L11 attention
print("\n" + "=" * 70)
print("Attention Weight Comparison: L0 vs L11")
print("=" * 70)

for component in ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']:
    print(f"\n{component}:")
    for layer_idx in [0, 11]:
        name = f"transformer.h.{layer_idx}.{component}"
        param = dict(model.named_parameters())[name]
        flat = param.data.flatten()

        print(f"  L{layer_idx}: mean={flat.mean().item():.4f}, std={flat.std().item():.4f}, "
              f"max={flat.abs().max().item():.4f}, outliers={((flat.abs() > flat.std()*3).float().mean()*100).item():.2f}%")

print("\nDone.")
