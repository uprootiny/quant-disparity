#!/usr/bin/env python3
"""
Exp-066: Why is Layer 0 uniquely critical?
Goal: Find what statistics distinguish L0 from other high-variance layers
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

print("=" * 70)
print("Why is Layer 0 Uniquely Critical?")
print("=" * 70)

def get_layer_stats(layer_idx):
    """Get comprehensive statistics for a layer"""
    prefix = f"transformer.h.{layer_idx}."
    weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            weights.append(param.data.flatten())

    if not weights:
        return {}

    all_w = torch.cat(weights)

    # Basic stats
    var = all_w.var().item()
    mean = all_w.mean().item()
    std = all_w.std().item()

    # Distribution stats
    skew = ((all_w - mean) / std).pow(3).mean().item()
    kurtosis = ((all_w - mean) / std).pow(4).mean().item() - 3

    # Outlier stats (values > 3 std)
    outlier_ratio = (all_w.abs() > 3 * std).float().mean().item()

    # Sparsity (values near zero)
    sparsity = (all_w.abs() < 0.01).float().mean().item()

    # Range stats
    max_val = all_w.abs().max().item()
    q99 = torch.quantile(all_w.abs(), 0.99).item()

    return {
        'variance': var,
        'std': std,
        'skewness': skew,
        'kurtosis': kurtosis,
        'outlier_ratio': outlier_ratio,
        'sparsity': sparsity,
        'max_abs': max_val,
        'q99': q99,
    }

# Collect stats for all layers
print("\n1. Layer Statistics Comparison")
print("-" * 80)

stats = {}
for i in range(12):
    stats[i] = get_layer_stats(i)

# Print comparison table
metrics = ['variance', 'skewness', 'kurtosis', 'outlier_ratio', 'sparsity', 'max_abs']
print(f"{'Layer':<8}", end='')
for m in metrics:
    print(f"{m:>12}", end='')
print()
print("-" * 80)

for i in range(12):
    print(f"L{i:<7}", end='')
    for m in metrics:
        val = stats[i][m]
        if m == 'outlier_ratio':
            print(f"{val*100:>11.3f}%", end='')
        elif m == 'sparsity':
            print(f"{val*100:>11.1f}%", end='')
        else:
            print(f"{val:>12.4f}", end='')
    print()

# Find what makes L0 and L11 unique
print("\n" + "=" * 70)
print("2. What Makes L0 and L11 Unique?")
print("-" * 70)

# Rank layers by each metric
print("\nRanking by metric (lower rank = higher value):")
for m in metrics:
    values = [(i, stats[i][m]) for i in range(12)]
    sorted_vals = sorted(values, key=lambda x: -x[1])
    ranks = {layer: rank for rank, (layer, _) in enumerate(sorted_vals, 1)}
    l0_rank = ranks[0]
    l11_rank = ranks[11]
    print(f"  {m:<15}: L0=#{l0_rank:<3} L11=#{l11_rank:<3}", end='')
    if l0_rank <= 3 or l11_rank <= 3:
        top3 = [l for l, _ in sorted_vals[:3]]
        print(f" (top-3: {top3})")
    else:
        print()

# Component-level analysis
print("\n" + "=" * 70)
print("3. Component-Level Statistics")
print("-" * 70)

def get_component_var(layer_idx, component):
    """Get variance for a specific component"""
    pattern = f"transformer.h.{layer_idx}.{component}"
    for name, param in model.named_parameters():
        if pattern in name and 'weight' in name:
            return param.data.var().item()
    return 0

components = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
print(f"{'Layer':<8}", end='')
for c in components:
    print(f"{c.split('.')[-1]:>12}", end='')
print()
print("-" * 60)

for i in [0, 5, 10, 11]:  # Representative layers
    print(f"L{i:<7}", end='')
    for c in components:
        var = get_component_var(i, c)
        print(f"{var:>12.6f}", end='')
    print()

# Key insight
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
Hypothesis: L0's criticality comes from its POSITION, not just statistics.

- L0 receives raw embeddings (token + position)
- Errors in L0 propagate through ALL subsequent layers
- L11 outputs to final LayerNorm and prediction head
- Both are "gateway" layers: input gateway (L0) and output gateway (L11)

Variance explains WITHIN-POSITION importance but not POSITIONAL importance.
""")
