#!/usr/bin/env python3
"""
EXP-013: BLOOM Component Analysis

Deeper analysis of BLOOM's layer structure.
Which components have outliers: Q, K, V, or MLP?

This helps understand the mechanism:
  - If outliers in attention (Q/K/V): language-specific attention patterns
  - If outliers in MLP: language-specific feature transformation
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def load_bloom():
    """Load BLOOM model."""
    try:
        from transformers import AutoModelForCausalLM
        print("Loading BLOOM-560M...")
        model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
            low_cpu_mem_usage=True,
        )
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None


def classify_weight(name):
    """Classify weight by component type."""
    name_lower = name.lower()

    if 'query' in name_lower or '.q.' in name_lower or 'q_proj' in name_lower:
        return 'query'
    elif 'key' in name_lower or '.k.' in name_lower or 'k_proj' in name_lower:
        return 'key'
    elif 'value' in name_lower or '.v.' in name_lower or 'v_proj' in name_lower:
        return 'value'
    elif 'dense_h_to_4h' in name_lower or 'mlp.c_fc' in name_lower or 'fc1' in name_lower:
        return 'mlp_up'
    elif 'dense_4h_to_h' in name_lower or 'mlp.c_proj' in name_lower or 'fc2' in name_lower:
        return 'mlp_down'
    elif 'query_key_value' in name_lower:
        return 'qkv_fused'
    elif 'dense' in name_lower and 'attention' in name_lower:
        return 'attn_out'
    elif 'embed' in name_lower:
        return 'embedding'
    elif 'ln' in name_lower or 'layernorm' in name_lower or 'norm' in name_lower:
        return 'layernorm'
    else:
        return 'other'


def analyze_components(model):
    """Analyze each component type separately."""

    # Per-layer, per-component stats
    layer_component_stats = {}

    # Global component stats
    component_weights = {}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue

        # Get layer number
        layer_id = None
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p == 'h' and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_id = int(parts[i + 1])
                break

        # Classify component
        component = classify_weight(name)

        # Extract weights
        w = param.detach().cpu().numpy().flatten()

        # Store for global analysis
        if component not in component_weights:
            component_weights[component] = []
        component_weights[component].extend(w)

        # Store for per-layer analysis
        if layer_id is not None:
            key = (layer_id, component)
            if key not in layer_component_stats:
                layer_component_stats[key] = []
            layer_component_stats[key].extend(w)

    # Compute statistics
    print("\n" + "="*60)
    print("GLOBAL COMPONENT ANALYSIS")
    print("="*60)

    global_stats = {}
    print(f"\n{'Component':<15} {'Kurtosis':<12} {'Max|W|':<12} {'Std':<12} {'N':<12}")
    print("-" * 60)

    for component, weights in sorted(component_weights.items()):
        if len(weights) < 100:
            continue
        w = np.array(weights)
        stats = {
            "kurtosis": float(sp_stats.kurtosis(w)),
            "max_abs": float(np.max(np.abs(w))),
            "std": float(np.std(w)),
            "n": len(w),
        }
        global_stats[component] = stats
        print(f"{component:<15} {stats['kurtosis']:<12.2f} {stats['max_abs']:<12.3f} "
              f"{stats['std']:<12.4f} {stats['n']:<12}")

    # Per-layer component analysis
    print("\n" + "="*60)
    print("PER-LAYER COMPONENT ANALYSIS (Outlier Layers)")
    print("="*60)

    outlier_layers = [5, 21, 22]  # Known outlier layers
    normal_layers = [0, 10, 15]   # Representative normal layers

    per_layer_stats = {}

    for layer_id in outlier_layers + normal_layers:
        per_layer_stats[layer_id] = {}
        for component in ['qkv_fused', 'attn_out', 'mlp_up', 'mlp_down']:
            key = (layer_id, component)
            if key in layer_component_stats:
                w = np.array(layer_component_stats[key])
                per_layer_stats[layer_id][component] = {
                    "kurtosis": float(sp_stats.kurtosis(w)),
                    "max_abs": float(np.max(np.abs(w))),
                }

    # Print outlier layer details
    print("\nOutlier Layers (5, 21, 22):")
    print("-" * 60)
    for layer_id in outlier_layers:
        print(f"\nLayer {layer_id}:")
        for comp, stats in per_layer_stats.get(layer_id, {}).items():
            print(f"  {comp:<12}: κ={stats['kurtosis']:>8.2f}, max|W|={stats['max_abs']:.3f}")

    print("\nNormal Layers (0, 10, 15):")
    print("-" * 60)
    for layer_id in normal_layers:
        print(f"\nLayer {layer_id}:")
        for comp, stats in per_layer_stats.get(layer_id, {}).items():
            print(f"  {comp:<12}: κ={stats['kurtosis']:>8.2f}, max|W|={stats['max_abs']:.3f}")

    # Identify which component has outliers
    print("\n" + "="*60)
    print("CONCLUSION: WHERE ARE THE OUTLIERS?")
    print("="*60)

    # Compare outlier vs normal layers per component
    for component in ['qkv_fused', 'attn_out', 'mlp_up', 'mlp_down']:
        outlier_kurt = np.mean([
            per_layer_stats.get(l, {}).get(component, {}).get('kurtosis', 0)
            for l in outlier_layers
        ])
        normal_kurt = np.mean([
            per_layer_stats.get(l, {}).get(component, {}).get('kurtosis', 0)
            for l in normal_layers
        ])

        ratio = outlier_kurt / (normal_kurt + 0.01)
        marker = "<<<" if ratio > 5 else ""
        print(f"{component:<12}: outlier_κ={outlier_kurt:>8.2f}, "
              f"normal_κ={normal_kurt:>8.2f}, ratio={ratio:>6.1f}x {marker}")

    return {
        "global": global_stats,
        "per_layer": {str(k): v for k, v in per_layer_stats.items()},
    }


def main():
    print("="*60)
    print("EXP-013: BLOOM Component Analysis")
    print("="*60)

    model = load_bloom()
    if model is None:
        return

    results = analyze_components(model)

    del model
    gc.collect()

    # Save
    Path("bloom_components.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to bloom_components.json")


if __name__ == "__main__":
    main()
