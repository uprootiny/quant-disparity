#!/usr/bin/env python3
"""
Investigate GPT-2's extreme outlier (κ=5910.6 in layer 2 MLP).

This is much higher than expected. What's happening?
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def main():
    print("="*60)
    print("GPT-2 Extreme Outlier Investigation")
    print("="*60)

    from huggingface_hub import hf_hub_download
    import torch

    print("\nLoading GPT-2...")
    path = hf_hub_download("openai-community/gpt2", "model.safetensors")

    from safetensors import safe_open
    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors\n")

    # Find the extreme outlier
    print("Searching for extreme outliers...")
    print("-"*70)

    outliers = []
    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        w = tensor.numpy().flatten()
        n_params = len(w)

        # For large tensors, sample
        if n_params > 100000:
            idx = np.random.choice(n_params, 100000, replace=False)
            w_sample = w[idx]
        else:
            w_sample = w

        try:
            kurt = float(sp_stats.kurtosis(w_sample))
            max_abs = float(np.max(np.abs(w)))
            std = float(np.std(w_sample))

            if np.isnan(kurt) or np.isinf(kurt):
                continue

            if kurt > 50:
                outliers.append({
                    "name": name,
                    "kurtosis": kurt,
                    "max_abs": max_abs,
                    "std": std,
                    "shape": list(tensor.shape),
                    "n_params": n_params,
                })
        except Exception as e:
            continue

    # Sort by kurtosis
    outliers.sort(key=lambda x: -x["kurtosis"])

    print(f"\nTop 10 outlier tensors:")
    print("-"*70)
    print(f"{'Rank':<5} {'Kurtosis':<12} {'Max|W|':<10} {'Shape':<20} {'Name'}")
    print("-"*70)

    for i, o in enumerate(outliers[:10]):
        print(f"{i+1:<5} {o['kurtosis']:<12.1f} {o['max_abs']:<10.4f} "
              f"{str(o['shape']):<20} {o['name'][:50]}")

    # Analyze the extreme outlier
    print("\n" + "="*60)
    print("DETAILED ANALYSIS: Extreme Outlier")
    print("="*60)

    if outliers:
        extreme = outliers[0]
        tensor = state_dict[extreme["name"]]
        w = tensor.numpy().flatten()

        print(f"\nTensor: {extreme['name']}")
        print(f"Shape: {extreme['shape']}")
        print(f"Kurtosis: {extreme['kurtosis']:.1f}")

        # Distribution analysis
        print("\n--- Distribution Statistics ---")
        print(f"Mean: {np.mean(w):.6f}")
        print(f"Std:  {np.std(w):.6f}")
        print(f"Min:  {np.min(w):.6f}")
        print(f"Max:  {np.max(w):.6f}")

        # Percentiles
        percentiles = [50, 90, 95, 99, 99.9, 99.99]
        print("\n--- Percentiles (absolute values) ---")
        abs_w = np.abs(w)
        for p in percentiles:
            val = np.percentile(abs_w, p)
            print(f"  {p:>6}%: {val:.6f}")

        # Outlier count (> 3σ, > 5σ, > 10σ)
        print("\n--- Outlier Counts ---")
        std = np.std(w)
        for threshold in [3, 5, 10, 20]:
            count = np.sum(np.abs(w) > threshold * std)
            pct = 100 * count / len(w)
            print(f"  > {threshold}σ: {count:>8} ({pct:.4f}%)")

        # Check for specific patterns
        print("\n--- Pattern Analysis ---")

        # Are outliers in specific positions?
        outlier_idx = np.where(np.abs(w) > 10 * std)[0]
        if len(outlier_idx) > 0:
            print(f"Extreme outliers (>10σ): {len(outlier_idx)} positions")

            # Reshape to original shape to find positions
            reshaped = tensor.numpy()
            print(f"Original shape: {reshaped.shape}")

            # Find row/column with most outliers
            if len(reshaped.shape) == 2:
                outlier_mask = np.abs(reshaped) > 10 * std
                row_counts = np.sum(outlier_mask, axis=1)
                col_counts = np.sum(outlier_mask, axis=0)

                print(f"Rows with outliers: {np.sum(row_counts > 0)}/{reshaped.shape[0]}")
                print(f"Cols with outliers: {np.sum(col_counts > 0)}/{reshaped.shape[1]}")

                # Top rows
                top_rows = np.argsort(row_counts)[-5:][::-1]
                print(f"\nTop outlier rows: {list(top_rows)} with counts {list(row_counts[top_rows])}")

                # Top cols
                top_cols = np.argsort(col_counts)[-5:][::-1]
                print(f"Top outlier cols: {list(top_cols)} with counts {list(col_counts[top_cols])}")

    # Compare layers
    print("\n" + "="*60)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*60)

    layer_max_kurt = {}
    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        # Extract layer number
        layer = None
        if ".h." in name:
            try:
                layer = int(name.split(".h.")[1].split(".")[0])
            except:
                pass

        if layer is None:
            continue

        w = tensor.numpy().flatten()
        if len(w) > 50000:
            idx = np.random.choice(len(w), 50000, replace=False)
            w = w[idx]

        try:
            kurt = float(sp_stats.kurtosis(w))
            if np.isnan(kurt) or np.isinf(kurt):
                continue

            if layer not in layer_max_kurt:
                layer_max_kurt[layer] = {"max": 0, "tensor": ""}
            if kurt > layer_max_kurt[layer]["max"]:
                layer_max_kurt[layer] = {"max": kurt, "tensor": name}
        except:
            continue

    print(f"\n{'Layer':<8} {'Max κ':<12} {'Tensor'}")
    print("-"*70)
    for layer in sorted(layer_max_kurt.keys()):
        info = layer_max_kurt[layer]
        marker = " <<<" if info["max"] > 100 else ""
        print(f"{layer:<8} {info['max']:<12.1f} {info['tensor'][:45]}{marker}")

    # Correlation with layer position
    layers = sorted(layer_max_kurt.keys())
    kurts = [layer_max_kurt[l]["max"] for l in layers]

    if len(layers) > 2:
        r, p = sp_stats.pearsonr(layers, kurts)
        print(f"\nLayer position vs kurtosis: r = {r:.3f}, p = {p:.4f}")

    # Save summary
    summary = {
        "model": "GPT-2-small",
        "extreme_outlier": outliers[0] if outliers else None,
        "n_outliers_gt50": len(outliers),
        "layer_max_kurtosis": {str(k): v for k, v in layer_max_kurt.items()},
    }
    Path("gpt2_analysis.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nSaved to gpt2_analysis.json")

    del state_dict
    gc.collect()


if __name__ == "__main__":
    main()
