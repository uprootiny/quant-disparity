#!/usr/bin/env python3
"""
Analyze weight statistics from BLOOM-560M (memory-efficient).

Usage:
    python3 weight_stats.py
"""

import json
from pathlib import Path
import gc

try:
    import torch
    from transformers import AutoModelForCausalLM
    from scipy import stats as sp_stats
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit(1)


def analyze_weights():
    print("=" * 60)
    print("WEIGHT DISTRIBUTION ANALYSIS (memory-efficient)")
    print("=" * 60)
    print()

    print("Loading BLOOM-560M...")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("Loaded.")
    print()

    results = {}

    # Aggregates
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    all_kurtosis = []

    print("Layer   Params      Kurt    Outlier%")
    print("-" * 45)

    for i, layer in enumerate(model.transformer.h):
        # Get MLP weights
        w1 = layer.mlp.dense_h_to_4h.weight.detach().cpu().numpy().flatten()
        w2 = layer.mlp.dense_4h_to_h.weight.detach().cpu().numpy().flatten()
        w = np.concatenate([w1, w2])

        # Statistics
        mean = float(w.mean())
        std = float(w.std())
        kurt = float(sp_stats.kurtosis(w))
        outlier = float((np.abs(w) > 3 * std).mean() * 100)

        print(f"{i:>5}   {len(w):>8}   {kurt:+.2f}    {outlier:.3f}%")

        results[f"layer_{i}"] = {
            "kurtosis": kurt,
            "outlier_pct": outlier,
            "n_params": len(w),
        }

        # Accumulate for aggregate
        total_sum += w.sum()
        total_sum_sq += (w ** 2).sum()
        total_count += len(w)
        all_kurtosis.append(kurt)

        # Free memory
        del w, w1, w2
        gc.collect()

    # Aggregate
    agg_mean = total_sum / total_count
    agg_var = (total_sum_sq / total_count) - agg_mean ** 2
    agg_std = float(np.sqrt(agg_var))

    print()
    print("=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(f"Total params: {total_count:,}")
    print(f"Mean std:     {agg_std:.6f}")
    print(f"Mean kurtosis: {np.mean(all_kurtosis):+.2f}")
    print(f"Kurtosis range: {min(all_kurtosis):+.2f} to {max(all_kurtosis):+.2f}")

    results["aggregate"] = {
        "mean_std": agg_std,
        "mean_kurtosis": float(np.mean(all_kurtosis)),
        "kurtosis_range": [min(all_kurtosis), max(all_kurtosis)],
        "total_params": total_count,
    }

    Path("weight_stats.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to weight_stats.json")

    return results


if __name__ == "__main__":
    analyze_weights()
