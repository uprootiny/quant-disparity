#!/usr/bin/env python3
"""
EXP-011: XLM-RoBERTa Weight Extraction

Extract per-layer weight statistics from XLM-RoBERTa.
Different architecture (encoder-only, masked LM) for comparison.

XLM-R Base: 270M parameters, 100 languages
XLM-R Large: 550M parameters, 100 languages
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats


def load_model(size="base"):
    """Load XLM-RoBERTa model weights only."""
    try:
        from transformers import AutoModel

        model_name = f"xlm-roberta-{size}"
        print(f"Loading {model_name} (weights only, CPU)...")

        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        return model
    except Exception as e:
        print(f"Error loading XLM-RoBERTa: {e}")
        return None


def extract_layer_stats(model):
    """Extract per-layer weight statistics from encoder."""
    stats = {}

    # XLM-R uses RoBERTa architecture
    if hasattr(model, 'encoder'):
        layers = model.encoder.layer
    elif hasattr(model, 'roberta'):
        layers = model.roberta.encoder.layer
    else:
        print("Unknown architecture")
        return stats

    print(f"Found {len(layers)} encoder layers")

    for i, layer in enumerate(layers):
        layer_weights = []

        for name, param in layer.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param.detach().cpu().numpy().flatten()
                layer_weights.extend(w)

        if layer_weights:
            w = np.array(layer_weights)
            stats[str(i)] = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "kurtosis": float(sp_stats.kurtosis(w)),
                "skew": float(sp_stats.skew(w)),
                "max_abs": float(np.max(np.abs(w))),
                "n_weights": len(w),
            }
            print(f"  Layer {i}: kurtosis={stats[str(i)]['kurtosis']:.2f}, "
                  f"max|W|={stats[str(i)]['max_abs']:.3f}")

    return stats


def analyze_patterns(stats, model_name):
    """Analyze kurtosis patterns."""
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS")
    print("=" * 60)

    kurtosis_vals = [s["kurtosis"] for s in stats.values()]
    max_abs_vals = [s["max_abs"] for s in stats.values()]

    print(f"\nKurtosis statistics:")
    print(f"  Mean: {np.mean(kurtosis_vals):.2f}")
    print(f"  Std:  {np.std(kurtosis_vals):.2f}")
    print(f"  Max:  {np.max(kurtosis_vals):.2f}")
    print(f"  Min:  {np.min(kurtosis_vals):.2f}")

    # Identify outlier layers
    threshold = np.mean(kurtosis_vals) + 2 * np.std(kurtosis_vals)
    outlier_layers = [k for k, v in stats.items() if v["kurtosis"] > threshold]

    print(f"\nOutlier layers (kurtosis > {threshold:.1f}):")
    print(f"  {outlier_layers if outlier_layers else 'None'}")

    # Multi-model comparison
    print("\n" + "-" * 60)
    print("CROSS-MODEL COMPARISON")
    print("-" * 60)

    models = {
        "BLOOM-560M": {"mean": 29.64, "max": 164.30, "std": 47.87, "outliers": 3},
        "XGLM-564M": {"mean": 0.64, "max": 1.94, "std": 0.45, "outliers": 0},
        model_name: {
            "mean": np.mean(kurtosis_vals),
            "max": np.max(kurtosis_vals),
            "std": np.std(kurtosis_vals),
            "outliers": len(outlier_layers),
        },
    }

    print(f"\n{'Model':<15} {'Mean κ':<10} {'Max κ':<10} {'Std κ':<10} {'Outliers':<10}")
    print("-" * 55)
    for name, m in models.items():
        print(f"{name:<15} {m['mean']:<10.2f} {m['max']:<10.2f} "
              f"{m['std']:<10.2f} {m['outliers']:<10}")

    return {
        "mean_kurtosis": np.mean(kurtosis_vals),
        "std_kurtosis": np.std(kurtosis_vals),
        "max_kurtosis": np.max(kurtosis_vals),
        "outlier_layers": outlier_layers,
        "outlier_threshold": threshold,
    }


def main():
    print("=" * 60)
    print("EXP-011: XLM-RoBERTa Weight Extraction")
    print("=" * 60)

    # Try base first (smaller)
    model = load_model("base")
    model_name = "XLM-R-base"

    if model is None:
        print("\nFailed to load model.")
        return

    print("\nExtracting weight statistics...")
    stats = extract_layer_stats(model)

    # Free memory
    del model

    if not stats:
        print("No statistics extracted.")
        return

    # Analyze patterns
    summary = analyze_patterns(stats, model_name)

    # Save results
    output = {
        "model": model_name,
        "architecture": "encoder-only (masked LM)",
        "per_layer": stats,
        "summary": summary,
    }

    Path("xlmr_architecture.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to xlmr_architecture.json")

    # Classification
    print("\n" + "=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)

    if summary["max_kurtosis"] > 50:
        pattern = "BLOOM-like (heavy outliers)"
    elif summary["max_kurtosis"] > 5:
        pattern = "Intermediate (moderate kurtosis)"
    else:
        pattern = "XGLM-like (near-Gaussian)"

    print(f"\n{model_name} pattern: {pattern}")
    print(f"""
Implications for LA-ACIQ:
  - If BLOOM-like: outlier mechanism may generalize
  - If XGLM-like: BLOOM-specific training artifact
  - If Intermediate: architecture-dependent threshold
""")


if __name__ == "__main__":
    main()
