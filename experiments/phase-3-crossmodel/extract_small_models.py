#!/usr/bin/env python3
"""
EXP-012: Small Model Survey

Extract weight statistics from smaller multilingual models
that fit in CPU memory.

Models:
  - mT5-small (300M) — encoder-decoder
  - distilbert-base-multilingual-cased (135M) — distilled BERT
  - bert-base-multilingual-cased (180M) — original mBERT
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def extract_stats(model, model_name):
    """Extract per-layer weight statistics."""
    stats = {}

    # Collect all named parameters with their layer info
    layer_weights = {}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue

        # Parse layer number from name
        parts = name.split('.')
        layer_id = None

        for i, p in enumerate(parts):
            if p.isdigit():
                layer_id = int(p)
                break
            if p in ['layer', 'block', 'h']:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_id = int(parts[i + 1])
                    break

        if layer_id is None:
            layer_id = 'other'

        layer_key = str(layer_id)
        if layer_key not in layer_weights:
            layer_weights[layer_key] = []

        w = param.detach().cpu().numpy().flatten()
        layer_weights[layer_key].extend(w)

    # Compute statistics per layer
    for layer_id, weights in layer_weights.items():
        if len(weights) < 100:
            continue
        w = np.array(weights)
        stats[layer_id] = {
            "kurtosis": float(sp_stats.kurtosis(w)),
            "max_abs": float(np.max(np.abs(w))),
            "std": float(np.std(w)),
            "n_weights": len(w),
        }

    return stats


def analyze_model(model_id, model_name):
    """Load and analyze a single model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    try:
        from transformers import AutoModel

        print(f"Loading {model_id}...")
        model = AutoModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )

        stats = extract_stats(model, model_name)

        # Free memory
        del model
        gc.collect()

        if not stats:
            print("No layer statistics extracted")
            return None

        # Summary
        kurtosis_vals = [s["kurtosis"] for s in stats.values() if isinstance(s, dict)]
        max_abs_vals = [s["max_abs"] for s in stats.values() if isinstance(s, dict)]

        print(f"\nLayers analyzed: {len(kurtosis_vals)}")
        print(f"Kurtosis — mean: {np.mean(kurtosis_vals):.2f}, "
              f"max: {np.max(kurtosis_vals):.2f}, "
              f"std: {np.std(kurtosis_vals):.2f}")
        print(f"Max|W|   — mean: {np.mean(max_abs_vals):.3f}, "
              f"max: {np.max(max_abs_vals):.3f}")

        # Identify outliers
        threshold = np.mean(kurtosis_vals) + 2 * np.std(kurtosis_vals)
        outliers = [k for k, v in stats.items()
                    if isinstance(v, dict) and v["kurtosis"] > threshold]
        print(f"Outlier layers (κ > {threshold:.1f}): {outliers if outliers else 'None'}")

        return {
            "model": model_name,
            "model_id": model_id,
            "per_layer": stats,
            "summary": {
                "n_layers": len(kurtosis_vals),
                "mean_kurtosis": float(np.mean(kurtosis_vals)),
                "max_kurtosis": float(np.max(kurtosis_vals)),
                "std_kurtosis": float(np.std(kurtosis_vals)),
                "outlier_layers": outliers,
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("="*60)
    print("EXP-012: Small Model Survey")
    print("="*60)

    models = [
        ("google/mt5-small", "mT5-small"),
        ("distilbert-base-multilingual-cased", "DistilmBERT"),
        ("bert-base-multilingual-cased", "mBERT"),
    ]

    results = {}

    for model_id, model_name in models:
        result = analyze_model(model_id, model_name)
        if result:
            results[model_name] = result
        gc.collect()

    # Cross-model comparison
    print("\n" + "="*60)
    print("CROSS-MODEL COMPARISON")
    print("="*60)

    # Add reference models
    reference = {
        "BLOOM-560M": {"mean": 29.64, "max": 164.30, "std": 47.87, "outliers": 3},
        "XGLM-564M": {"mean": 0.64, "max": 1.94, "std": 0.45, "outliers": 0},
        "XLM-R-base": {"mean": 5.10, "max": 9.81, "std": 1.54, "outliers": 1},
    }

    print(f"\n{'Model':<20} {'Mean κ':<10} {'Max κ':<10} {'Std κ':<10} {'Outliers':<10}")
    print("-" * 60)

    for name, data in reference.items():
        print(f"{name:<20} {data['mean']:<10.2f} {data['max']:<10.2f} "
              f"{data['std']:<10.2f} {data['outliers']:<10}")

    print("-" * 60)

    for name, result in results.items():
        s = result["summary"]
        print(f"{name:<20} {s['mean_kurtosis']:<10.2f} {s['max_kurtosis']:<10.2f} "
              f"{s['std_kurtosis']:<10.2f} {len(s['outlier_layers']):<10}")

    # Classification
    print("\n" + "="*60)
    print("CLASSIFICATION")
    print("="*60)

    for name, result in results.items():
        s = result["summary"]
        if s["max_kurtosis"] > 50:
            pattern = "BLOOM-like (heavy outliers)"
        elif s["max_kurtosis"] > 10:
            pattern = "Moderate (some outliers)"
        elif s["max_kurtosis"] > 3:
            pattern = "Mild (near-Gaussian)"
        else:
            pattern = "XGLM-like (Gaussian)"
        print(f"{name}: {pattern}")

    # Save
    Path("small_models_survey.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to small_models_survey.json")


if __name__ == "__main__":
    main()
