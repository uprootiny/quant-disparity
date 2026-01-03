#!/usr/bin/env python3
"""
EXP-010: mGPT Weight Extraction

Extract per-layer weight statistics from mGPT (Sberbank's multilingual GPT).
Compare kurtosis patterns with BLOOM to test generalization.

mGPT: 1.3B parameters, trained on 60 languages including Russian focus.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

# Lazy import to check availability
def load_model():
    """Load mGPT model weights only (no inference needed)."""
    try:
        from transformers import AutoModelForCausalLM
        print("Loading mGPT-1.3B (weights only, CPU)...")
        model = AutoModelForCausalLM.from_pretrained(
            "ai-forever/mGPT",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        return model
    except Exception as e:
        print(f"Error loading mGPT: {e}")
        return None


def extract_layer_stats(model):
    """Extract per-layer weight statistics."""
    stats = {}

    # Find transformer layers
    if hasattr(model, 'transformer'):
        layers = model.transformer.h
    elif hasattr(model, 'model'):
        layers = model.model.layers
    else:
        print("Unknown architecture, scanning all parameters...")
        layers = None

    if layers is not None:
        print(f"Found {len(layers)} transformer layers")

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
    else:
        # Fallback: scan all parameters
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                w = param.detach().cpu().numpy().flatten()
                layer_id = name.split('.')[0]  # rough grouping

                if layer_id not in stats:
                    stats[layer_id] = {
                        "weights": [],
                        "names": [],
                    }
                stats[layer_id]["weights"].extend(w)
                stats[layer_id]["names"].append(name)

    return stats


def analyze_patterns(stats):
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

    # Identify outlier layers (kurtosis > mean + 2*std)
    threshold = np.mean(kurtosis_vals) + 2 * np.std(kurtosis_vals)
    outlier_layers = [k for k, v in stats.items() if v["kurtosis"] > threshold]

    print(f"\nOutlier layers (kurtosis > {threshold:.1f}):")
    print(f"  {outlier_layers if outlier_layers else 'None'}")

    # Compare with BLOOM pattern
    print("\n" + "-" * 60)
    print("COMPARISON WITH BLOOM")
    print("-" * 60)

    bloom_stats = {
        "mean_kurtosis": 29.64,
        "std_kurtosis": 47.87,
        "max_kurtosis": 164.30,
        "outlier_layers": [5, 21, 22],
    }

    print(f"\n{'Metric':<20} {'mGPT':<12} {'BLOOM':<12} {'Ratio':<10}")
    print("-" * 54)
    print(f"{'Mean kurtosis':<20} {np.mean(kurtosis_vals):<12.2f} "
          f"{bloom_stats['mean_kurtosis']:<12.2f} "
          f"{np.mean(kurtosis_vals)/bloom_stats['mean_kurtosis']:<10.2f}")
    print(f"{'Max kurtosis':<20} {np.max(kurtosis_vals):<12.2f} "
          f"{bloom_stats['max_kurtosis']:<12.2f} "
          f"{np.max(kurtosis_vals)/bloom_stats['max_kurtosis']:<10.2f}")
    print(f"{'Std kurtosis':<20} {np.std(kurtosis_vals):<12.2f} "
          f"{bloom_stats['std_kurtosis']:<12.2f} "
          f"{np.std(kurtosis_vals)/bloom_stats['std_kurtosis']:<10.2f}")
    print(f"{'# Outlier layers':<20} {len(outlier_layers):<12} "
          f"{len(bloom_stats['outlier_layers']):<12}")

    return {
        "mean_kurtosis": np.mean(kurtosis_vals),
        "std_kurtosis": np.std(kurtosis_vals),
        "max_kurtosis": np.max(kurtosis_vals),
        "outlier_layers": outlier_layers,
        "outlier_threshold": threshold,
    }


def main():
    print("=" * 60)
    print("EXP-010: mGPT Weight Extraction")
    print("=" * 60)

    model = load_model()
    if model is None:
        print("\nFailed to load model. Try installing transformers:")
        print("  pip install transformers")
        return

    print("\nExtracting weight statistics...")
    stats = extract_layer_stats(model)

    # Free memory
    del model

    if not stats:
        print("No statistics extracted.")
        return

    # Analyze patterns
    summary = analyze_patterns(stats)

    # Save results
    output = {
        "model": "mGPT-1.3B",
        "per_layer": stats,
        "summary": summary,
    }

    Path("mgpt_architecture.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to mgpt_architecture.json")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if summary["max_kurtosis"] < 10:
        print("""
mGPT has NEAR-GAUSSIAN weights (like XGLM).
The outlier layer pattern is likely BLOOM-SPECIFIC.
""")
    elif len(summary["outlier_layers"]) > 0:
        print(f"""
mGPT has OUTLIER LAYERS: {summary["outlier_layers"]}
The pattern may generalize beyond BLOOM.
Further investigation needed.
""")
    else:
        print("""
mGPT has MODERATE kurtosis but no extreme outliers.
Intermediate between BLOOM and XGLM patterns.
""")


if __name__ == "__main__":
    main()
