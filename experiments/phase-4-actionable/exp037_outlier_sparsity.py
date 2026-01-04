#!/usr/bin/env python3
"""
EXP-037: Outlier Sparsity Patterns

Question: Are outliers sparse (few extreme) or dense (many moderate)?

Hypothesis H-037: Models with sparse outliers (few super weights) are
easier to fix than dense outlier models.

Method:
1. Count weights exceeding various thresholds (3σ, 5σ, 10σ)
2. Compute concentration ratio: top-1% / top-10% magnitude
3. Compare across models
4. Correlate sparsity with quantization-friendliness

Prediction: Sparse outliers → easier to preserve selectively.

Actionable outcome: Model selection criteria for quantization.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import kurtosis
import torch


def analyze_weight_sparsity(weights: np.ndarray, name: str) -> dict:
    """
    Analyze the sparsity pattern of outliers in a weight tensor.
    """
    flat = weights.flatten()
    n_total = len(flat)

    # Basic statistics
    mean = flat.mean()
    std = flat.std()

    # Count outliers at various thresholds
    thresholds = [3, 5, 7, 10, 15, 20]
    outlier_counts = {}
    for t in thresholds:
        count = np.sum(np.abs(flat - mean) > t * std)
        outlier_counts[f"{t}sigma"] = {
            "count": int(count),
            "percentage": count / n_total * 100
        }

    # Concentration ratio: how much of the "outlier mass" is in the top weights?
    abs_weights = np.abs(flat)
    sorted_weights = np.sort(abs_weights)[::-1]  # Descending

    top_01_pct = sorted_weights[:max(1, int(n_total * 0.0001))]  # Top 0.01%
    top_1_pct = sorted_weights[:max(1, int(n_total * 0.01))]    # Top 1%
    top_10_pct = sorted_weights[:max(1, int(n_total * 0.1))]    # Top 10%

    # Concentration ratios
    sum_all = abs_weights.sum()
    concentration = {
        "top_0.01%_mass": top_01_pct.sum() / sum_all * 100 if sum_all > 0 else 0,
        "top_1%_mass": top_1_pct.sum() / sum_all * 100 if sum_all > 0 else 0,
        "top_10%_mass": top_10_pct.sum() / sum_all * 100 if sum_all > 0 else 0,
    }

    # Sparsity index: ratio of top-1% to top-10% contribution
    # High ratio = sparse (few weights dominate)
    # Low ratio = dense (many weights contribute)
    sparsity_index = (concentration["top_1%_mass"] / concentration["top_10%_mass"]
                      if concentration["top_10%_mass"] > 0 else 0)

    # Super weight analysis: is there a single dominant weight?
    max_weight = abs_weights.max()
    second_max = np.partition(abs_weights, -2)[-2] if len(abs_weights) > 1 else 0
    super_weight_ratio = max_weight / second_max if second_max > 0 else float('inf')

    return {
        "name": name,
        "n_weights": n_total,
        "mean": float(mean),
        "std": float(std),
        "kurtosis": float(kurtosis(flat, fisher=True)),
        "outlier_counts": outlier_counts,
        "concentration": concentration,
        "sparsity_index": sparsity_index,
        "max_weight": float(max_weight),
        "super_weight_ratio": super_weight_ratio,
    }


def classify_sparsity(sparsity_index: float, super_weight_ratio: float) -> str:
    """
    Classify the outlier pattern.
    """
    if super_weight_ratio > 10:
        return "SUPER_WEIGHT"  # Single weight dominates
    elif sparsity_index > 0.5:
        return "SPARSE"        # Few weights dominate
    elif sparsity_index > 0.2:
        return "MODERATE"      # Mixed pattern
    else:
        return "DENSE"         # Many moderate outliers


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("EXP-037: Outlier Sparsity Patterns")
    print("=" * 60)

    results = {
        "experiment_id": "EXP-037",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-037: Sparse outliers are easier to fix than dense",
        "models": {}
    }

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    models_to_test = [
        ("gpt2", "GPT-2"),
        ("facebook/opt-125m", "OPT-125M"),
        ("EleutherAI/pythia-160m", "Pythia-160M"),
        ("bert-base-uncased", "BERT"),
    ]

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Analyzing: {model_label}")
        print("=" * 50)

        try:
            model = AutoModel.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        model_results = {
            "layers": {},
            "summary": {}
        }

        all_sparsity_indices = []
        all_super_ratios = []
        all_kurtoses = []

        # Analyze attention weights specifically
        print("\nAnalyzing weight matrices...")

        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            if param.dim() < 2:
                continue

            weights = param.detach().cpu().numpy()
            analysis = analyze_weight_sparsity(weights, name)

            # Store per-layer
            model_results["layers"][name] = analysis

            # Collect for summary
            all_sparsity_indices.append(analysis["sparsity_index"])
            all_super_ratios.append(analysis["super_weight_ratio"])
            all_kurtoses.append(analysis["kurtosis"])

        # Find most extreme layers
        sorted_by_kurtosis = sorted(
            model_results["layers"].items(),
            key=lambda x: x[1]["kurtosis"],
            reverse=True
        )

        print(f"\nTop 5 highest-kurtosis layers:")
        for name, analysis in sorted_by_kurtosis[:5]:
            k = analysis["kurtosis"]
            si = analysis["sparsity_index"]
            swr = analysis["super_weight_ratio"]
            pattern = classify_sparsity(si, swr)
            print(f"  {name.split('.')[-2]}.{name.split('.')[-1]}: κ={k:.1f}, SI={si:.3f}, SWR={swr:.1f} → {pattern}")

        # Model-level summary
        model_results["summary"] = {
            "n_layers": len(model_results["layers"]),
            "mean_sparsity_index": float(np.mean(all_sparsity_indices)),
            "max_sparsity_index": float(np.max(all_sparsity_indices)),
            "mean_super_weight_ratio": float(np.mean(all_super_ratios)),
            "max_super_weight_ratio": float(np.max(all_super_ratios)),
            "mean_kurtosis": float(np.mean(all_kurtoses)),
            "max_kurtosis": float(np.max(all_kurtoses)),
        }

        # Classify overall model pattern
        overall_pattern = classify_sparsity(
            model_results["summary"]["mean_sparsity_index"],
            model_results["summary"]["max_super_weight_ratio"]
        )
        model_results["summary"]["overall_pattern"] = overall_pattern

        print(f"\nModel Summary:")
        print(f"  Mean sparsity index: {model_results['summary']['mean_sparsity_index']:.3f}")
        print(f"  Max super weight ratio: {model_results['summary']['max_super_weight_ratio']:.1f}")
        print(f"  Mean kurtosis: {model_results['summary']['mean_kurtosis']:.1f}")
        print(f"  Overall pattern: {overall_pattern}")

        # Concentration analysis
        print(f"\nConcentration analysis (highest-κ layer):")
        top_layer = sorted_by_kurtosis[0]
        conc = top_layer[1]["concentration"]
        print(f"  Top 0.01% weights hold: {conc['top_0.01%_mass']:.2f}% of magnitude")
        print(f"  Top 1% weights hold: {conc['top_1%_mass']:.2f}% of magnitude")
        print(f"  Top 10% weights hold: {conc['top_10%_mass']:.2f}% of magnitude")

        results["models"][model_label] = model_results

        del model

    # Cross-model comparison
    print(f"\n{'='*60}")
    print("Cross-Model Comparison")
    print("=" * 60)

    print(f"\n{'Model':<15} {'Pattern':<12} {'Mean SI':<10} {'Max SWR':<10} {'Max κ':<10}")
    print("-" * 57)

    for model_label, data in results["models"].items():
        summary = data["summary"]
        print(f"{model_label:<15} {summary['overall_pattern']:<12} "
              f"{summary['mean_sparsity_index']:<10.3f} "
              f"{summary['max_super_weight_ratio']:<10.1f} "
              f"{summary['max_kurtosis']:<10.1f}")

    # Hypothesis evaluation
    print(f"\n{'='*60}")
    print("Hypothesis Evaluation")
    print("=" * 60)

    print("\nH-037: Sparse outliers are easier to fix than dense")
    print("\nFindings:")

    # Check if high-kurtosis models are sparse or dense
    high_k_models = [
        (label, data) for label, data in results["models"].items()
        if data["summary"]["max_kurtosis"] > 100
    ]

    if high_k_models:
        print(f"\nHigh-kurtosis models (κ > 100):")
        for label, data in high_k_models:
            pattern = data["summary"]["overall_pattern"]
            print(f"  {label}: {pattern}")

            if pattern in ["SUPER_WEIGHT", "SPARSE"]:
                print(f"    → GOOD: Few weights to preserve")
            else:
                print(f"    → HARD: Many weights to manage")

    # Actionable insight
    print(f"\n{'='*60}")
    print("Actionable Insight")
    print("=" * 60)

    sparse_models = [
        label for label, data in results["models"].items()
        if data["summary"]["overall_pattern"] in ["SUPER_WEIGHT", "SPARSE"]
    ]
    dense_models = [
        label for label, data in results["models"].items()
        if data["summary"]["overall_pattern"] in ["DENSE", "MODERATE"]
    ]

    print(f"\nQuantization-friendly (sparse outliers): {sparse_models}")
    print(f"Quantization-challenging (dense outliers): {dense_models}")

    print("\nRecommendation:")
    print("  - For SPARSE/SUPER_WEIGHT: Preserve top 0.01-0.1% weights in FP16")
    print("  - For DENSE: Need broader intervention (1-10% preservation or retraining)")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Save summary only (full results too large)
    summary_results = {
        "experiment_id": results["experiment_id"],
        "timestamp": results["timestamp"],
        "hypothesis": results["hypothesis"],
        "model_summaries": {
            label: data["summary"] for label, data in results["models"].items()
        }
    }

    output_file = output_dir / f"exp037_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
