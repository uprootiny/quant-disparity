#!/usr/bin/env python3
"""
Minimal memory extraction - load state dict only, no model instantiation.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_state_dict(model_id, model_name):
    """Load only state dict, not full model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name} (state dict only)")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download
        import torch

        # Download just the model weights file
        print(f"Downloading weights for {model_id}...")

        try:
            # Try safetensors first
            path = hf_hub_download(model_id, "model.safetensors")
            from safetensors import safe_open
            state_dict = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        except:
            # Fall back to pytorch
            path = hf_hub_download(model_id, "pytorch_model.bin")
            state_dict = torch.load(path, map_location="cpu")

        print(f"Loaded {len(state_dict)} tensors")

        # Analyze layer-wise
        layer_stats = {}

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            # Parse layer ID
            layer_id = "other"
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p.isdigit():
                    layer_id = p
                    break
                if p in ['layer', 'block', 'h', 'layers']:
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_id = parts[i + 1]
                        break

            if layer_id not in layer_stats:
                layer_stats[layer_id] = {"weights": [], "names": []}

            w = tensor.numpy().flatten()
            layer_stats[layer_id]["weights"].extend(w[:10000])  # Sample
            layer_stats[layer_id]["names"].append(name)

        # Compute stats
        stats = {}
        for layer_id, data in layer_stats.items():
            if len(data["weights"]) < 100:
                continue
            w = np.array(data["weights"])
            stats[layer_id] = {
                "kurtosis": float(sp_stats.kurtosis(w)),
                "max_abs": float(np.max(np.abs(w))),
                "std": float(np.std(w)),
                "n_weights": len(w),
            }

        # Clean up
        del state_dict
        gc.collect()

        # Summary
        if not stats:
            print("No valid layers found")
            return None

        kurtosis_vals = [s["kurtosis"] for s in stats.values()]

        print(f"\nLayers: {len(kurtosis_vals)}")
        print(f"Kurtosis — mean: {np.mean(kurtosis_vals):.2f}, "
              f"max: {np.max(kurtosis_vals):.2f}")

        # Outliers
        threshold = np.mean(kurtosis_vals) + 2 * np.std(kurtosis_vals)
        outliers = [k for k, v in stats.items() if v["kurtosis"] > threshold]
        print(f"Outlier layers: {outliers if outliers else 'None'}")

        return {
            "model": model_name,
            "per_layer": stats,
            "summary": {
                "mean_kurtosis": float(np.mean(kurtosis_vals)),
                "max_kurtosis": float(np.max(kurtosis_vals)),
                "std_kurtosis": float(np.std(kurtosis_vals)),
                "outlier_layers": outliers,
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*60)
    print("Minimal Memory Model Survey")
    print("="*60)

    models = [
        ("distilbert-base-multilingual-cased", "DistilmBERT"),
        ("google/mt5-small", "mT5-small"),
    ]

    results = {}
    for model_id, name in models:
        result = analyze_state_dict(model_id, name)
        if result:
            results[name] = result
        gc.collect()

    # Summary table
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        reference = {
            "BLOOM-560M": {"mean": 29.64, "max": 164.30},
            "XGLM-564M": {"mean": 0.64, "max": 1.94},
            "XLM-R-base": {"mean": 5.10, "max": 9.81},
        }

        print(f"\n{'Model':<20} {'Mean κ':<12} {'Max κ':<12} {'Pattern':<20}")
        print("-" * 64)

        for name, data in reference.items():
            pattern = "Heavy" if data['max'] > 50 else "Mild" if data['max'] > 5 else "Gaussian"
            print(f"{name:<20} {data['mean']:<12.2f} {data['max']:<12.2f} {pattern:<20}")

        print("-" * 64)

        for name, result in results.items():
            s = result["summary"]
            pattern = "Heavy" if s['max_kurtosis'] > 50 else "Mild" if s['max_kurtosis'] > 5 else "Gaussian"
            print(f"{name:<20} {s['mean_kurtosis']:<12.2f} {s['max_kurtosis']:<12.2f} {pattern:<20}")

        Path("minimal_survey.json").write_text(json.dumps(results, indent=2))
        print("\nSaved to minimal_survey.json")


if __name__ == "__main__":
    main()
