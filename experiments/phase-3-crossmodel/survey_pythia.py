#!/usr/bin/env python3
"""
Survey Pythia family models.

Pythia is ideal for analysis:
- Same training recipe across sizes
- Multiple checkpoints available
- Well-documented training process

We can see: Do outliers emerge with scale?
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_model(model_id, model_name):
    """Analyze a single model via state dict."""
    print(f"\n{'-'*50}")
    print(f"Analyzing: {model_name}")
    print("-"*50)

    try:
        from huggingface_hub import hf_hub_download
        import torch

        # Try to download weights
        try:
            path = hf_hub_download(model_id, "pytorch_model.bin")
            state_dict = torch.load(path, map_location="cpu")
        except:
            try:
                # Try safetensors
                path = hf_hub_download(model_id, "model.safetensors")
                from safetensors import safe_open
                state_dict = {}
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            except Exception as e:
                print(f"Could not load: {e}")
                return None

        # Analyze
        layer_kurtosis = {}

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            # Get layer number
            layer_id = "other"
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_id = parts[i + 1]
                    break
                if p == 'h' and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_id = parts[i + 1]
                    break

            w = tensor.numpy().flatten()
            # Sample for memory
            if len(w) > 50000:
                w = np.random.choice(w, 50000, replace=False)

            kurt = float(sp_stats.kurtosis(w))
            max_abs = float(np.max(np.abs(w)))

            if layer_id not in layer_kurtosis:
                layer_kurtosis[layer_id] = {"kurtosis": [], "max_abs": []}
            layer_kurtosis[layer_id]["kurtosis"].append(kurt)
            layer_kurtosis[layer_id]["max_abs"].append(max_abs)

        del state_dict
        gc.collect()

        # Aggregate per layer
        layer_stats = {}
        for layer_id, data in layer_kurtosis.items():
            layer_stats[layer_id] = {
                "kurtosis": float(np.mean(data["kurtosis"])),
                "max_abs": float(np.max(data["max_abs"])),
            }

        # Summary
        all_kurt = [v["kurtosis"] for v in layer_stats.values()]
        all_max = [v["max_abs"] for v in layer_stats.values()]

        summary = {
            "n_layers": len(all_kurt),
            "mean_kurtosis": float(np.mean(all_kurt)),
            "max_kurtosis": float(np.max(all_kurt)),
            "std_kurtosis": float(np.std(all_kurt)),
            "max_weight": float(np.max(all_max)),
        }

        # Outlier detection
        threshold = np.mean(all_kurt) + 2 * np.std(all_kurt)
        outliers = [k for k, v in layer_stats.items()
                    if v["kurtosis"] > threshold and k != "other"]
        summary["outlier_layers"] = outliers

        print(f"Layers: {summary['n_layers']}, "
              f"Mean κ: {summary['mean_kurtosis']:.2f}, "
              f"Max κ: {summary['max_kurtosis']:.2f}, "
              f"Outliers: {len(outliers)}")

        return {"summary": summary, "per_layer": layer_stats}

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("="*60)
    print("Pythia Family Survey")
    print("="*60)

    models = [
        ("EleutherAI/pythia-70m", "Pythia-70M"),
        ("EleutherAI/pythia-160m", "Pythia-160M"),
        ("EleutherAI/pythia-410m", "Pythia-410M"),
        ("gpt2", "GPT-2-small"),
    ]

    results = {}

    for model_id, name in models:
        result = analyze_model(model_id, name)
        if result:
            results[name] = result
        gc.collect()

    # Comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)

    # Reference models
    ref = {
        "BLOOM-560M": {"mean": 29.64, "max": 164.30, "outliers": 3},
        "XGLM-564M": {"mean": 0.64, "max": 1.94, "outliers": 0},
        "mT5-small": {"mean": 5.0, "max": 44.7, "outliers": 2},
    }

    print(f"\n{'Model':<18} {'Mean κ':<10} {'Max κ':<10} {'Outliers':<10} {'Pattern':<15}")
    print("-" * 70)

    for name, data in ref.items():
        pattern = "Heavy" if data['max'] > 50 else "Moderate" if data['max'] > 10 else "Gaussian"
        print(f"{name:<18} {data['mean']:<10.2f} {data['max']:<10.2f} "
              f"{data['outliers']:<10} {pattern:<15}")

    print("-" * 70)

    for name, result in results.items():
        s = result["summary"]
        pattern = "Heavy" if s['max_kurtosis'] > 50 else "Moderate" if s['max_kurtosis'] > 10 else "Gaussian"
        print(f"{name:<18} {s['mean_kurtosis']:<10.2f} {s['max_kurtosis']:<10.2f} "
              f"{len(s['outlier_layers']):<10} {pattern:<15}")

    # Scale analysis
    print("\n" + "="*60)
    print("SCALE ANALYSIS: Does kurtosis increase with model size?")
    print("="*60)

    pythia_sizes = [(70, "Pythia-70M"), (160, "Pythia-160M"), (410, "Pythia-410M")]
    for size, name in pythia_sizes:
        if name in results:
            s = results[name]["summary"]
            print(f"{name:<15}: max κ = {s['max_kurtosis']:.2f}")

    # Save
    output = {k: v["summary"] for k, v in results.items()}
    Path("pythia_survey.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to pythia_survey.json")


if __name__ == "__main__":
    main()
