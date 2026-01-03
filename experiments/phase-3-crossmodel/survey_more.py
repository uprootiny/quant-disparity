#!/usr/bin/env python3
"""
Extended model survey - OPT, Llama-like, others.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_model(model_id, model_name):
    """Analyze via state dict."""
    print(f"\n{'-'*50}")
    print(f"Analyzing: {model_name}")
    print("-"*50)

    try:
        from huggingface_hub import hf_hub_download
        import torch

        # Try safetensors first, then pytorch
        state_dict = None
        for fname in ["model.safetensors", "pytorch_model.bin"]:
            try:
                path = hf_hub_download(model_id, fname)
                if fname.endswith(".safetensors"):
                    from safetensors import safe_open
                    state_dict = {}
                    with safe_open(path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                else:
                    state_dict = torch.load(path, map_location="cpu")
                break
            except:
                continue

        if state_dict is None:
            print(f"Could not load weights")
            return None

        # Analyze
        all_kurtosis = []
        all_max_abs = []

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            w = tensor.numpy().flatten()
            if len(w) > 50000:
                w = np.random.choice(w, 50000, replace=False)

            try:
                kurt = float(sp_stats.kurtosis(w))
                max_abs = float(np.max(np.abs(w)))
                all_kurtosis.append(kurt)
                all_max_abs.append(max_abs)
            except:
                continue

        del state_dict
        gc.collect()

        if not all_kurtosis:
            return None

        summary = {
            "mean_kurtosis": float(np.mean(all_kurtosis)),
            "max_kurtosis": float(np.max(all_kurtosis)),
            "std_kurtosis": float(np.std(all_kurtosis)),
            "max_weight": float(np.max(all_max_abs)),
            "n_tensors": len(all_kurtosis),
        }

        print(f"Tensors: {summary['n_tensors']}, "
              f"Mean κ: {summary['mean_kurtosis']:.2f}, "
              f"Max κ: {summary['max_kurtosis']:.2f}")

        return summary

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("="*60)
    print("Extended Model Survey")
    print("="*60)

    models = [
        ("facebook/opt-125m", "OPT-125M"),
        ("facebook/opt-350m", "OPT-350M"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
        ("microsoft/phi-1", "Phi-1"),
        ("Qwen/Qwen2-0.5B", "Qwen2-0.5B"),
    ]

    results = {}

    for model_id, name in models:
        result = analyze_model(model_id, name)
        if result:
            results[name] = result
        gc.collect()

    # Full comparison
    print("\n" + "="*70)
    print("FULL MODEL COMPARISON")
    print("="*70)

    # Reference models
    ref = {
        "BLOOM-560M": {"mean": 29.64, "max": 164.30},
        "XGLM-564M": {"mean": 0.64, "max": 1.94},
        "GPT-2-small": {"mean": 18.38, "max": 92.31},
        "mT5-small": {"mean": 5.0, "max": 44.7},
        "Pythia-410M": {"mean": 2.23, "max": 13.84},
    }

    def classify(max_k):
        if max_k > 50: return "HEAVY"
        if max_k > 15: return "Moderate"
        if max_k > 5: return "Mild"
        return "Gaussian"

    print(f"\n{'Model':<20} {'Mean κ':<10} {'Max κ':<10} {'Class':<12}")
    print("-" * 55)

    for name, data in ref.items():
        cls = classify(data['max'])
        print(f"{name:<20} {data['mean']:<10.2f} {data['max']:<10.2f} {cls:<12}")

    print("-" * 55)

    for name, result in results.items():
        cls = classify(result['max_kurtosis'])
        print(f"{name:<20} {result['mean_kurtosis']:<10.2f} "
              f"{result['max_kurtosis']:<10.2f} {cls:<12}")

    # Count by category
    print("\n" + "="*60)
    print("SUMMARY BY CATEGORY")
    print("="*60)

    all_models = {**{k: {"max": v["max"]} for k, v in ref.items()},
                  **{k: {"max": v["max_kurtosis"]} for k, v in results.items()}}

    categories = {"HEAVY": [], "Moderate": [], "Mild": [], "Gaussian": []}
    for name, data in all_models.items():
        cat = classify(data["max"])
        categories[cat].append(name)

    for cat, models in categories.items():
        print(f"\n{cat}: {len(models)}")
        for m in models:
            print(f"  - {m}")

    # Save
    Path("extended_survey.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to extended_survey.json")


if __name__ == "__main__":
    main()
