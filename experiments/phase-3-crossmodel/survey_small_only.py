#!/usr/bin/env python3
"""
Survey only very small models that download quickly.
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

        # Try pytorch first for smaller models
        for fname in ["pytorch_model.bin", "model.safetensors"]:
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
        else:
            print(f"Could not load weights")
            return None

        # Analyze
        all_kurtosis = []

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            w = tensor.numpy().flatten()
            if len(w) > 30000:
                w = np.random.choice(w, 30000, replace=False)

            try:
                kurt = float(sp_stats.kurtosis(w))
                if not np.isnan(kurt) and not np.isinf(kurt):
                    all_kurtosis.append(kurt)
            except:
                continue

        del state_dict
        gc.collect()

        if not all_kurtosis:
            return None

        summary = {
            "mean_kurtosis": float(np.mean(all_kurtosis)),
            "max_kurtosis": float(np.max(all_kurtosis)),
        }

        print(f"Mean κ: {summary['mean_kurtosis']:.2f}, "
              f"Max κ: {summary['max_kurtosis']:.2f}")

        return summary

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("="*60)
    print("Small Model Survey")
    print("="*60)

    models = [
        ("facebook/opt-125m", "OPT-125M"),
        ("sshleifer/tiny-gpt2", "Tiny-GPT2"),
        ("prajjwal1/bert-tiny", "BERT-Tiny"),
    ]

    results = {}

    for model_id, name in models:
        result = analyze_model(model_id, name)
        if result:
            results[name] = result
        gc.collect()

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    ref = {
        "BLOOM-560M": 164.30,
        "XGLM-564M": 1.94,
        "GPT-2-small": 92.31,
        "mT5-small": 44.7,
        "Pythia-410M": 13.84,
        "XLM-R-base": 9.81,
        "DistilmBERT": 10.79,
    }

    def classify(max_k):
        if max_k > 50: return "HEAVY"
        if max_k > 15: return "Moderate"
        if max_k > 5: return "Mild"
        return "Gaussian"

    print(f"\n{'Model':<20} {'Max κ':<12} {'Class':<12}")
    print("-" * 45)

    for name, max_k in sorted(ref.items(), key=lambda x: -x[1]):
        print(f"{name:<20} {max_k:<12.2f} {classify(max_k):<12}")

    print("-" * 45)

    for name, result in results.items():
        max_k = result['max_kurtosis']
        print(f"{name:<20} {max_k:<12.2f} {classify(max_k):<12}")

    Path("small_survey.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
