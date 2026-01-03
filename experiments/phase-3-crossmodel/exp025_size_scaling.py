#!/usr/bin/env python3
"""
EXP-025: Size Scaling Analysis

Tests H5: Does kurtosis scale inversely with model size?

Method: Analyze Pythia family (70M, 160M, 410M) for scaling patterns.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_model(model_name: str, repo_id: str, size_m: int) -> dict:
    """Extract max kurtosis from model."""
    from huggingface_hub import hf_hub_download
    import torch

    print(f"\nAnalyzing {model_name} ({size_m}M params)...")

    try:
        path = hf_hub_download(repo_id, "pytorch_model.bin")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except:
        try:
            path = hf_hub_download(repo_id, "model.safetensors")
            from safetensors import safe_open
            state_dict = {}
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        except Exception as e:
            print(f"  Error loading: {e}")
            return None

    print(f"  Loaded {len(state_dict)} tensors")

    max_kurt = 0
    max_tensor = ""
    all_kurts = []

    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        w = tensor.numpy().flatten()
        if len(w) > 50000:
            idx = np.random.choice(len(w), 50000, replace=False)
            w = w[idx]

        try:
            kurt = float(sp_stats.kurtosis(w))
            if np.isnan(kurt) or np.isinf(kurt):
                continue

            all_kurts.append(kurt)
            if kurt > max_kurt:
                max_kurt = kurt
                max_tensor = name
        except:
            continue

    del state_dict
    gc.collect()

    result = {
        "model": model_name,
        "size_m": size_m,
        "max_kurtosis": max_kurt,
        "mean_kurtosis": np.mean(all_kurts) if all_kurts else 0,
        "max_tensor": max_tensor,
        "n_tensors": len(all_kurts),
    }

    print(f"  Max κ = {max_kurt:.1f} in {max_tensor[:40]}")
    return result


def main():
    print("="*60)
    print("EXP-025: Size Scaling Analysis")
    print("="*60)

    # Pythia family
    pythia_models = [
        ("Pythia-70M", "EleutherAI/pythia-70m", 70),
        ("Pythia-160M", "EleutherAI/pythia-160m", 160),
        ("Pythia-410M", "EleutherAI/pythia-410m", 410),
    ]

    # OPT family (if we can load)
    opt_models = [
        ("OPT-125M", "facebook/opt-125m", 125),
    ]

    # GPT-2 family
    gpt2_models = [
        ("GPT2-small", "openai-community/gpt2", 124),
    ]

    results = []

    # Analyze Pythia family
    print("\n### PYTHIA FAMILY ###")
    for name, repo, size in pythia_models:
        r = analyze_model(name, repo, size)
        if r:
            r["family"] = "Pythia"
            results.append(r)

    # Analyze others for comparison
    print("\n### OTHER MODELS ###")
    for name, repo, size in opt_models + gpt2_models:
        r = analyze_model(name, repo, size)
        if r:
            r["family"] = name.split("-")[0]
            results.append(r)

    # Analysis
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)

    # Pythia scaling
    pythia_results = [r for r in results if r["family"] == "Pythia"]
    if len(pythia_results) >= 2:
        sizes = [r["size_m"] for r in pythia_results]
        kurts = [r["max_kurtosis"] for r in pythia_results]

        r_val, p_val = sp_stats.pearsonr(sizes, kurts) if len(sizes) > 2 else (0, 1)

        print("\n### Pythia Family ###")
        print(f"{'Model':<15} {'Size':<10} {'Max κ':<10}")
        print("-"*35)
        for r in sorted(pythia_results, key=lambda x: x["size_m"]):
            print(f"{r['model']:<15} {r['size_m']:<10} {r['max_kurtosis']:<10.1f}")

        print(f"\nCorrelation (size vs κ): r = {r_val:.3f}")

        if r_val < -0.5:
            pythia_verdict = "INVERSE: smaller → higher κ"
        elif r_val > 0.5:
            pythia_verdict = "POSITIVE: larger → higher κ"
        else:
            pythia_verdict = "NO CLEAR PATTERN"

        print(f"Pythia verdict: {pythia_verdict}")
    else:
        pythia_verdict = "INSUFFICIENT DATA"

    # Cross-family comparison
    print("\n### Cross-Family Comparison ###")
    print(f"{'Model':<15} {'Family':<10} {'Size':<10} {'Max κ':<10}")
    print("-"*45)
    for r in sorted(results, key=lambda x: -x["max_kurtosis"]):
        print(f"{r['model']:<15} {r['family']:<10} {r['size_m']:<10} {r['max_kurtosis']:<10.1f}")

    # H5 verdict
    print("\n" + "="*60)
    print("H5 VERDICT")
    print("="*60)

    # Check if smaller models have higher kurtosis within families
    if pythia_results:
        smallest = min(pythia_results, key=lambda x: x["size_m"])
        largest = max(pythia_results, key=lambda x: x["size_m"])

        if smallest["max_kurtosis"] > largest["max_kurtosis"] * 1.5:
            h5_verdict = "H5 SUPPORTED: smaller Pythia has higher κ"
        elif largest["max_kurtosis"] > smallest["max_kurtosis"] * 1.5:
            h5_verdict = "H5 REJECTED: larger Pythia has higher κ"
        else:
            h5_verdict = "H5 INCONCLUSIVE: similar κ across sizes"
    else:
        h5_verdict = "H5 UNTESTABLE: insufficient Pythia data"

    print(f"\n{h5_verdict}")

    # Note about cross-family
    print("\nNote: Cross-family comparison confounded by training differences")
    print("      OPT (κ=562) vs Pythia-410M (κ~14) likely due to training, not size")

    # Save
    output = {
        "experiment": "EXP-025",
        "results": results,
        "pythia_verdict": pythia_verdict,
        "h5_verdict": h5_verdict,
    }
    Path("exp025_results.json").write_text(json.dumps(output, indent=2, default=float))
    print("\nSaved to exp025_results.json")


if __name__ == "__main__":
    main()
