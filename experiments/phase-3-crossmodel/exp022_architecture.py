#!/usr/bin/env python3
"""
EXP-022: Architectural Comparison

Tests:
- H1: Does tensor dimension predict kurtosis?
- H4: Does component type (attention vs MLP) predict kurtosis?

Method:
Extract weight shapes and kurtosis from OPT, BLOOM, GPT-2.
Compare kurtosis by component type and parameter count.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_model(model_name: str, repo_id: str) -> dict:
    """Extract per-tensor stats: shape, kurtosis, component type."""
    from huggingface_hub import hf_hub_download
    import torch

    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print('='*60)

    # Download and load
    try:
        path = hf_hub_download(repo_id, "pytorch_model.bin")
    except:
        path = hf_hub_download(repo_id, "model.safetensors")
        from safetensors import safe_open
        state_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        print(f"Loaded {len(state_dict)} tensors (safetensors)")

    if 'state_dict' not in dir():
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(state_dict)} tensors")

    results = []

    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        # Classify component type
        name_lower = name.lower()
        if "embed" in name_lower:
            component = "embedding"
        elif "fc1" in name_lower or "fc2" in name_lower or "mlp" in name_lower:
            if "fc1" in name_lower or "dense_h" in name_lower or "up" in name_lower:
                component = "mlp_up"
            elif "fc2" in name_lower or "dense_4h" in name_lower or "down" in name_lower:
                component = "mlp_down"
            else:
                component = "mlp_other"
        elif "q_proj" in name_lower or "query" in name_lower:
            component = "attn_query"
        elif "k_proj" in name_lower or "key" in name_lower:
            component = "attn_key"
        elif "v_proj" in name_lower or "value" in name_lower:
            component = "attn_value"
        elif "out_proj" in name_lower or "o_proj" in name_lower:
            component = "attn_out"
        elif "attn" in name_lower or "attention" in name_lower:
            component = "attn_other"
        elif "ln" in name_lower or "layernorm" in name_lower or "norm" in name_lower:
            component = "layernorm"
        elif "lm_head" in name_lower or "head" in name_lower:
            component = "lm_head"
        else:
            component = "other"

        # Extract layer number if present
        layer_num = None
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break
        if layer_num is None:
            # Try "layers.X" or "h.X" patterns
            import re
            match = re.search(r'(?:layers?|h)\.(\d+)', name)
            if match:
                layer_num = int(match.group(1))

        # Compute kurtosis
        w = tensor.numpy().flatten()
        n_params = len(w)

        # Sample if too large
        if n_params > 100000:
            idx = np.random.choice(n_params, 100000, replace=False)
            w_sample = w[idx]
        else:
            w_sample = w

        try:
            kurt = float(sp_stats.kurtosis(w_sample))
            max_abs = float(np.max(np.abs(w)))
            std = float(np.std(w_sample))

            if np.isnan(kurt) or np.isinf(kurt):
                continue

            results.append({
                "name": name,
                "component": component,
                "layer": layer_num,
                "shape": list(tensor.shape),
                "n_params": n_params,
                "kurtosis": kurt,
                "max_abs": max_abs,
                "std": std,
            })
        except Exception as e:
            continue

    del state_dict
    gc.collect()

    return {
        "model": model_name,
        "repo": repo_id,
        "tensors": results,
    }


def analyze_by_component(data: dict) -> dict:
    """Aggregate kurtosis statistics by component type."""
    component_stats = {}

    for tensor in data["tensors"]:
        comp = tensor["component"]
        if comp not in component_stats:
            component_stats[comp] = {
                "kurtosis_values": [],
                "n_params_values": [],
                "count": 0,
            }
        component_stats[comp]["kurtosis_values"].append(tensor["kurtosis"])
        component_stats[comp]["n_params_values"].append(tensor["n_params"])
        component_stats[comp]["count"] += 1

    # Compute summary stats
    for comp, stats in component_stats.items():
        kurts = stats["kurtosis_values"]
        params = stats["n_params_values"]
        stats["max_kurtosis"] = max(kurts)
        stats["mean_kurtosis"] = np.mean(kurts)
        stats["median_kurtosis"] = np.median(kurts)
        stats["total_params"] = sum(params)
        stats["mean_params"] = np.mean(params)
        # Remove raw lists for cleaner output
        del stats["kurtosis_values"]
        del stats["n_params_values"]

    return component_stats


def analyze_by_layer(data: dict) -> dict:
    """Aggregate kurtosis by layer position."""
    layer_stats = {}

    for tensor in data["tensors"]:
        layer = tensor["layer"]
        if layer is None:
            continue
        if layer not in layer_stats:
            layer_stats[layer] = []
        layer_stats[layer].append(tensor["kurtosis"])

    return {
        layer: {
            "max": max(kurts),
            "mean": float(np.mean(kurts)),
            "count": len(kurts),
        }
        for layer, kurts in layer_stats.items()
    }


def test_h1_dimension(all_data: list) -> dict:
    """H1: Does dimension (n_params) predict kurtosis?"""
    print("\n" + "="*60)
    print("Testing H1: Dimension → Kurtosis")
    print("="*60)

    # Collect outlier tensors (κ > 20) from all models
    outliers = []
    for model_data in all_data:
        for tensor in model_data["tensors"]:
            if tensor["kurtosis"] > 20:
                outliers.append({
                    "model": model_data["model"],
                    "name": tensor["name"],
                    "n_params": tensor["n_params"],
                    "kurtosis": tensor["kurtosis"],
                })

    if len(outliers) < 3:
        return {"status": "INSUFFICIENT_DATA", "n_outliers": len(outliers)}

    n_params = [o["n_params"] for o in outliers]
    kurtosis = [o["kurtosis"] for o in outliers]

    r, p = sp_stats.pearsonr(n_params, kurtosis)

    print(f"\nOutlier tensors (κ > 20): {len(outliers)}")
    print(f"Correlation (n_params vs kurtosis): r = {r:.3f}, p = {p:.4f}")

    # Prediction: smaller tensors → higher kurtosis (negative correlation)
    if r < -0.3 and p < 0.05:
        verdict = "H1 SUPPORTED: smaller tensors have higher kurtosis"
    elif r > 0.3 and p < 0.05:
        verdict = "H1 REJECTED: larger tensors have higher kurtosis"
    else:
        verdict = "H1 INCONCLUSIVE: no significant correlation"

    print(f"Verdict: {verdict}")

    return {
        "n_outliers": len(outliers),
        "correlation_r": r,
        "correlation_p": p,
        "verdict": verdict,
        "outliers": outliers[:10],  # Top 10 for reference
    }


def test_h4_component(all_data: list) -> dict:
    """H4: Does component type (attention vs MLP) predict kurtosis?"""
    print("\n" + "="*60)
    print("Testing H4: Component Type → Kurtosis")
    print("="*60)

    # Aggregate by component across all models
    component_max = {}

    for model_data in all_data:
        model = model_data["model"]
        for tensor in model_data["tensors"]:
            comp = tensor["component"]
            key = f"{model}_{comp}"
            if key not in component_max:
                component_max[key] = {"model": model, "component": comp, "max_kurt": 0}
            if tensor["kurtosis"] > component_max[key]["max_kurt"]:
                component_max[key]["max_kurt"] = tensor["kurtosis"]

    # Group by model
    model_comparison = {}
    for key, data in component_max.items():
        model = data["model"]
        if model not in model_comparison:
            model_comparison[model] = {"attn_max": 0, "mlp_max": 0}

        if data["component"].startswith("attn"):
            if data["max_kurt"] > model_comparison[model]["attn_max"]:
                model_comparison[model]["attn_max"] = data["max_kurt"]
        elif data["component"].startswith("mlp"):
            if data["max_kurt"] > model_comparison[model]["mlp_max"]:
                model_comparison[model]["mlp_max"] = data["max_kurt"]

    print("\nMax kurtosis by component type:")
    print("-" * 50)
    print(f"{'Model':<15} {'Attention':<12} {'MLP':<12} {'Higher':<10}")
    print("-" * 50)

    attn_wins = 0
    mlp_wins = 0

    for model, stats in model_comparison.items():
        attn = stats["attn_max"]
        mlp = stats["mlp_max"]
        if attn > mlp:
            higher = "ATTN"
            attn_wins += 1
        elif mlp > attn:
            higher = "MLP"
            mlp_wins += 1
        else:
            higher = "TIE"
        print(f"{model:<15} {attn:<12.1f} {mlp:<12.1f} {higher:<10}")

    print("-" * 50)
    print(f"Attention wins: {attn_wins}, MLP wins: {mlp_wins}")

    if attn_wins > mlp_wins * 2:
        verdict = "H4 SUPPORTED: attention has higher kurtosis"
    elif mlp_wins > attn_wins * 2:
        verdict = "H4 REJECTED: MLP has higher kurtosis"
    else:
        verdict = "H4 INCONCLUSIVE: no clear pattern"

    print(f"\nVerdict: {verdict}")

    return {
        "model_comparison": model_comparison,
        "attn_wins": attn_wins,
        "mlp_wins": mlp_wins,
        "verdict": verdict,
    }


def main():
    print("="*60)
    print("EXP-022: Architectural Comparison")
    print("="*60)

    # Models to analyze
    models = [
        ("OPT-125M", "facebook/opt-125m"),
        ("BLOOM-560M", "bigscience/bloom-560m"),
        ("GPT2-small", "openai-community/gpt2"),
    ]

    all_data = []

    for name, repo in models:
        try:
            data = analyze_model(name, repo)
            all_data.append(data)

            # Print component summary
            comp_stats = analyze_by_component(data)
            print(f"\n{name} - By Component:")
            for comp, stats in sorted(comp_stats.items(), key=lambda x: -x[1]["max_kurtosis"]):
                if stats["max_kurtosis"] > 5:
                    print(f"  {comp:<15}: max κ = {stats['max_kurtosis']:6.1f}, "
                          f"mean = {stats['mean_kurtosis']:6.1f}, "
                          f"count = {stats['count']}")

            # Print layer summary
            layer_stats = analyze_by_layer(data)
            if layer_stats:
                print(f"\n{name} - By Layer (top 5 by max κ):")
                for layer, stats in sorted(layer_stats.items(), key=lambda x: -x[1]["max"])[:5]:
                    print(f"  Layer {layer:2d}: max κ = {stats['max']:6.1f}")

        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_data) < 2:
        print("\nInsufficient data for hypothesis testing")
        return

    # Test hypotheses
    h1_result = test_h1_dimension(all_data)
    h4_result = test_h4_component(all_data)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    results = {
        "experiment": "EXP-022",
        "models_analyzed": [d["model"] for d in all_data],
        "H1_dimension": h1_result,
        "H4_component": h4_result,
    }

    # Save results
    Path("exp022_results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nResults saved to exp022_results.json")

    # Print key findings
    print("\nKey Findings:")
    print(f"  H1 (dimension): {h1_result['verdict']}")
    print(f"  H4 (component): {h4_result['verdict']}")


if __name__ == "__main__":
    main()
