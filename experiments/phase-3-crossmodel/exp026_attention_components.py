#!/usr/bin/env python3
"""
EXP-026: Attention Component Analysis

Why do outliers appear in different attention components?
- OPT: out_proj (output projection)
- BLOOM: query projection
- Pythia: query_key_value (fused QKV)
- GPT-2: c_proj (output projection)

Hypothesis: The component with outliers depends on architecture specifics.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_attention_components(model_name: str, repo_id: str) -> dict:
    """Analyze kurtosis by attention component type."""
    from huggingface_hub import hf_hub_download
    import torch

    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print('='*50)

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
            print(f"  Error: {e}")
            return None

    print(f"  Loaded {len(state_dict)} tensors")

    # Categorize attention components
    components = {
        "query": [],
        "key": [],
        "value": [],
        "qkv_fused": [],
        "output": [],
        "other_attn": [],
    }

    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        name_lower = name.lower()

        # Skip non-attention
        if not any(x in name_lower for x in ["attn", "attention", "self_attn"]):
            continue

        # Categorize
        if "query_key_value" in name_lower or "qkv" in name_lower:
            category = "qkv_fused"
        elif "q_proj" in name_lower or "query" in name_lower or ".q." in name_lower:
            category = "query"
        elif "k_proj" in name_lower or "key" in name_lower or ".k." in name_lower:
            category = "key"
        elif "v_proj" in name_lower or "value" in name_lower or ".v." in name_lower:
            category = "value"
        elif "out_proj" in name_lower or "o_proj" in name_lower or "c_proj" in name_lower:
            category = "output"
        elif "attn" in name_lower or "attention" in name_lower:
            category = "other_attn"
        else:
            continue

        # Compute kurtosis
        w = tensor.numpy().flatten()
        if len(w) > 50000:
            idx = np.random.choice(len(w), 50000, replace=False)
            w = w[idx]

        try:
            kurt = float(sp_stats.kurtosis(w))
            if not np.isnan(kurt) and not np.isinf(kurt):
                components[category].append({
                    "name": name,
                    "kurtosis": kurt,
                    "shape": list(tensor.shape),
                })
        except:
            continue

    del state_dict
    gc.collect()

    # Summarize
    summary = {"model": model_name}
    print(f"\n  {'Component':<15} {'Count':<8} {'Max κ':<10} {'Mean κ'}")
    print("  " + "-"*45)

    max_component = None
    max_kurt = 0

    for comp, tensors in components.items():
        if not tensors:
            continue

        kurts = [t["kurtosis"] for t in tensors]
        max_k = max(kurts)
        mean_k = np.mean(kurts)

        summary[comp] = {
            "count": len(tensors),
            "max_kurtosis": max_k,
            "mean_kurtosis": mean_k,
            "max_tensor": max(tensors, key=lambda x: x["kurtosis"])["name"],
        }

        marker = " <<<" if max_k > 50 else ""
        print(f"  {comp:<15} {len(tensors):<8} {max_k:<10.1f} {mean_k:.1f}{marker}")

        if max_k > max_kurt:
            max_kurt = max_k
            max_component = comp

    summary["outlier_component"] = max_component
    summary["max_kurtosis"] = max_kurt

    return summary


def main():
    print("="*60)
    print("EXP-026: Attention Component Analysis")
    print("="*60)

    models = [
        ("OPT-125M", "facebook/opt-125m"),
        ("BLOOM-560M", "bigscience/bloom-560m"),
        ("GPT2-small", "openai-community/gpt2"),
        ("Pythia-410M", "EleutherAI/pythia-410m"),
        ("XGLM-564M", "facebook/xglm-564M"),
    ]

    results = []
    for name, repo in models:
        r = analyze_attention_components(name, repo)
        if r:
            results.append(r)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Outlier Location by Model")
    print("="*60)

    print(f"\n{'Model':<15} {'Outlier Component':<20} {'Max κ'}")
    print("-"*50)

    component_counts = {}
    for r in results:
        comp = r.get("outlier_component", "—")
        kurt = r.get("max_kurtosis", 0)
        print(f"{r['model']:<15} {comp:<20} {kurt:.1f}")

        if comp:
            component_counts[comp] = component_counts.get(comp, 0) + 1

    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)

    print("\nOutlier component frequency:")
    for comp, count in sorted(component_counts.items(), key=lambda x: -x[1]):
        print(f"  {comp}: {count} models")

    # Architecture patterns
    print("\nArchitecture patterns:")
    fused_models = [r["model"] for r in results if r.get("outlier_component") == "qkv_fused"]
    output_models = [r["model"] for r in results if r.get("outlier_component") == "output"]
    query_models = [r["model"] for r in results if r.get("outlier_component") == "query"]

    if fused_models:
        print(f"  QKV fused: {', '.join(fused_models)}")
    if output_models:
        print(f"  Output projection: {', '.join(output_models)}")
    if query_models:
        print(f"  Query projection: {', '.join(query_models)}")

    # Save
    output = {
        "experiment": "EXP-026",
        "results": results,
        "component_counts": component_counts,
    }
    Path("exp026_results.json").write_text(json.dumps(output, indent=2, default=float))
    print("\nSaved to exp026_results.json")


if __name__ == "__main__":
    main()
