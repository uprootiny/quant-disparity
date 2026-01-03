#!/usr/bin/env python3
"""
EXP-024: Layer Position Analysis

Tests H3: Do early layers have higher kurtosis than late layers
due to gradient accumulation dynamics?

Method:
1. For each model, extract max kurtosis per layer
2. Compute correlation: layer_index vs kurtosis
3. Compare early (< 25%) vs late (> 75%) layers
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_layers(model_name: str, repo_id: str) -> dict:
    """Extract per-layer max kurtosis."""
    from huggingface_hub import hf_hub_download
    import torch

    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print('='*50)

    # Load model
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
            print(f"Could not load: {e}")
            return None

    print(f"Loaded {len(state_dict)} tensors")

    # Extract layer info
    layer_kurts = {}

    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        # Extract layer number - try multiple patterns
        layer = None
        import re

        patterns = [
            r'layers?\.(\d+)\.',
            r'\.h\.(\d+)\.',
            r'block\.(\d+)\.',
            r'decoder\.layers\.(\d+)\.',
            r'encoder\.layers\.(\d+)\.',
        ]

        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                layer = int(match.group(1))
                break

        if layer is None:
            continue

        # Compute kurtosis
        w = tensor.numpy().flatten()
        if len(w) > 50000:
            idx = np.random.choice(len(w), 50000, replace=False)
            w = w[idx]

        try:
            kurt = float(sp_stats.kurtosis(w))
            if np.isnan(kurt) or np.isinf(kurt):
                continue

            if layer not in layer_kurts:
                layer_kurts[layer] = []
            layer_kurts[layer].append({"tensor": name, "kurtosis": kurt})
        except:
            continue

    del state_dict
    gc.collect()

    if not layer_kurts:
        print("No layer data found")
        return None

    # Compute per-layer max
    n_layers = max(layer_kurts.keys()) + 1
    layer_max = {
        layer: max(k["kurtosis"] for k in kurts)
        for layer, kurts in layer_kurts.items()
    }

    # Print layer summary
    print(f"\n{'Layer':<8} {'Max κ':<12} {'Position'}")
    print("-"*35)
    for layer in sorted(layer_max.keys()):
        kurt = layer_max[layer]
        pct = 100 * layer / (n_layers - 1) if n_layers > 1 else 50
        pos = "EARLY" if pct < 25 else "LATE" if pct > 75 else "MID"
        marker = " <<<" if kurt > 50 else ""
        print(f"{layer:<8} {kurt:<12.1f} {pos:>6} ({pct:.0f}%){marker}")

    # Correlation analysis
    layers = list(layer_max.keys())
    kurts = [layer_max[l] for l in layers]

    r, p = sp_stats.pearsonr(layers, kurts) if len(layers) > 2 else (0, 1)

    print(f"\nCorrelation (layer vs kurtosis): r = {r:.3f}, p = {p:.4f}")

    # Compare early vs late
    quarter = n_layers // 4
    early_layers = [l for l in layers if l < quarter]
    late_layers = [l for l in layers if l > 3 * quarter]

    early_max = max([layer_max[l] for l in early_layers]) if early_layers else 0
    late_max = max([layer_max[l] for l in late_layers]) if late_layers else 0

    print(f"Early layers (< 25%): max κ = {early_max:.1f}")
    print(f"Late layers (> 75%):  max κ = {late_max:.1f}")

    return {
        "model": model_name,
        "n_layers": n_layers,
        "layer_max_kurtosis": layer_max,
        "correlation_r": r,
        "correlation_p": p,
        "early_max": early_max,
        "late_max": late_max,
    }


def main():
    print("="*60)
    print("EXP-024: Layer Position Analysis")
    print("="*60)

    models = [
        ("OPT-125M", "facebook/opt-125m"),
        ("BLOOM-560M", "bigscience/bloom-560m"),
        ("GPT2-small", "openai-community/gpt2"),
    ]

    results = []

    for name, repo in models:
        try:
            data = analyze_layers(name, repo)
            if data:
                results.append(data)
        except Exception as e:
            print(f"Error with {name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Layer Position → Kurtosis (H3)")
    print("="*60)

    print(f"\n{'Model':<15} {'r':<8} {'p':<10} {'Early max':<12} {'Late max'}")
    print("-"*60)

    early_higher = 0
    late_higher = 0

    for r in results:
        early = r["early_max"]
        late = r["late_max"]
        higher = "EARLY" if early > late else "LATE" if late > early else "TIE"

        if early > late:
            early_higher += 1
        elif late > early:
            late_higher += 1

        print(f"{r['model']:<15} {r['correlation_r']:<8.3f} "
              f"{r['correlation_p']:<10.4f} {early:<12.1f} {late:<12.1f} → {higher}")

    # Overall verdict
    print("-"*60)

    mean_r = np.mean([r["correlation_r"] for r in results]) if results else 0

    if mean_r < -0.3:
        verdict = "H3 SUPPORTED: early layers have higher kurtosis"
    elif mean_r > 0.3:
        verdict = "H3 REJECTED: late layers have higher kurtosis"
    else:
        verdict = "H3 INCONCLUSIVE: no consistent pattern"

    # Alternative check: which has more extreme outliers?
    print(f"\nEarly has highest outlier: {early_higher}/{len(results)} models")
    print(f"Late has highest outlier:  {late_higher}/{len(results)} models")

    if early_higher > late_higher:
        alt_verdict = "EARLY layers tend to have worst outliers"
    elif late_higher > early_higher:
        alt_verdict = "LATE layers tend to have worst outliers"
    else:
        alt_verdict = "No clear position pattern"

    print(f"\nOverall: {alt_verdict}")
    print(f"H3 verdict: {verdict}")

    # Save
    output = {
        "experiment": "EXP-024",
        "models": results,
        "summary": {
            "mean_correlation": mean_r,
            "early_higher_count": early_higher,
            "late_higher_count": late_higher,
            "verdict": verdict,
        }
    }
    Path("exp024_results.json").write_text(json.dumps(output, indent=2, default=float))
    print("\nSaved to exp024_results.json")


if __name__ == "__main__":
    main()
