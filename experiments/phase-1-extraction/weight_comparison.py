#!/usr/bin/env python3
"""
EXP-005: BLOOM vs XGLM Weight Distribution Comparison

Compares layer-by-layer weight kurtosis to understand
why BLOOM shows language-dependent patterns but XGLM doesn't.

Stages:
  0: Load existing kurtosis data
  1: Layer-by-layer comparison
  2: Pattern analysis

Usage:
    python3 weight_comparison.py --stage 0
    python3 weight_comparison.py --stage 1
    python3 weight_comparison.py --stage 2
"""

import argparse
import json
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as sp_stats
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# BLOOM layer kurtosis from weight_stats.json
BLOOM_KURTOSIS = {
    0: 8.13, 1: 4.71, 2: 1.74, 3: 2.64, 4: 76.28, 5: 125.92,
    6: 44.80, 7: 39.05, 8: 8.03, 9: 8.98, 10: 5.47, 11: 3.44,
    12: 2.84, 13: 1.48, 14: 0.96, 15: 0.98, 16: 1.37, 17: 1.81,
    18: 2.99, 19: 4.63, 20: 16.37, 21: 148.20, 22: 164.30, 23: 36.31,
}


def stage_0():
    """Stage 0: Load and verify data."""
    print("=" * 60)
    print("STAGE 0: Load Kurtosis Data")
    print("=" * 60)

    # Load XGLM kurtosis
    try:
        xglm_data = json.loads(Path("xglm_layer_kurtosis.json").read_text())
        xglm_kurtosis = {int(k): v for k, v in xglm_data.items()}
        print(f"XGLM: {len(xglm_kurtosis)} layers loaded")
    except FileNotFoundError:
        print("[!] xglm_layer_kurtosis.json not found")
        return None

    print(f"BLOOM: {len(BLOOM_KURTOSIS)} layers (hardcoded)")
    print()

    # Quick stats
    bloom_vals = list(BLOOM_KURTOSIS.values())
    xglm_vals = list(xglm_kurtosis.values())

    print("Summary Statistics:")
    print("-" * 40)
    print(f"         BLOOM          XGLM")
    print(f"Min:     {min(bloom_vals):.2f}          {min(xglm_vals):.2f}")
    print(f"Max:     {max(bloom_vals):.2f}        {max(xglm_vals):.2f}")
    print(f"Mean:    {np.mean(bloom_vals):.2f}         {np.mean(xglm_vals):.2f}")
    print(f"Median:  {np.median(bloom_vals):.2f}          {np.median(xglm_vals):.2f}")
    print(f"Std:     {np.std(bloom_vals):.2f}         {np.std(xglm_vals):.2f}")

    print("\n[OK] Stage 0 passed.")
    return {"bloom": BLOOM_KURTOSIS, "xglm": xglm_kurtosis}


def stage_1():
    """Stage 1: Layer-by-layer comparison."""
    print("=" * 60)
    print("STAGE 1: Layer-by-Layer Comparison")
    print("=" * 60)

    data = stage_0()
    if not data:
        return None

    bloom = data["bloom"]
    xglm = data["xglm"]

    print()
    print("Layer  BLOOM    XGLM     Ratio    Category")
    print("-" * 55)

    ratios = []
    categories = {"extreme": [], "high": [], "normal": [], "low": []}

    for layer in sorted(bloom.keys()):
        b = bloom[layer]
        x = xglm.get(layer, 0)
        ratio = b / x if x > 0 else float('inf')
        ratios.append(ratio)

        # Categorize
        if b > 100:
            cat = "EXTREME"
            categories["extreme"].append(layer)
        elif b > 30:
            cat = "HIGH"
            categories["high"].append(layer)
        elif b > 5:
            cat = "normal"
            categories["normal"].append(layer)
        else:
            cat = "low"
            categories["low"].append(layer)

        print(f"{layer:>5}  {b:>7.2f}  {x:>6.2f}   {ratio:>6.1f}x   {cat}")

    print()
    print("Categories:")
    print(f"  EXTREME (>100): {categories['extreme']}")
    print(f"  HIGH (30-100):  {categories['high']}")
    print(f"  Normal (5-30):  {categories['normal']}")
    print(f"  Low (<5):       {categories['low']}")

    return categories


def stage_2():
    """Stage 2: Pattern analysis."""
    print("=" * 60)
    print("STAGE 2: Pattern Analysis")
    print("=" * 60)

    data = stage_0()
    if not data:
        return None

    bloom = data["bloom"]

    print()
    print("BLOOM Layer Position vs Kurtosis:")
    print("-" * 40)

    layers = list(range(24))
    kurtosis = [bloom[l] for l in layers]

    # Correlation with position
    r, p = sp_stats.pearsonr(layers, kurtosis)
    print(f"Correlation (layer vs kurtosis): r={r:.3f}, p={p:.4f}")

    # Early vs Late
    early = [bloom[l] for l in range(12)]
    late = [bloom[l] for l in range(12, 24)]

    print()
    print(f"Early layers (0-11): mean={np.mean(early):.2f}, max={max(early):.2f}")
    print(f"Late layers (12-23): mean={np.mean(late):.2f}, max={max(late):.2f}")

    # Extreme layer positions
    extreme = [l for l in layers if bloom[l] > 100]
    print()
    print(f"Extreme layers (>100): {extreme}")
    print(f"  Positions: layer 5 (early-mid), layers 21-22 (late)")

    # Pattern check
    print()
    print("=" * 60)
    print("PATTERN ANALYSIS")
    print("=" * 60)

    if len(extreme) > 0:
        early_extreme = [l for l in extreme if l < 12]
        late_extreme = [l for l in extreme if l >= 12]

        print(f"Extreme in early half: {early_extreme}")
        print(f"Extreme in late half:  {late_extreme}")

        if early_extreme and late_extreme:
            print("\n[*] Pattern: BIMODAL — extreme layers in both early AND late")
            print("    Interpretation: Not just depth-related")
        elif late_extreme:
            print("\n[*] Pattern: LATE — extreme layers cluster at end")
        else:
            print("\n[*] Pattern: EARLY — extreme layers cluster at start")

    # XGLM comparison
    xglm = data["xglm"]
    xglm_extreme = [l for l in range(24) if xglm.get(l, 0) > 5]
    print()
    print(f"XGLM layers with kurtosis > 5: {xglm_extreme}")
    if not xglm_extreme:
        print("    XGLM has NO extreme layers — uniformly Gaussian")

    # Save analysis
    analysis = {
        "bloom_extreme_layers": [l for l in layers if bloom[l] > 100],
        "bloom_high_layers": [l for l in layers if 30 < bloom[l] <= 100],
        "pattern": "bimodal" if extreme else "uniform",
        "bloom_mean_kurtosis": float(np.mean(kurtosis)),
        "xglm_mean_kurtosis": float(np.mean(list(xglm.values()))),
    }
    Path("weight_comparison.json").write_text(json.dumps(analysis, indent=2))
    print("\nSaved to weight_comparison.json")

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    args = parser.parse_args()

    if args.stage == 0:
        stage_0()
    elif args.stage == 1:
        stage_1()
    elif args.stage == 2:
        stage_2()


if __name__ == "__main__":
    main()
