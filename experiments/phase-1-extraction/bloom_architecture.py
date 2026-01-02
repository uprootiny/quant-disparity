#!/usr/bin/env python3
"""
EXP-006: BLOOM Architecture Analysis

Investigate why layers 4-7 and 20-23 have extreme kurtosis.

Stages:
  0: Examine model config
  1: Compare weight shapes
  2: Analyze extreme vs normal layers

Usage:
    python3 bloom_architecture.py --stage 0
    python3 bloom_architecture.py --stage 1
    python3 bloom_architecture.py --stage 2
"""

import argparse
import json
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    import numpy as np
    from scipy import stats as sp_stats
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing: {e}")
    HAS_DEPS = False

# Known extreme layers from EXP-005
EXTREME_LAYERS = [5, 21, 22]
HIGH_LAYERS = [4, 6, 7, 23]
NORMAL_LAYERS = [0, 8, 9, 10, 20]
LOW_LAYERS = [1, 2, 3, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def stage_0():
    """Stage 0: Examine model configuration."""
    print("=" * 60)
    print("STAGE 0: BLOOM Model Configuration")
    print("=" * 60)

    config = AutoConfig.from_pretrained("bigscience/bloom-560m")

    print()
    print("Model Configuration:")
    print("-" * 40)
    print(f"Model type:        {config.model_type}")
    print(f"Hidden size:       {config.hidden_size}")
    print(f"Num layers:        {config.n_layer}")
    print(f"Num attention heads: {config.n_head}")
    print(f"Vocab size:        {config.vocab_size}")

    # BLOOM-specific
    if hasattr(config, 'apply_residual_connection_post_layernorm'):
        print(f"Post-LN residual:  {config.apply_residual_connection_post_layernorm}")
    if hasattr(config, 'hidden_dropout'):
        print(f"Hidden dropout:    {config.hidden_dropout}")

    print()
    print("MLP Configuration:")
    print("-" * 40)
    # BLOOM uses 4x hidden size for MLP intermediate
    mlp_size = config.hidden_size * 4
    print(f"MLP intermediate:  {mlp_size}")
    print(f"Dense H→4H:        ({config.hidden_size}, {mlp_size})")
    print(f"Dense 4H→H:        ({mlp_size}, {config.hidden_size})")

    print()
    print("[OK] Stage 0 passed.")
    return config


def stage_1():
    """Stage 1: Compare weight shapes across layers."""
    print("=" * 60)
    print("STAGE 1: Weight Shape Analysis")
    print("=" * 60)

    print("Loading BLOOM-560M...")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("Loaded.")
    print()

    print("Layer Weight Shapes:")
    print("-" * 60)
    print("Layer  H→4H Shape       4H→H Shape       Same?")

    shapes = {}
    for i, layer in enumerate(model.transformer.h):
        h_to_4h = layer.mlp.dense_h_to_4h.weight.shape
        h4_to_h = layer.mlp.dense_4h_to_h.weight.shape

        same = "YES" if h_to_4h == (4096, 1024) and h4_to_h == (1024, 4096) else "NO"
        shapes[i] = {"h_to_4h": list(h_to_4h), "4h_to_h": list(h4_to_h)}

        # Mark extreme layers
        marker = ""
        if i in EXTREME_LAYERS:
            marker = " [EXTREME]"
        elif i in HIGH_LAYERS:
            marker = " [HIGH]"

        print(f"{i:>5}  {str(h_to_4h):<16} {str(h4_to_h):<16} {same}{marker}")

    # Check if all shapes are identical
    all_same = len(set(str(v) for v in shapes.values())) == 1
    print()
    print(f"All layers same shape: {all_same}")

    if all_same:
        print("→ Weight shape is NOT the cause of extreme kurtosis")

    return shapes


def stage_2():
    """Stage 2: Analyze weight statistics (memory-efficient)."""
    print("=" * 60)
    print("STAGE 2: Detailed Weight Analysis (memory-efficient)")
    print("=" * 60)

    import gc

    print("Loading BLOOM-560M...")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("Loaded.")
    print()

    # Analyze one layer at a time
    print("Per-layer statistics:")
    print("-" * 60)
    print("Layer  Mean|W|   Max|W|   Kurtosis  Category")

    layer_stats = {}
    for i in range(24):
        layer = model.transformer.h[i]
        w1 = layer.mlp.dense_h_to_4h.weight.detach().cpu().numpy().flatten()
        w2 = layer.mlp.dense_4h_to_h.weight.detach().cpu().numpy().flatten()
        w = np.concatenate([w1, w2])

        mean_abs = float(np.mean(np.abs(w)))
        max_abs = float(np.max(np.abs(w)))
        kurt = float(sp_stats.kurtosis(w))

        if i in EXTREME_LAYERS:
            cat = "EXTREME"
        elif i in HIGH_LAYERS:
            cat = "HIGH"
        elif i in NORMAL_LAYERS:
            cat = "normal"
        else:
            cat = "low"

        layer_stats[i] = {"mean_abs": mean_abs, "max_abs": max_abs, "kurtosis": kurt, "category": cat}
        print(f"{i:>5}  {mean_abs:.5f}  {max_abs:.5f}  {kurt:+7.2f}   {cat}")

        del w, w1, w2
        gc.collect()

    # Compare categories
    print()
    print("=" * 60)
    print("CATEGORY COMPARISON")
    print("=" * 60)

    for cat in ["EXTREME", "HIGH", "normal", "low"]:
        layers = [i for i, s in layer_stats.items() if s["category"] == cat]
        if not layers:
            continue
        mean_k = np.mean([layer_stats[i]["kurtosis"] for i in layers])
        mean_m = np.mean([layer_stats[i]["max_abs"] for i in layers])
        print(f"{cat:<8}: layers={layers}, mean_kurt={mean_k:.1f}, mean_max={mean_m:.4f}")

    # Save
    Path("bloom_architecture.json").write_text(json.dumps(layer_stats, indent=2))
    print("\nSaved to bloom_architecture.json")

    return layer_stats


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
