#!/usr/bin/env python3
"""
EXP-007: Activation × Outlier Weight Connection

Tests if languages that activate outlier-heavy layers less have higher degradation.

Uses existing data from EXP-003 and EXP-006.

Stages:
  0: Load and verify data
  1: Compute outlier layer activation fraction
  2: Correlate with degradation

Usage:
    python3 activation_outlier.py --stage 0
    python3 activation_outlier.py --stage 1
    python3 activation_outlier.py --stage 2
"""

import argparse
import json
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Outlier layers from EXP-006
OUTLIER_LAYERS = [5, 21, 22]  # max|W| > 2.5
HIGH_LAYERS = [4, 6, 7, 23]   # max|W| > 1.5

DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}


def stage_0():
    """Stage 0: Load and verify data."""
    print("=" * 60)
    print("STAGE 0: Load Data")
    print("=" * 60)

    # Check for activation data from EXP-003
    # We need per-layer activations, but layer_activations.json only has weighted_kurtosis
    # Let me check what we have

    files = {
        "layer_activations.json": "Per-language weighted kurtosis",
        "bloom_architecture.json": "Per-layer weight stats",
        "weight_stats.json": "Per-layer kurtosis",
    }

    for f, desc in files.items():
        path = Path(f)
        if path.exists():
            data = json.loads(path.read_text())
            print(f"[OK] {f}: {len(data)} entries — {desc}")
        else:
            print(f"[!] {f}: NOT FOUND")

    print()
    print("Note: Need per-layer activations per language.")
    print("Will need to re-run layer activation analysis with detailed output.")

    return True


def stage_1():
    """Stage 1: Compute outlier layer activation fraction."""
    print("=" * 60)
    print("STAGE 1: Compute Outlier Layer Activation")
    print("=" * 60)

    # We need to load the model and recompute activations
    # Let me do this efficiently

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[!] Need torch and transformers")
        return None

    SAMPLES = {
        "eng": "London is the capital and largest city of England.",
        "fra": "Paris est la capitale de la France.",
        "deu": "Berlin ist die Hauptstadt Deutschlands.",
        "ara": "القاهرة هي عاصمة مصر.",
        "heb": "ירושלים היא בירת ישראל.",
        "jpn": "東京は日本の首都です。",
        "zho": "北京是中国的首都。",
        "kor": "서울은 대한민국의 수도이다.",
        "rus": "Москва столица России.",
        "hin": "दिल्ली भारत की राजधानी है।",
        "tha": "กรุงเทพเป็นเมืองหลวง",
        "vie": "Hanoi la thu do Viet Nam.",
        "fin": "Helsinki on Suomen paakaupunki.",
        "tur": "Ankara Turkiye'nin baskentidir.",
    }

    print("Loading BLOOM-560M...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Loaded.")
    print()

    results = {}

    for lang in sorted(SAMPLES.keys()):
        text = SAMPLES[lang]
        inputs = tokenizer(text, return_tensors="pt")

        # Collect per-layer activations
        activations = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[layer_idx] = output.abs().mean().item()
            return hook_fn

        hooks = []
        for i, layer in enumerate(model.transformer.h):
            h = layer.mlp.register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks:
            h.remove()

        # Compute outlier layer fraction
        total_act = sum(activations.values())
        outlier_act = sum(activations[i] for i in OUTLIER_LAYERS)
        high_act = sum(activations[i] for i in HIGH_LAYERS)

        outlier_frac = outlier_act / total_act
        high_frac = high_act / total_act
        combined_frac = (outlier_act + high_act) / total_act

        results[lang] = {
            "outlier_frac": outlier_frac,
            "high_frac": high_frac,
            "combined_frac": combined_frac,
            "total_activation": total_act,
        }

        print(f"{lang}: outlier={outlier_frac:.3f}, high={high_frac:.3f}, combined={combined_frac:.3f}")

    Path("outlier_activation.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to outlier_activation.json")

    return results


def stage_2():
    """Stage 2: Correlate with degradation."""
    print("=" * 60)
    print("STAGE 2: Correlation Analysis")
    print("=" * 60)

    try:
        results = json.loads(Path("outlier_activation.json").read_text())
    except FileNotFoundError:
        print("[!] Run stage 1 first")
        return None

    if not HAS_SCIPY:
        print("[!] Need scipy for correlation")
        return None

    langs = sorted(set(results.keys()) & set(DEGRADATION.keys()))

    outlier_frac = [results[l]["outlier_frac"] for l in langs]
    combined_frac = [results[l]["combined_frac"] for l in langs]
    degradation = [DEGRADATION[l] for l in langs]

    r_outlier, p_outlier = sp_stats.pearsonr(outlier_frac, degradation)
    r_combined, p_combined = sp_stats.pearsonr(combined_frac, degradation)

    print()
    print("Lang   Outlier%  Combined%  Degradation")
    print("-" * 45)
    for l in langs:
        o = results[l]["outlier_frac"] * 100
        c = results[l]["combined_frac"] * 100
        d = DEGRADATION[l]
        print(f"{l}    {o:5.1f}%    {c:5.1f}%     {d:.3f}")

    print()
    print("=" * 60)
    print("CORRELATION RESULTS")
    print("=" * 60)
    print(f"Outlier layers (5,21,22):  r = {r_outlier:+.3f}, p = {p_outlier:.4f}")
    print(f"Combined (4-7,20-23):      r = {r_combined:+.3f}, p = {p_combined:.4f}")

    if r_outlier < 0 and p_outlier < 0.05:
        print("\n[*] CONFIRMED: Less outlier activation → more degradation")
    elif r_outlier < 0:
        print("\n[~] Trend in expected direction but not significant")
    else:
        print("\n[!] Unexpected: positive or no correlation")

    # Save
    correlation = {
        "outlier": {"r": float(r_outlier), "p": float(p_outlier)},
        "combined": {"r": float(r_combined), "p": float(p_combined)},
    }
    Path("outlier_correlation.json").write_text(json.dumps(correlation, indent=2))

    return correlation


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
