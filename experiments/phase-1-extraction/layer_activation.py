#!/usr/bin/env python3
"""
Per-Language Layer Activation Analysis

Measures which layers activate most for each language.
Hypothesis: High-degradation languages rely more on high-kurtosis layers.

Stages:
  0: Single language, single layer (sanity check)
  1: Single language, all layers
  2: All languages, all layers
  3: Correlation with degradation

Usage:
    python3 layer_activation.py --stage 0
    python3 layer_activation.py --stage 1 --lang eng
    python3 layer_activation.py --stage 2
    python3 layer_activation.py --stage 3
"""

import argparse
import json
import gc
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing: {e}")
    HAS_DEPS = False

# Sample sentences (native content)
SAMPLES = {
    "eng": "London is the capital and largest city of England and the United Kingdom.",
    "fra": "Paris est la capitale de la France et le chef-lieu de la region.",
    "deu": "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.",
    "ara": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها.",
    "heb": "ירושלים היא בירת ישראל והעיר הגדולה ביותר בה.",
    "jpn": "東京は日本の首都であり、世界最大の都市圏を形成している。",
    "zho": "北京是中华人民共和国的首都，是全国政治中心。",
    "kor": "서울은 대한민국의 수도이자 최대 도시이다.",
    "rus": "Москва является столицей Российской Федерации.",
    "hin": "नई दिल्ली भारत की राजधानी है।",
    "tha": "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
    "vie": "Hanoi la thu do cua Viet Nam.",
    "fin": "Helsinki on Suomen paakaupunki.",
    "tur": "Ankara Turkiye'nin baskentidir.",
}

# Degradation from Marchisio (for correlation)
DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}

# Layer kurtosis from Phase 1 weight analysis
LAYER_KURTOSIS = {
    0: 8.13, 1: 4.71, 2: 1.74, 3: 2.64, 4: 76.28, 5: 125.92,
    6: 44.80, 7: 39.05, 8: 8.03, 9: 8.98, 10: 5.47, 11: 3.44,
    12: 2.84, 13: 1.48, 14: 0.96, 15: 0.98, 16: 1.37, 17: 1.81,
    18: 2.99, 19: 4.63, 20: 16.37, 21: 148.20, 22: 164.30, 23: 36.31,
}


def load_model():
    """Load model and tokenizer."""
    print("Loading BLOOM-560M...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Loaded.")
    return model, tokenizer


def stage_0():
    """Stage 0: Sanity check - single language, single layer."""
    print("=" * 60)
    print("STAGE 0: Sanity Check")
    print("=" * 60)

    if not HAS_DEPS:
        return False

    model, tokenizer = load_model()

    text = SAMPLES["eng"]
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Text: {text[:40]}...")
    print(f"Tokens: {inputs['input_ids'].shape[1]}")

    # Hook for layer 0 only
    activation_magnitude = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # Mean absolute activation
        mag = output.abs().mean().item()
        activation_magnitude.append(mag)

    # Register hook on layer 0 MLP
    hook = model.transformer.h[0].mlp.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(**inputs)

    hook.remove()

    print(f"Layer 0 mean activation magnitude: {activation_magnitude[0]:.4f}")
    print("[OK] Stage 0 passed.")
    return True


def stage_1(lang="eng"):
    """Stage 1: Single language, all layers."""
    print("=" * 60)
    print(f"STAGE 1: All Layers for '{lang}'")
    print("=" * 60)

    if not HAS_DEPS:
        return None

    model, tokenizer = load_model()

    text = SAMPLES.get(lang, SAMPLES["eng"])
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Text: {text[:40]}...")
    print(f"Tokens: {inputs['input_ids'].shape[1]}")
    print()

    # Hooks for all layers
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

    # Results
    print("Layer  Activation  Kurtosis")
    print("-" * 35)
    for i in sorted(activations.keys()):
        act = activations[i]
        kurt = LAYER_KURTOSIS[i]
        print(f"{i:>5}   {act:.4f}      {kurt:+.1f}")

    # Weighted kurtosis by activation
    total_act = sum(activations.values())
    weighted_kurt = sum(activations[i] * LAYER_KURTOSIS[i] for i in activations) / total_act

    print()
    print(f"Activation-weighted kurtosis: {weighted_kurt:.2f}")

    return {"activations": activations, "weighted_kurtosis": weighted_kurt}


def stage_2():
    """Stage 2: All languages, all layers."""
    print("=" * 60)
    print("STAGE 2: All Languages")
    print("=" * 60)

    if not HAS_DEPS:
        return None

    model, tokenizer = load_model()

    results = {}

    for lang in sorted(SAMPLES.keys()):
        text = SAMPLES[lang]
        inputs = tokenizer(text, return_tensors="pt")

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

        # Weighted kurtosis
        total_act = sum(activations.values())
        weighted_kurt = sum(activations[i] * LAYER_KURTOSIS[i] for i in activations) / total_act

        results[lang] = {
            "weighted_kurtosis": weighted_kurt,
            "total_activation": total_act,
            "tokens": inputs['input_ids'].shape[1],
        }

        print(f"{lang}: weighted_kurt={weighted_kurt:.2f}, tokens={inputs['input_ids'].shape[1]}")

        gc.collect()

    # Save
    Path("layer_activations.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to layer_activations.json")
    return results


def stage_3():
    """Stage 3: Correlate with degradation."""
    print("=" * 60)
    print("STAGE 3: Correlation Analysis")
    print("=" * 60)

    # Load results
    try:
        results = json.loads(Path("layer_activations.json").read_text())
    except FileNotFoundError:
        print("Run stage 2 first!")
        return None

    from scipy import stats as sp_stats

    # Get common languages
    langs = sorted(set(results.keys()) & set(DEGRADATION.keys()))

    weighted_kurt = [results[l]["weighted_kurtosis"] for l in langs]
    degradation = [DEGRADATION[l] for l in langs]

    r, p = sp_stats.pearsonr(weighted_kurt, degradation)

    print()
    print("Lang   W.Kurt   Degradation")
    print("-" * 35)
    for l in langs:
        print(f"{l}    {results[l]['weighted_kurtosis']:.2f}      {DEGRADATION[l]:.3f}")

    print()
    print("=" * 60)
    print(f"CORRELATION: r = {r:+.3f}, p = {p:.4f}")
    print("=" * 60)

    if abs(r) > 0.5 and p < 0.05:
        print("[*] SIGNIFICANT: Activation-weighted kurtosis correlates with degradation!")
    else:
        print("[ ] Not significant at p<0.05")

    # Save
    correlation = {"r": r, "p": p, "n": len(langs)}
    Path("correlation.json").write_text(json.dumps(correlation, indent=2))

    return correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--lang", default="eng")
    args = parser.parse_args()

    if args.stage == 0:
        stage_0()
    elif args.stage == 1:
        stage_1(args.lang)
    elif args.stage == 2:
        stage_2()
    elif args.stage == 3:
        stage_3()


if __name__ == "__main__":
    main()
