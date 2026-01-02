#!/usr/bin/env python3
"""
XGLM Validation: Confirm BLOOM findings on second model.

XGLM-564M is Meta's multilingual model, different architecture from BLOOM.
If r=-0.77 replicates, finding is robust.

Stages:
  0: Test model loading
  1: Single language, all layers
  2: All languages
  3: Compare with BLOOM

Usage:
    python3 xglm_validation.py --stage 0
    python3 xglm_validation.py --stage 1 --lang eng
    python3 xglm_validation.py --stage 2
    python3 xglm_validation.py --stage 3
"""

import argparse
import json
import gc
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from scipy import stats as sp_stats
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing: {e}")
    HAS_DEPS = False

# Same samples as BLOOM analysis
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

DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}


def stage_0():
    """Stage 0: Test XGLM model loading."""
    print("=" * 60)
    print("STAGE 0: XGLM Model Loading Test")
    print("=" * 60)

    if not HAS_DEPS:
        return False

    print("Loading facebook/xglm-564M...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
        print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")

        model = AutoModelForCausalLM.from_pretrained(
            "facebook/xglm-564M",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params/1e6:.1f}M params")

        # Quick forward pass
        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"  Forward pass: {outputs.logits.shape}")

        # Check architecture
        n_layers = len(model.model.layers)
        print(f"  Layers: {n_layers}")

        print("\n[OK] Stage 0 passed.")
        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def get_layer_kurtosis(model):
    """Extract per-layer weight kurtosis from XGLM."""
    print("Computing per-layer weight kurtosis...")
    kurtosis = {}

    for i, layer in enumerate(model.model.layers):
        # XGLM uses different naming: fc1, fc2 for MLP
        w1 = layer.fc1.weight.detach().cpu().numpy().flatten()
        w2 = layer.fc2.weight.detach().cpu().numpy().flatten()
        w = np.concatenate([w1, w2])

        kurt = float(sp_stats.kurtosis(w))
        kurtosis[i] = kurt

        del w, w1, w2
        gc.collect()

    return kurtosis


def stage_1(lang="eng"):
    """Stage 1: Single language, all layers."""
    print("=" * 60)
    print(f"STAGE 1: XGLM All Layers for '{lang}'")
    print("=" * 60)

    if not HAS_DEPS:
        return None

    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/xglm-564M",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Get layer kurtosis
    layer_kurtosis = get_layer_kurtosis(model)
    print(f"Layer kurtosis range: {min(layer_kurtosis.values()):.1f} to {max(layer_kurtosis.values()):.1f}")
    print()

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
    for i, layer in enumerate(model.model.layers):
        h = layer.fc2.register_forward_hook(make_hook(i))
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
        kurt = layer_kurtosis[i]
        print(f"{i:>5}   {act:.4f}      {kurt:+.1f}")

    # Weighted kurtosis
    total_act = sum(activations.values())
    weighted_kurt = sum(activations[i] * layer_kurtosis[i] for i in activations) / total_act

    print()
    print(f"Activation-weighted kurtosis: {weighted_kurt:.2f}")

    return {
        "activations": activations,
        "layer_kurtosis": layer_kurtosis,
        "weighted_kurtosis": weighted_kurt
    }


def stage_2():
    """Stage 2: All languages."""
    print("=" * 60)
    print("STAGE 2: XGLM All Languages")
    print("=" * 60)

    if not HAS_DEPS:
        return None

    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/xglm-564M",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Get layer kurtosis once
    layer_kurtosis = get_layer_kurtosis(model)
    print(f"Layer kurtosis range: {min(layer_kurtosis.values()):.1f} to {max(layer_kurtosis.values()):.1f}")
    print()

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
        for i, layer in enumerate(model.model.layers):
            h = layer.fc2.register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks:
            h.remove()

        # Weighted kurtosis
        total_act = sum(activations.values())
        weighted_kurt = sum(activations[i] * layer_kurtosis[i] for i in activations) / total_act

        results[lang] = {
            "weighted_kurtosis": weighted_kurt,
            "tokens": inputs['input_ids'].shape[1],
        }

        print(f"{lang}: w.kurt={weighted_kurt:.2f}, tokens={inputs['input_ids'].shape[1]}")

        gc.collect()

    # Save
    Path("xglm_activations.json").write_text(json.dumps(results, indent=2))
    Path("xglm_layer_kurtosis.json").write_text(json.dumps(layer_kurtosis, indent=2))
    print("\nSaved to xglm_activations.json")

    return results


def stage_3():
    """Stage 3: Compare with BLOOM."""
    print("=" * 60)
    print("STAGE 3: BLOOM vs XGLM Comparison")
    print("=" * 60)

    # Load both results
    try:
        bloom = json.loads(Path("layer_activations.json").read_text())
        xglm = json.loads(Path("xglm_activations.json").read_text())
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        print("Run stage 2 for both models first.")
        return None

    langs = sorted(set(bloom.keys()) & set(xglm.keys()) & set(DEGRADATION.keys()))

    bloom_kurt = [bloom[l]["weighted_kurtosis"] for l in langs]
    xglm_kurt = [xglm[l]["weighted_kurtosis"] for l in langs]
    degradation = [DEGRADATION[l] for l in langs]

    r_bloom, p_bloom = sp_stats.pearsonr(bloom_kurt, degradation)
    r_xglm, p_xglm = sp_stats.pearsonr(xglm_kurt, degradation)

    print()
    print("Lang   BLOOM    XGLM    Degradation")
    print("-" * 45)
    for l in langs:
        print(f"{l}    {bloom[l]['weighted_kurtosis']:.2f}    {xglm[l]['weighted_kurtosis']:.2f}      {DEGRADATION[l]:.3f}")

    print()
    print("=" * 60)
    print("CORRELATION COMPARISON")
    print("=" * 60)
    print(f"BLOOM: r = {r_bloom:+.3f}, p = {p_bloom:.4f}")
    print(f"XGLM:  r = {r_xglm:+.3f}, p = {p_xglm:.4f}")

    if r_bloom < 0 and r_xglm < 0:
        print("\n[*] BOTH NEGATIVE: Finding replicates across models!")
    elif (r_bloom < 0) != (r_xglm < 0):
        print("\n[!] DIFFERENT SIGNS: Finding may be model-specific.")
    else:
        print("\n[ ] Both positive or non-significant.")

    # Save comparison
    comparison = {
        "bloom": {"r": float(r_bloom), "p": float(p_bloom)},
        "xglm": {"r": float(r_xglm), "p": float(p_xglm)},
        "replicates": bool(r_bloom < 0 and r_xglm < 0),
    }
    Path("model_comparison.json").write_text(json.dumps(comparison, indent=2))

    return comparison


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
