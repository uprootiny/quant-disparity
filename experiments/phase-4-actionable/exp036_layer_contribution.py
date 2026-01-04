#!/usr/bin/env python3
"""
EXP-036: Layer Contribution Analysis

Question: Which layers contribute most to each language's performance?

Hypothesis H-036: Languages rely on different layer subsets; overlap with
outlier layers predicts vulnerability.

Method:
1. Compute layer-wise activations per language
2. Measure activation magnitude distribution per layer
3. Identify "critical layers" per language (top 20% by activation)
4. Correlate with outlier layer locations

Prediction: Low-resource languages have critical layers overlapping with high-κ layers.

Actionable outcome: Layer-specific quantization strategies.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, kurtosis
import torch

# Test sentences per language
TEST_SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "fr": "Le rapide renard brun saute par-dessus le chien paresseux.",
    "es": "El rápido zorro marrón salta sobre el perro perezoso.",
    "zh": "敏捷的棕色狐狸跳过懒惰的狗。",
    "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "he": "השועל החום המהיר קופץ מעל הכלב העצלן.",
    "ru": "Быстрая коричневая лиса прыгает через ленивую собаку.",
}

RESOURCE_LEVELS = {
    "en": 1.0, "de": 0.8, "fr": 0.75, "es": 0.6,
    "zh": 0.5, "ar": 0.35, "he": 0.2, "ru": 0.45,
}


def get_layer_activations(model, tokenizer, text: str) -> dict:
    """
    Get activation magnitudes per layer for a given text.
    Returns dict: layer_idx -> mean activation magnitude
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    layer_activations = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Mean absolute activation
            layer_activations[layer_idx] = hidden.abs().mean().item()
        return hook

    # Register hooks
    hooks = []
    if hasattr(model, 'transformer'):
        # GPT-2 style
        for idx, layer in enumerate(model.transformer.h):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))
    elif hasattr(model, 'encoder'):
        # BERT style
        for idx, layer in enumerate(model.encoder.layer):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))
    elif hasattr(model, 'layers'):
        # OPT style
        for idx, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Wrapped OPT
        for idx, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))

    # Forward pass
    with torch.no_grad():
        try:
            model(**inputs)
        except Exception:
            pass

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return layer_activations


def get_layer_kurtosis(model) -> dict:
    """Get kurtosis of attention weights per layer."""
    layer_kurtosis = {}

    for name, param in model.named_parameters():
        if 'weight' not in name or 'attn' not in name.lower():
            continue

        # Extract layer number
        import re
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(1))
            weights = param.detach().cpu().numpy().flatten()
            k = kurtosis(weights, fisher=True)

            if layer_idx not in layer_kurtosis:
                layer_kurtosis[layer_idx] = []
            layer_kurtosis[layer_idx].append(k)

    # Max kurtosis per layer
    return {layer: max(vals) for layer, vals in layer_kurtosis.items()}


def identify_critical_layers(activations: dict, threshold_pct: float = 20) -> set:
    """
    Identify layers with highest activation contribution.
    Returns set of layer indices in top threshold_pct.
    """
    if not activations:
        return set()

    sorted_layers = sorted(activations.items(), key=lambda x: x[1], reverse=True)
    n_critical = max(1, int(len(sorted_layers) * threshold_pct / 100))

    return set(layer for layer, _ in sorted_layers[:n_critical])


def identify_outlier_layers(kurtosis_dict: dict, threshold: float = 50) -> set:
    """Identify layers with kurtosis above threshold."""
    return set(layer for layer, k in kurtosis_dict.items() if k > threshold)


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("EXP-036: Layer Contribution Analysis")
    print("=" * 60)

    results = {
        "experiment_id": "EXP-036",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-036: Critical layer overlap with outlier layers predicts vulnerability",
        "models": {}
    }

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    models_to_test = [
        ("gpt2", "GPT-2"),
        ("bert-base-multilingual-cased", "mBERT"),
    ]

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Analyzing: {model_label}")
        print("=" * 50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(model_name)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Get layer kurtosis
        print("\n1. Computing layer kurtosis...")
        layer_kurtosis = get_layer_kurtosis(model)

        if layer_kurtosis:
            print(f"   Layers with κ > 50: {[l for l, k in layer_kurtosis.items() if k > 50]}")
            outlier_layers = identify_outlier_layers(layer_kurtosis, threshold=50)
        else:
            print("   No attention layers found with standard naming")
            outlier_layers = set()

        # Get activations per language
        print("\n2. Computing layer activations per language...")
        lang_activations = {}
        lang_critical_layers = {}

        for lang, text in TEST_SENTENCES.items():
            activations = get_layer_activations(model, tokenizer, text)
            if activations:
                lang_activations[lang] = activations
                lang_critical_layers[lang] = identify_critical_layers(activations)
                print(f"   {lang}: Critical layers = {sorted(lang_critical_layers[lang])}")

        # Compute overlap with outlier layers
        print("\n3. Computing overlap with outlier layers...")
        lang_overlap = {}

        for lang in lang_activations:
            if outlier_layers and lang_critical_layers[lang]:
                overlap = lang_critical_layers[lang] & outlier_layers
                overlap_pct = len(overlap) / len(lang_critical_layers[lang]) * 100
                lang_overlap[lang] = {
                    "critical_layers": sorted(lang_critical_layers[lang]),
                    "overlap_with_outliers": sorted(overlap),
                    "overlap_percentage": overlap_pct,
                    "resource_level": RESOURCE_LEVELS.get(lang, 0.5)
                }
                print(f"   {lang}: {overlap_pct:.1f}% overlap with outlier layers")

        # Correlation analysis
        print(f"\n{'='*50}")
        print("Correlation Analysis")
        print("=" * 50)

        if len(lang_overlap) >= 4:
            resources = [lang_overlap[l]["resource_level"] for l in lang_overlap]
            overlaps = [lang_overlap[l]["overlap_percentage"] for l in lang_overlap]

            r, p = pearsonr(resources, overlaps)

            print(f"\nResource level vs outlier overlap:")
            print(f"  r = {r:.4f}, p = {p:.4f}")
            print(f"\n  Prediction: r < 0 (low resource → more overlap)")
            print(f"  Result: {'SUPPORTED' if r < 0 else 'NOT SUPPORTED'}")

            results["models"][model_label] = {
                "layer_kurtosis": {str(k): v for k, v in layer_kurtosis.items()},
                "outlier_layers": sorted(outlier_layers),
                "language_analysis": lang_overlap,
                "correlation": {"r": r, "p": p},
                "hypothesis_supported": r < 0
            }
        else:
            print("   Insufficient data for correlation")
            results["models"][model_label] = {
                "layer_kurtosis": {str(k): v for k, v in layer_kurtosis.items()},
                "outlier_layers": sorted(outlier_layers),
                "language_analysis": lang_overlap,
            }

        # Visualization of layer importance per language
        print(f"\n{'='*50}")
        print("Layer Importance by Language")
        print("=" * 50)

        if lang_activations:
            # Get all layers
            all_layers = set()
            for acts in lang_activations.values():
                all_layers.update(acts.keys())
            all_layers = sorted(all_layers)

            print(f"\n{'Layer':<8}", end="")
            for lang in sorted(lang_activations.keys()):
                print(f"{lang:<8}", end="")
            print("κ")
            print("-" * (8 + 8 * len(lang_activations) + 8))

            for layer in all_layers:
                print(f"{layer:<8}", end="")
                for lang in sorted(lang_activations.keys()):
                    act = lang_activations[lang].get(layer, 0)
                    # Normalize to percentage of max
                    max_act = max(lang_activations[lang].values())
                    pct = act / max_act * 100 if max_act > 0 else 0
                    marker = "██" if pct > 80 else "▓▓" if pct > 60 else "▒▒" if pct > 40 else "░░"
                    print(f"{marker:<8}", end="")
                k = layer_kurtosis.get(layer, 0)
                print(f"{k:.1f}")

        del model
        del tokenizer

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"exp036_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
