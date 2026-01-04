#!/usr/bin/env python3
"""
EXP-039: Simulated Intervention Study

Question: Does preserving outlier weights reduce disparity?

Hypothesis H-039: Keeping top-k outliers in FP16 disproportionately helps
low-resource languages.

Method:
1. Identify top-k outlier weights (k = 0.01%, 0.1%, 1%)
2. Simulate quantization with outlier preservation
3. Measure degradation per language
4. Compare disparity ratio (low-resource / high-resource degradation)

Prediction: Higher k → lower disparity ratio.

Actionable outcome: Optimal k for fairness-efficiency tradeoff.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import copy

# Test sentences
TEST_SENTENCES = {
    "en": [
        "The weather is nice today.",
        "She reads books every day.",
        "We need water to live.",
    ],
    "de": [
        "Das Wetter ist heute schön.",
        "Sie liest jeden Tag Bücher.",
        "Wir brauchen Wasser zum Leben.",
    ],
    "fr": [
        "Le temps est beau aujourd'hui.",
        "Elle lit des livres chaque jour.",
        "Nous avons besoin d'eau pour vivre.",
    ],
    "zh": [
        "今天天气很好。",
        "她每天都读书。",
        "我们需要水来生活。",
    ],
    "he": [
        "מזג האוויר יפה היום.",
        "היא קוראת ספרים כל יום.",
        "אנחנו צריכים מים כדי לחיות.",
    ],
    "ar": [
        "الطقس جميل اليوم.",
        "تقرأ الكتب كل يوم.",
        "نحتاج الماء للعيش.",
    ],
    "sw": [
        "Hali ya hewa ni nzuri leo.",
        "Anasoma vitabu kila siku.",
        "Tunahitaji maji ili kuishi.",
    ],
}

RESOURCE_LEVELS = {
    "en": 1.0, "de": 0.8, "fr": 0.75,
    "zh": 0.5, "he": 0.2, "ar": 0.35, "sw": 0.05,
}


def identify_top_k_weights(model, k_percentage: float) -> set:
    """
    Identify the top-k% weights by magnitude.
    Returns set of (param_name, flat_index) tuples.
    """
    all_weights = []

    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        weights = param.detach().cpu().numpy().flatten()
        for idx, w in enumerate(weights):
            all_weights.append((abs(w), name, idx))

    # Sort by magnitude
    all_weights.sort(key=lambda x: x[0], reverse=True)

    # Take top k%
    n_keep = max(1, int(len(all_weights) * k_percentage / 100))
    top_k = set((name, idx) for _, name, idx in all_weights[:n_keep])

    return top_k


def quantize_with_preservation(model, bits: int, preserved_weights: set):
    """
    Quantize model weights, but preserve specified weights in FP16.
    Modifies model in place.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue

            flat = param.view(-1)
            abs_max = flat.abs().max()

            if abs_max == 0:
                continue

            scale = abs_max / (2 ** (bits - 1) - 1)

            for idx in range(len(flat)):
                if (name, idx) in preserved_weights:
                    continue  # Keep original

                # Quantize and dequantize
                q = torch.round(flat[idx] / scale)
                q = torch.clamp(q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                flat[idx] = q * scale


def compute_mlm_loss(model, tokenizer, text: str) -> float:
    """Compute masked language model loss for a sentence."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]

    total_loss = 0.0
    n_tokens = 0

    for i in range(1, min(input_ids.shape[1] - 1, 15)):
        original = input_ids[0, i].item()
        if original in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue

        masked_input = input_ids.clone()
        masked_input[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input, attention_mask=inputs["attention_mask"])
            logits = outputs.logits

        loss = F.cross_entropy(logits[0, i:i+1], torch.tensor([original]))
        total_loss += loss.item()
        n_tokens += 1

    return total_loss / n_tokens if n_tokens > 0 else float('inf')


def compute_language_losses(model, tokenizer, sentences_dict: dict) -> dict:
    """Compute average loss per language."""
    lang_losses = {}
    for lang, sentences in sentences_dict.items():
        losses = []
        for sent in sentences:
            try:
                loss = compute_mlm_loss(model, tokenizer, sent)
                if not np.isinf(loss):
                    losses.append(loss)
            except Exception:
                pass
        if losses:
            lang_losses[lang] = np.mean(losses)
    return lang_losses


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("EXP-039: Simulated Intervention Study")
    print("=" * 60)

    results = {
        "experiment_id": "EXP-039",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-039: Preserving top-k outliers helps low-resource languages more",
        "findings": {}
    }

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    model_name = "bert-base-multilingual-cased"
    model_label = "mBERT"

    print(f"\nLoading {model_label}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_original = AutoModelForMaskedLM.from_pretrained(model_name)
        model_original.eval()
    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    # Baseline: FP16 losses
    print("\n1. Computing FP16 baseline losses...")
    baseline_losses = compute_language_losses(model_original, tokenizer, TEST_SENTENCES)

    print("   FP16 losses:")
    for lang in sorted(baseline_losses.keys()):
        print(f"   {lang}: {baseline_losses[lang]:.4f}")

    # Test different k values
    k_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    bits = 4  # INT4 quantization

    print(f"\n2. Testing preservation at k = {k_values}%...")

    intervention_results = {}

    for k in k_values:
        print(f"\n   k = {k}%:")

        # Identify weights to preserve
        preserved = identify_top_k_weights(model_original, k)
        print(f"   Preserving {len(preserved):,} weights")

        # Create fresh copy and quantize
        model_quant = copy.deepcopy(model_original)
        quantize_with_preservation(model_quant, bits, preserved)

        # Compute losses
        quant_losses = compute_language_losses(model_quant, tokenizer, TEST_SENTENCES)

        # Compute degradation per language
        degradation = {}
        for lang in baseline_losses:
            if lang in quant_losses:
                deg = (quant_losses[lang] - baseline_losses[lang]) / baseline_losses[lang]
                degradation[lang] = deg

        # Compute disparity ratio
        high_resource_deg = np.mean([degradation[l] for l in degradation
                                     if RESOURCE_LEVELS.get(l, 0) > 0.5])
        low_resource_deg = np.mean([degradation[l] for l in degradation
                                    if RESOURCE_LEVELS.get(l, 0) <= 0.5])

        disparity_ratio = low_resource_deg / high_resource_deg if high_resource_deg > 0 else float('inf')

        intervention_results[k] = {
            "n_preserved": len(preserved),
            "degradation": degradation,
            "high_resource_avg_deg": high_resource_deg,
            "low_resource_avg_deg": low_resource_deg,
            "disparity_ratio": disparity_ratio,
        }

        print(f"   High-resource avg degradation: {high_resource_deg*100:.1f}%")
        print(f"   Low-resource avg degradation: {low_resource_deg*100:.1f}%")
        print(f"   Disparity ratio: {disparity_ratio:.2f}x")

        del model_quant

    # Also test no preservation (pure quantization)
    print(f"\n   k = 0% (no preservation):")
    model_quant = copy.deepcopy(model_original)
    quantize_with_preservation(model_quant, bits, set())

    quant_losses = compute_language_losses(model_quant, tokenizer, TEST_SENTENCES)

    degradation = {}
    for lang in baseline_losses:
        if lang in quant_losses:
            deg = (quant_losses[lang] - baseline_losses[lang]) / baseline_losses[lang]
            degradation[lang] = deg

    high_resource_deg = np.mean([degradation[l] for l in degradation
                                 if RESOURCE_LEVELS.get(l, 0) > 0.5])
    low_resource_deg = np.mean([degradation[l] for l in degradation
                                if RESOURCE_LEVELS.get(l, 0) <= 0.5])
    disparity_ratio = low_resource_deg / high_resource_deg if high_resource_deg > 0 else float('inf')

    intervention_results[0] = {
        "n_preserved": 0,
        "degradation": degradation,
        "high_resource_avg_deg": high_resource_deg,
        "low_resource_avg_deg": low_resource_deg,
        "disparity_ratio": disparity_ratio,
    }

    print(f"   High-resource avg degradation: {high_resource_deg*100:.1f}%")
    print(f"   Low-resource avg degradation: {low_resource_deg*100:.1f}%")
    print(f"   Disparity ratio: {disparity_ratio:.2f}x")

    del model_quant
    del model_original

    # Summary
    print(f"\n{'='*60}")
    print("Summary: Preservation Effect on Disparity")
    print("=" * 60)

    print(f"\n{'k%':<10} {'Preserved':<12} {'HR Deg%':<10} {'LR Deg%':<10} {'Disparity':<10}")
    print("-" * 52)

    for k in sorted(intervention_results.keys()):
        r = intervention_results[k]
        print(f"{k:<10} {r['n_preserved']:<12,} "
              f"{r['high_resource_avg_deg']*100:<10.1f} "
              f"{r['low_resource_avg_deg']*100:<10.1f} "
              f"{r['disparity_ratio']:<10.2f}x")

    # Hypothesis evaluation
    print(f"\n{'='*60}")
    print("Hypothesis Evaluation")
    print("=" * 60)

    print("\nH-039: Higher k → lower disparity ratio")

    # Check if disparity decreases as k increases
    k_vals = sorted([k for k in intervention_results.keys() if k > 0])
    disparity_vals = [intervention_results[k]["disparity_ratio"] for k in k_vals]

    if len(k_vals) >= 3:
        r, p = pearsonr(k_vals, disparity_vals)
        print(f"\nCorrelation (k vs disparity): r = {r:.4f}, p = {p:.4f}")
        print(f"Prediction: r < 0 (more preservation → less disparity)")

        supported = r < 0
        print(f"Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

        results["findings"] = {
            "intervention_results": {str(k): v for k, v in intervention_results.items()},
            "correlation": {"r": r, "p": p},
            "hypothesis_supported": supported
        }
    else:
        results["findings"] = {
            "intervention_results": {str(k): v for k, v in intervention_results.items()},
        }

    # Optimal k recommendation
    print(f"\n{'='*60}")
    print("Actionable Recommendation")
    print("=" * 60)

    # Find k that achieves ~1.5x disparity (acceptable) with minimum preservation
    target_disparity = 1.5
    for k in sorted(intervention_results.keys()):
        if intervention_results[k]["disparity_ratio"] <= target_disparity:
            print(f"\nTo achieve disparity ratio ≤ {target_disparity}x:")
            print(f"  → Preserve top {k}% of weights ({intervention_results[k]['n_preserved']:,} weights)")
            break
    else:
        print(f"\nNo tested k achieves disparity ≤ {target_disparity}x")
        print("Consider: larger k values or alternative strategies")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"exp039_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
