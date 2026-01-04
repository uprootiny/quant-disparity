#!/usr/bin/env python3
"""
EXP-039 v2: Intervention Study (Optimized)

Question: Does preserving outlier weights reduce disparity?

Optimizations:
- In-place weight modification (no deep copy)
- Batched weight processing
- Streaming language evaluation
- Incremental result reporting

Method:
1. Compute baseline losses per language
2. For each k in [0, 0.1, 1, 5, 10]:
   a. Identify preservation mask
   b. Quantize in-place
   c. Measure losses
   d. Restore weights
3. Compute disparity ratios
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import gc

# Test sentences - short for speed
TEST_SENTENCES = {
    "en": ["The cat sits.", "Dogs run fast.", "Birds fly high."],
    "de": ["Die Katze sitzt.", "Hunde laufen schnell.", "Vögel fliegen hoch."],
    "fr": ["Le chat dort.", "Les chiens courent.", "Les oiseaux volent."],
    "zh": ["猫坐着。", "狗跑得快。", "鸟飞得高。"],
    "he": ["החתול יושב.", "כלבים רצים.", "ציפורים עפות."],
    "ar": ["القطة تجلس.", "الكلاب تركض.", "الطيور تطير."],
}

RESOURCE_LEVELS = {
    "en": 1.0, "de": 0.8, "fr": 0.75,
    "zh": 0.5, "he": 0.2, "ar": 0.35,
}


class WeightPreserver:
    """Efficiently manage weight preservation and restoration."""

    def __init__(self, model):
        self.model = model
        self.original_weights = {}
        self.weight_info = []

        # Catalog all weight tensors
        print("   Cataloging weights...")
        total_params = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.weight_info.append({
                    'name': name,
                    'shape': param.shape,
                    'numel': param.numel(),
                })
                total_params += param.numel()

        print(f"   Total weight parameters: {total_params:,}")
        self.total_params = total_params

    def backup_weights(self):
        """Store original weights."""
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.original_weights[name] = param.data.clone()

    def restore_weights(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])

    def get_global_threshold(self, preserve_pct: float) -> float:
        """Compute magnitude threshold for top k% preservation."""
        if preserve_pct <= 0:
            return float('inf')

        # Sample weights to estimate threshold efficiently
        all_magnitudes = []
        sample_rate = max(1, self.total_params // 100000)  # Sample ~100k weights

        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                flat = param.data.view(-1).abs()
                sampled = flat[::sample_rate].cpu().numpy()
                all_magnitudes.extend(sampled)

        all_magnitudes = np.array(all_magnitudes)
        threshold_idx = int(len(all_magnitudes) * (1 - preserve_pct / 100))
        threshold = np.partition(all_magnitudes, threshold_idx)[threshold_idx]

        return float(threshold)

    def quantize_with_preservation(self, bits: int, preserve_pct: float) -> dict:
        """Quantize weights, preserving top k% by magnitude."""
        threshold = self.get_global_threshold(preserve_pct)

        preserved_count = 0
        quantized_count = 0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' not in name or not param.requires_grad:
                    continue

                flat = param.data.view(-1)
                abs_vals = flat.abs()

                # Create preservation mask
                mask = abs_vals >= threshold

                # Quantize non-preserved weights
                abs_max = abs_vals.max()
                if abs_max > 0:
                    scale = abs_max / (2 ** (bits - 1) - 1)
                    quantized = torch.round(flat / scale)
                    quantized = torch.clamp(quantized,
                                           -(2 ** (bits - 1)),
                                           2 ** (bits - 1) - 1)
                    dequantized = quantized * scale

                    # Apply: keep original where mask=True, else dequantized
                    flat.data = torch.where(mask, flat, dequantized)

                preserved_count += mask.sum().item()
                quantized_count += (~mask).sum().item()

        return {
            'preserved': preserved_count,
            'quantized': quantized_count,
            'preserve_pct_actual': preserved_count / self.total_params * 100
        }


def compute_loss_batch(model, tokenizer, texts: list) -> float:
    """Compute average loss for a batch of texts."""
    total_loss = 0
    n_valid = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt",
                          padding=True, truncation=True, max_length=32)

        if hasattr(model, 'lm_head'):
            # Causal LM
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    n_valid += 1
        else:
            # Just measure activation magnitude as proxy
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    # Use negative activation magnitude as "loss" proxy
                    total_loss += -outputs.last_hidden_state.abs().mean().item()
                    n_valid += 1

    return total_loss / n_valid if n_valid > 0 else float('inf')


def run_experiment():
    """Main experiment with streaming results."""
    print("=" * 60)
    print("EXP-039 v2: Intervention Study (Optimized)")
    print("=" * 60)

    results = {
        "experiment_id": "EXP-039-v2",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-039: Preserving top-k outliers helps low-resource languages more",
        "interventions": {}
    }

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    # Use GPT-2 for causal LM loss
    model_name = "gpt2"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Initialize weight preserver
    preserver = WeightPreserver(model)
    preserver.backup_weights()

    # Baseline losses
    print("\n1. Computing baseline (FP32) losses...")
    baseline_losses = {}
    for lang, texts in TEST_SENTENCES.items():
        loss = compute_loss_batch(model, tokenizer, texts)
        baseline_losses[lang] = loss
        print(f"   {lang}: {loss:.4f}")

    # Test preservation levels
    k_values = [0, 0.1, 1, 5, 10]
    bits = 4

    print(f"\n2. Testing INT{bits} with preservation levels: {k_values}%")
    print("-" * 70)

    for k in k_values:
        print(f"\n   k = {k}%:")

        # Restore to original weights
        preserver.restore_weights()

        # Quantize with preservation
        stats = preserver.quantize_with_preservation(bits, k)
        print(f"   Preserved: {stats['preserved']:,} ({stats['preserve_pct_actual']:.2f}%)")

        # Compute losses
        quant_losses = {}
        for lang, texts in TEST_SENTENCES.items():
            loss = compute_loss_batch(model, tokenizer, texts)
            quant_losses[lang] = loss

        # Compute degradation
        degradation = {}
        for lang in baseline_losses:
            if lang in quant_losses and baseline_losses[lang] != 0:
                deg = (quant_losses[lang] - baseline_losses[lang]) / abs(baseline_losses[lang])
                degradation[lang] = deg

        # Print per-language results
        print(f"   Degradation: ", end="")
        for lang in sorted(degradation.keys()):
            print(f"{lang}:{degradation[lang]*100:+.1f}% ", end="")
        print()

        # Compute disparity
        high_res = [degradation[l] for l in degradation if RESOURCE_LEVELS.get(l, 0) > 0.5]
        low_res = [degradation[l] for l in degradation if RESOURCE_LEVELS.get(l, 0) <= 0.5]

        if high_res and low_res:
            hr_avg = np.mean([abs(d) for d in high_res])
            lr_avg = np.mean([abs(d) for d in low_res])
            disparity = lr_avg / hr_avg if hr_avg > 0 else float('inf')

            print(f"   High-resource avg: {hr_avg*100:.1f}%, Low-resource avg: {lr_avg*100:.1f}%")
            print(f"   Disparity ratio: {disparity:.2f}x")

            results["interventions"][str(k)] = {
                "preserved_count": stats['preserved'],
                "preserved_pct": stats['preserve_pct_actual'],
                "degradation": degradation,
                "high_resource_deg": hr_avg,
                "low_resource_deg": lr_avg,
                "disparity_ratio": disparity,
            }

        # Force garbage collection
        gc.collect()

    # Restore weights at end
    preserver.restore_weights()

    # Analysis
    print(f"\n{'='*60}")
    print("3. Summary")
    print("=" * 60)

    print(f"\n{'k%':<8} {'Preserved':<15} {'HR Deg':<12} {'LR Deg':<12} {'Disparity':<10}")
    print("-" * 57)

    for k in sorted(results["interventions"].keys(), key=float):
        r = results["interventions"][k]
        print(f"{k:<8} {r['preserved_count']:<15,} "
              f"{r['high_resource_deg']*100:<12.1f}% "
              f"{r['low_resource_deg']*100:<12.1f}% "
              f"{r['disparity_ratio']:<10.2f}x")

    # Hypothesis evaluation
    print(f"\n{'='*60}")
    print("4. Hypothesis Evaluation")
    print("=" * 60)

    k_vals = [float(k) for k in results["interventions"].keys()]
    disparities = [results["interventions"][str(int(k) if k == int(k) else k)]["disparity_ratio"]
                   for k in sorted(k_vals)]

    # Check if disparity decreases as k increases
    if len(k_vals) >= 3:
        from scipy.stats import pearsonr
        r, p = pearsonr(sorted(k_vals), disparities)

        print(f"\nCorrelation (k vs disparity): r = {r:.4f}, p = {p:.4f}")
        print(f"Prediction: r < 0 (more preservation → less disparity)")

        supported = r < 0
        results["hypothesis_supported"] = supported
        results["correlation"] = {"r": r, "p": p}

        print(f"\nH-039: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

        if supported:
            # Find optimal k
            min_disp_k = sorted(k_vals)[np.argmin(disparities)]
            print(f"\nOptimal preservation level: {min_disp_k}%")
            print(f"Achieves disparity ratio: {min(disparities):.2f}x")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"exp039v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
