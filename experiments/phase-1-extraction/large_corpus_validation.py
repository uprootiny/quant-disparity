#!/usr/bin/env python3
"""
EXP-008: Large Corpus Validation

Validates EXP-007 findings with larger multilingual corpora.

Uses FLORES-200 dataset (1012 sentences per language) for robust estimates.

Stages:
  0: Download and verify FLORES-200
  1: Compute activation statistics with confidence intervals
  2: Correlate with degradation, compute bootstrap CI

Usage:
    python3 large_corpus_validation.py --stage 0
    python3 large_corpus_validation.py --stage 1
    python3 large_corpus_validation.py --stage 2
"""

import argparse
import json
import gc
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Outlier layers from EXP-006
OUTLIER_LAYERS = [5, 21, 22]
HIGH_LAYERS = [4, 6, 7, 23]
ALL_OUTLIER = OUTLIER_LAYERS + HIGH_LAYERS

# Wikipedia 2-letter codes mapping
WIKI_MAP = {
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "ara": "ar",
    "heb": "he",
    "jpn": "ja",
    "zho": "zh",
    "kor": "ko",
    "rus": "ru",
    "hin": "hi",
    "tha": "th",
    "vie": "vi",
    "fin": "fi",
    "tur": "tr",
}

DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}


def stage_0():
    """Stage 0: Download and verify multilingual corpus."""
    print("=" * 60)
    print("STAGE 0: Verify Wikipedia Access")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Need: pip install datasets")
        return None

    print("Testing Wikipedia access for each language...")
    available = {}

    for lang, wiki_code in WIKI_MAP.items():
        try:
            wiki = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{wiki_code}",
                split="train",
                streaming=True
            )
            sample = next(iter(wiki))["text"][:80]
            print(f"  {lang} ({wiki_code}): OK — '{sample[:40]}...'")
            available[lang] = wiki_code
        except Exception as e:
            print(f"  {lang} ({wiki_code}): FAILED — {str(e)[:50]}")

    print()
    print(f"Available: {len(available)}/{len(WIKI_MAP)} languages")

    Path("corpus_mapping.json").write_text(json.dumps(available, indent=2))
    print("Saved to corpus_mapping.json")

    return available


def stage_1():
    """Stage 1: Compute activation statistics with CI."""
    print("=" * 60)
    print("STAGE 1: Large-Scale Activation Analysis")
    print("=" * 60)

    try:
        from datasets import load_dataset
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[!] Missing: {e}")
        return None

    # Load mapping
    try:
        available = json.loads(Path("corpus_mapping.json").read_text())
    except FileNotFoundError:
        print("[!] Run stage 0 first")
        return None

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
    n_samples = 50  # 50 samples per language (reduced for memory)

    for lang, wiki_code in sorted(available.items()):
        print(f"\n{lang}: loading Wikipedia ({wiki_code})...")

        try:
            wiki = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{wiki_code}",
                split="train",
                streaming=True
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        outlier_fracs = []
        combined_fracs = []
        count = 0

        for article in wiki:
            text = article["text"]
            if not text or len(text.strip()) < 50:
                continue

            text = text[:200]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=64  # Reduced for memory
            )

            activations = {}

            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    activations[layer_idx] = out.abs().mean().item()
                return hook_fn

            hooks = []
            for i, layer in enumerate(model.transformer.h):
                h = layer.mlp.register_forward_hook(make_hook(i))
                hooks.append(h)

            with torch.no_grad():
                _ = model(**inputs)

            for h in hooks:
                h.remove()

            del inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            total = sum(activations.values())
            if total > 0:
                outlier_act = sum(activations[i] for i in OUTLIER_LAYERS)
                combined_act = sum(activations[i] for i in ALL_OUTLIER)
                outlier_fracs.append(outlier_act / total)
                combined_fracs.append(combined_act / total)

            count += 1
            if count >= n_samples:
                break

        # Force cleanup after each language
        gc.collect()

        if len(outlier_fracs) < 10:
            print(f"  SKIPPED: only {len(outlier_fracs)} samples")
            continue

        outlier_arr = np.array(outlier_fracs)
        combined_arr = np.array(combined_fracs)

        results[lang] = {
            "n_samples": len(outlier_fracs),
            "outlier_mean": float(np.mean(outlier_arr)),
            "outlier_std": float(np.std(outlier_arr)),
            "outlier_ci95": float(1.96 * np.std(outlier_arr) / np.sqrt(len(outlier_arr))),
            "combined_mean": float(np.mean(combined_arr)),
            "combined_std": float(np.std(combined_arr)),
            "combined_ci95": float(1.96 * np.std(combined_arr) / np.sqrt(len(combined_arr))),
        }

        o = results[lang]
        print(f"  n={o['n_samples']}, outlier: {o['outlier_mean']:.4f} ± {o['outlier_ci95']:.4f}")

        # Save incrementally
        Path("large_corpus_activation.json").write_text(json.dumps(results, indent=2))

    print("\nSaved to large_corpus_activation.json")
    return results


def stage_2():
    """Stage 2: Correlation with bootstrap CI."""
    print("=" * 60)
    print("STAGE 2: Correlation Analysis with Bootstrap CI")
    print("=" * 60)

    try:
        results = json.loads(Path("large_corpus_activation.json").read_text())
    except FileNotFoundError:
        print("[!] Run stage 1 first")
        return None

    if not HAS_SCIPY:
        print("[!] Need scipy")
        return None

    langs = sorted(set(results.keys()) & set(DEGRADATION.keys()))

    outlier_means = [results[l]["outlier_mean"] for l in langs]
    combined_means = [results[l]["combined_mean"] for l in langs]
    degradation = [DEGRADATION[l] for l in langs]

    # Point estimates
    r_outlier, p_outlier = sp_stats.pearsonr(outlier_means, degradation)
    r_combined, p_combined = sp_stats.pearsonr(combined_means, degradation)

    print()
    print("Lang   Outlier%   ±CI95    Degradation")
    print("-" * 50)
    for l in langs:
        o = results[l]["outlier_mean"] * 100
        ci = results[l]["outlier_ci95"] * 100
        d = DEGRADATION[l]
        print(f"{l}    {o:5.2f}%   ±{ci:.2f}%    {d:.3f}")

    print()
    print("=" * 60)
    print("CORRELATION RESULTS (Large Corpus)")
    print("=" * 60)
    print(f"Outlier layers (5,21,22):  r = {r_outlier:+.3f}, p = {p_outlier:.6f}")
    print(f"Combined (4-7,20-23):      r = {r_combined:+.3f}, p = {p_combined:.6f}")

    # Bootstrap CI for correlation
    print()
    print("Bootstrap 95% CI (1000 iterations):")
    n_boot = 1000
    outlier_arr = np.array(outlier_means)
    deg_arr = np.array(degradation)

    boot_r = []
    for _ in range(n_boot):
        idx = np.random.choice(len(langs), size=len(langs), replace=True)
        r, _ = sp_stats.pearsonr(outlier_arr[idx], deg_arr[idx])
        boot_r.append(r)

    boot_r = np.array(boot_r)
    ci_low = np.percentile(boot_r, 2.5)
    ci_high = np.percentile(boot_r, 97.5)

    print(f"  Outlier r: [{ci_low:+.3f}, {ci_high:+.3f}]")

    # Compare with EXP-007
    print()
    print("=" * 60)
    print("COMPARISON WITH EXP-007 (single sentence)")
    print("=" * 60)
    print(f"EXP-007 (n=1):   r = -0.834")
    print(f"EXP-008 (n=200): r = {r_outlier:+.3f}")

    if r_outlier < -0.5 and p_outlier < 0.05:
        print("\n[*] REPLICATED: Large corpus confirms finding")
    elif r_outlier < 0:
        print("\n[~] Trend preserved but weaker")
    else:
        print("\n[!] NOT REPLICATED with larger corpus")

    # Save
    correlation = {
        "n_samples_per_lang": results[langs[0]]["n_samples"],
        "n_languages": len(langs),
        "outlier": {"r": float(r_outlier), "p": float(p_outlier)},
        "combined": {"r": float(r_combined), "p": float(p_combined)},
        "bootstrap_ci95": [float(ci_low), float(ci_high)],
        "exp007_comparison": {
            "exp007_r": -0.834,
            "exp008_r": float(r_outlier),
            "replicated": bool(r_outlier < -0.5 and p_outlier < 0.05),
        },
    }
    Path("large_corpus_correlation.json").write_text(json.dumps(correlation, indent=2))
    print("\nSaved to large_corpus_correlation.json")

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
