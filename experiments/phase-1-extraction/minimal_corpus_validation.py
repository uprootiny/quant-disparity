#!/usr/bin/env python3
"""
EXP-008b: Minimal Large Corpus Validation

Process ONE language at a time, release model, reload for next.
Trades speed for memory efficiency.
"""

import argparse
import json
import gc
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

OUTLIER_LAYERS = [5, 21, 22]
HIGH_LAYERS = [4, 6, 7, 23]
ALL_OUTLIER = OUTLIER_LAYERS + HIGH_LAYERS

WIKI_MAP = {
    "eng": "en", "fra": "fr", "deu": "de", "ara": "ar",
    "heb": "he", "jpn": "ja", "zho": "zh", "kor": "ko",
    "rus": "ru", "hin": "hi", "tha": "th", "vie": "vi",
    "fin": "fi", "tur": "tr",
}

DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}


def process_single_language(lang, wiki_code, n_samples=50):
    """Process a single language, return stats, free memory."""
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"  Loading Wikipedia ({wiki_code})...")
    wiki = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{wiki_code}",
        split="train",
        streaming=True
    )

    outlier_fracs = []
    combined_fracs = []
    count = 0

    for article in wiki:
        text = article["text"]
        if not text or len(text.strip()) < 50:
            continue

        text = text[:150]  # Short snippets
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)

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

        total = sum(activations.values())
        if total > 0:
            outlier_act = sum(activations[i] for i in OUTLIER_LAYERS)
            combined_act = sum(activations[i] for i in ALL_OUTLIER)
            outlier_fracs.append(outlier_act / total)
            combined_fracs.append(combined_act / total)

        count += 1
        if count >= n_samples:
            break

    # Explicit cleanup
    del model, tokenizer, wiki
    gc.collect()

    if len(outlier_fracs) < 10:
        return None

    outlier_arr = np.array(outlier_fracs)
    combined_arr = np.array(combined_fracs)

    return {
        "n_samples": len(outlier_fracs),
        "outlier_mean": float(np.mean(outlier_arr)),
        "outlier_std": float(np.std(outlier_arr)),
        "outlier_ci95": float(1.96 * np.std(outlier_arr) / np.sqrt(len(outlier_arr))),
        "combined_mean": float(np.mean(combined_arr)),
        "combined_std": float(np.std(combined_arr)),
        "combined_ci95": float(1.96 * np.std(combined_arr) / np.sqrt(len(combined_arr))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", help="Single language to process")
    parser.add_argument("--all", action="store_true", help="Process all languages")
    parser.add_argument("--correlate", action="store_true", help="Run correlation")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    results_file = Path("large_corpus_activation.json")

    if args.lang:
        if args.lang not in WIKI_MAP:
            print(f"Unknown language: {args.lang}")
            sys.exit(1)

        print(f"\n{args.lang}:")
        result = process_single_language(args.lang, WIKI_MAP[args.lang], args.samples)

        if result:
            # Load existing, update, save
            if results_file.exists():
                results = json.loads(results_file.read_text())
            else:
                results = {}
            results[args.lang] = result
            results_file.write_text(json.dumps(results, indent=2))
            print(f"  outlier: {result['outlier_mean']:.4f} ± {result['outlier_ci95']:.4f}")
        else:
            print("  FAILED")

    elif args.all:
        for lang, wiki_code in sorted(WIKI_MAP.items()):
            print(f"\n{'='*40}")
            print(f"{lang}:")
            result = process_single_language(lang, wiki_code, args.samples)

            if result:
                if results_file.exists():
                    results = json.loads(results_file.read_text())
                else:
                    results = {}
                results[lang] = result
                results_file.write_text(json.dumps(results, indent=2))
                print(f"  outlier: {result['outlier_mean']:.4f} ± {result['outlier_ci95']:.4f}")
            else:
                print("  FAILED")

    elif args.correlate:
        if not results_file.exists():
            print("No results file")
            sys.exit(1)

        results = json.loads(results_file.read_text())
        langs = sorted(set(results.keys()) & set(DEGRADATION.keys()))

        if len(langs) < 5:
            print(f"Only {len(langs)} languages, need more data")
            sys.exit(1)

        outlier_means = [results[l]["outlier_mean"] for l in langs]
        degradation = [DEGRADATION[l] for l in langs]

        r, p = sp_stats.pearsonr(outlier_means, degradation)

        print(f"\nCorrelation (n={len(langs)} languages):")
        print(f"  r = {r:+.3f}, p = {p:.4f}")
        print()

        for l in langs:
            o = results[l]["outlier_mean"] * 100
            ci = results[l]["outlier_ci95"] * 100
            d = DEGRADATION[l]
            print(f"  {l}: {o:.2f}% ± {ci:.2f}% → {d:.3f}")

        # Bootstrap CI
        n_boot = 1000
        outlier_arr = np.array(outlier_means)
        deg_arr = np.array(degradation)
        boot_r = []
        for _ in range(n_boot):
            idx = np.random.choice(len(langs), size=len(langs), replace=True)
            r_boot, _ = sp_stats.pearsonr(outlier_arr[idx], deg_arr[idx])
            boot_r.append(r_boot)
        ci_low = np.percentile(boot_r, 2.5)
        ci_high = np.percentile(boot_r, 97.5)

        print(f"\n  Bootstrap 95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")

        # Compare with EXP-007
        print(f"\n  EXP-007 (n=1/lang): r = -0.834")
        print(f"  EXP-008 (n={results[langs[0]]['n_samples']}/lang): r = {r:+.3f}")

        if r < -0.5 and p < 0.05:
            print("\n  [*] REPLICATED")
        elif r < 0:
            print("\n  [~] Trend preserved")
        else:
            print("\n  [!] NOT replicated")


if __name__ == "__main__":
    main()
