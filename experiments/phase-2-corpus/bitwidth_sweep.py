#!/usr/bin/env python3
"""
EXP-009: Bit-Width Sweep

Tests quantization degradation across bit-widths to find per-language thresholds.

Hypothesis: Languages with lower outlier activation will hit quality cliffs
at higher bit-widths.

Predictions:
  - ara, hin: threshold > 4 bits
  - eng, fra: threshold < 3.5 bits

Stages:
  0: Load corpus and compute baseline perplexity
  1: Quantize model at each bit-width, measure perplexity
  2: Find threshold (bit-width where degradation > 10%)

Usage:
    python3 bitwidth_sweep.py --stage 0
    python3 bitwidth_sweep.py --stage 1 --bits 8
    python3 bitwidth_sweep.py --stage 2
"""

import argparse
import json
import gc
from pathlib import Path
from datetime import datetime

import numpy as np

# Languages with collected corpus
CORPUS_LANGS = ["ara", "eng", "fra", "hin", "jpn", "zho"]

# Expected outlier activation (from EXP-007)
OUTLIER_ACTIVATION = {
    "ara": 0.177, "eng": 0.205, "fra": 0.202,
    "hin": 0.172, "jpn": 0.181, "zho": 0.183,
}

# Bit-widths to test
BIT_WIDTHS = [8, 4, 3, 2]


def load_corpus_samples(lang, n_samples=100, max_chars=500):
    """Load samples from corpus."""
    corpus_file = Path(f"corpus/{lang}.txt")
    if not corpus_file.exists():
        return None

    text = corpus_file.read_text(encoding="utf-8")
    docs = text.split("<|endofdoc|>")
    docs = [d.strip()[:max_chars] for d in docs if len(d.strip()) > 100]

    return docs[:n_samples]


def compute_perplexity(model, tokenizer, texts, max_length=128):
    """Compute mean perplexity over texts."""
    import torch

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

            n_tokens = inputs["input_ids"].shape[1]
            total_loss += loss * n_tokens
            total_tokens += n_tokens

    return np.exp(total_loss / total_tokens)


def stage_0():
    """Stage 0: Baseline perplexity (FP32)."""
    print("=" * 60)
    print("STAGE 0: Baseline Perplexity (FP32)")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[!] Missing: {e}")
        return None

    print("Loading BLOOM-560M (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print("Loaded.")
    print()

    results = {}

    for lang in CORPUS_LANGS:
        samples = load_corpus_samples(lang)
        if not samples:
            print(f"{lang}: No corpus found")
            continue

        print(f"{lang}: computing perplexity on {len(samples)} samples...")
        ppl = compute_perplexity(model, tokenizer, samples)
        results[lang] = {"baseline_ppl": float(ppl), "n_samples": len(samples)}
        print(f"  PPL = {ppl:.2f}")

    Path("baseline_perplexity.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to baseline_perplexity.json")

    del model
    gc.collect()

    return results


def stage_1(bits):
    """Stage 1: Quantized perplexity at given bit-width."""
    print("=" * 60)
    print(f"STAGE 1: Quantized Perplexity (INT{bits})")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"[!] Missing: {e}")
        return None

    # BitsAndBytes quantization config
    if bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        print(f"[!] BitsAndBytes doesn't support {bits}-bit directly")
        print("    Using simulated quantization instead")
        quant_config = None

    print(f"Loading BLOOM-560M (INT{bits})...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # Simulated quantization for INT3/INT2
        model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        # Apply simulated quantization
        print("Applying simulated quantization...")
        simulate_quantization(model, bits)

    print("Loaded.")
    print()

    results = {}

    for lang in CORPUS_LANGS:
        samples = load_corpus_samples(lang)
        if not samples:
            continue

        print(f"{lang}: computing perplexity...")
        ppl = compute_perplexity(model, tokenizer, samples)
        results[lang] = {"ppl": float(ppl), "bits": bits}
        print(f"  PPL = {ppl:.2f}")

    output_file = f"perplexity_int{bits}.json"
    Path(output_file).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {output_file}")

    del model
    gc.collect()

    return results


def simulate_quantization(model, bits):
    """Simulate low-bit quantization by rounding weights."""
    import torch

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            # Compute scale
            max_val = param.data.abs().max()
            n_levels = 2 ** bits
            scale = max_val / (n_levels / 2)

            # Quantize and dequantize
            param.data = torch.round(param.data / scale) * scale


def stage_2():
    """Stage 2: Find thresholds and correlate with outlier activation."""
    print("=" * 60)
    print("STAGE 2: Threshold Analysis")
    print("=" * 60)

    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("[!] Need scipy")
        return None

    # Load baseline
    try:
        baseline = json.loads(Path("baseline_perplexity.json").read_text())
    except FileNotFoundError:
        print("[!] Run stage 0 first")
        return None

    # Load all bit-width results
    results = {}
    for bits in BIT_WIDTHS:
        try:
            data = json.loads(Path(f"perplexity_int{bits}.json").read_text())
            for lang, d in data.items():
                if lang not in results:
                    results[lang] = {}
                results[lang][bits] = d["ppl"]
        except FileNotFoundError:
            print(f"[!] Missing perplexity_int{bits}.json")

    if not results:
        print("[!] No quantized results found. Run stage 1 for each bit-width.")
        return None

    # Compute degradation and find thresholds
    print("\nDegradation by bit-width:")
    print("-" * 60)
    print(f"{'Lang':<6} {'Outlier%':<10} " + " ".join(f"INT{b:<4}" for b in BIT_WIDTHS))
    print("-" * 60)

    thresholds = {}

    for lang in sorted(results.keys()):
        if lang not in baseline:
            continue

        base_ppl = baseline[lang]["baseline_ppl"]
        outlier = OUTLIER_ACTIVATION.get(lang, 0) * 100

        row = f"{lang:<6} {outlier:<10.1f}"
        threshold = None

        for bits in BIT_WIDTHS:
            if bits in results[lang]:
                ppl = results[lang][bits]
                degrad = (ppl - base_ppl) / base_ppl * 100
                row += f" {degrad:+5.1f}%"

                # Threshold = first bit-width with >10% degradation
                if degrad > 10 and threshold is None:
                    threshold = bits
            else:
                row += f" {'—':>6}"

        thresholds[lang] = threshold
        print(row)

    # Correlate thresholds with outlier activation
    langs_with_thresh = [l for l in thresholds if thresholds[l] is not None]
    if len(langs_with_thresh) >= 4:
        outlier_vals = [OUTLIER_ACTIVATION[l] for l in langs_with_thresh]
        thresh_vals = [thresholds[l] for l in langs_with_thresh]

        r, p = sp_stats.pearsonr(outlier_vals, thresh_vals)

        print()
        print("=" * 60)
        print("CORRELATION: Outlier Activation vs Threshold")
        print("=" * 60)
        print(f"r = {r:+.3f}, p = {p:.4f}")

        if r < 0 and p < 0.1:
            print("\n[*] CONFIRMED: Lower outlier activation → higher bit threshold needed")
        else:
            print("\n[~] Trend not significant (may need more data points)")

    # Save
    analysis = {
        "baseline": baseline,
        "quantized": results,
        "thresholds": thresholds,
        "timestamp": datetime.now().isoformat(),
    }
    Path("bitwidth_analysis.json").write_text(json.dumps(analysis, indent=2))
    print("\nSaved to bitwidth_analysis.json")

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--bits", type=int, default=8, help="Bit-width for stage 1")
    args = parser.parse_args()

    if args.stage == 0:
        stage_0()
    elif args.stage == 1:
        stage_1(args.bits)
    elif args.stage == 2:
        stage_2()


if __name__ == "__main__":
    main()
