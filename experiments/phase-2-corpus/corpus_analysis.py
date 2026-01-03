#!/usr/bin/env python3
"""
Corpus Analysis

Analyze linguistic properties of collected corpus.
Compute statistics that may correlate with quantization behavior.
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import stats as sp_stats

LANGS = ["ara", "eng", "fra", "hin", "jpn", "zho"]

DEGRADATION = {
    "ara": 0.025, "eng": 0.005, "fra": 0.007,
    "hin": 0.021, "jpn": 0.022, "zho": 0.013,
}


def analyze_corpus(lang):
    """Compute corpus statistics."""
    corpus_file = Path(f"corpus/{lang}.txt")
    if not corpus_file.exists():
        return None

    text = corpus_file.read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split("<|endofdoc|>") if d.strip()]

    # Basic stats
    n_docs = len(docs)
    n_chars = sum(len(d) for d in docs)
    n_words = sum(len(d.split()) for d in docs)

    # Character-level stats
    all_text = " ".join(docs)
    char_counts = Counter(all_text)
    unique_chars = len(char_counts)

    # Character entropy (approximates script complexity)
    total_chars = sum(char_counts.values())
    char_probs = np.array([c / total_chars for c in char_counts.values()])
    char_entropy = -np.sum(char_probs * np.log2(char_probs + 1e-10))

    # Average word length (approximates morphological complexity)
    words = all_text.split()
    avg_word_len = np.mean([len(w) for w in words if w])

    # Punctuation density
    punct_count = sum(1 for c in all_text if c in ".,;:!?\"'()[]{}")
    punct_density = punct_count / len(all_text)

    # Type-token ratio (vocabulary richness, sampled)
    sample_words = words[:10000]
    ttr = len(set(sample_words)) / len(sample_words) if sample_words else 0

    return {
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_words": n_words,
        "unique_chars": unique_chars,
        "char_entropy": char_entropy,
        "avg_word_len": avg_word_len,
        "punct_density": punct_density,
        "type_token_ratio": ttr,
    }


def main():
    print("=" * 60)
    print("CORPUS ANALYSIS")
    print("=" * 60)

    results = {}

    print("\nBasic Statistics:")
    print("-" * 70)
    print(f"{'Lang':<6} {'Docs':<8} {'Words':<10} {'Chars':<10} {'Entropy':<8} {'TTR':<6}")
    print("-" * 70)

    for lang in LANGS:
        stats = analyze_corpus(lang)
        if stats:
            results[lang] = stats
            print(f"{lang:<6} {stats['n_docs']:<8} {stats['n_words']:<10} "
                  f"{stats['n_chars']:<10} {stats['char_entropy']:<8.2f} "
                  f"{stats['type_token_ratio']:<6.3f}")

    # Correlation with degradation
    print("\n" + "=" * 60)
    print("CORRELATION WITH DEGRADATION")
    print("=" * 60)

    langs_common = [l for l in LANGS if l in results and l in DEGRADATION]

    correlations = {}

    for metric in ["char_entropy", "avg_word_len", "punct_density", "type_token_ratio"]:
        vals = [results[l][metric] for l in langs_common]
        degrad = [DEGRADATION[l] for l in langs_common]

        r, p = sp_stats.pearsonr(vals, degrad)
        correlations[metric] = {"r": r, "p": p}

        sig = "*" if p < 0.05 else ""
        print(f"{metric:<20}: r = {r:+.3f}, p = {p:.3f} {sig}")

    # Character entropy might relate to tokenization efficiency
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
Character entropy measures script complexity:
  - High entropy (CJK, Arabic): more unique characters
  - Low entropy (Latin): simpler character set

This may correlate with tokenization efficiency and model behavior.
However, with n=6, statistical power is limited.
""")

    # Save
    Path("corpus_analysis.json").write_text(json.dumps({
        "per_language": results,
        "correlations": correlations,
    }, indent=2))
    print("Saved to corpus_analysis.json")


if __name__ == "__main__":
    main()
