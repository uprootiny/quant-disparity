#!/usr/bin/env python3
"""
Weight Distribution Analysis — Banner et al. Framework

Analyzes weight distributions for language-activated neurons.
Computes optimal clipping thresholds and correlates with degradation.

Usage:
    python3 distrib_analysis.py
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from scipy import stats as sp_stats
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[!] scipy not available, using approximations")


@dataclass
class WeightStats:
    """Weight tensor statistics."""
    mean: float
    std: float
    kurtosis: float      # Excess kurtosis (0 = Gaussian)
    skewness: float
    outlier_ratio: float # Fraction beyond 3 sigma
    n_params: int


@dataclass
class ClippingResult:
    """Optimal clipping analysis."""
    alpha_opt: float     # Optimal threshold
    alpha_sigma: float   # alpha / sigma ratio
    clip_error: float
    quant_error: float
    total_error: float


# =============================================================================
# DATA
# =============================================================================

# Mock weight statistics per language
# TODO: Replace with real extraction from BLOOM-560M
WEIGHT_STATS = {
    # Latin script, analytic — near Gaussian
    "eng": WeightStats(0.001, 0.042, 0.3, 0.1, 0.004, 1000000),
    "fra": WeightStats(0.002, 0.044, 0.4, 0.15, 0.005, 950000),
    "deu": WeightStats(0.001, 0.046, 0.6, 0.2, 0.006, 920000),
    "vie": WeightStats(0.000, 0.041, 0.2, 0.05, 0.003, 800000),

    # Non-Latin — heavy-tailed (more outliers)
    "ara": WeightStats(0.003, 0.055, 2.1, 0.4, 0.018, 650000),
    "heb": WeightStats(0.002, 0.052, 1.8, 0.35, 0.015, 600000),
    "jpn": WeightStats(0.002, 0.058, 2.4, 0.5, 0.022, 700000),
    "zho": WeightStats(0.001, 0.054, 1.9, 0.3, 0.016, 750000),
    "kor": WeightStats(0.002, 0.053, 2.0, 0.38, 0.017, 680000),
    "rus": WeightStats(0.001, 0.048, 1.2, 0.25, 0.010, 850000),
    "hin": WeightStats(0.003, 0.056, 2.2, 0.45, 0.020, 620000),
    "tha": WeightStats(0.002, 0.057, 2.3, 0.48, 0.021, 580000),

    # Agglutinative, Latin — intermediate
    "fin": WeightStats(0.002, 0.050, 1.4, 0.3, 0.012, 720000),
    "tur": WeightStats(0.001, 0.049, 1.3, 0.28, 0.011, 740000),
}

# Degradation from Marchisio et al. (2024)
DEGRADATION = {
    "eng": -0.005, "fra": -0.007, "deu": -0.008, "vie": -0.009,
    "ara": -0.025, "heb": -0.020, "jpn": -0.022, "zho": -0.013,
    "kor": -0.018, "rus": -0.012, "hin": -0.021, "tha": -0.020,
    "fin": -0.016, "tur": -0.015,
}


# =============================================================================
# BANNER FRAMEWORK
# =============================================================================

def gaussian_clip_error(alpha: float, sigma: float) -> float:
    """Expected clipping error for Gaussian. Banner Eq. 7."""
    if HAS_SCIPY:
        z = alpha / sigma
        sf = sp_stats.norm.sf(z)
        pdf = sp_stats.norm.pdf(z)
        return 2 * sigma**2 * (z * sf + pdf)
    else:
        z = alpha / sigma
        tail = math.exp(-z**2 / 2) / (z * math.sqrt(2 * math.pi))
        return 2 * sigma**2 * tail * (z + 1/z)


def quant_error(alpha: float, bits: int) -> float:
    """Quantization noise. Delta^2 / 12."""
    n_levels = 2**bits - 1
    delta = 2 * alpha / n_levels
    return delta**2 / 12


def optimal_clipping(sigma: float, bits: int = 4) -> ClippingResult:
    """Find optimal clipping threshold. Banner Sec. 3.1."""
    def total(alpha):
        return gaussian_clip_error(alpha, sigma) + quant_error(alpha, bits)

    if HAS_SCIPY:
        result = optimize.minimize_scalar(total, bounds=(0.5*sigma, 6*sigma), method='bounded')
        alpha_opt = result.x
    else:
        best_alpha, best_err = sigma, float('inf')
        for alpha in [sigma * x / 10 for x in range(5, 61)]:
            err = total(alpha)
            if err < best_err:
                best_alpha, best_err = alpha, err
        alpha_opt = best_alpha

    c_err = gaussian_clip_error(alpha_opt, sigma)
    q_err = quant_error(alpha_opt, bits)

    return ClippingResult(
        alpha_opt=alpha_opt,
        alpha_sigma=alpha_opt / sigma,
        clip_error=c_err,
        quant_error=q_err,
        total_error=c_err + q_err
    )


def kurtosis_adjusted_clipping(sigma: float, kurtosis: float, bits: int = 4) -> ClippingResult:
    """Adjust clipping for heavy tails."""
    base = optimal_clipping(sigma, bits)
    adjustment = 1 + 0.1 * max(0, kurtosis)
    alpha_adj = base.alpha_opt * adjustment

    c_err = gaussian_clip_error(alpha_adj, sigma)
    q_err = quant_error(alpha_adj, bits)

    return ClippingResult(
        alpha_opt=alpha_adj,
        alpha_sigma=alpha_adj / sigma,
        clip_error=c_err,
        quant_error=q_err,
        total_error=c_err + q_err
    )


# =============================================================================
# CORRELATION
# =============================================================================

def correlate(xs: list, ys: list) -> tuple:
    """Pearson correlation."""
    if HAS_SCIPY:
        r, p = sp_stats.pearsonr(xs, ys)
        return float(r), float(p)
    else:
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        sx = math.sqrt(sum((x-mx)**2 for x in xs)/n)
        sy = math.sqrt(sum((y-my)**2 for y in ys)/n)
        if sx == 0 or sy == 0:
            return 0.0, 1.0
        r = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (n * sx * sy)
        return r, 0.05 if abs(r) > 0.5 else 0.5


# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 70)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    # Analyze each language
    results = {}
    for lang, stats in WEIGHT_STATS.items():
        gauss = optimal_clipping(stats.std)
        adj = kurtosis_adjusted_clipping(stats.std, stats.kurtosis)
        results[lang] = {
            'stats': stats,
            'gaussian': gauss,
            'adjusted': adj,
        }

    # Print table
    print("Lang   std     kurt   alpha_g  alpha_adj  alpha/sigma")
    print("-" * 60)
    for lang in sorted(results):
        r = results[lang]
        s, g, a = r['stats'], r['gaussian'], r['adjusted']
        print(f"{lang}    {s.std:.3f}   {s.kurtosis:+.1f}   {g.alpha_opt:.4f}   {a.alpha_opt:.4f}     {a.alpha_sigma:.2f}")
    print()

    # Correlations
    langs = sorted(set(results) & set(DEGRADATION))
    kurtosis = [results[l]['stats'].kurtosis for l in langs]
    outlier = [results[l]['stats'].outlier_ratio for l in langs]
    alpha_sig = [results[l]['adjusted'].alpha_sigma for l in langs]
    degrad = [abs(DEGRADATION[l]) for l in langs]

    r_k, p_k = correlate(kurtosis, degrad)
    r_o, p_o = correlate(outlier, degrad)
    r_a, p_a = correlate(alpha_sig, degrad)

    print("CORRELATIONS WITH DEGRADATION")
    print("-" * 50)
    print(f"kurtosis      r={r_k:+.3f}  p={p_k:.4f}  {'[*]' if p_k < 0.05 else ''}")
    print(f"outlier_ratio r={r_o:+.3f}  p={p_o:.4f}  {'[*]' if p_o < 0.05 else ''}")
    print(f"alpha/sigma   r={r_a:+.3f}  p={p_a:.4f}  {'[*]' if p_a < 0.05 else ''}")
    print()

    # Two regimes
    low_k = [l for l in langs if results[l]['stats'].kurtosis < 1.0]
    high_k = [l for l in langs if results[l]['stats'].kurtosis >= 1.5]

    print("QUANTIZATION REGIMES")
    print("-" * 50)
    print(f"Low kurtosis (<1.0):  {low_k}")
    print(f"  -> alpha/sigma ~ {sum(results[l]['gaussian'].alpha_sigma for l in low_k)/len(low_k):.2f}")
    print(f"High kurtosis (>=1.5): {high_k}")
    print(f"  -> alpha/sigma ~ {sum(results[l]['adjusted'].alpha_sigma for l in high_k)/len(high_k):.2f}")
    print()

    print("=" * 70)

    # Save results
    out = {
        'correlations': {
            'kurtosis': {'r': r_k, 'p': p_k},
            'outlier_ratio': {'r': r_o, 'p': p_o},
            'alpha_sigma': {'r': r_a, 'p': p_a},
        },
        'regimes': {
            'low_kurtosis': low_k,
            'high_kurtosis': high_k,
        }
    }
    Path('results.json').write_text(json.dumps(out, indent=2))
    print("Results written to results.json")


if __name__ == "__main__":
    run()
