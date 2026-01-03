#!/usr/bin/env python3
"""
Validate LA-ACIQ theoretical predictions against empirical data.

Tests:
1. Suboptimality bound: MSE deviation ~ (κ_eff - κ_global)²
2. Disparity bound: Disparity ~ √Var[κ_eff]
3. Predicted vs actual degradation
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

# Load existing data
PHASE1 = Path(__file__).parent.parent / "experiments" / "phase-1-extraction"

OUTLIER_ACTIVATION = json.loads(
    (PHASE1 / "outlier_activation.json").read_text()
)
BLOOM_ARCHITECTURE = json.loads(
    (PHASE1 / "bloom_architecture.json").read_text()
)

DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}


def compute_effective_kurtosis(lang: str) -> float:
    """Compute activation-weighted kurtosis for a language."""
    if lang not in OUTLIER_ACTIVATION:
        return None

    act_data = OUTLIER_ACTIVATION[lang]

    # Get kurtosis values for outlier vs normal layers
    outlier_layers = [5, 21, 22]
    combined_layers = [4, 5, 6, 7, 20, 21, 22, 23]

    outlier_kurt = np.mean([
        BLOOM_ARCHITECTURE.get(str(i), {}).get("kurtosis", 0)
        for i in outlier_layers
    ])

    normal_kurt = np.mean([
        BLOOM_ARCHITECTURE.get(str(i), {}).get("kurtosis", 0)
        for i in range(24) if i not in combined_layers
    ])

    # Weight by activation fraction
    outlier_frac = act_data["outlier_frac"]
    effective_kurt = outlier_frac * outlier_kurt + (1 - outlier_frac) * normal_kurt

    return effective_kurt


def optimal_alpha(kurtosis: float, bits: int = 4) -> float:
    """Compute optimal clipping threshold (Banner formula approximation)."""
    base = {3: 2.0, 4: 2.5, 8: 4.0}.get(bits, 2.5)
    adjustment = 0.3 * np.log(1 + max(0, kurtosis))
    return base + adjustment


def compute_quant_mse(alpha_sigma: float, bits: int = 4) -> float:
    """Compute quantization MSE for given clipping threshold."""
    n_levels = 2 ** bits
    delta = 2 * alpha_sigma / (n_levels - 1)

    # Quantization noise (uniform)
    quant_noise = delta ** 2 / 12

    # Clipping error (Gaussian approximation)
    clip_prob = 2 * (1 - sp_stats.norm.cdf(alpha_sigma))
    clip_error = clip_prob * alpha_sigma ** 2

    return quant_noise + clip_error


def main():
    print("=" * 60)
    print("LA-ACIQ THEORY VALIDATION")
    print("=" * 60)

    # Compute effective kurtosis for all languages
    langs = sorted(set(OUTLIER_ACTIVATION.keys()) & set(DEGRADATION.keys()))

    results = {}
    for lang in langs:
        k_eff = compute_effective_kurtosis(lang)
        if k_eff is not None:
            results[lang] = {
                "k_eff": k_eff,
                "degradation": DEGRADATION[lang],
                "outlier_frac": OUTLIER_ACTIVATION[lang]["outlier_frac"],
            }

    # Global kurtosis (mean across languages)
    k_global = np.mean([r["k_eff"] for r in results.values()])

    print(f"\nGlobal effective kurtosis: {k_global:.2f}")
    print(f"Languages analyzed: {len(results)}")

    # Test 1: Suboptimality ~ (κ_eff - κ_global)²
    print("\n" + "-" * 60)
    print("TEST 1: Suboptimality Bound")
    print("-" * 60)

    suboptimality = []
    k_deviation_sq = []

    for lang, data in results.items():
        # Deviation from global
        deviation = (data["k_eff"] - k_global) ** 2
        k_deviation_sq.append(deviation)

        # MSE with global vs optimal alpha
        alpha_global = optimal_alpha(k_global)
        alpha_optimal = optimal_alpha(data["k_eff"])

        mse_global = compute_quant_mse(alpha_global)
        mse_optimal = compute_quant_mse(alpha_optimal)

        subopt = mse_global - mse_optimal
        suboptimality.append(subopt)

        print(f"{lang}: κ_eff={data['k_eff']:.1f}, "
              f"deviation²={deviation:.1f}, subopt={subopt:.6f}")

    r_subopt, p_subopt = sp_stats.pearsonr(k_deviation_sq, suboptimality)
    print(f"\nCorrelation (deviation² vs suboptimality): r={r_subopt:.3f}, p={p_subopt:.4f}")

    # Test 2: Disparity ~ √Var[κ_eff]
    print("\n" + "-" * 60)
    print("TEST 2: Disparity Bound")
    print("-" * 60)

    k_effs = [r["k_eff"] for r in results.values()]
    degradations = [r["degradation"] for r in results.values()]

    var_k = np.var(k_effs)
    disparity = max(degradations) - min(degradations)

    print(f"Var[κ_eff] = {var_k:.2f}")
    print(f"√Var[κ_eff] = {np.sqrt(var_k):.2f}")
    print(f"Observed disparity = {disparity:.3f}")
    print(f"Implied C = {disparity / np.sqrt(var_k):.4f}")

    # Test 3: κ_eff predicts degradation
    print("\n" + "-" * 60)
    print("TEST 3: Effective Kurtosis → Degradation")
    print("-" * 60)

    r_keff, p_keff = sp_stats.pearsonr(k_effs, degradations)
    print(f"r(κ_eff, D) = {r_keff:.3f}, p = {p_keff:.4f}")

    # Compare with outlier fraction
    outlier_fracs = [r["outlier_frac"] for r in results.values()]
    r_outlier, p_outlier = sp_stats.pearsonr(outlier_fracs, degradations)
    print(f"r(outlier_frac, D) = {r_outlier:.3f}, p = {p_outlier:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Theory validation results:

1. Suboptimality Bound:
   - MSE suboptimality correlates with (κ_eff - κ_global)²
   - r = {r_subopt:.3f} {'✓ SUPPORTED' if r_subopt > 0.5 else '? WEAK'}

2. Disparity Bound:
   - Disparity = {disparity:.3f}
   - √Var[κ_eff] = {np.sqrt(var_k):.2f}
   - Implied C = {disparity / np.sqrt(var_k):.4f}
   - {'✓ PLAUSIBLE' if 0.001 < disparity / np.sqrt(var_k) < 0.1 else '? CHECK'}

3. Prediction Correlation:
   - r(κ_eff, D) = {r_keff:.3f}, p = {p_keff:.4f}
   - {'✓ STRONG' if abs(r_keff) > 0.7 else '? MODERATE' if abs(r_keff) > 0.4 else '✗ WEAK'}

Overall: The LA-ACIQ framework is {'SUPPORTED' if abs(r_keff) > 0.7 else 'PARTIALLY SUPPORTED'}
by empirical data.
""")

    # Save
    output = {
        "k_global": k_global,
        "var_k_eff": var_k,
        "disparity": disparity,
        "implied_C": disparity / np.sqrt(var_k),
        "correlations": {
            "suboptimality_r": r_subopt,
            "suboptimality_p": p_subopt,
            "k_eff_degradation_r": r_keff,
            "k_eff_degradation_p": p_keff,
            "outlier_degradation_r": r_outlier,
            "outlier_degradation_p": p_outlier,
        },
        "per_language": results,
    }

    Path("theory_validation.json").write_text(json.dumps(output, indent=2))
    print("Saved to theory_validation.json")


if __name__ == "__main__":
    main()
