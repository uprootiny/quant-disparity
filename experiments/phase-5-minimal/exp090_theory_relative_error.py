#!/usr/bin/env python3
"""
Exp-090 / T-002: Relative Quantization Error Analysis

Hypothesis H1.2: High-variance weights have lower RELATIVE error
because scale factors are larger.

Tests if relative_error = MSE/signal_power correlates with criticality.
"""
import numpy as np

print("=" * 70)
print("EXP-090 / T-002: RELATIVE QUANTIZATION ERROR ANALYSIS")
print("=" * 70)

# Layer statistics from our experiments
LAYER_STATS = {
    0:  {'variance': 0.039, 'max_abs': 0.89},
    1:  {'variance': 0.015, 'max_abs': 0.67},
    2:  {'variance': 0.008, 'max_abs': 0.54},
    3:  {'variance': 0.010, 'max_abs': 0.58},
    4:  {'variance': 0.011, 'max_abs': 0.61},
    5:  {'variance': 0.012, 'max_abs': 0.63},
    6:  {'variance': 0.013, 'max_abs': 0.65},
    7:  {'variance': 0.014, 'max_abs': 0.68},
    8:  {'variance': 0.016, 'max_abs': 0.71},
    9:  {'variance': 0.019, 'max_abs': 0.76},
    10: {'variance': 0.022, 'max_abs': 0.82},
    11: {'variance': 0.026, 'max_abs': 0.91},
}

# Disparity when protected (lower = more critical)
LAYER_DISPARITY = {
    0: 2.6, 1: 381.0, 2: 795.0, 3: 188.0, 4: 156.0, 5: 167.0,
    6: 145.0, 7: 178.0, 8: 134.0, 9: 89.0, 10: 67.0, 11: 55.0,
}

print("\n1. COMPUTING QUANTIZATION ERROR METRICS")
print("-" * 50)

def compute_quantization_error(variance, max_abs, bits=4):
    """
    Compute relative quantization error for a weight distribution.

    Assumes Gaussian distribution with given variance and max.
    INT4 quantization: scale = max / 7, quantize to [-8, 7]
    """
    # Scale factor
    max_val = 7  # INT4
    scale = max_abs / max_val

    # Quantization step size
    delta = scale  # Step between levels

    # Quantization noise variance (uniform quantization)
    # E[(w - Q(w))^2] ≈ Δ²/12 for uniform quantization
    quant_noise_var = (delta ** 2) / 12

    # Clipping error (for Gaussian, ~0.3% of values clipped at 2.5σ)
    # Approximate clipping error contribution
    std = np.sqrt(variance)
    # At max_abs = ~3σ typically, clipping is small
    clip_prob = 2 * (1 - 0.997)  # ~0.3% beyond 3σ
    clip_error = clip_prob * variance  # Approximate

    # Total MSE
    total_mse = quant_noise_var + clip_error

    # Signal power
    signal_power = variance

    # Relative error
    relative_error = total_mse / signal_power if signal_power > 0 else float('inf')

    return {
        'scale': scale,
        'delta': delta,
        'quant_noise_var': quant_noise_var,
        'clip_error': clip_error,
        'total_mse': total_mse,
        'signal_power': signal_power,
        'relative_error': relative_error,
    }

results = []
print(f"{'Layer':<6} {'Variance':<10} {'Scale':<10} {'Rel.Error':<12} {'Disparity':<10}")
print("-" * 50)

for layer in range(12):
    stats = LAYER_STATS[layer]
    error_metrics = compute_quantization_error(stats['variance'], stats['max_abs'])
    disparity = LAYER_DISPARITY[layer]

    results.append({
        'layer': layer,
        'variance': stats['variance'],
        'relative_error': error_metrics['relative_error'],
        'disparity': disparity,
    })

    print(f"L{layer:<5} {stats['variance']:<10.4f} {error_metrics['scale']:<10.4f} "
          f"{error_metrics['relative_error']:<12.4f} {disparity:<10.1f}")

print("\n2. CORRELATION ANALYSIS")
print("-" * 50)

variances = [r['variance'] for r in results]
rel_errors = [r['relative_error'] for r in results]
disparities = [r['disparity'] for r in results]
log_disparities = [np.log(d + 1) for d in disparities]

# Correlations
corr_var_disp = np.corrcoef(variances, log_disparities)[0, 1]
corr_err_disp = np.corrcoef(rel_errors, log_disparities)[0, 1]
corr_var_err = np.corrcoef(variances, rel_errors)[0, 1]

print(f"Correlation (variance vs log-disparity): r = {corr_var_disp:.3f}")
print(f"Correlation (rel_error vs log-disparity): r = {corr_err_disp:.3f}")
print(f"Correlation (variance vs rel_error): r = {corr_var_err:.3f}")

print("\n3. HYPOTHESIS TEST")
print("-" * 50)

# H1.2: High variance → low relative error → more critical
# If true: corr_var_err < 0 AND corr_err_disp > 0

print(f"""
Hypothesis H1.2: High-variance weights have lower relative error

Test 1: Variance negatively correlates with relative error
  Correlation: r = {corr_var_err:.3f}
  Result: {'SUPPORTED' if corr_var_err < -0.5 else 'NOT SUPPORTED'}

Test 2: Relative error correlates with disparity
  Correlation: r = {corr_err_disp:.3f}
  Result: {'SUPPORTED' if corr_err_disp > 0.5 else 'NOT SUPPORTED'}

Overall: {'H1.2 SUPPORTED' if corr_var_err < -0.5 and corr_err_disp > 0.5 else 'H1.2 NOT FULLY SUPPORTED'}
""")

print("\n4. ALTERNATIVE ANALYSIS")
print("-" * 50)

# The issue: variance and relative_error are mathematically related
# rel_error ∝ 1/variance (by definition)
# So correlation is expected!

print("""
NOTE: Relative error = MSE / variance is mathematically inverse to variance.
This makes the correlation trivial, not informative.

Better test: Does relative error BEYOND mathematical expectation predict disparity?

Computing "excess relative error" = actual - expected(1/variance)
""")

# Fit expected relationship: rel_error = k / variance
# Then compute residuals
expected_rel_errors = [0.01 / v for v in variances]  # Simple inverse model
excess_errors = [r - e for r, e in zip(rel_errors, expected_rel_errors)]

corr_excess_disp = np.corrcoef(excess_errors, log_disparities)[0, 1]
print(f"Correlation (excess_error vs log-disparity): r = {corr_excess_disp:.3f}")

print("\n5. REVISED INTERPRETATION")
print("-" * 50)

print(f"""
FINDING:

The direct relative_error → disparity correlation (r = {corr_err_disp:.3f}) is
largely explained by the mathematical relationship: rel_error ∝ 1/variance.

The excess error (beyond mathematical expectation) shows {'strong' if abs(corr_excess_disp) > 0.5 else 'weak'}
correlation with disparity (r = {corr_excess_disp:.3f}).

CONCLUSION:

H1.2 is {'partially supported' if corr_var_err < 0 else 'not supported'}:
- High variance DOES mean lower relative error (by math)
- But this doesn't fully explain why variance predicts criticality
- Other factors must contribute (information content? language-specific usage?)

NEXT: Test H1.1 (information content) and H1.3 (language-specific activation)
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Exp-090 / T-002 Results:

1. Variance → Relative Error: r = {corr_var_err:.3f} (expected inverse relationship)
2. Relative Error → Disparity: r = {corr_err_disp:.3f}
3. Excess Error → Disparity: r = {corr_excess_disp:.3f}

Interpretation:
- Mathematical relationship explains some of variance-criticality correlation
- But not all: excess error shows {'significant' if abs(corr_excess_disp) > 0.3 else 'minimal'} additional signal
- Need to test information content (H1.1) and language activation (H1.3) hypotheses
""")
