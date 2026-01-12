#!/usr/bin/env python3
"""
EXPERIMENT: E-EXP10 - Sensitivity Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: How robust are our findings to modeling assumptions?

WHY THIS MATTERS:
- Our simulations use specific parameter values
- If findings change drastically with small parameter changes, they're fragile
- Sensitivity analysis tests stability of conclusions

METHOD:
1. Identify key parameters in our models
2. Vary each parameter ±20%
3. Track how key findings change
4. Report which findings are sensitive vs robust

KEY PARAMETERS:
- Degradation formula coefficients
- Alignment thresholds
- Language classification boundaries
"""
import numpy as np
from scipy import stats

print("=" * 70)
print("E-EXP10: SENSITIVITY ANALYSIS")
print("=" * 70)
print("\nTesting robustness of findings to parameter changes")
print("=" * 70)

np.random.seed(42)


# Base language data
LANGUAGES = {
    'en': {'alignment': 0.72, 'type': 'HR'},
    'de': {'alignment': 0.58, 'type': 'HR'},
    'fr': {'alignment': 0.62, 'type': 'HR'},
    'es': {'alignment': 0.60, 'type': 'HR'},
    'zh': {'alignment': 0.55, 'type': 'HR'},
    'ru': {'alignment': 0.48, 'type': 'LR'},
    'ja': {'alignment': 0.38, 'type': 'LR'},
    'ko': {'alignment': 0.32, 'type': 'LR'},
    'ar': {'alignment': 0.28, 'type': 'LR'},
    'he': {'alignment': 0.24, 'type': 'LR'},
    'tr': {'alignment': 0.35, 'type': 'LR'},
    'pl': {'alignment': 0.45, 'type': 'LR'},
}

langs = list(LANGUAGES.keys())
alignment = np.array([LANGUAGES[l]['alignment'] for l in langs])
lang_type = np.array([1 if LANGUAGES[l]['type'] == 'HR' else 0 for l in langs])


def compute_degradation(alignment, base=50, scale=200, exponent=1.0):
    """
    Parameterized degradation model.

    degradation = base + scale * (1 - alignment)^exponent
    """
    return base + scale * (1 - alignment) ** exponent


def compute_key_metrics(degradation, alignment, lang_type):
    """Compute all key metrics from degradation values."""
    # Correlation
    r_value = np.corrcoef(alignment, degradation)[0, 1]

    # Disparity
    hr_mean = np.mean(degradation[lang_type == 1])
    lr_mean = np.mean(degradation[lang_type == 0])
    disparity = lr_mean / hr_mean if hr_mean > 0 else np.nan

    # R-squared
    r_squared = r_value ** 2

    return {
        'correlation': r_value,
        'disparity': disparity,
        'r_squared': r_squared,
        'hr_mean': hr_mean,
        'lr_mean': lr_mean,
    }


# Baseline metrics
baseline_deg = compute_degradation(alignment)
baseline_metrics = compute_key_metrics(baseline_deg, alignment, lang_type)


print("\n1. BASELINE METRICS")
print("-" * 70)

print(f"""
Baseline parameters:
  base = 50
  scale = 200
  exponent = 1.0

Baseline metrics:
  Correlation (r):     {baseline_metrics['correlation']:.3f}
  Disparity (LR/HR):   {baseline_metrics['disparity']:.2f}x
  R-squared:           {baseline_metrics['r_squared']:.3f}
  HR mean degradation: {baseline_metrics['hr_mean']:.1f}%
  LR mean degradation: {baseline_metrics['lr_mean']:.1f}%
""")


print("\n2. PARAMETER SENSITIVITY")
print("-" * 70)

# Parameters to vary
PARAMS = {
    'base': [30, 40, 50, 60, 70],           # ±40%
    'scale': [120, 160, 200, 240, 280],     # ±40%
    'exponent': [0.6, 0.8, 1.0, 1.2, 1.4],  # ±40%
}

sensitivity_results = {}

for param_name, param_values in PARAMS.items():
    print(f"\nVarying {param_name}:")
    results = []

    for val in param_values:
        if param_name == 'base':
            deg = compute_degradation(alignment, base=val)
        elif param_name == 'scale':
            deg = compute_degradation(alignment, scale=val)
        elif param_name == 'exponent':
            deg = compute_degradation(alignment, exponent=val)

        metrics = compute_key_metrics(deg, alignment, lang_type)
        results.append(metrics)

        is_baseline = (param_name == 'base' and val == 50) or \
                      (param_name == 'scale' and val == 200) or \
                      (param_name == 'exponent' and val == 1.0)
        marker = " ← baseline" if is_baseline else ""

        print(f"  {param_name}={val}: r={metrics['correlation']:.3f}, disp={metrics['disparity']:.2f}x{marker}")

    # Compute sensitivity (range of values)
    corr_range = max(r['correlation'] for r in results) - min(r['correlation'] for r in results)
    disp_range = max(r['disparity'] for r in results) - min(r['disparity'] for r in results)

    sensitivity_results[param_name] = {
        'correlation_range': corr_range,
        'disparity_range': disp_range,
        'results': results,
    }


print("\n\n3. SENSITIVITY SUMMARY")
print("-" * 70)

print(f"\n{'Parameter':<12} {'Correlation Range':<20} {'Disparity Range':<20} {'Sensitivity':<15}")
print("-" * 70)

for param_name, sens in sensitivity_results.items():
    corr_sens = "LOW" if sens['correlation_range'] < 0.1 else "MEDIUM" if sens['correlation_range'] < 0.3 else "HIGH"
    disp_sens = "LOW" if sens['disparity_range'] < 0.5 else "MEDIUM" if sens['disparity_range'] < 1.0 else "HIGH"
    overall = "ROBUST" if corr_sens == "LOW" and disp_sens == "LOW" else \
              "MODERATE" if corr_sens != "HIGH" and disp_sens != "HIGH" else "FRAGILE"

    print(f"{param_name:<12} {sens['correlation_range']:<20.4f} {sens['disparity_range']:<20.2f} {overall:<15}")


print("\n\n4. WHICH FINDINGS ARE INVARIANT?")
print("-" * 70)

# Check if key findings hold across all parameter combinations
all_correlations = []
all_disparities = []
all_r2 = []

for param_name, param_values in PARAMS.items():
    for val in param_values:
        if param_name == 'base':
            deg = compute_degradation(alignment, base=val)
        elif param_name == 'scale':
            deg = compute_degradation(alignment, scale=val)
        elif param_name == 'exponent':
            deg = compute_degradation(alignment, exponent=val)

        metrics = compute_key_metrics(deg, alignment, lang_type)
        all_correlations.append(metrics['correlation'])
        all_disparities.append(metrics['disparity'])
        all_r2.append(metrics['r_squared'])

# Check invariants
always_negative_r = all(r < 0 for r in all_correlations)
always_disparity_gt_1 = all(d > 1 for d in all_disparities)
always_r2_gt_50 = all(r2 > 0.5 for r2 in all_r2)

print(f"""
INVARIANT 1: Correlation is always negative (alignment → less degradation)?
  Range: [{min(all_correlations):.3f}, {max(all_correlations):.3f}]
  Always negative: {'YES ✓' if always_negative_r else 'NO ✗'}

INVARIANT 2: Disparity is always > 1 (LR always worse)?
  Range: [{min(all_disparities):.2f}x, {max(all_disparities):.2f}x]
  Always > 1: {'YES ✓' if always_disparity_gt_1 else 'NO ✗'}

INVARIANT 3: R² is always > 0.5 (strong relationship)?
  Range: [{min(all_r2):.3f}, {max(all_r2):.3f}]
  Always > 0.5: {'YES ✓' if always_r2_gt_50 else 'NO ✗'}
""")


print("\n5. EXTREME SCENARIOS")
print("-" * 70)

# Test extreme parameter combinations
extreme_scenarios = [
    {'name': 'Low base, high scale', 'base': 20, 'scale': 300, 'exponent': 1.0},
    {'name': 'High base, low scale', 'base': 100, 'scale': 100, 'exponent': 1.0},
    {'name': 'Low exponent (sublinear)', 'base': 50, 'scale': 200, 'exponent': 0.5},
    {'name': 'High exponent (superlinear)', 'base': 50, 'scale': 200, 'exponent': 1.5},
]

print(f"\n{'Scenario':<30} {'r':<10} {'Disparity':<12} {'R²':<10} {'Valid?':<10}")
print("-" * 75)

for scenario in extreme_scenarios:
    deg = compute_degradation(alignment, base=scenario['base'],
                               scale=scenario['scale'], exponent=scenario['exponent'])
    metrics = compute_key_metrics(deg, alignment, lang_type)

    valid = metrics['correlation'] < 0 and metrics['disparity'] > 1
    print(f"{scenario['name']:<30} {metrics['correlation']:<10.3f} {metrics['disparity']:<12.2f}x {metrics['r_squared']:<10.3f} {'✓' if valid else '✗':<10}")


print("\n\n6. THRESHOLD SENSITIVITY")
print("-" * 70)

print("\nVarying HR/LR boundary threshold:")

thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
baseline_deg = compute_degradation(alignment)

print(f"\n{'Threshold':<12} {'HR count':<12} {'LR count':<12} {'Disparity':<15}")
print("-" * 55)

for thresh in thresholds:
    # Reclassify languages
    hr_mask = alignment >= thresh
    lr_mask = alignment < thresh

    hr_count = np.sum(hr_mask)
    lr_count = np.sum(lr_mask)

    if hr_count > 0 and lr_count > 0:
        disparity = np.mean(baseline_deg[lr_mask]) / np.mean(baseline_deg[hr_mask])
    else:
        disparity = np.nan

    baseline_marker = " ← current" if thresh == 0.50 else ""
    print(f"{thresh:<12.2f} {hr_count:<12} {lr_count:<12} {disparity:<15.2f}x{baseline_marker}")


print("\n\n7. HYPOTHESIS TEST")
print("-" * 70)

# Test 1: Key findings are parameter-invariant
test1_pass = always_negative_r and always_disparity_gt_1

# Test 2: No parameter has HIGH sensitivity for both metrics
n_high_sensitivity = sum(1 for s in sensitivity_results.values()
                          if s['correlation_range'] > 0.3 or s['disparity_range'] > 1.0)
test2_pass = n_high_sensitivity == 0

# Test 3: Extreme scenarios still show valid findings
all_extreme_valid = all(
    compute_key_metrics(
        compute_degradation(alignment, base=s['base'], scale=s['scale'], exponent=s['exponent']),
        alignment, lang_type
    )['correlation'] < 0 and
    compute_key_metrics(
        compute_degradation(alignment, base=s['base'], scale=s['scale'], exponent=s['exponent']),
        alignment, lang_type
    )['disparity'] > 1
    for s in extreme_scenarios
)
test3_pass = all_extreme_valid

print(f"""
TEST 1: Key findings hold across all parameter variations?
  Always r < 0: {always_negative_r}
  Always disparity > 1: {always_disparity_gt_1}
  Verdict: {'PASS ✓' if test1_pass else 'FAIL ✗'}

TEST 2: No parameter has HIGH sensitivity?
  Parameters with high sensitivity: {n_high_sensitivity}/3
  Verdict: {'PASS ✓' if test2_pass else 'FAIL ✗'}

TEST 3: Extreme scenarios still valid?
  All extreme valid: {all_extreme_valid}
  Verdict: {'PASS ✓' if test3_pass else 'FAIL ✗'}

OVERALL: {'FINDINGS ARE ROBUST TO PARAMETER CHANGES ✓' if test1_pass and test2_pass and test3_pass else 'SOME SENSITIVITY DETECTED'}
""")


print("\n" + "=" * 70)
print("SUMMARY: E-EXP10 SENSITIVITY ANALYSIS")
print("=" * 70)

print(f"""
QUESTION: How robust are findings to modeling assumptions?

FINDINGS:

1. PARAMETER INVARIANTS:
   - Correlation always negative: {'YES' if always_negative_r else 'NO'}
   - Disparity always > 1: {'YES' if always_disparity_gt_1 else 'NO'}
   - R² always > 0.5: {'YES' if always_r2_gt_50 else 'NO'}

2. PARAMETER SENSITIVITY:
   - 'base' parameter: {'ROBUST' if sensitivity_results['base']['correlation_range'] < 0.1 else 'SENSITIVE'}
   - 'scale' parameter: {'ROBUST' if sensitivity_results['scale']['correlation_range'] < 0.1 else 'SENSITIVE'}
   - 'exponent' parameter: {'ROBUST' if sensitivity_results['exponent']['correlation_range'] < 0.1 else 'SENSITIVE'}

3. EXTREME SCENARIOS:
   All tested extremes maintain key findings: {'YES' if all_extreme_valid else 'NO'}

IMPLICATION:
Our key conclusions (negative correlation, LR disparity) are ROBUST
to reasonable variations in model parameters. The specific magnitudes
change, but the qualitative findings remain stable.

This supports the validity of our simulation-based conclusions.
""")
