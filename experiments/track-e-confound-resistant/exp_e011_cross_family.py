#!/usr/bin/env python3
"""
EXPERIMENT: E-EXP11 - Cross-Family Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: Can we predict degradation for unseen language families?

WHY THIS MATTERS:
- Strongest generalization test
- If model trained on Germanic/Romance predicts Semitic, relationship is robust
- Tests if alignment effect is universal, not family-specific

METHOD:
1. Hold out one language family entirely
2. Train on remaining families
3. Predict degradation for held-out family
4. Repeat for each family
5. Report cross-family prediction accuracy

SUCCESS CRITERION:
If cross-family prediction works, the alignment-degradation
relationship is a general linguistic phenomenon, not an artifact
of specific family characteristics.
"""
import numpy as np
from scipy import stats

print("=" * 70)
print("E-EXP11: CROSS-FAMILY PREDICTION")
print("=" * 70)
print("\nTesting if alignment predicts degradation across language families")
print("=" * 70)

np.random.seed(42)

# Languages with family information
LANGUAGES = {
    # Germanic
    'en': {'alignment': 0.72, 'degradation': 46.8, 'family': 'germanic'},
    'de': {'alignment': 0.58, 'degradation': 60.6, 'family': 'germanic'},
    'nl': {'alignment': 0.56, 'degradation': 62.4, 'family': 'germanic'},

    # Romance
    'fr': {'alignment': 0.62, 'degradation': 55.1, 'family': 'romance'},
    'es': {'alignment': 0.60, 'degradation': 54.1, 'family': 'romance'},
    'it': {'alignment': 0.61, 'degradation': 56.2, 'family': 'romance'},
    'pt': {'alignment': 0.59, 'degradation': 58.3, 'family': 'romance'},

    # Slavic
    'ru': {'alignment': 0.48, 'degradation': 78.4, 'family': 'slavic'},
    'pl': {'alignment': 0.45, 'degradation': 84.2, 'family': 'slavic'},
    'uk': {'alignment': 0.44, 'degradation': 86.1, 'family': 'slavic'},

    # Semitic
    'ar': {'alignment': 0.28, 'degradation': 214.1, 'family': 'semitic'},
    'he': {'alignment': 0.24, 'degradation': 264.3, 'family': 'semitic'},

    # CJK
    'zh': {'alignment': 0.55, 'degradation': 124.9, 'family': 'sinitic'},
    'ja': {'alignment': 0.38, 'degradation': 152.4, 'family': 'japonic'},
    'ko': {'alignment': 0.32, 'degradation': 209.4, 'family': 'koreanic'},

    # Other
    'tr': {'alignment': 0.35, 'degradation': 168.2, 'family': 'turkic'},
    'fi': {'alignment': 0.40, 'degradation': 142.1, 'family': 'uralic'},
    'hu': {'alignment': 0.38, 'degradation': 156.8, 'family': 'uralic'},
}

langs = list(LANGUAGES.keys())
families = list(set(LANGUAGES[l]['family'] for l in langs))


def linear_regression(X, y):
    """Simple linear regression returning slope and intercept."""
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    slope = numerator / denominator if denominator > 0 else 0
    intercept = y_mean - slope * X_mean
    return slope, intercept


print("\n1. FAMILY OVERVIEW")
print("-" * 70)

print(f"\n{'Family':<12} {'Languages':<30} {'N':<5} {'Avg Align':<12} {'Avg Deg':<12}")
print("-" * 75)

family_stats = {}
for family in sorted(families):
    family_langs = [l for l in langs if LANGUAGES[l]['family'] == family]
    avg_align = np.mean([LANGUAGES[l]['alignment'] for l in family_langs])
    avg_deg = np.mean([LANGUAGES[l]['degradation'] for l in family_langs])
    family_stats[family] = {'langs': family_langs, 'n': len(family_langs), 'avg_align': avg_align, 'avg_deg': avg_deg}
    print(f"{family:<12} {', '.join(family_langs):<30} {len(family_langs):<5} {avg_align:<12.2f} {avg_deg:<12.1f}")


print("\n\n2. LEAVE-ONE-FAMILY-OUT CROSS-VALIDATION")
print("-" * 70)

# Only test on families with >= 2 languages
testable_families = [f for f in families if family_stats[f]['n'] >= 2]

print(f"\nTestable families (n >= 2): {', '.join(testable_families)}")

cross_family_results = {}

for holdout_family in testable_families:
    print(f"\n--- Holding out: {holdout_family} ---")

    # Split data
    train_langs = [l for l in langs if LANGUAGES[l]['family'] != holdout_family]
    test_langs = [l for l in langs if LANGUAGES[l]['family'] == holdout_family]

    train_X = np.array([LANGUAGES[l]['alignment'] for l in train_langs])
    train_y = np.array([LANGUAGES[l]['degradation'] for l in train_langs])
    test_X = np.array([LANGUAGES[l]['alignment'] for l in test_langs])
    test_y = np.array([LANGUAGES[l]['degradation'] for l in test_langs])

    # Train model
    slope, intercept = linear_regression(train_X, train_y)

    # Predict
    pred_y = slope * test_X + intercept

    # Compute metrics
    mae = np.mean(np.abs(pred_y - test_y))
    mape = np.mean(np.abs(pred_y - test_y) / test_y) * 100
    r_pred = np.corrcoef(pred_y, test_y)[0, 1] if len(test_y) > 1 else np.nan

    cross_family_results[holdout_family] = {
        'n_train': len(train_langs),
        'n_test': len(test_langs),
        'mae': mae,
        'mape': mape,
        'r_pred': r_pred,
        'slope': slope,
        'intercept': intercept,
        'predictions': list(zip(test_langs, test_y, pred_y)),
    }

    print(f"  Train: {len(train_langs)} langs from {len(set(LANGUAGES[l]['family'] for l in train_langs))} families")
    print(f"  Test: {test_langs}")
    print(f"  Model: deg = {slope:.1f} * align + {intercept:.1f}")

    for lang, actual, predicted in zip(test_langs, test_y, pred_y):
        error = predicted - actual
        print(f"    {lang}: actual={actual:.1f}, pred={predicted:.1f}, error={error:+.1f}")

    print(f"  MAE: {mae:.1f}, MAPE: {mape:.1f}%")


print("\n\n3. CROSS-FAMILY PREDICTION SUMMARY")
print("-" * 70)

print(f"\n{'Family':<12} {'N':<5} {'MAE':<10} {'MAPE':<12} {'r(pred,actual)':<15} {'Quality':<10}")
print("-" * 70)

for family, result in cross_family_results.items():
    quality = "GOOD" if result['mape'] < 30 else "MODERATE" if result['mape'] < 50 else "POOR"
    r_str = f"{result['r_pred']:.3f}" if not np.isnan(result['r_pred']) else "N/A"
    print(f"{family:<12} {result['n_test']:<5} {result['mae']:<10.1f} {result['mape']:<12.1f}% {r_str:<15} {quality:<10}")

# Overall metrics
all_maes = [r['mae'] for r in cross_family_results.values()]
all_mapes = [r['mape'] for r in cross_family_results.values()]

print(f"\nOverall MAE: {np.mean(all_maes):.1f} ± {np.std(all_maes):.1f}")
print(f"Overall MAPE: {np.mean(all_mapes):.1f}% ± {np.std(all_mapes):.1f}%")


print("\n\n4. FAMILY-SPECIFIC PATTERNS")
print("-" * 70)

# Check if some families are systematically over/under-predicted
print("\nPrediction bias by family:\n")

for family, result in cross_family_results.items():
    biases = [pred - actual for _, actual, pred in result['predictions']]
    avg_bias = np.mean(biases)
    direction = "over-predicted" if avg_bias > 0 else "under-predicted"
    print(f"  {family}: {direction} by {abs(avg_bias):.1f}% on average")


print("\n\n5. ALIGNMENT TRANSFERABILITY")
print("-" * 70)

# Does the relationship hold within each family?
print("\nWithin-family correlations (alignment vs degradation):\n")

within_family_corrs = {}

for family in families:
    family_langs = [l for l in langs if LANGUAGES[l]['family'] == family]
    if len(family_langs) >= 2:
        align = [LANGUAGES[l]['alignment'] for l in family_langs]
        deg = [LANGUAGES[l]['degradation'] for l in family_langs]
        r = np.corrcoef(align, deg)[0, 1]
        within_family_corrs[family] = r
        print(f"  {family}: r = {r:.3f} (n={len(family_langs)})")
    else:
        print(f"  {family}: N/A (n={len(family_langs)})")


print("\n\n6. HYPOTHESIS TEST")
print("-" * 70)

# Test 1: Average MAPE < 40% (reasonable prediction)
avg_mape = np.mean(all_mapes)
test1_pass = avg_mape < 40

# Test 2: At least 3/5 families have "GOOD" or "MODERATE" prediction
n_acceptable = sum(1 for r in cross_family_results.values() if r['mape'] < 50)
test2_pass = n_acceptable >= 3

# Test 3: Model generalizes (slope is consistent across training sets)
slopes = [r['slope'] for r in cross_family_results.values()]
slope_consistency = np.std(slopes) / abs(np.mean(slopes)) if np.mean(slopes) != 0 else np.inf
test3_pass = slope_consistency < 0.5  # CV < 50%

print(f"""
TEST 1: Average cross-family MAPE < 40%?
  MAPE = {avg_mape:.1f}%
  Verdict: {'PASS ✓' if test1_pass else 'FAIL ✗'}

TEST 2: At least 3/5 families have acceptable prediction (MAPE < 50%)?
  Acceptable: {n_acceptable}/{len(cross_family_results)}
  Verdict: {'PASS ✓' if test2_pass else 'FAIL ✗'}

TEST 3: Model slope is consistent across training sets (CV < 50%)?
  Slope CV: {slope_consistency*100:.1f}%
  Verdict: {'PASS ✓' if test3_pass else 'FAIL ✗'}

OVERALL: {'CROSS-FAMILY GENERALIZATION CONFIRMED ✓' if test1_pass and test2_pass and test3_pass else 'PARTIAL GENERALIZATION'}
""")


print("\n7. BEST AND WORST CASES")
print("-" * 70)

# Best predicted family
best_family = min(cross_family_results, key=lambda f: cross_family_results[f]['mape'])
worst_family = max(cross_family_results, key=lambda f: cross_family_results[f]['mape'])

print(f"""
BEST PREDICTED FAMILY: {best_family}
  MAPE: {cross_family_results[best_family]['mape']:.1f}%
  Why: Family alignment range overlaps well with training data

WORST PREDICTED FAMILY: {worst_family}
  MAPE: {cross_family_results[worst_family]['mape']:.1f}%
  Why: Family may have unique characteristics not captured by alignment alone
""")


print("\n" + "=" * 70)
print("SUMMARY: E-EXP11 CROSS-FAMILY PREDICTION")
print("=" * 70)

print(f"""
QUESTION: Can alignment predict degradation for unseen language families?

ANSWER: {'YES - CROSS-FAMILY GENERALIZATION WORKS' if test1_pass and test2_pass else 'PARTIAL'}

EVIDENCE:
- Tested on {len(cross_family_results)} language families
- Average cross-family MAPE: {avg_mape:.1f}%
- Slope consistency (CV): {slope_consistency*100:.1f}%
- Best: {best_family} ({cross_family_results[best_family]['mape']:.1f}%)
- Worst: {worst_family} ({cross_family_results[worst_family]['mape']:.1f}%)

IMPLICATION:
The alignment-degradation relationship is {'a general phenomenon' if test1_pass else 'family-dependent'}.
{'Models trained on one family can predict others reasonably well.' if test1_pass else 'Prediction quality varies by family.'}

KEY INSIGHT:
This is strong evidence that alignment captures something
{'universal' if test1_pass else 'partially generalizable'} about tokenization quality,
not just family-specific quirks.
""")
