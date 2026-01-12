#!/usr/bin/env python3
"""
EXPERIMENT: E-EXP6 - Held-Out Language Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: Can we predict degradation for NEW languages using alignment?

HYPOTHESIS: E-H5 (generalization)
- If alignment-degradation relationship is genuine, it should generalize
- Train model on subset of languages
- Predict degradation for held-out languages
- Good prediction = relationship is robust, not overfitting

METHOD:
1. Split languages into train (10) and test (5) sets
2. Fit alignment → degradation model on train set
3. Predict degradation for test languages
4. Compare prediction to "actual" degradation
5. Repeat with different splits

WHY THIS IS INFORMATIVE:
- If predictions work, relationship generalizes
- If predictions fail, we may be overfitting
- Cross-validation is standard for robustness
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

print("=" * 70)
print("E-EXP6: HELD-OUT LANGUAGE PREDICTION")
print("=" * 70)
print("\nTesting if alignment-degradation relationship generalizes")
print("=" * 70)

np.random.seed(42)

# Full language dataset
LANGUAGES = {
    'en': {'alignment': 0.72, 'degradation': 46.8, 'family': 'germanic'},
    'de': {'alignment': 0.58, 'degradation': 60.6, 'family': 'germanic'},
    'fr': {'alignment': 0.62, 'degradation': 55.1, 'family': 'romance'},
    'es': {'alignment': 0.60, 'degradation': 54.1, 'family': 'romance'},
    'it': {'alignment': 0.61, 'degradation': 56.2, 'family': 'romance'},
    'pt': {'alignment': 0.59, 'degradation': 58.3, 'family': 'romance'},
    'nl': {'alignment': 0.56, 'degradation': 62.4, 'family': 'germanic'},
    'zh': {'alignment': 0.55, 'degradation': 124.9, 'family': 'sinitic'},
    'ru': {'alignment': 0.48, 'degradation': 78.4, 'family': 'slavic'},
    'pl': {'alignment': 0.45, 'degradation': 84.2, 'family': 'slavic'},
    'uk': {'alignment': 0.44, 'degradation': 86.1, 'family': 'slavic'},
    'ja': {'alignment': 0.38, 'degradation': 152.4, 'family': 'japonic'},
    'ko': {'alignment': 0.32, 'degradation': 209.4, 'family': 'koreanic'},
    'tr': {'alignment': 0.35, 'degradation': 168.2, 'family': 'turkic'},
    'ar': {'alignment': 0.28, 'degradation': 214.1, 'family': 'semitic'},
    'he': {'alignment': 0.24, 'degradation': 264.3, 'family': 'semitic'},
    'fi': {'alignment': 0.40, 'degradation': 142.1, 'family': 'uralic'},
    'hu': {'alignment': 0.38, 'degradation': 156.8, 'family': 'uralic'},
}

langs = list(LANGUAGES.keys())
n = len(langs)

# Prepare data
X = np.array([[LANGUAGES[l]['alignment']] for l in langs])
y = np.array([LANGUAGES[l]['degradation'] for l in langs])


print("\n1. FULL DATASET OVERVIEW")
print("-" * 70)

print(f"\n{'Language':<10} {'Alignment':<12} {'Degradation':<15} {'Family':<12}")
print("-" * 55)

for lang in sorted(langs, key=lambda l: LANGUAGES[l]['alignment'], reverse=True):
    data = LANGUAGES[lang]
    print(f"{lang:<10} {data['alignment']:<12.2f} {data['degradation']:<15.1f} {data['family']:<12}")


print(f"\n\nTotal languages: {n}")
print(f"Alignment range: {X.min():.2f} - {X.max():.2f}")
print(f"Degradation range: {y.min():.1f} - {y.max():.1f}")


print("\n\n2. LEAVE-ONE-OUT CROSS-VALIDATION")
print("-" * 70)

loo = LeaveOneOut()
predictions = cross_val_predict(LinearRegression(), X, y, cv=loo)

print(f"\n{'Language':<10} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Pct Error':<12}")
print("-" * 60)

errors = []
pct_errors = []

for i, lang in enumerate(langs):
    actual = y[i]
    pred = predictions[i]
    error = pred - actual
    pct_error = abs(error) / actual * 100
    errors.append(error)
    pct_errors.append(pct_error)

    marker = "★" if pct_error > 30 else ""
    print(f"{lang:<10} {actual:<12.1f} {pred:<12.1f} {error:>+10.1f}  {pct_error:>10.1f}% {marker}")


print("\n\n3. PREDICTION QUALITY METRICS")
print("-" * 70)

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(np.array(errors)**2))
mape = np.mean(pct_errors)
r2_cv = 1 - np.sum(np.array(errors)**2) / np.sum((y - y.mean())**2)

# Compare to baseline (predict mean)
baseline_errors = y - y.mean()
baseline_mae = np.mean(np.abs(baseline_errors))
skill_score = 1 - mae / baseline_mae

print(f"""
LEAVE-ONE-OUT CROSS-VALIDATION RESULTS:

  Mean Absolute Error (MAE):  {mae:.1f}%
  Root Mean Square Error:     {rmse:.1f}%
  Mean Absolute Pct Error:    {mape:.1f}%
  Cross-validated R²:         {r2_cv:.3f}

COMPARISON TO BASELINE:
  Baseline MAE (predict mean): {baseline_mae:.1f}%
  Skill score vs baseline:     {skill_score:.3f}
  (1 = perfect, 0 = no better than mean)
""")


print("\n4. TRAIN-TEST SPLIT ANALYSIS")
print("-" * 70)

# Multiple random splits
n_splits = 10
split_results = []

print(f"\nRunning {n_splits} random 70/30 splits...\n")

for split_i in range(n_splits):
    # Random split
    indices = np.random.permutation(n)
    train_size = int(0.7 * n)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit and predict
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    test_mae = np.mean(np.abs(y_pred - y_test))
    test_r2 = 1 - np.sum((y_pred - y_test)**2) / np.sum((y_test - y_test.mean())**2)

    split_results.append({'mae': test_mae, 'r2': test_r2})

avg_mae = np.mean([r['mae'] for r in split_results])
avg_r2 = np.mean([r['r2'] for r in split_results])
std_mae = np.std([r['mae'] for r in split_results])
std_r2 = np.std([r['r2'] for r in split_results])

print(f"""
70/30 SPLIT RESULTS (averaged over {n_splits} splits):

  Test MAE:  {avg_mae:.1f}% ± {std_mae:.1f}%
  Test R²:   {avg_r2:.3f} ± {std_r2:.3f}
""")


print("\n5. FAMILY-HOLDOUT TEST")
print("-" * 70)

print("\nHolding out entire language families...\n")

families = list(set(LANGUAGES[l]['family'] for l in langs))

for holdout_family in ['semitic', 'slavic', 'uralic']:
    train_idx = [i for i, l in enumerate(langs) if LANGUAGES[l]['family'] != holdout_family]
    test_idx = [i for i, l in enumerate(langs) if LANGUAGES[l]['family'] == holdout_family]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    family_mae = np.mean(np.abs(y_pred - y_test))
    family_mape = np.mean(np.abs(y_pred - y_test) / y_test) * 100

    test_langs = [langs[i] for i in test_idx]

    print(f"  Holdout family: {holdout_family}")
    print(f"  Test languages: {', '.join(test_langs)}")
    print(f"  MAE: {family_mae:.1f}%, MAPE: {family_mape:.1f}%")
    print()


print("\n6. HYPOTHESIS TEST")
print("-" * 70)

# Test 1: LOO R² > 0.7 (good predictive power)
test1_pass = r2_cv > 0.7

# Test 2: Skill score > 0.5 (better than baseline)
test2_pass = skill_score > 0.5

# Test 3: MAPE < 30% (reasonable accuracy)
test3_pass = mape < 30

print(f"""
TEST 1: LOO Cross-validated R² > 0.7?
  R² = {r2_cv:.3f}
  Verdict: {'PASS ✓' if test1_pass else 'FAIL ✗'}

TEST 2: Skill score > 0.5 (better than predicting mean)?
  Skill = {skill_score:.3f}
  Verdict: {'PASS ✓' if test2_pass else 'FAIL ✗'}

TEST 3: Mean Absolute Percentage Error < 30%?
  MAPE = {mape:.1f}%
  Verdict: {'PASS ✓' if test3_pass else 'FAIL ✗'}

OVERALL: {'GENERALIZATION CONFIRMED ✓' if test1_pass and test2_pass else 'PARTIAL'}
""")


print("\n7. OUTLIER ANALYSIS")
print("-" * 70)

# Identify languages where prediction fails
outliers = [(langs[i], pct_errors[i]) for i in range(n) if pct_errors[i] > 25]

print("\nLanguages with >25% prediction error:\n")
if outliers:
    for lang, err in sorted(outliers, key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {err:.1f}% error")
        print(f"    Possible reason: {LANGUAGES[lang]['family']} family may have unique characteristics")
else:
    print("  None - model generalizes well to all languages")


print("\n" + "=" * 70)
print("SUMMARY: E-EXP6 HELD-OUT LANGUAGE PREDICTION")
print("=" * 70)

print(f"""
QUESTION: Can alignment predict degradation for NEW languages?

ANSWER: {'YES - RELATIONSHIP GENERALIZES' if test1_pass and test2_pass else 'PARTIAL'}

EVIDENCE:
- Leave-One-Out R²: {r2_cv:.3f}
- Skill score vs baseline: {skill_score:.3f}
- Mean Absolute Pct Error: {mape:.1f}%
- 70/30 split R²: {avg_r2:.3f} ± {std_r2:.3f}

INTERPRETATION:
- Alignment-degradation relationship is {'robust' if test1_pass else 'weak'}
- Model can predict new languages with {mape:.0f}% average error
- {'Relationship is genuine, not overfitting' if test1_pass else 'May be overfitting to training languages'}

LIMITATION:
- n={n} languages is still small
- Some families underrepresented
- Real validation needs truly new languages

IMPLICATION:
Alignment metric can be used to {'PREDICT' if test1_pass else 'estimate'}
quantization sensitivity for new languages before deployment.
""")
