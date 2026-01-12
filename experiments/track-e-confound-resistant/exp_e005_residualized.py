#!/usr/bin/env python3
"""
EXPERIMENT: E-EXP5 - Residualized Alignment Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: Does alignment predict degradation BEYOND confounds?

METHOD:
1. Regress degradation on ALL confounds (training data, vocab, benchmarks)
2. Extract residuals (what confounds can't explain)
3. Test if alignment predicts the residuals
4. If yes → alignment has independent effect

This is the STATISTICAL approach to confound control.
Complements E-EXP3's DESIGN approach (within-language).
"""
import numpy as np
from scipy import stats
from numpy.linalg import lstsq

print("=" * 70)
print("E-EXP5: RESIDUALIZED ALIGNMENT ANALYSIS")
print("=" * 70)
print("\nTesting if alignment predicts degradation beyond confounds")
print("=" * 70)

# Full language dataset
LANGUAGES = {
    'en': {'alignment': 0.72, 'degradation': 46.8, 'training_gb': 500, 'vocab_cov': 0.95, 'bench_qual': 0.95},
    'de': {'alignment': 0.58, 'degradation': 60.6, 'training_gb': 80, 'vocab_cov': 0.88, 'bench_qual': 0.92},
    'fr': {'alignment': 0.62, 'degradation': 55.1, 'training_gb': 70, 'vocab_cov': 0.90, 'bench_qual': 0.93},
    'es': {'alignment': 0.60, 'degradation': 54.1, 'training_gb': 65, 'vocab_cov': 0.89, 'bench_qual': 0.91},
    'zh': {'alignment': 0.55, 'degradation': 124.9, 'training_gb': 100, 'vocab_cov': 0.75, 'bench_qual': 0.88},
    'ru': {'alignment': 0.48, 'degradation': 78.4, 'training_gb': 45, 'vocab_cov': 0.82, 'bench_qual': 0.85},
    'ja': {'alignment': 0.38, 'degradation': 152.4, 'training_gb': 50, 'vocab_cov': 0.70, 'bench_qual': 0.82},
    'ko': {'alignment': 0.32, 'degradation': 209.4, 'training_gb': 25, 'vocab_cov': 0.65, 'bench_qual': 0.78},
    'ar': {'alignment': 0.28, 'degradation': 214.1, 'training_gb': 20, 'vocab_cov': 0.60, 'bench_qual': 0.75},
    'he': {'alignment': 0.24, 'degradation': 264.3, 'training_gb': 8, 'vocab_cov': 0.55, 'bench_qual': 0.72},
    'tr': {'alignment': 0.35, 'degradation': 168.2, 'training_gb': 15, 'vocab_cov': 0.62, 'bench_qual': 0.76},
    'pl': {'alignment': 0.45, 'degradation': 84.2, 'training_gb': 15, 'vocab_cov': 0.78, 'bench_qual': 0.80},
}

langs = list(LANGUAGES.keys())
n = len(langs)

# Extract arrays
alignment = np.array([LANGUAGES[l]['alignment'] for l in langs])
degradation = np.array([LANGUAGES[l]['degradation'] for l in langs])
training = np.array([LANGUAGES[l]['training_gb'] for l in langs])
vocab = np.array([LANGUAGES[l]['vocab_cov'] for l in langs])
benchmark = np.array([LANGUAGES[l]['bench_qual'] for l in langs])


print("\n1. RAW CORRELATIONS")
print("-" * 70)

r_align, p_align = stats.pearsonr(alignment, degradation)
r_train, p_train = stats.pearsonr(training, degradation)
r_vocab, p_vocab = stats.pearsonr(vocab, degradation)
r_bench, p_bench = stats.pearsonr(benchmark, degradation)

print(f"""
Variable             r with degradation    p-value
─────────────────────────────────────────────────────
Alignment            {r_align:>8.3f}           {p_align:.4f}
Training data (GB)   {r_train:>8.3f}           {p_train:.4f}
Vocab coverage       {r_vocab:>8.3f}           {p_vocab:.4f}
Benchmark quality    {r_bench:>8.3f}           {p_bench:.4f}
""")


print("\n2. CONFOUND INTERCORRELATIONS")
print("-" * 70)

print("\nHow correlated are confounds with alignment?")
r_a_t, _ = stats.pearsonr(alignment, training)
r_a_v, _ = stats.pearsonr(alignment, vocab)
r_a_b, _ = stats.pearsonr(alignment, benchmark)

print(f"""
Alignment vs Training:   r = {r_a_t:.3f}
Alignment vs Vocab:      r = {r_a_v:.3f}
Alignment vs Benchmark:  r = {r_a_b:.3f}

PROBLEM: High multicollinearity - hard to separate effects
""")


print("\n3. RESIDUALIZED REGRESSION")
print("-" * 70)

print("\nStep 1: Regress degradation on confounds ONLY (exclude alignment)")

# Build design matrix: [intercept, training, vocab, benchmark]
X_confounds = np.column_stack([
    np.ones(n),
    (training - training.mean()) / training.std(),
    (vocab - vocab.mean()) / vocab.std(),
    (benchmark - benchmark.mean()) / benchmark.std(),
])

y = (degradation - degradation.mean()) / degradation.std()

# Fit confound-only model
beta_confounds, residuals, rank, s = lstsq(X_confounds, y, rcond=None)

# Get predictions and residuals
y_pred_confounds = X_confounds @ beta_confounds
residuals = y - y_pred_confounds

print(f"""
Confound-only model coefficients:
  Intercept:      {beta_confounds[0]:.3f}
  Training:       {beta_confounds[1]:.3f}
  Vocab coverage: {beta_confounds[2]:.3f}
  Benchmark:      {beta_confounds[3]:.3f}

R² of confound model: {1 - np.var(residuals)/np.var(y):.3f}
""")


print("\nStep 2: Does alignment predict the RESIDUALS?")

# Standardize alignment
alignment_std = (alignment - alignment.mean()) / alignment.std()

# Correlate alignment with residuals
r_resid, p_resid = stats.pearsonr(alignment_std, residuals)

print(f"""
Correlation: alignment with residuals
  r = {r_resid:.3f}
  p = {p_resid:.4f}

If r is significant, alignment explains variance
BEYOND what confounds explain.
""")


print("\n4. FULL REGRESSION (Confounds + Alignment)")
print("-" * 70)

# Build full design matrix
X_full = np.column_stack([
    np.ones(n),
    (training - training.mean()) / training.std(),
    (vocab - vocab.mean()) / vocab.std(),
    (benchmark - benchmark.mean()) / benchmark.std(),
    alignment_std,
])

# Fit full model
beta_full, _, _, _ = lstsq(X_full, y, rcond=None)
y_pred_full = X_full @ beta_full
residuals_full = y - y_pred_full

r2_confounds = 1 - np.var(residuals)/np.var(y)
r2_full = 1 - np.var(residuals_full)/np.var(y)
r2_increment = r2_full - r2_confounds

print(f"""
Full model coefficients:
  Intercept:      {beta_full[0]:.3f}
  Training:       {beta_full[1]:.3f}
  Vocab coverage: {beta_full[2]:.3f}
  Benchmark:      {beta_full[3]:.3f}
  Alignment:      {beta_full[4]:.3f}

R² comparison:
  Confounds only: {r2_confounds:.3f}
  With alignment: {r2_full:.3f}
  Increment:      {r2_increment:.3f}

Does alignment ADD predictive power? {'YES ✓' if r2_increment > 0.01 else 'NO ✗'}
""")


print("\n5. HYPOTHESIS TEST")
print("-" * 70)

# Test 1: Alignment-residual correlation significant
test1_pass = abs(r_resid) > 0.3 and p_resid < 0.1  # Relaxed for small sample

# Test 2: R² increment meaningful
test2_pass = r2_increment > 0.02

# Test 3: Alignment coefficient in full model non-zero
test3_pass = abs(beta_full[4]) > 0.1

print(f"""
TEST 1: Alignment predicts residuals (beyond confounds)?
  r(alignment, residuals) = {r_resid:.3f}
  p = {p_resid:.4f}
  Verdict: {'PASS ✓' if test1_pass else 'FAIL ✗'}

TEST 2: Alignment adds predictive power (R² increment > 0.02)?
  ΔR² = {r2_increment:.3f}
  Verdict: {'PASS ✓' if test2_pass else 'FAIL ✗'}

TEST 3: Alignment coefficient is non-trivial?
  β(alignment) = {beta_full[4]:.3f}
  Verdict: {'PASS ✓' if test3_pass else 'FAIL ✗'}

OVERALL: {'ALIGNMENT HAS INDEPENDENT EFFECT ✓' if test1_pass or test2_pass else 'CANNOT CONFIRM INDEPENDENT EFFECT'}
""")


print("\n6. RESIDUAL VISUALIZATION")
print("-" * 70)

print("\nResiduals (what confounds can't explain) vs Alignment:\n")

# Sort by alignment
sorted_idx = np.argsort(alignment)

print(f"{'Lang':<6} {'Align':<8} {'Residual':<12} {'Direction':<10}")
print("-" * 50)

for idx in sorted_idx:
    lang = langs[idx]
    al = alignment[idx]
    res = residuals[idx]
    direction = "↓ less" if res < 0 else "↑ more"

    print(f"{lang:<6} {al:<8.2f} {res:<12.3f} {direction:<10}")


print("\n\n7. INTERPRETATION")
print("-" * 70)

print(f"""
WHAT THIS ANALYSIS SHOWS:

1. CONFOUNDS EXPLAIN MOST VARIANCE:
   R² from confounds alone: {r2_confounds:.3f}
   Most degradation variance is explained by:
   - Training data, vocab coverage, benchmark quality

2. ALIGNMENT ADDS {'SOMETHING' if r2_increment > 0.01 else 'LITTLE'}:
   R² increment from alignment: {r2_increment:.3f}
   Alignment-residual correlation: r = {r_resid:.3f}

3. HONEST CONCLUSION:
   {"Alignment has a small but detectable independent effect" if test1_pass or test2_pass else
    "Alignment effect may be fully explained by confounds"}

IMPORTANT CAVEAT:
Small sample size (n={n}) limits statistical power.
High multicollinearity makes coefficient estimation unstable.
Results should be interpreted cautiously.
""")


print("\n" + "=" * 70)
print("SUMMARY: E-EXP5 RESIDUALIZED ANALYSIS")
print("=" * 70)

print(f"""
QUESTION: Does alignment predict degradation beyond confounds?

ANSWER: {'SMALL INDEPENDENT EFFECT DETECTED' if test1_pass or test2_pass else 'CANNOT CONFIRM'}

EVIDENCE:
- Raw alignment-degradation: r = {r_align:.3f}
- After controlling confounds: r = {r_resid:.3f}
- R² increment: {r2_increment:.3f}

INTERPRETATION:
- Confounds (training, vocab, benchmark) explain most variance
- Alignment {'adds modest predictive power' if r2_increment > 0.01 else 'may not add predictive power'}
- Effect is {'detectable but small' if test1_pass else 'not clearly detectable'}

LIMITATION:
Small sample size and high multicollinearity limit confidence.

RECOMMENDATION:
Combine with E-EXP3 (within-language) for triangulation.
If both show effect, confidence increases substantially.
""")
