#!/usr/bin/env python3
"""
EXPERIMENT: D3 - Instrumental Variable Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: Can we find instruments that affect alignment but not training data?

WHY THIS MATTERS:
- Alignment is confounded with training data investment
- Instrumental variables could break this confound
- A valid instrument Z must:
  1. Correlate with alignment (relevance)
  2. NOT correlate with degradation except through alignment (exclusion)
  3. NOT correlate with confounds (independence)

CANDIDATE INSTRUMENTS:
- Writing direction (RTL vs LTR)
- Script complexity (number of unique characters)
- Morphological typology index
- Language family (exogenous to investment decisions?)
- Geographic isolation

METHOD:
1. Define candidate instruments
2. Test relevance (correlation with alignment)
3. Test independence (correlation with confounds)
4. If valid instrument found, run 2SLS
"""
import numpy as np
from scipy import stats

print("=" * 70)
print("D3: INSTRUMENTAL VARIABLE SEARCH")
print("=" * 70)
print("\nSearching for valid instruments to break confound")
print("=" * 70)

np.random.seed(42)

# Language data with potential instruments
LANGUAGES = {
    'en': {
        'alignment': 0.72, 'degradation': 46.8, 'training_gb': 500,
        'rtl': 0, 'script_chars': 26, 'morph_index': 0.3, 'family_id': 1,
        'speakers_log': np.log10(1500), 'gdp_per_speaker': 50000,
    },
    'de': {
        'alignment': 0.58, 'degradation': 60.6, 'training_gb': 80,
        'rtl': 0, 'script_chars': 30, 'morph_index': 0.5, 'family_id': 1,
        'speakers_log': np.log10(100), 'gdp_per_speaker': 45000,
    },
    'fr': {
        'alignment': 0.62, 'degradation': 55.1, 'training_gb': 70,
        'rtl': 0, 'script_chars': 40, 'morph_index': 0.4, 'family_id': 2,
        'speakers_log': np.log10(280), 'gdp_per_speaker': 35000,
    },
    'zh': {
        'alignment': 0.55, 'degradation': 124.9, 'training_gb': 100,
        'rtl': 0, 'script_chars': 5000, 'morph_index': 0.1, 'family_id': 3,
        'speakers_log': np.log10(1100), 'gdp_per_speaker': 12000,
    },
    'ru': {
        'alignment': 0.48, 'degradation': 78.4, 'training_gb': 45,
        'rtl': 0, 'script_chars': 33, 'morph_index': 0.7, 'family_id': 4,
        'speakers_log': np.log10(250), 'gdp_per_speaker': 15000,
    },
    'ja': {
        'alignment': 0.38, 'degradation': 152.4, 'training_gb': 50,
        'rtl': 0, 'script_chars': 2000, 'morph_index': 0.6, 'family_id': 5,
        'speakers_log': np.log10(125), 'gdp_per_speaker': 40000,
    },
    'ko': {
        'alignment': 0.32, 'degradation': 209.4, 'training_gb': 25,
        'rtl': 0, 'script_chars': 40, 'morph_index': 0.8, 'family_id': 6,
        'speakers_log': np.log10(80), 'gdp_per_speaker': 35000,
    },
    'ar': {
        'alignment': 0.28, 'degradation': 214.1, 'training_gb': 20,
        'rtl': 1, 'script_chars': 28, 'morph_index': 0.9, 'family_id': 7,
        'speakers_log': np.log10(400), 'gdp_per_speaker': 8000,
    },
    'he': {
        'alignment': 0.24, 'degradation': 264.3, 'training_gb': 8,
        'rtl': 1, 'script_chars': 22, 'morph_index': 0.85, 'family_id': 7,
        'speakers_log': np.log10(9), 'gdp_per_speaker': 45000,
    },
    'tr': {
        'alignment': 0.35, 'degradation': 168.2, 'training_gb': 15,
        'rtl': 0, 'script_chars': 29, 'morph_index': 0.9, 'family_id': 8,
        'speakers_log': np.log10(80), 'gdp_per_speaker': 10000,
    },
    'pl': {
        'alignment': 0.45, 'degradation': 84.2, 'training_gb': 15,
        'rtl': 0, 'script_chars': 32, 'morph_index': 0.7, 'family_id': 4,
        'speakers_log': np.log10(45), 'gdp_per_speaker': 18000,
    },
    'fi': {
        'alignment': 0.40, 'degradation': 142.1, 'training_gb': 10,
        'rtl': 0, 'script_chars': 29, 'morph_index': 0.95, 'family_id': 9,
        'speakers_log': np.log10(5), 'gdp_per_speaker': 50000,
    },
}

langs = list(LANGUAGES.keys())
n = len(langs)

# Extract arrays
alignment = np.array([LANGUAGES[l]['alignment'] for l in langs])
degradation = np.array([LANGUAGES[l]['degradation'] for l in langs])
training = np.array([LANGUAGES[l]['training_gb'] for l in langs])

# Candidate instruments
instruments = {
    'rtl': np.array([LANGUAGES[l]['rtl'] for l in langs]),
    'script_chars': np.array([np.log10(LANGUAGES[l]['script_chars']) for l in langs]),
    'morph_index': np.array([LANGUAGES[l]['morph_index'] for l in langs]),
    'family_id': np.array([LANGUAGES[l]['family_id'] for l in langs]),
    'speakers_log': np.array([LANGUAGES[l]['speakers_log'] for l in langs]),
    'gdp_per_speaker': np.array([np.log10(LANGUAGES[l]['gdp_per_speaker']) for l in langs]),
}


print("\n1. CANDIDATE INSTRUMENTS")
print("-" * 70)

print("""
For a valid instrument Z, we need:
  1. RELEVANCE: Z correlates with alignment (the endogenous variable)
  2. EXCLUSION: Z doesn't affect degradation except through alignment
  3. INDEPENDENCE: Z is uncorrelated with confounds (training data)

Testing candidates:
""")


print("\n2. INSTRUMENT VALIDITY TESTS")
print("-" * 70)

print(f"\n{'Instrument':<18} {'r(Z,align)':<12} {'r(Z,train)':<12} {'r(Z,deg)':<12} {'Verdict':<15}")
print("-" * 70)

valid_instruments = []

for name, Z in instruments.items():
    # Relevance: correlation with alignment
    r_align, p_align = stats.pearsonr(Z, alignment)

    # Independence: correlation with confound (training data)
    r_train, p_train = stats.pearsonr(Z, training)

    # Direct effect: correlation with degradation
    r_deg, p_deg = stats.pearsonr(Z, degradation)

    # Validity criteria
    relevant = abs(r_align) > 0.3 and p_align < 0.1
    independent = abs(r_train) < 0.5 or p_train > 0.1
    # Exclusion is hard to test directly, but we check if Z correlates with deg
    # more than expected from its correlation with alignment

    if relevant and independent:
        verdict = "CANDIDATE ✓"
        valid_instruments.append(name)
    elif relevant and not independent:
        verdict = "CONFOUNDED"
    elif not relevant:
        verdict = "WEAK"
    else:
        verdict = "INVALID"

    print(f"{name:<18} {r_align:>+.3f}       {r_train:>+.3f}       {r_deg:>+.3f}       {verdict:<15}")


print("\n\n3. DETAILED ANALYSIS OF CANDIDATES")
print("-" * 70)

if valid_instruments:
    print(f"\nPotentially valid instruments: {', '.join(valid_instruments)}\n")

    for iv_name in valid_instruments:
        Z = instruments[iv_name]
        print(f"\n--- {iv_name.upper()} ---")

        r_align, _ = stats.pearsonr(Z, alignment)
        r_train, _ = stats.pearsonr(Z, training)
        r_deg, _ = stats.pearsonr(Z, degradation)

        # First stage F-statistic (instrument strength)
        # Regress alignment on instrument
        X_iv = np.column_stack([np.ones(n), Z])
        beta_first, _, _, _ = np.linalg.lstsq(X_iv, alignment, rcond=None)
        predicted_align = X_iv @ beta_first
        residuals_first = alignment - predicted_align

        ss_model = np.sum((predicted_align - np.mean(alignment))**2)
        ss_resid = np.sum(residuals_first**2)
        f_stat = (ss_model / 1) / (ss_resid / (n - 2))

        print(f"  First-stage F-statistic: {f_stat:.2f}")
        print(f"  (Rule of thumb: F > 10 for strong instrument)")

        # Expected correlation with degradation (if exclusion holds)
        # If Z only affects deg through alignment:
        # r(Z, deg) ≈ r(Z, align) * r(align, deg)
        r_align_deg, _ = stats.pearsonr(alignment, degradation)
        expected_r_deg = r_align * r_align_deg
        excess_r = abs(r_deg) - abs(expected_r_deg)

        print(f"  Expected r(Z, deg) if exclusion holds: {expected_r_deg:.3f}")
        print(f"  Actual r(Z, deg): {r_deg:.3f}")
        print(f"  Excess correlation: {excess_r:.3f}")

        if excess_r > 0.2:
            print(f"  WARNING: Possible direct effect (exclusion may fail)")
        else:
            print(f"  Exclusion plausible: no strong direct effect")
else:
    print("\nNo valid instruments found that satisfy all criteria.")


print("\n\n4. MORPHOLOGICAL INDEX DEEP DIVE")
print("-" * 70)

# Morphological index is theoretically the best candidate
# It should affect alignment (complex morphology → poor tokenization)
# But shouldn't directly affect training data investment

morph = instruments['morph_index']

print("""
MORPHOLOGICAL INDEX as instrument:

Theory: High morphological complexity (agglutinative, fusional)
leads to poor BPE alignment because:
- More morphemes per word
- More allomorphic variation
- Less predictable segmentation

This is EXOGENOUS to:
- Training data investment (companies don't invest less because of morphology)
- Benchmark quality (benchmarks test meaning, not morphology)
""")

r_morph_align, p_morph_align = stats.pearsonr(morph, alignment)
r_morph_train, p_morph_train = stats.pearsonr(morph, training)

print(f"""
Correlations:
  Morph → Alignment: r = {r_morph_align:.3f} (p = {p_morph_align:.4f})
  Morph → Training:  r = {r_morph_train:.3f} (p = {p_morph_train:.4f})

Assessment:
  Relevance: {'STRONG' if abs(r_morph_align) > 0.5 else 'MODERATE' if abs(r_morph_align) > 0.3 else 'WEAK'}
  Independence: {'GOOD' if abs(r_morph_train) < 0.3 else 'CONCERNING' if abs(r_morph_train) < 0.5 else 'FAILS'}
""")


print("\n5. POTENTIAL 2SLS ESTIMATE")
print("-" * 70)

# If we had a valid instrument, we could run 2SLS
# For illustration, use morph_index

print("""
If morphological index is valid, we can estimate:

  Stage 1: alignment_hat = α + β * morph_index
  Stage 2: degradation = γ + δ * alignment_hat

The coefficient δ is the causal effect of alignment.
""")

# First stage
X_first = np.column_stack([np.ones(n), morph])
beta_first, _, _, _ = np.linalg.lstsq(X_first, alignment, rcond=None)
alignment_hat = X_first @ beta_first

# Second stage
X_second = np.column_stack([np.ones(n), alignment_hat])
beta_second, _, _, _ = np.linalg.lstsq(X_second, degradation, rcond=None)

# OLS for comparison
X_ols = np.column_stack([np.ones(n), alignment])
beta_ols, _, _, _ = np.linalg.lstsq(X_ols, degradation, rcond=None)

print(f"""
ESTIMATES:

  OLS (potentially biased):
    Effect of alignment: {beta_ols[1]:.1f}

  2SLS (using morph_index as instrument):
    Effect of alignment: {beta_second[1]:.1f}

  Difference: {abs(beta_second[1] - beta_ols[1]):.1f}
  Direction: {'2SLS shows STRONGER effect' if abs(beta_second[1]) > abs(beta_ols[1]) else '2SLS shows WEAKER effect'}
""")


print("\n6. HYPOTHESIS TEST")
print("-" * 70)

# Test 1: At least one candidate instrument found
test1_pass = len(valid_instruments) > 0

# Test 2: Morphological index is relevant
test2_pass = abs(r_morph_align) > 0.3

# Test 3: Morphological index is independent of training
test3_pass = abs(r_morph_train) < 0.5

print(f"""
TEST 1: At least one candidate instrument found?
  Candidates: {valid_instruments if valid_instruments else 'None'}
  Verdict: {'PASS ✓' if test1_pass else 'FAIL ✗'}

TEST 2: Morphological index is relevant (|r| > 0.3)?
  r(morph, align) = {r_morph_align:.3f}
  Verdict: {'PASS ✓' if test2_pass else 'FAIL ✗'}

TEST 3: Morphological index is independent (|r| < 0.5)?
  r(morph, train) = {r_morph_train:.3f}
  Verdict: {'PASS ✓' if test3_pass else 'FAIL ✗'}

OVERALL: {'VALID INSTRUMENT FOUND ✓' if test1_pass and test2_pass and test3_pass else 'NO CLEAR INSTRUMENT'}
""")


print("\n7. LIMITATIONS")
print("-" * 70)

print("""
IMPORTANT CAVEATS:

1. EXCLUSION RESTRICTION IS UNTESTABLE
   We cannot prove that morph_index doesn't directly affect degradation.
   It's a theoretical assumption.

2. SAMPLE SIZE IS SMALL
   n=12 is marginal for 2SLS, which has higher variance than OLS.
   First-stage F-statistic should be > 10 for reliable inference.

3. MORPH_INDEX IS CONSTRUCTED
   Our morphological complexity index is somewhat arbitrary.
   A better index from linguistics literature might help.

4. POTENTIAL FOR MANY INSTRUMENTS
   Testing multiple instruments raises multiple comparison concerns.
   Should pre-register which instruments to use.

RECOMMENDATION:
- Morphological index is the BEST AVAILABLE instrument
- Use it with caution and report both OLS and 2SLS
- Acknowledge that exclusion restriction is an assumption
- Prioritize within-language evidence (confound-free) over IV approach
""")


print("\n" + "=" * 70)
print("SUMMARY: D3 INSTRUMENTAL VARIABLE SEARCH")
print("=" * 70)

print(f"""
QUESTION: Can we find instruments to break the alignment-training confound?

ANSWER: {'PARTIAL SUCCESS' if test2_pass and test3_pass else 'NO VALID INSTRUMENT'}

BEST CANDIDATE: Morphological Index
  - Relevance: r(morph, align) = {r_morph_align:.3f}
  - Independence: r(morph, train) = {r_morph_train:.3f}
  - First-stage F: {f_stat:.2f}

2SLS ESTIMATE:
  - OLS effect: {beta_ols[1]:.1f}
  - 2SLS effect: {beta_second[1]:.1f}

LIMITATIONS:
  - Exclusion restriction is untestable
  - Small sample size
  - Requires theoretical justification

PRACTICAL RECOMMENDATION:
  While morphological index shows promise as an instrument,
  the within-language evidence (E-EXP3, D2) is cleaner.
  Use IV as supplementary evidence, not primary.
""")
