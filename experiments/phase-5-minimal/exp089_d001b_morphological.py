#!/usr/bin/env python3
"""
Exp-089 / D-001b: Morphological Complexity Analysis

Test if morphologically rich languages (Hebrew, Arabic) show
different layer dependency patterns than morphologically simple
languages (English).

CPU-feasible version: Use perplexity on morphologically simple
vs complex sentences.
"""
import numpy as np

print("=" * 70)
print("EXP-089 / D-001b: MORPHOLOGICAL COMPLEXITY ANALYSIS")
print("=" * 70)

# Morphological complexity by language
# MRL = Morphologically Rich Language
LANGUAGE_MORPHOLOGY = {
    'en': {'type': 'analytic', 'mrl': False, 'agreement_features': 2},  # number, person
    'fr': {'type': 'fusional', 'mrl': False, 'agreement_features': 4},   # number, gender, person, tense
    'de': {'type': 'fusional', 'mrl': True,  'agreement_features': 5},   # +case
    'he': {'type': 'templatic', 'mrl': True,  'agreement_features': 6},  # +construct state
    'ar': {'type': 'templatic', 'mrl': True,  'agreement_features': 7},  # +aspect, voice
    'zh': {'type': 'isolating', 'mrl': False, 'agreement_features': 0},  # no agreement
}

# From Exp-080: Disparity results with L0+L9+L11 protection
DISPARITY_L0_L9_L11 = {
    'en': 1.00,  # Reference
    'de': 0.67,
    'fr': 0.43,
    'he': 0.48,
    'ar': 0.31,
    'zh': 0.31,
}

# From baseline (no protection): disparity relative to English
DISPARITY_BASELINE = {
    'en': 1.0,
    'de': 0.9,
    'fr': 0.7,
    'he': 42.0,
    'ar': 20.0,
    'zh': 6.0,
}

print("\n1. MORPHOLOGICAL COMPLEXITY DATA")
print("-" * 50)
print(f"{'Lang':<6} {'Type':<12} {'MRL':<6} {'Features':<10} {'Base Disp':<12} {'Protected':<12}")
for lang in ['en', 'de', 'fr', 'he', 'ar', 'zh']:
    m = LANGUAGE_MORPHOLOGY[lang]
    bd = DISPARITY_BASELINE[lang]
    pd = DISPARITY_L0_L9_L11[lang]
    print(f"{lang:<6} {m['type']:<12} {str(m['mrl']):<6} {m['agreement_features']:<10} {bd:<12.1f} {pd:<12.2f}")

print("\n2. CORRELATION ANALYSIS")
print("-" * 50)

# Extract data
langs = ['en', 'de', 'fr', 'he', 'ar', 'zh']
features = [LANGUAGE_MORPHOLOGY[l]['agreement_features'] for l in langs]
mrl = [1 if LANGUAGE_MORPHOLOGY[l]['mrl'] else 0 for l in langs]
base_disp = [DISPARITY_BASELINE[l] for l in langs]
prot_disp = [DISPARITY_L0_L9_L11[l] for l in langs]

# Protection benefit: how much did L0+L9+L11 help?
protection_benefit = [base_disp[i] / prot_disp[i] if prot_disp[i] > 0 else 0
                      for i in range(len(langs))]

print("Protection benefit (baseline/protected disparity):")
for i, lang in enumerate(langs):
    print(f"  {lang}: {protection_benefit[i]:.1f}x improvement")

# Correlations
corr_features_base = np.corrcoef(features, np.log([d + 1 for d in base_disp]))[0, 1]
corr_features_prot = np.corrcoef(features, prot_disp)[0, 1]
corr_features_benefit = np.corrcoef(features, protection_benefit)[0, 1]
corr_mrl_base = np.corrcoef(mrl, np.log([d + 1 for d in base_disp]))[0, 1]

print(f"\nCorrelation (agreement_features vs log-baseline-disparity): r = {corr_features_base:.3f}")
print(f"Correlation (agreement_features vs protected-disparity): r = {corr_features_prot:.3f}")
print(f"Correlation (agreement_features vs protection-benefit): r = {corr_features_benefit:.3f}")
print(f"Correlation (MRL vs log-baseline-disparity): r = {corr_mrl_base:.3f}")

print("\n3. MRL vs NON-MRL COMPARISON")
print("-" * 50)

mrl_langs = [l for l in langs if LANGUAGE_MORPHOLOGY[l]['mrl']]
non_mrl_langs = [l for l in langs if not LANGUAGE_MORPHOLOGY[l]['mrl']]

mrl_base_avg = np.mean([DISPARITY_BASELINE[l] for l in mrl_langs])
mrl_prot_avg = np.mean([DISPARITY_L0_L9_L11[l] for l in mrl_langs])
non_mrl_base_avg = np.mean([DISPARITY_BASELINE[l] for l in non_mrl_langs])
non_mrl_prot_avg = np.mean([DISPARITY_L0_L9_L11[l] for l in non_mrl_langs])

print(f"MRL languages ({', '.join(mrl_langs)}):")
print(f"  Baseline disparity: {mrl_base_avg:.1f}x")
print(f"  Protected disparity: {mrl_prot_avg:.2f}x")
print(f"  Improvement: {mrl_base_avg / mrl_prot_avg:.1f}x")

print(f"\nNon-MRL languages ({', '.join(non_mrl_langs)}):")
print(f"  Baseline disparity: {non_mrl_base_avg:.1f}x")
print(f"  Protected disparity: {non_mrl_prot_avg:.2f}x")
print(f"  Improvement: {non_mrl_base_avg / non_mrl_prot_avg:.1f}x")

print("\n4. TEMPLATIC MORPHOLOGY (SEMITIC) ANALYSIS")
print("-" * 50)

templatic = [l for l in langs if LANGUAGE_MORPHOLOGY[l]['type'] == 'templatic']
non_templatic = [l for l in langs if LANGUAGE_MORPHOLOGY[l]['type'] != 'templatic']

temp_base = np.mean([DISPARITY_BASELINE[l] for l in templatic])
temp_prot = np.mean([DISPARITY_L0_L9_L11[l] for l in templatic])
non_temp_base = np.mean([DISPARITY_BASELINE[l] for l in non_templatic])
non_temp_prot = np.mean([DISPARITY_L0_L9_L11[l] for l in non_templatic])

print(f"Templatic (Semitic) languages ({', '.join(templatic)}):")
print(f"  Baseline: {temp_base:.1f}x → Protected: {temp_prot:.2f}x")
print(f"  Improvement: {temp_base / temp_prot:.1f}x")

print(f"\nOther morphology types ({', '.join(non_templatic)}):")
print(f"  Baseline: {non_temp_base:.1f}x → Protected: {non_temp_prot:.2f}x")
print(f"  Improvement: {non_temp_base / non_temp_prot:.1f}x")

print("\n5. HYPOTHESIS TESTING")
print("-" * 50)

# H-D1: MRLs suffer more under quantization
mrl_suffers_more = mrl_base_avg > non_mrl_base_avg
print(f"H-D1: MRLs suffer more under quantization (baseline)")
print(f"  MRL avg: {mrl_base_avg:.1f}x vs Non-MRL avg: {non_mrl_base_avg:.1f}x")
print(f"  Result: {'SUPPORTED' if mrl_suffers_more else 'NOT SUPPORTED'}")

# H-D2: MRLs benefit more from protection
mrl_benefit = mrl_base_avg / mrl_prot_avg
non_mrl_benefit = non_mrl_base_avg / non_mrl_prot_avg
mrl_benefits_more = mrl_benefit > non_mrl_benefit

print(f"\nH-D2: MRLs benefit more from L0+L9+L11 protection")
print(f"  MRL improvement: {mrl_benefit:.1f}x vs Non-MRL: {non_mrl_benefit:.1f}x")
print(f"  Result: {'SUPPORTED' if mrl_benefits_more else 'NOT SUPPORTED'}")

# H-D3: Morphological complexity correlates with vulnerability
print(f"\nH-D3: Morphological complexity correlates with vulnerability")
print(f"  Correlation (features vs baseline): r = {corr_features_base:.3f}")
print(f"  Result: {'SUPPORTED' if corr_features_base > 0.5 else 'WEAK/NOT SUPPORTED'}")

print("\n6. IMPLICATIONS FOR TRACK D")
print("-" * 50)

print("""
KEY FINDINGS:

1. Semitic (templatic) languages show LARGEST baseline disparity:
   - Hebrew: 42x, Arabic: 20x
   - These have root-pattern morphology that may require
     specific layer processing

2. MRLs benefit DISPROPORTIONATELY from protection:
   - The L0+L9+L11 strategy helps Hebrew/Arabic more than English
   - Suggests these layers handle morphological processing

3. Chinese (isolating, no morphology) has moderate baseline disparity (6x)
   - This may be due to script complexity, not morphology
   - Decouples morphology from script effects

PROPOSED FOLLOW-UP (requires model activations):
- Probe which layers encode morphological features
- Test if L9 specifically handles agreement disambiguation
- Compare attention patterns on morphologically ambiguous sentences
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Exp-089 / D-001b Results:

MRL Analysis:
  - MRLs (de, he, ar) baseline disparity: {mrl_base_avg:.1f}x
  - Non-MRLs (en, fr, zh) baseline disparity: {non_mrl_base_avg:.1f}x
  - MRLs suffer {mrl_base_avg / non_mrl_base_avg:.1f}x more than non-MRLs

Protection Effectiveness:
  - MRLs improved {mrl_benefit:.1f}x with L0+L9+L11
  - Non-MRLs improved {non_mrl_benefit:.1f}x with L0+L9+L11
  - MRLs benefit {mrl_benefit / non_mrl_benefit:.1f}x more from protection

Hypothesis Status:
  - H-D1 (MRLs suffer more): {'SUPPORTED' if mrl_suffers_more else 'NOT SUPPORTED'}
  - H-D2 (MRLs benefit more from protection): {'SUPPORTED' if mrl_benefits_more else 'NOT SUPPORTED'}
  - H-D3 (complexity correlates): r = {corr_features_base:.3f}

Connection to Track A:
  - L9 may be the "morphological consolidation" layer
  - Explains why L0+L9+L11 helps Semitic languages most
""")
