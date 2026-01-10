# Track D Assumptions: Linguistic Analysis

*What we assume about language typology and its relationship to model behavior*

**Target Lab:** Goldberg Lab (Multilingual NLP, Linguistics)

---

## Track-Specific Assumptions

### D-1: Alignment is THE Root Cause
**Assumption:** BPE-morpheme alignment is the primary cause of quantization disparity.

**Could be wrong if:**
- Alignment is proxy for training data
- Multiple causes contribute equally
- Our alignment metric is flawed

**Evidence:** CONTESTED - E14 showed critical confounds; E15 showed 3/4 tests pass

**Risk Level:** HIGH - this is our central claim and it's contested

---

### D-2: Morphology Predicts Tokenization Quality
**Assumption:** Language morphological type (analytic, fusional, agglutinative, templatic) predicts BPE behavior.

**Could be wrong if:**
- Morphology is one factor among many
- BPE behavior is vocabulary-dependent
- Our morphology categorization is too coarse

**Evidence:** E3 showed strong family clustering (F=35.71); E5 showed root-heavy model best

**Risk Level:** LOW - typological patterns are robust

---

### D-3: Language Families are Meaningful Units
**Assumption:** Languages within the same family behave similarly; family membership predicts sensitivity.

**Could be wrong if:**
- Within-family variance is high
- Family membership is confounded with geography/resources
- Our family assignments are incorrect

**Evidence:** E3 showed within-family variance < between-family variance (F=35.71)

**Risk Level:** LOW - statistically validated

---

### D-4: Root Alignment is Most Critical (for Semitic)
**Assumption:** For templatic languages, root misalignment matters more than prefix/suffix misalignment.

**Could be wrong if:**
- All position misalignments are equally bad
- Prefix misalignment dominates (Hebrew definiteness)
- Our decomposition is too simplistic

**Evidence:** E5 showed root-heavy model best predicts degradation (r=0.921)

**Risk Level:** MODERATE - based on one decomposition

---

### D-5: Tokenizer Intervention Would Help
**Assumption:** Morphology-aware tokenization would reduce disparity at the source.

**Could be wrong if:**
- Tokenization is not the real bottleneck
- Re-training is impractical
- Benefits don't transfer to quantized models

**Evidence:** E8 simulated 35% Semitic reduction; not empirically validated

**Risk Level:** HIGH - only simulated, not tested

---

## Track D Specific Confounds

### CRITICAL: Alignment-Training Data Collinearity
E14 showed:
- r(alignment, vocab_coverage) = 0.966
- r(alignment, benchmark_quality) = 0.987
- Partial correlations collapse when controlling for these

This means: **We cannot definitively say alignment is causal**

### MITIGATING EVIDENCE (E15):
- Within-family correlation: r = -0.828 (alignment predicts within controlled groups)
- Outlier analysis: Dutch beats Russian despite less training data
- Mechanistic test: L0 contribution correlates with alignment, not training data

---

## Refined Position

**Original claim:** "Alignment causes disparity"

**Revised claim:** "Alignment is a strong predictor of disparity, with mechanistic evidence supporting an independent effect beyond training data confounds. However, causation is not definitively established due to high collinearity with vocabulary coverage and benchmark quality."

---

## What Would Falsify Track D Claims?

1. Finding high-alignment LR language with high degradation
2. Showing alignment-degradation correlation disappears when controlling for confounds
3. Demonstrating language family clustering is spurious
4. Finding that tokenizer changes don't affect quantization sensitivity
