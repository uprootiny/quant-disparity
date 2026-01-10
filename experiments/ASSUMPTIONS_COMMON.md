# Common Assumptions Across All Tracks

*Explicit statements of what we assume to be true. If any of these are wrong, our conclusions may be invalid.*

**Last Updated:** After E15 Confound Analysis

---

## Core Assumptions About the Problem

### A1: Quantization Causes Measurable Degradation
**Assumption:** Reducing precision from FP16 to INT4/INT8 degrades model performance in measurable ways (perplexity increase, accuracy drop).

**Could be wrong if:**
- Degradation is within noise of measurement
- Models have enough redundancy to perfectly compensate
- Our metrics don't capture actual quality

**Status:** VALIDATED - degradation is consistently measurable

---

### A2: Degradation Varies by Language
**Assumption:** Different languages experience different amounts of degradation under the same quantization.

**Could be wrong if:**
- Observed differences are measurement noise
- Benchmark difficulty varies (not model behavior)
- We're measuring benchmark artifacts, not model quality

**Status:** VALIDATED but CONFOUNDED - differences exist but may reflect benchmark quality

---

### A3: BPE Alignment is a Valid Proxy for Tokenization Quality
**Assumption:** Our "alignment score" (morpheme-token correspondence) captures something real about tokenization quality.

**Could be wrong if:**
- Alignment is just one aspect of tokenization quality
- Other factors (subword frequency, OOV rate) matter more
- Our alignment metric is poorly calibrated

**Status:** UNCERTAIN - high collinearity with vocab coverage suggests they may measure the same thing

---

### A4: Perplexity Change Reflects Real-World Impact
**Assumption:** Perplexity degradation translates to real downstream task degradation.

**Could be wrong if:**
- Perplexity doesn't correlate with task performance
- Different tasks have different sensitivity
- Generation quality degrades differently than perplexity suggests

**Status:** PARTIALLY VALIDATED - we tested multiple metrics but all are proxies

---

## Assumptions About Mechanism

### A5: Layer Position Determines Function
**Assumption:** Layer 0 primarily encodes tokenization, Layer 11 primarily encodes semantics, etc.

**Could be wrong if:**
- Functions are distributed across layers
- Position-function mapping varies by language
- Our layer importance scores are model-specific

**Status:** VALIDATED - probing literature supports this, our experiments confirm

---

### A6: Quantization Affects All Layers Similarly
**Assumption:** The quantization algorithm treats all layers equally, so any differential impact is due to layer content, not layer treatment.

**Could be wrong if:**
- Quantization algorithms adapt per-layer
- Hardware effects differ by layer
- Our INT4 simulation differs from real quantization

**Status:** ASSUMED - we haven't validated on actual quantization hardware

---

### A7: Language Representations are Separable
**Assumption:** Different languages have somewhat distinct representations that can be analyzed separately.

**Could be wrong if:**
- Multilingual models share all representations
- Language-specific patterns are emergent illusions
- Cross-lingual interference dominates

**Status:** UNCERTAIN - some evidence for language-specific neurons, but incomplete

---

## Assumptions About Data

### A8: Benchmark Quality is Uniform
**Assumption:** FLORES, mC4, and other benchmarks have comparable quality across languages.

**Could be wrong if:**
- LR benchmarks have more noise
- Translation quality varies
- Domain mismatch affects some languages more

**Status:** LIKELY VIOLATED - E14 showed benchmark quality is a critical confounder

---

### A9: Training Data Quantity Doesn't Explain Everything
**Assumption:** Alignment has effect beyond just being a proxy for training data amount.

**Could be wrong if:**
- Alignment and training data are perfect proxies
- All "alignment" effects are really "data quantity" effects
- We're measuring the same thing twice

**Status:** PARTIALLY VALIDATED - E15 showed 3/4 confound-resistant tests pass

---

### A10: Our Language Sample is Representative
**Assumption:** The ~15 languages we study represent the diversity of world languages.

**Could be wrong if:**
- We're missing important language types
- Our sample is biased toward "model-friendly" languages
- Survivorship bias (only languages in models are testable)

**Status:** UNCERTAIN - we cover major families but many gaps remain

---

## Assumptions About Causation

### A11: Alignment → Degradation (Causal Direction)
**Assumption:** Poor alignment CAUSES higher degradation, not the reverse.

**Could be wrong if:**
- Both caused by third factor (training data)
- The relationship is just correlation
- Causation runs in different direction

**Status:** PARTIALLY VALIDATED - mechanism is plausible, but confounds exist

---

### A12: Gateway-Bottleneck Pattern is Universal
**Assumption:** The L0-L9-L11 importance pattern holds across different models.

**Could be wrong if:**
- Pattern is specific to our model architecture
- Different model sizes have different patterns
- Training procedure affects layer importance

**Status:** NEEDS VALIDATION - only tested on one model family

---

## Known Violations and Caveats

1. **Benchmark quality varies** - we've documented this but can't fully correct
2. **Training data is confounded** - high collinearity with alignment
3. **Only one model family tested** - generalization is unvalidated
4. **Simulated quantization** - real hardware may differ
5. **Limited language sample** - many language types untested

---

## Recommendations

### For Claims, Use This Hierarchy:

**Can claim confidently:**
- Gateway layers matter (mechanistic evidence)
- Families cluster (robust to confounds)
- Scaling paradox exists (architectural)

**Can claim with caveats:**
- Alignment predicts degradation (confounds acknowledged)
- LR languages suffer more (partially confounded)

**Cannot claim:**
- Alignment is THE root cause
- Specific degradation percentages are precise
- Findings generalize to all models
