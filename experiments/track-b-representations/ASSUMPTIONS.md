# Track B Assumptions: Representation Analysis

*What we assume about how languages are represented in models*

**Target Lab:** Belinkov Lab (Model Analysis, Probing)

---

## Track-Specific Assumptions

### B-1: Representation Redundancy Varies by Language
**Assumption:** High-resource languages have more redundant representations; low-resource have sparser, more fragile representations.

**Could be wrong if:**
- Redundancy is uniform across languages
- "Redundancy" is not well-defined
- Our measurement of redundancy is flawed

**Evidence:** E11 confirmed 3.3x representation damage for LR; E7 showed HR benefits more from model scale

**Risk Level:** LOW - multiple evidence sources

---

### B-2: Representation Quality Correlates with Training Data
**Assumption:** Languages with more training data develop better, more robust representations.

**Could be wrong if:**
- Quality is determined by typological fit to model
- Some languages are inherently harder to represent
- Quality ≠ robustness

**Evidence:** CONFOUNDED - training data correlates with everything

**Risk Level:** HIGH - critical confounder

---

### B-3: Probing Reveals True Representations
**Assumption:** Linear probes accurately measure what the model "knows" about each language.

**Could be wrong if:**
- Probes learn their own representations
- Non-linear structure is missed
- Probing itself is biased toward certain languages

**Evidence:** Literature debates probe validity; we assume probes are useful but imperfect

**Risk Level:** MODERATE

---

### B-4: Representation Damage is Measurable Pre-Deployment
**Assumption:** We can predict representation damage from model analysis without full deployment.

**Could be wrong if:**
- Damage only manifests in specific contexts
- Interaction effects dominate
- Our damage metrics miss important aspects

**Evidence:** Our simulations correlate with expected patterns

**Risk Level:** MODERATE

---

### B-5: Cross-Lingual Transfer Relies on Shared Representations
**Assumption:** Zero-shot transfer works because languages share representations; damaging these affects transfer more.

**Could be wrong if:**
- Transfer uses language-specific features
- Shared representations are more robust, not less
- Transfer mechanism is different than assumed

**Evidence:** E12 showed transfer disparity > monolingual disparity (supports assumption)

**Risk Level:** LOW

---

## Track B Specific Confounds

1. **Training data quantity** - explains most representation quality
2. **Tokenization quality** - affects what representations can form
3. **Benchmark domain** - may measure domain fit, not representation quality

---

## What Would Falsify Track B Claims?

1. LR languages with equally redundant representations as HR
2. Representation damage not predicting performance degradation
3. Cross-lingual transfer being MORE robust than monolingual
