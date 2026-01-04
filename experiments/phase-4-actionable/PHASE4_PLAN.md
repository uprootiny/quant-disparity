# Phase 4: Toward Actionable Techniques

## Objective

Move from correlation to causation, from observation to intervention, from understanding to engineering.

---

## What We Know (High Confidence)

| Finding | Evidence | Implication |
|---------|----------|-------------|
| r=-0.834 outlier-disparity correlation | Bootstrap CI, permutation test | Outliers explain disparity |
| Outliers in attention projections | 3/3 HEAVY models | Target attention for fixes |
| 82x growth during training | Checkpoint analysis | Training dynamics matter |
| Attention sinks ≠ outliers | r=-0.23 | Distinct phenomena |
| Fertility ≠ degradation | r=-0.07 | Not a tokenization problem |

---

## What We Don't Know (Prioritized)

### Priority 1: Prediction

**Q1:** Can we predict degradation BEFORE quantizing?
- If yes → enables pre-deployment diagnostics
- Actionable: Flag vulnerable model-language pairs

**Q2:** What pre-quantization metrics predict post-quantization quality?
- Candidates: kurtosis overlap, activation patterns, weight distribution

### Priority 2: Mechanism

**Q3:** Which specific weights matter most?
- Super weights (0.01%) vs general outliers (1%)?
- Per-layer or distributed?

**Q4:** Why are some low-resource languages more resilient?
- Not all low-resource languages degrade equally
- What factors beyond resource level?

### Priority 3: Intervention

**Q5:** Can targeted weight preservation reduce disparity?
- Keep top-k outliers in FP16
- Language-aware clipping thresholds

**Q6:** Which layers should we protect?
- Early, middle, or late?
- Same for all languages?

---

## Experiment Series

### EXP-035: Pre-Quantization Degradation Predictor

**Question:** Can we predict which languages will degrade most before quantizing?

**Hypothesis H-035:** Pre-quantization kurtosis overlap predicts post-quantization degradation.

**Method:**
1. Compute language-specific activation patterns on probe sentences
2. Measure overlap with high-kurtosis weight regions
3. Simulate quantization and measure degradation
4. Correlate pre-quant overlap with post-quant degradation

**Prediction:** r > 0.5 between overlap and degradation.

**Actionable outcome:** Diagnostic tool for model deployment.

---

### EXP-036: Layer Contribution Analysis

**Question:** Which layers contribute most to each language's performance?

**Hypothesis H-036:** Languages rely on different layer subsets; overlap with outlier layers predicts vulnerability.

**Method:**
1. Compute layer-wise activations per language
2. Measure activation magnitude distribution per layer
3. Identify "critical layers" per language (top 20% by activation)
4. Correlate with outlier layer locations

**Prediction:** Low-resource languages have critical layers overlapping with high-κ layers.

**Actionable outcome:** Layer-specific quantization strategies.

---

### EXP-037: Outlier Sparsity Patterns

**Question:** Are outliers sparse (few extreme) or dense (many moderate)?

**Hypothesis H-037:** Models with sparse outliers (few super weights) are easier to fix than dense outlier models.

**Method:**
1. Count weights exceeding various thresholds (3σ, 5σ, 10σ)
2. Compute concentration ratio: top-1% / top-10% magnitude
3. Compare across models
4. Correlate sparsity with quantization-friendliness

**Prediction:** Sparse outliers → easier to preserve selectively.

**Actionable outcome:** Model selection criteria for quantization.

---

### EXP-038: Language Resilience Factors

**Question:** What distinguishes resilient vs vulnerable languages?

**Hypothesis H-038:** Resilience correlates with representation redundancy (multiple pathways encode same information).

**Method:**
1. Identify "resilient" low-resource languages (degrade less than expected)
2. Identify "vulnerable" high-resource languages (degrade more than expected)
3. Compare their activation patterns, layer usage, head utilization
4. Find discriminating features

**Prediction:** Resilient languages use more distributed representations.

**Actionable outcome:** Training recommendations for robust multilingual models.

---

### EXP-039: Simulated Intervention Study

**Question:** Does preserving outlier weights reduce disparity?

**Hypothesis H-039:** Keeping top-k outliers in FP16 disproportionately helps low-resource languages.

**Method:**
1. Identify top-k outlier weights (k = 0.01%, 0.1%, 1%)
2. Simulate quantization with outlier preservation
3. Measure degradation per language
4. Compare disparity ratio (low-resource / high-resource degradation)

**Prediction:** Higher k → lower disparity ratio.

**Actionable outcome:** Optimal k for fairness-efficiency tradeoff.

---

## Success Criteria

| Experiment | Success Threshold | Impact |
|------------|-------------------|--------|
| EXP-035 | r > 0.5 | Enables pre-deployment diagnostics |
| EXP-036 | Significant layer-language interaction | Layer-specific quantization |
| EXP-037 | Clear sparsity pattern | Model selection criteria |
| EXP-038 | Discriminating features found | Training recommendations |
| EXP-039 | k < 1% achieves parity | Practical intervention |

---

## Execution Order

1. **EXP-037** (Sparsity) - Quick, informs other experiments
2. **EXP-036** (Layer contribution) - Foundation for intervention
3. **EXP-035** (Predictor) - Combines insights from 036, 037
4. **EXP-038** (Resilience) - Deep analysis
5. **EXP-039** (Intervention) - Validates actionable technique

---

*Created: 2026-01-04*
