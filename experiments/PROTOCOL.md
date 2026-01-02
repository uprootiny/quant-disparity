# Experimental Protocol Log

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PROTOCOL LOG                                              rev. 2026-01-02 │
│  Pre-registered hypotheses, predictions, and results                       │
└────────────────────────────────────────────────────────────────────────────┘
```

## Experiment Registry

| ID | Date | Hypothesis | Prediction | Result | Status |
|----|------|------------|------------|--------|--------|
| EXP-001 | 2026-01-02 | Fertility→Degradation | r > 0.7 | r = 0.34 | FALSIFIED |
| EXP-002 | 2026-01-02 | Kurtosis→Degradation (mock) | r > 0.7 | r = 0.92 | PASSED |
| EXP-003 | 2026-01-02 | Layer activation→Degradation | r > 0.5 | r = -0.77 | SURPRISING |
| EXP-004 | 2026-01-02 | XGLM replication | r < 0 | — | PENDING |

---

## EXP-001: Tokenization Fertility

**Pre-registration:** 2026-01-02
**Status:** FALSIFIED

### Hypothesis
Tokenization fertility (tokens per word) directly predicts quantization degradation.

### Prediction
- r > 0.7 between fertility and degradation
- High-fertility languages (Arabic, Japanese) degrade more

### Method
- Compute fertility using GPT-2 and BLOOM tokenizers
- Correlate with Marchisio degradation values

### Result
- GPT-2: r = 0.34 (p > 0.05)
- BLOOM: r = 0.36 (p > 0.05)

### Conclusion
FALSIFIED. Vocabulary coverage confounds fertility measurement.

---

## EXP-002: Weight Kurtosis (Mock Data)

**Pre-registration:** 2026-01-02
**Status:** PASSED (but mock data)

### Hypothesis
Weight distribution kurtosis predicts quantization degradation.

### Prediction
- r > 0.7 between kurtosis and degradation
- High-kurtosis languages degrade more

### Method
- Use mock weight statistics (designed with expected patterns)
- Correlate with degradation

### Result
- r = 0.92 (p < 0.0001)

### Conclusion
PASSED but with mock data. Need validation with real weights.

### Caveat
Mock data may be circular (designed with expected outcome).

---

## EXP-003: Layer Activation Pattern (BLOOM)

**Pre-registration:** 2026-01-02
**Status:** SIGNIFICANT (unexpected direction)

### Hypothesis
Languages relying more on high-kurtosis layers degrade more.

### Prediction
- r > 0.5 between activation-weighted kurtosis and degradation
- Languages activating layers 5, 21, 22 (high kurtosis) degrade more

### Method
1. Extract per-layer weight kurtosis from BLOOM-560M
2. For each language, compute activation magnitude per layer
3. Compute weighted kurtosis: Σ(act[i] × kurt[i]) / Σ(act[i])
4. Correlate with degradation

### Result
- r = -0.77 (p = 0.0014)
- NEGATIVE correlation

### Conclusion
SIGNIFICANT but OPPOSITE direction. Languages with LOWER weighted kurtosis degrade MORE.

### Interpretation
Under-represented languages (ara, hin) rely more proportionally on early layers.
Early layer representations may be more fragile for these languages.

### Data
| Language | W.Kurtosis | Degradation |
|----------|------------|-------------|
| eng | 43.02 | 0.005 |
| fra | 43.02 | 0.007 |
| ara | 38.83 | 0.025 |
| hin | 37.52 | 0.021 |

---

## EXP-004: XGLM Replication

**Pre-registration:** 2026-01-02
**Status:** COMPLETE — NULL HYPOTHESIS ACCEPTED

### Hypothesis
The negative correlation (r = -0.77) found in BLOOM replicates in XGLM.

### Prediction (pre-registered)
- **Primary:** r < 0 (same direction as BLOOM)
- **Secondary:** |r| > 0.5 (similar magnitude)
- **Null:** r ≈ 0 or r > 0 (finding is BLOOM-specific)

### Method
1. Load XGLM-564M (Meta's multilingual model)
2. Extract per-layer weight kurtosis
3. For each language, compute activation-weighted kurtosis
4. Correlate with Marchisio degradation values
5. Compare with BLOOM results

### Results

| Model | r | p | Significant? |
|-------|---|---|--------------|
| BLOOM | -0.766 | 0.0014 | YES |
| XGLM | +0.377 | 0.1844 | NO |

### Key Observations

1. **XGLM has near-Gaussian weights:** Layer kurtosis 0.2–1.9 (vs BLOOM's 0.96–164)
2. **XGLM has uniform activation patterns:** W.kurt range 0.59–0.65 (spread 0.06)
3. **BLOOM has variable patterns:** W.kurt range 37.5–43.0 (spread 5.5)

### Conclusion

**NULL HYPOTHESIS ACCEPTED.** The negative correlation is BLOOM-specific.

Possible explanations:
1. BLOOM's heavy-tailed weights create language-dependent activation patterns
2. XGLM's near-Gaussian weights don't differentiate languages
3. The effect depends on training data distribution (BLOOM has more English)

### Decision
Per decision criteria: r_xglm > 0 → **BLOOM-SPECIFIC**

Next step: Investigate why BLOOM shows the pattern but XGLM doesn't.

---

## Roadmap Reference

From `experiments/README.md`:

```
Phase 0: Proof of concept     [COMPLETE]
Phase 1: Real weight analysis [IN PROGRESS - XGLM validation]
Phase 2: Layer sensitivity    [BLOCKED on Phase 1]
Phase 3: Algorithm development [FUTURE]
```

Current position: Phase 1, Experiment 4 (XGLM replication)

---

## Decision Log Cross-Reference

- D001: Pivot from fertility → weight distribution
- D006: Shift to per-layer analysis
- D007: Document surprising negative correlation

---

*Last updated: 2026-01-02*
