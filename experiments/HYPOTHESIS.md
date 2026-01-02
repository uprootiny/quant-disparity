# Hypothesis Evolution Loop

```
┌────────────────────────────────────────────────────────────────────────────┐
│  HYPOTHESIS TRACKER                                        rev. 2026-01-03 │
│  Iterative refinement based on experimental evidence                       │
└────────────────────────────────────────────────────────────────────────────┘
```

## Current Hypothesis (v3)

**Statement:**
Quantization degradation disparity across languages is determined by the
interaction between:
1. Layer-wise weight kurtosis (clumpiness)
2. Language-specific activation patterns
3. Quantization bit-width

**Formal model:**

```
degradation(lang, bits) = f(outlier_activation(lang), kurtosis_profile, bits)

where:
  outlier_activation(lang) = Σ_i∈outlier_layers activation(lang, i) / Σ_j activation(lang, j)
  kurtosis_profile = {kurt(layer_i) for i in 0..23}
  bits ∈ {2, 3, 4, 8, 16}
```

**Predicted relationship:**
```
∂degradation/∂bits < 0  (more bits = less degradation)
∂degradation/∂outlier_activation < 0  (more outlier use = less degradation)
∃ threshold(lang) where degradation spikes sharply
```

---

## Hypothesis Evolution

### v1: Tokenization Fertility (FALSIFIED)
> Languages with higher tokenization fertility degrade more under quantization.

**Evidence:** r = 0.34 (not significant)
**Reason for failure:** Vocabulary coverage confounds fertility.

### v2: Weight Kurtosis (SUPPORTED with caveats)
> Languages activating high-kurtosis layers more experience more degradation.

**Evidence:** r = -0.77 (significant but NEGATIVE)
**Revision needed:** Direction was wrong. Under-represented languages
DON'T use high-kurtosis layers, and that's the problem.

### v3: Outlier Layer Activation (CURRENT)
> Languages with LESS activation in outlier-heavy layers (5, 21, 22)
> degrade MORE because they lack robust representations that survive clipping.

**Evidence:**
- r = -0.834, p = 0.0002
- Bootstrap 95% CI: [-0.93, -0.65]
- Leave-one-out stable

---

## Instrumental Predictions

### Prediction 1: Quantization Threshold

For INT4 quantization (4-bit), we predict a "quality cliff" at:

| Language | Outlier% | Predicted threshold (bits) |
|----------|----------|---------------------------|
| eng | 20.5% | 3.0 (robust to INT3) |
| fra | 20.2% | 3.2 |
| deu | 19.0% | 3.5 |
| ara | 17.7% | 4.2 (needs INT5 for quality) |
| hin | 17.2% | 4.5 |

**Formula (to be calibrated):**
```
threshold_bits(lang) ≈ 6.0 - 15 × outlier_activation(lang)
```

### Prediction 2: Degradation Curve Shape

```
         Degradation
              ▲
              │     ara ────────┐
              │                  ╲
              │     hin ─────────╲──┐
              │                   ╲  ╲
              │                    ╲  ╲
              │     eng ────────────╲──╲─┐
              │                      ╲  ╲ ╲
              │                       ╲  ╲ ╲
              └──────────────────────────────▶ Bits
                 2    3    4    5    6    7    8

Low-outlier languages (ara, hin): steep cliff at ~4 bits
High-outlier languages (eng, fra): gradual decline, no cliff until ~3 bits
```

### Prediction 3: Clumpiness-Conditioned Distribution

Given corpus "clumpiness" C (measured as weighted kurtosis of activations):

```
degradation ~ Normal(μ(C, bits), σ(C))

where:
  μ(C, bits) = α + β₁×C + β₂×bits + β₃×C×bits
  σ(C) = σ₀ × exp(-γ×C)  # higher clumpiness = less variance
```

From our data:
- β₃ (interaction term) should be POSITIVE
- Interpretation: clumpy weights + few bits = catastrophic

---

## Testable Next Steps

### EXP-009: Bit-width Sweep

Run BLOOM-560M at multiple quantization levels:
- INT8, INT4, INT3, INT2
- Measure per-language perplexity on new corpus
- Fit degradation curves, find thresholds

**Pre-registered prediction:**
- ara, hin threshold > 4 bits
- eng, fra threshold < 3.5 bits
- Threshold negatively correlated with outlier_activation

### EXP-010: Cross-Model Validation

Test on other BLOOM sizes (1B, 3B, 7B):
- Does outlier pattern scale?
- Does correlation hold?

### EXP-011: Intervention Study

If hypothesis is correct, we can IMPROVE low-resource language
quantization by:
1. Per-layer quantization (protect outlier layers)
2. Activation-aware clipping (adjust α per language)
3. Language-specific calibration data

---

## Confidence Assessment

| Component | Confidence | Evidence Quality |
|-----------|------------|------------------|
| Negative correlation | HIGH | r=-0.83, p<0.001, robust to LOO |
| BLOOM-specific | HIGH | XGLM shows opposite pattern |
| Mechanism (outlier weights) | MEDIUM | Consistent but not causal |
| Quantitative predictions | LOW | Need calibration experiments |

---

## Decision Points

**D010:** Should we proceed with bit-width sweep (EXP-009)?
- Requires: quantized model inference
- Risk: computational cost
- Value: calibrates threshold predictions

**D011:** Should we build corpus first?
- Corpus enables better perplexity measurement
- But bit-width sweep could use existing data

**Recommended:** Parallel track — continue corpus while planning EXP-009.

---

*Last updated: 2026-01-03*
