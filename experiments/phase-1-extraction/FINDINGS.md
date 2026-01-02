# Phase 1 Findings

## Weight Statistics from BLOOM-560M

**Date:** 2026-01-02
**Model:** bigscience/bloom-560m (559M params)

### Per-Layer MLP Weight Kurtosis

| Layer | Kurtosis | Outlier% | Notes |
|-------|----------|----------|-------|
| 0 | +8.13 | 1.31% | |
| 1 | +4.71 | 0.93% | |
| 2 | +1.74 | 0.55% | |
| 3 | +2.64 | 0.56% | |
| 4 | +76.28 | 0.61% | Very high |
| 5 | **+125.92** | 0.83% | **Extreme** |
| 6 | +44.80 | 0.75% | High |
| 7 | +39.05 | 1.53% | High |
| 8 | +8.03 | 2.19% | |
| 9 | +8.98 | 2.54% | Highest outlier% |
| 10 | +5.47 | 1.09% | |
| 11 | +3.44 | 0.93% | |
| 12 | +2.84 | 0.65% | |
| 13 | +1.48 | 0.62% | |
| 14 | +0.96 | 0.58% | Near Gaussian |
| 15 | +0.98 | 0.57% | Near Gaussian |
| 16 | +1.37 | 0.56% | |
| 17 | +1.81 | 0.56% | |
| 18 | +2.99 | 0.56% | |
| 19 | +4.63 | 0.58% | |
| 20 | +16.37 | 0.63% | |
| 21 | **+148.20** | 0.60% | **Extreme** |
| 22 | **+164.30** | 0.55% | **Highest** |
| 23 | +36.31 | 0.58% | High |

### Aggregate

- **Total MLP params:** 201,326,592
- **Mean std:** 0.018
- **Mean kurtosis:** +29.64
- **Kurtosis range:** +0.96 to +164.30

### Key Observations

1. **Weights are globally heavy-tailed.** The model has mean kurtosis +30, far above Gaussian (0).

2. **Extreme layers:** Layers 5, 21, 22 have kurtosis >100. These are likely to be most sensitive to quantization.

3. **Near-Gaussian layers:** Layers 14-17 have kurtosis ~1, near Gaussian. These should quantize well.

4. **Bimodal pattern:** Early-to-mid layers (4-7) and late layers (21-23) have highest kurtosis.

### Hypothesis Refinement

**Original hypothesis:** Languages with high-kurtosis weight distributions degrade more.

**Problem:** All languages share the same weights. Kurtosis is a model property, not per-language.

**Refined hypothesis:** Languages that activate neurons in high-kurtosis layers degrade more.

- Languages needing layers 5, 21, 22 for processing → more sensitive
- Languages primarily using layers 14-17 → more robust

**Next step:** Analyze which layers activate most for each language.

### Implications for Quantization

Following Banner et al.:

| Layer Type | Kurtosis | Recommended α*/σ |
|------------|----------|------------------|
| Near-Gaussian (14-17) | ~1 | 3.5 (standard) |
| Heavy-tail (5, 21, 22) | >100 | 5.0+ (adjusted) |

**Proposal:** Layer-specific clipping thresholds based on measured kurtosis, not language.

---

## Per-Language Layer Activation Analysis

**Date:** 2026-01-02

### Method

For each language, computed activation-weighted kurtosis:

```
weighted_kurt = Σ(activation[i] × kurtosis[i]) / Σ(activation[i])
```

This measures how much each language relies on high-kurtosis vs low-kurtosis layers.

### Results

| Language | W.Kurtosis | Degradation |
|----------|------------|-------------|
| eng | 43.02 | 0.005 |
| fra | 43.02 | 0.007 |
| deu | 40.57 | 0.008 |
| zho | 40.53 | 0.013 |
| jpn | 40.27 | 0.022 |
| vie | 40.14 | 0.009 |
| fin | 40.06 | 0.016 |
| rus | 39.55 | 0.012 |
| heb | 39.28 | 0.020 |
| tur | 39.05 | 0.015 |
| ara | 38.83 | 0.025 |
| tha | 38.14 | 0.020 |
| kor | 38.05 | 0.018 |
| hin | 37.52 | 0.021 |

### Correlation

```
r = -0.766, p = 0.0014 [SIGNIFICANT]
```

**The correlation is NEGATIVE.** Languages with LOWER activation-weighted kurtosis degrade MORE.

### Interpretation

This is counterintuitive. We expected high-kurtosis reliance → more degradation.

Instead: **Low-kurtosis reliance → more degradation.**

Possible explanations:

1. **Representation quality hypothesis:** Well-represented languages (eng, fra) have richer representations in late (high-kurtosis) layers. These representations are more redundant/robust.

2. **Early layer fragility:** Under-represented languages rely more on early layers. Early layer representations may be more compressed and fragile.

3. **Training data effect:** BLOOM was trained heavily on English/French. Their representations in late layers are more robust to quantization noise.

### Revised Hypothesis

> Languages that proportionally rely more on EARLY layers (lower weighted kurtosis) degrade more under quantization, possibly because early layer representations are more fragile for underrepresented languages.

### Next Steps

1. Test whether this holds on other models (XGLM, mT5)
2. Investigate early vs late layer sensitivity directly
3. Check if this correlates with training data volume
