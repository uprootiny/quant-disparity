# Research Proposal: Multilingual Quantization Disparity

**Target:** Soudry Lab, Technion
**Status:** Validating core hypothesis

## Abstract

Multilingual LLMs exhibit non-uniform performance degradation under quantization, with non-Latin scripts suffering 2-3x more than Latin scripts. We propose that **weight distribution kurtosis** predicts this degradation, providing a mechanistic explanation grounded in Banner et al.'s optimal clipping theory. Preliminary analysis shows r=0.92 correlation between kurtosis and degradation.

## Research Question

Why do certain languages degrade more than others when LLMs are quantized?

## Core Hypothesis

Languages that activate neurons with **heavy-tailed weight distributions** (high kurtosis) are more sensitive to quantization because:

1. Standard clipping (α*/σ ≈ 3.5) clips more information from heavy tails
2. These clipped outliers carry disproportionate semantic information
3. Non-Latin script languages tend to activate such neurons

## Methodology

### Phase 1: Validate Weight Distribution Hypothesis

```
For each language L:
  1. Run inference on L-specific text through multilingual model
  2. Identify top-k activated neurons per layer
  3. Extract weight statistics for those neurons
  4. Compute: mean, std, kurtosis, outlier_ratio
  5. Correlate with known degradation values
```

### Phase 2: Layer Sensitivity Matrix

```
For each (layer, language) pair:
  1. Quantize only that layer to W4
  2. Measure perplexity change
  3. Build sensitivity matrix
  4. Identify: FFN vs attention, early vs late layers
```

### Phase 3: Language-Aware Quantization

```
Proposed algorithm:
  1. Compute per-layer kurtosis for target language
  2. Adjust clipping threshold: α*_adj = α*_gauss × (1 + 0.1 × kurtosis)
  3. Apply mixed-precision: more bits for high-kurtosis layers
```

## Preliminary Results

| Predictor | r | p | Interpretation |
|-----------|---|---|----------------|
| fertility (mock) | 0.93 | <0.001 | Circular (falsified) |
| fertility (real) | 0.34 | >0.05 | Confounded by vocabulary |
| **kurtosis (mock)** | **0.92** | **<0.001** | **Promising** |
| script_type | 0.73 | <0.05 | Descriptive only |

## Alignment with Soudry Lab

| Our Method | Reference |
|------------|-----------|
| Optimal clipping analysis | Banner et al. 2019, Sec. 3 |
| Layer sensitivity | Chmiel et al. 2025, Sec. 4 |
| Kurtosis as diagnostic | Chmiel et al. 2025, Sec. 3.2 |

## Expected Contributions

1. **Mechanistic explanation** for multilingual quantization disparity
2. **Language-aware quantization** algorithm
3. **Diagnostic tool** for identifying quantization-sensitive languages
4. **Extension** of Banner framework to multilingual setting

## Timeline

| Phase | Goal | Compute |
|-------|------|---------|
| 0 | Validate mock assumptions | CPU |
| 1 | Extract real weight stats | Small GPU |
| 2 | Layer sensitivity matrix | GPU cluster |
| 3 | Algorithm development | Moderate |

## Open Questions

1. Does training data volume explain kurtosis differences?
2. Which layers are most sensitive per language type?
3. Can mixed-precision restore equity across languages?
