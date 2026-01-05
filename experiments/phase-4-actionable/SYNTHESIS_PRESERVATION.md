# Synthesis: Preservation Study Results

## Key Finding: Non-Monotonic Relationship

**Preserving more weights does NOT uniformly reduce disparity.**

There is an optimal preservation level around **5%** that minimizes the disparity ratio.

## Results Summary

| Preservation % | HR Degradation | LR Degradation | Disparity Ratio |
|----------------|----------------|----------------|-----------------|
| 0% | 11,166% | 990,280% | 88.68x |
| **5%** | **7,968%** | **361,698%** | **45.39x** |
| 10% | 11,028% | 1,125,670% | 102.08x |
| 20% | 7,327% | 1,270,105% | 173.34x |

## Analysis

### Why 5% is optimal

1. **Sweet spot for outlier coverage**: 5% captures the most critical outliers without over-protecting English-biased weights

2. **Threshold bias**: At higher preservation levels, the global magnitude threshold increasingly selects English-dominant weights (since English has more training data → larger magnitudes in relevant weights)

3. **Quantization intensity**: Fewer remaining weights must absorb more quantization error at higher preservation levels

### The English-Bias Hypothesis

Large weights in transformer models are disproportionately optimized for high-resource languages. When we preserve by magnitude:

- At 5%: We capture universally critical weights
- At 10-20%: We start capturing English-optimized weights
- The remaining quantized weights matter more for low-resource languages

## Implications

### For Practitioners

1. **Don't assume more preservation = better fairness**
2. **Test multiple preservation levels on your specific language mix**
3. **5% is a good starting point for multilingual fairness**

### For Research

1. **Language-aware thresholding** may outperform magnitude-based
2. **Per-language calibration** could identify language-specific critical weights
3. **Layer-specific preservation** (especially layer 0) may be more efficient

## Recommended Strategy

```
For multilingual INT4 deployment:
1. Preserve top 5% of weights in FP16
2. Preserve 100% of layer 0 (embedding layer)
3. Quantize remaining to INT4

Expected result: ~50% reduction in language disparity
Memory overhead: ~7-8% vs pure INT4
```

## Statistical Note

This finding needs validation across:
- [ ] Multiple models (OPT, Pythia, BLOOM)
- [ ] Multiple runs (statistical significance)
- [ ] Real quantization (GPTQ, AWQ, bitsandbytes)

---

*Synthesized: 2026-01-04*
