# Cross-Model Survey: Findings

## Models Analyzed

| Model | Type | Size | Max κ | Mean κ | Pattern |
|-------|------|------|-------|--------|---------|
| BLOOM-560M | Decoder | 560M | 164.3 | 29.6 | Heavy (MLP layers) |
| XGLM-564M | Decoder | 564M | 1.9 | 0.6 | Gaussian |
| XLM-R-base | Encoder | 270M | 9.8 | 5.1 | Mild (last layer) |
| DistilmBERT | Encoder | 135M | 10.8 | 2.1 | Mild |
| mT5-small | Enc-Dec | 300M | 44.7 | ~5 | Moderate (decoder attn) |

## Key Observations

### 1. Outlier Location Varies by Architecture

| Model | Outlier Location | Component |
|-------|------------------|-----------|
| BLOOM | Layers 5, 21, 22 | MLP (dense_h_to_4h) |
| XLM-R | Layer 11 | Last encoder layer |
| mT5 | Decoder blocks 2-5 | Attention output (o.weight) |

### 2. Decoder-Heavy Models Have More Outliers

```
BLOOM (decoder-only):    max κ = 164
mT5 (decoder):           max κ = 44.7
mT5 (encoder):           max κ = 21.3
XLM-R (encoder-only):    max κ = 9.8
DistilmBERT (encoder):   max κ = 10.8
```

**Hypothesis:** Autoregressive decoding creates conditions for outlier formation.

### 3. XGLM is Anomalously Gaussian

Despite being decoder-only like BLOOM, XGLM has near-Gaussian weights.

Possible explanations:
- Different training data distribution
- Different optimizer settings
- Different architectural choices (activation, normalization)

### 4. mT5's Decoder Pattern

Top outlier tensors in mT5:
```
decoder.block.5.layer.1.EncDecAttention.o.weight  κ=44.7
decoder.block.2.layer.2.DenseReluDense.wo.weight  κ=42.4
decoder.block.2.layer.0.SelfAttention.o.weight    κ=37.7
decoder.block.4.layer.1.EncDecAttention.o.weight  κ=35.0
decoder.block.4.layer.0.SelfAttention.o.weight    κ=31.9
```

Outliers concentrated in:
- Cross-attention output (EncDecAttention.o)
- Self-attention output (SelfAttention.o)
- MLP output (DenseReluDense.wo)

**Insight:** Attention OUTPUT projections are outlier-prone in mT5.

---

## Implications for LA-ACIQ

### Generalization Question

```
Does the activation-degradation correlation hold for models with different
outlier patterns?

BLOOM:  Outliers in MLP → language-dependent MLP activation → disparity
mT5:    Outliers in decoder attention → language-dependent attention → disparity?
```

### Prediction

If outlier location matters:
- BLOOM disparity linked to MLP activation patterns
- mT5 disparity would be linked to attention patterns
- Different calibration strategies needed

### What We'd Need to Test

1. Extract mT5 per-language activation patterns
2. Compute correlation with degradation
3. If r < -0.5: same mechanism as BLOOM
4. If r ≈ 0: different mechanism, outlier location matters

---

## Updated Model Classification

### Category A: Heavy Outliers (Potential Disparity)
- BLOOM-560M
- mT5-small (decoder)

### Category B: Mild Outliers (Possible Disparity)
- XLM-R-base
- DistilmBERT

### Category C: Gaussian (No Disparity Mechanism)
- XGLM-564M

---

## Open Questions

1. **Why is XGLM Gaussian?**
   - Same architecture family as BLOOM
   - But completely different weight distribution
   - Training data? Optimizer? Regularization?

2. **Does decoder depth correlate with outliers?**
   - mT5 has outliers in blocks 2-5 (middle-late)
   - BLOOM has outliers in layers 5, 21, 22 (early + late)
   - No clear depth pattern

3. **Is attention output special?**
   - mT5 outliers concentrated in .o.weight (output projection)
   - BLOOM outliers in MLP
   - Different components, same phenomenon?

---

*Updated: 2026-01-03*
