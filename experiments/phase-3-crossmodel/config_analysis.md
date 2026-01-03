# BLOOM vs XGLM: Configuration Analysis

## Key Findings

### Architectural Differences

| Parameter | BLOOM-560M | XGLM-564M | Significance |
|-----------|------------|-----------|--------------|
| Layers | 24 | 24 | Same |
| Vocab Size | 250,880 | 256,008 | Similar |
| Attention in FP32 | **Yes** | No | Stability concern |
| Dropout | Low/None | 0.1 | Regularization |
| scale_embedding | No | **Yes** | Gradient scaling |

### Critical Difference: attention_softmax_in_fp32

```
BLOOM: attention_softmax_in_fp32 = True
XGLM: Not specified (default False)
```

**Interpretation:** BLOOM team needed FP32 attention to stabilize training.
This suggests BLOOM's training dynamics were pushing towards instability.
The same dynamics may have created outlier weights.

### Regularization: Dropout

```
XGLM: dropout = 0.1, attention_dropout = 0.1
BLOOM: Not visible in config (likely minimal)
```

**Interpretation:** Dropout acts as regularization, preventing weight explosion.
XGLM's dropout may prevent outlier formation.

### Embedding Scaling

```
XGLM: scale_embedding = True (multiply by √d_model)
BLOOM: Not present
```

**Interpretation:** Embedding scaling normalizes gradient magnitudes.
Without it, gradients may be larger, contributing to outliers.

---

## Hypothesis: Why BLOOM Has Outliers

### Primary Hypothesis: Training Instability

```
BLOOM training characteristics:
1. Required FP32 attention → sign of instability
2. Minimal dropout → no weight regularization
3. No embedding scaling → larger gradients
4. Tensor parallelism → complex gradient synchronization

Result: Weights in certain layers grew large (outliers)
```

### Secondary Hypothesis: Data Imbalance

```
BLOOM: ~46 languages, heavily English-weighted
XGLM: 30 languages, claimed more balanced

If English dominates:
→ Model specializes early layers for English
→ Other languages forced into generic representations
→ Creates the activation pattern we observe
```

---

## Implications for Our Theory

### Why XGLM is Gaussian

XGLM avoided outliers through:
1. **Dropout regularization** (0.1) limiting weight growth
2. **Embedding scaling** normalizing gradients
3. **Stable attention** (no FP32 workaround needed)

### Prediction for New Models

Models likely to have outliers:
- Minimal dropout
- No embedding scaling
- Training stability issues (FP32 hacks)
- Imbalanced training data

Models likely to be Gaussian:
- Standard dropout (0.1+)
- Embedding scaling
- Stable training (no FP32 needed)
- Balanced data

---

## Evidence We'd Need

1. **BLOOM training logs** — did loss spike at layers 5, 21, 22?
2. **BLOOM data distribution** — English % vs others
3. **XGLM training details** — dropout schedule, data balance
4. **Ablation** — train model with/without dropout, check kurtosis

---

## Summary

```
BLOOM outliers likely caused by:
  └── Training instability (attention_softmax_in_fp32 = True)
      └── Minimal regularization (no visible dropout)
          └── Large gradient variance
              └── Weight outliers in specific layers

XGLM avoided this via:
  └── Standard dropout (0.1)
      └── Embedding scaling
          └── Stable training
              └── Gaussian weights
```

This explains why our disparity mechanism is BLOOM-specific:
- The mechanism requires outliers (C1)
- XGLM's training prevented outliers
- No outliers → no differential activation → no disparity

---

*Analysis date: 2026-01-03*
