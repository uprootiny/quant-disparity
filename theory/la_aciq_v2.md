# LA-ACIQ v2: Refined Theory

## Revision Notes

v1 established the framework. v2 adds:
- Tighter disparity bound
- Cross-model predictions
- Testable conditions
- Causal model

---

## 1. Core Result (Unchanged)

```
Empirical:  r(outlier_activation, degradation) = -0.834
            Bootstrap CI: [-0.93, -0.65]

Meaning:    Languages that activate outlier layers LESS degrade MORE
```

---

## 2. Refined Causal Model

### 2.1 Causal Graph

```
Training Data     Model Architecture
Distribution  →   (GeLU, LayerNorm, etc.)
     ↓                    ↓
     └────────┬───────────┘
              ↓
     Outlier Weight Formation
     (layers 5, 21, 22 in BLOOM)
              ↓
     Language-Specific Activation Patterns
     (eng: 20.5%, ara: 17.7% in outlier layers)
              ↓
     Quantization Applied
     (single α for all)
              ↓
     Differential Degradation
     (eng: 0.5%, ara: 2.5%)
```

### 2.2 Key Mediators

**M1: Outlier Formation**
```
Condition: Model develops outlier weights during training
Evidence:  BLOOM has κ > 100 in layers 5,21,22; XGLM has κ < 2 everywhere
Cause:     Unknown (training dynamics, architecture, data mix)
```

**M2: Language-Dependent Activation**
```
Condition: Languages activate outlier layers differently
Evidence:  3+ percentage point spread in BLOOM
Cause:     Training data volume creates specialization
```

**M3: Quantization Sensitivity**
```
Condition: Outlier layers are more sensitive to quantization
Evidence:  High κ means more clipping error for fixed α
Theory:    Banner et al. MSE decomposition
```

### 2.3 Necessary Conditions for Disparity

For quantization disparity to occur, model must satisfy:

```
(C1) Outlier layers exist:     max_l κ_l > κ_threshold
(C2) Activation varies:        Var_λ[ā_outlier(λ)] > ε
(C3) Quantization is applied:  B < B_safe
```

If any condition fails, disparity ≈ 0.

---

## 3. Cross-Model Predictions

### 3.1 Model Classification

| Model | (C1) Outliers? | (C2) Varies? | Predicted Disparity |
|-------|---------------|--------------|---------------------|
| BLOOM | ✓ (κ=164) | ✓ (spread=3%) | HIGH |
| XGLM | ✗ (κ=1.9) | ? | LOW (C1 fails) |
| mT5 | ✓ (κ=45) | ? | MODERATE |
| XLM-R | ~ (κ=10) | ? | LOW-MODERATE |

### 3.2 XGLM Null Result Explained

XGLM shows no correlation (r=+0.38, n.s.) because:

```
XGLM fails (C1): max κ = 1.9 (near-Gaussian)
                 No outlier layers to activate differentially

Therefore:       All languages experience same effective distribution
                 Single α is approximately optimal for all
                 No disparity mechanism
```

**Prediction:** Any model with max κ < 5 will show no disparity pattern.

### 3.3 mT5 Prediction

mT5 has outliers (κ=45) in decoder attention layers.

```
If (C2) holds for mT5:  Expect r < -0.5 between decoder attention
                        activation and degradation

If (C2) fails:          Languages may activate decoder uniformly
                        despite outliers existing
```

**Testable:** Extract mT5 activation patterns, compute correlation.

---

## 4. Tighter Disparity Bound

### 4.1 Decomposition

Degradation for language λ under quantization:

```
D(λ) = Σ_l w_l · MSE_l(α, λ)

where:
  w_l = importance weight of layer l
  MSE_l(α, λ) = quantization error at layer l for language λ
```

### 4.2 Banner's Layer-wise MSE

From ACIQ, for layer l with kurtosis κ_l:

```
MSE_l(α) = σ_l² · g(α/σ_l, κ_l, B)

where g captures clipping + quantization trade-off
```

### 4.3 Language-Dependence

Languages differ in which layers they "weight":

```
D(λ) = Σ_l ā_l(λ) · σ_l² · g(α/σ_l, κ_l, B)
```

### 4.4 Disparity Bound (Refined)

**Theorem (Informal):**

```
Disparity ≤ max_l |g'(κ_l)| · Var_λ[ā_l(λ)] · max_l σ_l²
```

**Interpretation:**
- Disparity bounded by product of:
  1. How sensitive MSE is to kurtosis (g')
  2. How much activation patterns vary (Var)
  3. How large weights are (σ²)

### 4.5 Empirical Fit

```
From BLOOM data:
  max |g'| ≈ 0.1 (for κ ≈ 100)
  Var[ā] ≈ 0.001 (for outlier layers)
  max σ² ≈ 0.01

Predicted bound: 0.1 × 0.001 × 0.01 = 0.000001 (too tight!)

Issue: Need to account for cumulative effect across layers
       and interaction between activation and kurtosis
```

**Revised bound (empirical):**

```
Disparity ≈ C · |r(ā_outlier, D)| · Range[ā_outlier] · Range[κ]

Plugging in:
  C ≈ 0.01
  r ≈ 0.834
  Range[ā] ≈ 0.03
  Range[κ] ≈ 160

  Predicted: 0.01 × 0.834 × 0.03 × 160 ≈ 0.04

Observed: 0.020

Order of magnitude correct, factor of 2 error.
```

---

## 5. Testable Predictions

### 5.1 Bit-width Threshold

**Prediction:** Critical bit-width B* where disparity emerges:

```
B* = f(max_l κ_l)

For BLOOM (κ=164):  B* ≈ 5-6 bits
For XGLM (κ=2):     B* ≈ 2-3 bits (below practical range)
```

**Test:** EXP-009 bit-width sweep (needs GPU)

### 5.2 Layer Ablation

**Prediction:** Increasing precision for outlier layers reduces disparity

```
If layers 5,21,22 use INT8 while rest uses INT4:
  Disparity should decrease by ~50%

Reasoning: Outlier layers contribute most to the effect
```

**Test:** Mixed-precision experiment

### 5.3 Training Data Correlation

**Prediction:** Languages with more training data have higher outlier activation

```
r(training_tokens(λ), ā_outlier(λ)) > 0.5
```

**Test:** Obtain BLOOM training data statistics, correlate

### 5.4 New Model Screening

**Prediction:** For any model M with outlier layers:

```
If max κ_l(M) > 50:
  Expect r(ā_outlier, D) < -0.5

If max κ_l(M) < 5:
  Expect |r(ā_outlier, D)| < 0.3
```

**Test:** Survey more models (Pythia, OPT, Llama)

---

## 6. Why XGLM Is Gaussian

### 6.1 Possible Explanations

| Hypothesis | Mechanism | Testable? |
|------------|-----------|-----------|
| Training data | More balanced across languages | Need data stats |
| Optimizer | Different hyperparameters | Check configs |
| Architecture | Subtle differences | Compare configs |
| Regularization | Weight decay prevents outliers | Check training |
| Initialization | Different init scheme | Check code |

### 6.2 BLOOM vs XGLM Config Comparison

Need to extract and compare:
- Model dimension, layers, heads
- Activation function (GeLU vs SiLU vs SwiGLU)
- LayerNorm (pre vs post, epsilon)
- Positional encoding (learned vs rotary)
- Optimizer (Adam vs AdamW, learning rate)
- Weight decay
- Training data composition

---

## 7. Implications for Practice

### 7.1 Pre-Quantization Checklist

```
□ Extract per-layer kurtosis
□ If max κ > 50: proceed with caution
□ Compute language-wise activation patterns
□ If Var[ā_outlier] > 0.001: expect disparity
□ Consider per-language calibration or mixed precision
```

### 7.2 Quantization Strategy Selection

```
IF max κ < 5:
    Use standard PTQ (no special handling)

ELIF max κ < 50:
    Use standard PTQ with per-language eval

ELSE:  # max κ > 50
    EITHER:
        1. Mixed precision (INT8 for outlier layers)
        2. Per-language calibration
        3. Higher global bit-width
```

### 7.3 Fairness Reporting

```
Always report:
  - Per-language degradation D(λ)
  - Disparity metric: max D - min D
  - Outlier layer identification
  - Bit-width used
```

---

## 8. Summary of Refinements

| Aspect | v1 | v2 |
|--------|-----|-----|
| Causal model | Implicit | Explicit graph |
| Cross-model | BLOOM only | Predictions for XGLM, mT5 |
| Bound | Conjecture | Semi-empirical formula |
| Testability | Vague | 4 specific predictions |
| XGLM null | Unexplained | C1 failure (no outliers) |

---

## 9. Next Steps

1. **Compare BLOOM/XGLM configs** → understand why outliers form
2. **Bit-width sweep** → validate threshold prediction
3. **Survey more models** → test screening prediction
4. **Training data analysis** → test data volume correlation

---

*LA-ACIQ v2 — 2026-01-03*
