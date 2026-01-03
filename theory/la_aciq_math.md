# LA-ACIQ: Mathematical Summary

## Core Definitions

**Weight kurtosis at layer l:**
```
κ_l = E[(W_l - μ_l)⁴] / σ_l⁴ - 3
```

**Activation fraction for language λ at layer l:**
```
ā_l(λ) = ||h_l(λ)|| / Σ_j ||h_j(λ)||
```

**Effective kurtosis:**
```
κ_eff(λ) = Σ_l ā_l(λ) · κ_l
```

---

## ACIQ Framework (Banner 2019)

**Quantization MSE:**
```
MSE(α) = E_clip(α) + E_quant(α)

E_clip(α) = E[(|X| - α)² · 𝟙_{|X|>α}]
E_quant(α) = (2α)² / (12 · (2^B - 1)²)
```

**Optimal clipping (Gaussian):**
```
α*/σ ≈ 2.5 + 0.3 · ln(1 + κ)    [4-bit]
```

---

## LA-ACIQ Extension

**Standard (language-blind):**
```
α* = σ · f(κ_global, B)
```

**Language-aware:**
```
α*(λ) = σ · f(κ_eff(λ), B)
```

**Suboptimality of single α:**
```
MSE_actual(λ) - MSE_optimal(λ) ∝ (κ_eff(λ) - κ_global)²
```

---

## Disparity Analysis

**Definition:**
```
Disparity = max_λ D(λ) - min_λ D(λ)
```

**Empirical finding:**
```
D(λ) ∝ -κ_eff(λ)     [r = -0.838]
```

**Disparity bound (conjecture):**
```
Disparity ≤ C · √Var_λ[κ_eff(λ)]
```

---

## Key Results

| Metric | Value | p-value |
|--------|-------|---------|
| r(outlier_frac, D) | -0.834 | 0.0002 |
| r(κ_eff, D) | -0.838 | <0.001 |
| Bootstrap CI | [-0.93, -0.65] | — |
| Permutation p | 0.0001 | — |

---

## Calibration Strategies

**A. Per-language calibration:**
```
For λ ∈ Λ:
    α*(λ) = calibrate(model, data_λ)

Overhead: O(|Λ|) forward passes
```

**B. Layer-wise mixed precision:**
```
bits(l) = 8 if κ_l > τ else 4

Overhead: None (compile-time)
```

**C. Adaptive (input-dependent):**
```
α*(x) = α*(detect_language(x))

Overhead: O(1) lookup + detection
```

---

## Predictions

1. **Per-language α reduces disparity:**
   ```
   Disparity(LA-ACIQ) < Disparity(ACIQ)
   ```

2. **Bit-width threshold correlates with κ_eff:**
   ```
   B_threshold(λ) ∝ log(κ_eff(λ))
   ```

3. **Training data volume → κ_eff:**
   ```
   More data → higher outlier activation → higher κ_eff
   ```
