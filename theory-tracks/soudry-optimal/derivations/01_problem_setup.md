# LA-ACIQ Derivation: Problem Setup

## 1. Notation

| Symbol | Definition | Domain |
|--------|------------|--------|
| W | Weight matrix | ℝ^{d×d} |
| B | Bit-width | {2, 3, 4, 8, 16} |
| α | Clipping threshold | ℝ₊ |
| Q_α | Quantization operator | ℝ → Q_B |
| κ | Excess kurtosis | ℝ (≥ -2) |
| λ | Language index | Λ = {1, ..., L} |
| a_l(λ) | Activation at layer l for language λ | [0, 1] |

## 2. The Quantization Operator

**Definition.** Uniform symmetric quantization with clipping:

$$Q_\alpha(x) = \alpha \cdot \text{round}\left( \frac{\text{clip}(x, -\alpha, \alpha)}{\alpha} \cdot (2^{B-1} - 1) \right) \cdot \frac{1}{2^{B-1} - 1}$$

**Properties:**
- Range: [-α, α] mapped to 2^B discrete levels
- Step size: Δ = 2α / (2^B - 1)
- Symmetric around 0

## 3. The Objective

**Goal:** Find α* minimizing expected quantization error:

$$\alpha^* = \arg\min_{\alpha > 0} \mathbb{E}_{X \sim P}[(X - Q_\alpha(X))^2]$$

**LA-ACIQ Extension:** Find α*(λ) for each language:

$$\alpha^*(\lambda) = \arg\min_{\alpha > 0} \mathbb{E}_{X \sim P_\lambda}[(X - Q_\alpha(X))^2]$$

where P_λ is the effective weight distribution for language λ.

## 4. Key Assumption

**Effective Distribution.** Language λ experiences an effective distribution:

$$P_\lambda(x) = \sum_l \bar{a}_l(\lambda) \cdot P_l(x)$$

where:
- P_l(x) is the weight distribution at layer l
- ā_l(λ) is the normalized activation for language λ at layer l

This is a mixture distribution weighted by activation patterns.

## 5. What We Need to Derive

1. **Closed-form α*(λ)** — Express optimal clipping as function of distribution parameters
2. **Dependence on κ_eff(λ)** — Show that effective kurtosis is the key statistic
3. **Disparity bound** — Upper bound on max_λ MSE(λ) - min_λ MSE(λ)
4. **Optimality conditions** — KKT conditions for the LA-ACIQ problem

## 6. Starting Point: Banner's Result

**Theorem (Banner 2019).** For X ~ N(0, σ²), the optimal clipping satisfies:

$$\frac{\alpha^*}{\sigma} \approx 2.5 + 0.3 \ln(1 + \max(0, \kappa))$$

for INT4 quantization.

**Our task:** Extend this to:
1. Mixture distributions (across layers)
2. Language-dependent mixtures
3. Prove optimality, not just approximate
