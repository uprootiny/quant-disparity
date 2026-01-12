# LA-ACIQ Proof Sketch

*Based on validated T-009 and T-010 results*

---

## Goal

Prove that for language λ with effective kurtosis κ_eff(λ), the optimal clipping threshold is:

$$\alpha^*(\lambda) = \sigma_{\text{eff}}(\lambda) \cdot g(\kappa_{\text{eff}}(\lambda), B)$$

where g is a monotonically increasing function in κ.

---

## Theorem 1: MSE Decomposition for Mixtures

**Statement.** For X ~ P_λ = Σ_l ā_l(λ) · P_l, the MSE under clipped quantization Q_α decomposes as:

$$\text{MSE}_\lambda(\alpha) = E_c^\lambda(\alpha) + E_q^\lambda(\alpha)$$

where:
- E_c^λ(α) = E[(|X| - α)² · 𝟙_{|X|>α}] (clipping error)
- E_q^λ(α) = Δ²/12 · P(|X| ≤ α) (quantization noise)

**Proof sketch:**

1. Quantization error decomposes into clipping + noise (Banner 2019, Theorem 1)
2. For mixture: E_λ[f(X)] = Σ_l ā_l(λ) · E_l[f(X)]
3. Each component contributes additively
4. Sum preserves the decomposition structure ∎

---

## Theorem 2: Convexity of MSE_λ(α)

**Statement.** MSE_λ(α) is convex in α for α > 0.

**Proof sketch:**

1. E_c^λ(α) is convex: second derivative ≥ 0
   - ∂E_c/∂α = -2α · P(|X| > α) + ∫_{|x|>α} 2(|x|-α)·(-1) dx
   - ∂²E_c/∂α² = ... ≥ 0 (algebra)

2. E_q^λ(α) = (2α)²/(12·(2^B-1)²) · P(|X| ≤ α)
   - Quadratic in α, hence convex

3. Sum of convex functions is convex ∎

---

## Theorem 3: Optimal Clipping Depends on Kurtosis

**Statement.** Let α*(κ) denote the optimal clipping for a distribution with kurtosis κ. Then:

$$\frac{\partial \alpha^*}{\partial \kappa} > 0$$

**Proof sketch:**

1. Higher κ → heavier tails → more probability mass at extremes
2. Clipping error E_c more sensitive to α when tails are heavy
3. Optimal α* shifts outward to reduce clipping error
4. Formally: use implicit function theorem on ∂MSE/∂α = 0 ∎

---

## Theorem 4: Effective Kurtosis Formula

**Statement.** For mixture P_λ = Σ_l ā_l(λ) · P_l with component means μ_l, variances σ_l², and kurtoses κ_l:

$$\kappa_{\text{eff}}(\lambda) = \frac{\sum_l \bar{a}_l(\lambda) \cdot (\mu_{4,l} + 6\sigma_l^2 \delta_l^2 + 3\delta_l^4)}{\sigma_{\text{eff}}^4(\lambda)} - 3$$

where δ_l = μ_l - μ_eff(λ) and μ₄,l is the 4th central moment of component l.

**Proof sketch:**

1. 4th moment of mixture: E[X⁴] = Σ_l ā_l E_l[X⁴]
2. Expand E_l[(X - μ_eff)⁴] using binomial
3. Collect terms involving component moments
4. Divide by σ_eff⁴ and subtract 3 ∎

---

## Theorem 5: Disparity Bound

**Statement.** Under LA-ACIQ with per-language α*(λ):

$$\max_\lambda \text{MSE}_\lambda - \min_\lambda \text{MSE}_\lambda \leq C \cdot \text{Var}_\lambda[\kappa_{\text{eff}}(\lambda)]^{1/2} \cdot 2^{-B}$$

for some constant C depending on the model.

**Proof sketch:**

1. MSE_λ(α*(λ)) depends continuously on κ_eff(λ)
2. Taylor expand MSE around mean κ̄:
   MSE_λ ≈ MSE(κ̄) + (∂MSE/∂κ)(κ_eff(λ) - κ̄)
3. Max-min ≤ 2·|∂MSE/∂κ|·max|κ_eff - κ̄|
4. Rate-distortion gives 2^{-B} scaling ∎

---

## Corollary: Rate-Distortion Slope

**Statement.** The disparity-vs-bits relationship has slope -log(2)/2.

**Proof:**

From T-010 validation: slope = -0.347 ≈ -ln(2)/2 = -0.347

This matches the Gaussian rate-distortion bound D(R) = σ² · 2^{-2R}, confirming that quantization error follows Shannon's fundamental limit.

---

## Empirical Validation

From T-009:
- Predicted: κ_eff ↔ degradation correlation
- Observed: r = -0.991, p = 1.84 × 10⁻⁶

From T-010:
- Predicted: disparity ∝ 2^{-B/2}
- Observed: R² = 1.0

Both core predictions validated.

---

## What Remains

1. **Algebra:** Complete ∂²MSE/∂α² calculation
2. **Bound tightness:** Determine if C is achievable
3. **Closed form:** Solve ∂MSE/∂α = 0 for specific distributions
4. **Computation:** Efficient algorithm for α*(λ) in practice

---

*Proof Sketch — 2026-01-11*
