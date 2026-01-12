# Track S Literature Review: Optimization Theory Foundations

*Prerequisites for deriving LA-ACIQ with proof of optimality*

---

## 1. Convex Optimization (Boyd & Vandenberghe, 2004)

### Key Concepts We Need

#### 1.1 Convex Functions

**Definition.** f: ℝⁿ → ℝ is convex if:
$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$
for all x, y and θ ∈ [0,1].

**Relevance:** Our MSE function MSE(α) must be shown convex in α for unique optimum.

#### 1.2 First-Order Conditions

**Theorem.** If f is convex and differentiable, then x* is a global minimizer iff:
$$\nabla f(x^*) = 0$$

**Relevance:** We need to solve ∂MSE/∂α = 0 for optimal α*.

#### 1.3 KKT Conditions

For constrained optimization:
$$\min f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \quad h_j(x) = 0$$

**KKT conditions:**
1. Stationarity: ∇f + Σ μᵢ∇gᵢ + Σ λⱼ∇hⱼ = 0
2. Primal feasibility: gᵢ(x) ≤ 0, hⱼ(x) = 0
3. Dual feasibility: μᵢ ≥ 0
4. Complementary slackness: μᵢgᵢ(x) = 0

**Relevance:** LA-ACIQ has constraint α > 0. KKT gives necessary conditions.

#### 1.4 Duality

**Lagrangian:**
$$L(x, μ, λ) = f(x) + \sum_i μ_i g_i(x) + \sum_j λ_j h_j(x)$$

**Dual function:**
$$g(μ, λ) = \inf_x L(x, μ, λ)$$

**Relevance:** Duality might provide alternative derivation of α*.

### Specific Results to Use

| Result | Reference | Our Use |
|--------|-----------|---------|
| Convexity of quadratic | B&V §3.1.4 | MSE is quadratic in Q(x) |
| Sum preserves convexity | B&V §3.2.1 | E[·] preserves convexity |
| Composition rules | B&V §3.2.4 | For clipping function |
| Minimization conditions | B&V §4.2 | Finding α* |

---

## 2. Probability Theory: Moments and Distributions

### Key Concepts We Need

#### 2.1 Moments

**Definition.** The k-th moment of X:
$$m_k = \mathbb{E}[X^k]$$

**Central moments:**
$$\mu_k = \mathbb{E}[(X - \mathbb{E}[X])^k]$$

**Standardized moments:**
- Variance: σ² = μ₂
- Skewness: γ₁ = μ₃/σ³
- Kurtosis: κ = μ₄/σ⁴ - 3 (excess kurtosis)

**Relevance:** ACIQ shows α* depends on κ.

#### 2.2 Moment Generating Functions

**Definition.**
$$M_X(t) = \mathbb{E}[e^{tX}]$$

**Property:** Moments from derivatives:
$$\mathbb{E}[X^k] = M_X^{(k)}(0)$$

**Relevance:** May help derive effective distribution properties.

#### 2.3 Mixture Distributions

**Definition.** If X is drawn from distribution Pᵢ with probability wᵢ:
$$P(X \leq x) = \sum_i w_i P_i(X \leq x)$$

**Moment properties:**
$$\mathbb{E}[X] = \sum_i w_i \mathbb{E}_i[X]$$
$$\text{Var}(X) = \sum_i w_i \text{Var}_i(X) + \sum_i w_i (\mathbb{E}_i[X] - \mathbb{E}[X])^2$$

**Relevance:** P_λ is a mixture across layers. We need κ of mixture.

#### 2.4 Kurtosis of Mixtures

**Theorem.** For mixture X = Σ wᵢXᵢ:

$$\kappa_X = \frac{\sum_i w_i \mu_{4,i} + \text{cross terms}}{\sigma_X^4} - 3$$

**Cross terms involve:**
- Within-component 4th moments
- Between-component variance
- Depends on how "separated" components are

**Relevance:** This is exactly what we need for κ_eff(λ).

### Distributions We Encounter

| Distribution | κ | PDF | Where |
|--------------|---|-----|-------|
| Gaussian | 0 | exp(-x²/2σ²)/√(2πσ²) | Baseline |
| Laplace | 3 | exp(-|x|/b)/2b | Some weights |
| Student's t | 6/(ν-4) for ν>4 | ... | Heavy tails |
| Mixture | Variable | Σ wᵢpᵢ(x) | Effective dist |

---

## 3. Rate-Distortion Theory (Cover & Thomas, Ch. 10)

### Key Concepts We Need

#### 3.1 Rate-Distortion Function

**Definition.** Minimum rate to achieve distortion D:
$$R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

**Inverse:** Minimum distortion at rate R:
$$D(R) = \min_{\text{codes with rate } R} \mathbb{E}[d(X, \hat{X})]$$

#### 3.2 Gaussian Rate-Distortion

**Theorem.** For X ~ N(0, σ²) with squared error distortion:
$$D(R) = \sigma^2 \cdot 2^{-2R}$$

**Interpretation:**
- Each bit halves the distortion (for Gaussian)
- At rate R bits, achieve distortion σ²/4^R

**Relevance:** Quantization to B bits ≈ rate R = B. Sets fundamental limit.

#### 3.3 Non-Gaussian Sources

**General result:**
$$D(R) \geq \sigma^2 \cdot 2^{-2R}$$

with equality only for Gaussian.

**Implication:** Heavy-tailed (high κ) sources have WORSE rate-distortion.

**Relevance:** Explains why high-κ layers are harder to quantize.

#### 3.4 Relationship to Quantization

**Scalar quantization:**
- Rate = B bits per sample
- Distortion = MSE of quantizer
- Uniform quantization is near-optimal for uniform sources
- For Gaussian, Lloyd-Max quantizer is optimal

**Our setting:**
- We use uniform quantization with clipping
- Clipping makes it work for non-uniform sources
- α* trades clipping error vs quantization noise

### The Connection We Need to Make

**Rate-distortion says:** At B bits, minimum distortion is D(B).

**Our question:** Given language λ with effective distribution P_λ:
1. What is D_λ(B)?
2. How does it depend on κ_eff(λ)?
3. What α*(λ) achieves D_λ(B)?

**Conjecture:** D_λ(B) increases with κ_eff(λ), explaining disparity.

---

## 4. ACIQ (Banner et al., 2019)

### The Paper's Contributions

#### 4.1 MSE Decomposition

**Theorem 1 (Banner).** For clipped uniform quantization:
$$\text{MSE}(\alpha) = \underbrace{\mathbb{E}[(|X| - \alpha)^2 \cdot \mathbf{1}_{|X| > \alpha}]}_{\text{clipping error } E_c(\alpha)} + \underbrace{\frac{\Delta^2}{12}}_{\text{quantization noise } E_q(\alpha)}$$

where Δ = 2α/(2^B - 1).

**Key insight:** These terms have opposite dependence on α:
- E_c(α) ↓ as α ↑ (less clipping)
- E_q(α) ↑ as α ↑ (larger step size)

#### 4.2 Optimal Clipping

**For Gaussian X ~ N(0, σ²):**

$$\frac{\partial E_c}{\partial \alpha} = -2\alpha \cdot P(|X| > \alpha) = -2\alpha \cdot 2(1 - \Phi(\alpha/\sigma))$$

$$\frac{\partial E_q}{\partial \alpha} = \frac{2\alpha}{3(2^B - 1)^2}$$

Setting total derivative to zero gives transcendental equation for α*/σ.

**Numerical result for INT4:** α*/σ ≈ 2.83 for Gaussian.

#### 4.3 Kurtosis Correction

**For non-Gaussian with kurtosis κ:**

$$\frac{\alpha^*}{\sigma} \approx c_B + d_B \cdot \ln(1 + \max(0, \kappa))$$

where c₄ ≈ 2.5, d₄ ≈ 0.3 for INT4.

**Intuition:** Higher kurtosis → more outliers → need wider clipping.

### What We Extend

| ACIQ | LA-ACIQ Extension |
|------|-------------------|
| Single distribution P | Per-language P_λ |
| Global κ | Effective κ_eff(λ) |
| Single α* | Per-language α*(λ) |
| Optimize MSE | Optimize disparity-weighted MSE |

---

## 5. Synthesis: The Derivation Path

### Step 1: Effective Distribution

Show that language λ experiences:
$$P_\lambda(x) = \sum_l \bar{a}_l(\lambda) \cdot P_l(x)$$

### Step 2: Effective Kurtosis

Derive:
$$\kappa_{\text{eff}}(\lambda) = f(\{\bar{a}_l(\lambda)\}, \{\kappa_l\}, \{\sigma_l\})$$

Show this is well-defined and computable.

### Step 3: LA-ACIQ Formula

Apply Banner's kurtosis correction with κ_eff:
$$\alpha^*(\lambda) = \sigma_{\text{eff}}(\lambda) \cdot g(\kappa_{\text{eff}}(\lambda), B)$$

### Step 4: Disparity Bound

Prove:
$$\max_\lambda \text{MSE}(\lambda) - \min_\lambda \text{MSE}(\lambda) \leq C \cdot \text{Var}_\lambda[\kappa_{\text{eff}}(\lambda)]^{1/2}$$

### Step 5: Optimality

Show that α*(λ) is:
1. Unique (convexity)
2. Global minimum (first-order conditions)
3. Achieves the bound (or close to it)

---

## 6. Open Mathematical Questions

### Q1: Convexity of MSE(α) for Mixtures

ACIQ shows convexity for single distribution. Does it hold for mixtures?

**Approach:** Show MSE is sum of convex functions (one per component).

### Q2: Closed-Form κ_eff

Can we get exact formula for kurtosis of mixture?

**Approach:** Use moment-generating functions or direct computation.

### Q3: Tightness of Disparity Bound

Is C in our bound tight? What's the best achievable C?

**Approach:** Construct worst-case distribution, compute bound.

### Q4: Computational Complexity

How expensive is computing α*(λ) for all languages?

**Approach:** Analyze calibration cost, propose approximations.

---

## References

1. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

2. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.

3. Banner, R., Nahshan, Y., & Soudry, D. (2019). Post-training 4-bit quantization of convolution networks for rapid-deployment. *NeurIPS*.

4. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.

---

*Track S Literature Review — 2026-01-11*
