# Deep Foundations: Tracing the Theory Several Layers Down

*Understanding what our theory rests upon*

---

## Layer 0: Our Claims

### LA-ACIQ
```
Claim: Languages have different effective kurtosis κ_eff(λ)
       Therefore optimal clipping α*(λ) differs by language
       Using single α* creates disparity
```

### Gateway-Bottleneck Model
```
Claim: Layers L0, L9, L11 are critical for LR languages
       Protecting them reduces disparity by 41%
       L0 = input gateway, L9 = bottleneck, L11 = output gateway
```

### Redundancy Hypothesis
```
Claim: HR languages have redundant representations
       Quantization noise is absorbed by redundancy
       LR languages lack redundancy → higher degradation
```

---

## Layer 1: Immediate Foundations

### 1.1 ACIQ (Banner et al., 2019)

**What it provides:**
- Optimal clipping formula: α* = σ · f(κ, B)
- MSE decomposition: MSE = E_clip + E_quant
- Proof that kurtosis determines optimal threshold

**The key equation:**

For quantization with clipping threshold α:
$$\text{MSE}(\alpha) = \underbrace{\mathbb{E}[(|X| - \alpha)^2 \cdot \mathbf{1}_{|X| > \alpha}]}_{\text{clipping error}} + \underbrace{\frac{\Delta^2}{12}}_{\text{quantization noise}}$$

where Δ = 2α/(2^B - 1).

**The key insight:**
- Clipping error ↓ as α ↑ (less clipping)
- Quantization noise ↑ as α ↑ (larger step size)
- Optimal α* balances these

**For Gaussian:** α*/σ ≈ 2.5 (INT4)
**For heavy-tailed:** α*/σ increases with kurtosis

**What we inherit:**
- The framework for analyzing quantization error
- The dependence on distribution shape (kurtosis)
- The methodology for deriving optimal parameters

**What we add:**
- Different languages induce different effective distributions
- Therefore different languages need different α*

---

### 1.2 Information Bottleneck (Tishby et al., 1999)

**What it provides:**
- Framework for understanding compression in neural networks
- Trade-off between compression and prediction

**The key equation:**

$$\min_{p(t|x)} I(X; T) - \beta I(T; Y)$$

where:
- X = input
- T = representation (compressed)
- Y = target
- β = trade-off parameter

**The key insight:**
- Mid-layers compress representations
- Compression removes irrelevant information
- Over-compression loses relevant information

**What we inherit:**
- The bottleneck concept for layer L9
- The idea that compression has a cost

**What we add:**
- LR languages have less redundancy to compress
- The "bottleneck" layer is where disparity concentrates

---

### 1.3 Probing Classifiers (Belinkov & Glass, 2019)

**What it provides:**
- Methodology for measuring what representations encode
- Layer-wise analysis of linguistic features

**The key insight:**
- Train classifier on hidden states → if it succeeds, model encodes that feature
- Different layers encode different features (surface → syntax → semantics)

**What we inherit:**
- The probing methodology for Track B
- The layer hierarchy interpretation

**What we add:**
- Quantization damages probing accuracy differentially
- LR languages lose more encoded information

---

## Layer 2: Deeper Foundations

### 2.1 Rate-Distortion Theory (Shannon, 1959)

**What it provides:**
- Fundamental limits on lossy compression
- Trade-off between rate (bits) and distortion (error)

**The key equation:**

For Gaussian source with variance σ²:
$$D(R) = \sigma^2 \cdot 2^{-2R}$$

where:
- R = rate in bits
- D = distortion (MSE)

**The key insight:**
- At rate R, minimum achievable distortion is D(R)
- Halving R roughly doubles D (for Gaussian)
- Non-Gaussian sources have different D(R) curves

**What we inherit:**
- The fundamental limit on quantization error
- The exponential relationship between bits and error

**What we add:**
- Languages have different "source" distributions
- Same rate (bit-width) → different distortion per language

**Critical connection:**
Our redundancy bound: ε_ℓ ≤ C_b / (R_ℓ + δ)

This is a rate-distortion statement: less redundancy (lower R) → higher distortion (ε).

---

### 2.2 Optimal Transport Theory

**What it provides:**
- Framework for measuring distance between distributions
- Wasserstein distance for comparing representations

**The key equation:**

$$W_p(P, Q) = \left( \inf_{\gamma \in \Gamma(P,Q)} \int \|x - y\|^p d\gamma(x,y) \right)^{1/p}$$

**What we inherit:**
- Framework for measuring representation damage
- Cross-lingual alignment as transport cost

**What we add:**
- Quantization increases transport distance
- LR languages have higher transport cost increase

---

### 2.3 Heavy-Tailed Statistics

**What it provides:**
- Theory of distributions with extreme values
- Kurtosis as measure of tail heaviness

**The key equation:**

$$\kappa = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} - 3$$

For Gaussian: κ = 0
For heavy-tailed: κ > 0 (potentially >> 0)

**The key insight:**
- Heavy tails contain rare but important values
- Standard clipping destroys these outliers
- Higher κ → need wider clipping range

**What we inherit:**
- Kurtosis as the key statistic for quantization
- The Banner insight that κ determines α*

**What we add:**
- Effective kurtosis: κ_eff(λ) = Σ_l ā_l(λ) · κ_l
- Languages experience different effective κ

---

### 2.4 Implicit Bias in Gradient Descent (Soudry et al., 2018)

**What it provides:**
- Theory of what GD converges to
- Max-margin solution emerges implicitly

**The key insight:**
- For separable data, GD finds max-margin classifier
- This happens without explicit regularization
- Training dynamics shape model structure

**What we inherit:**
- Understanding that training shapes weight structure
- Outlier layers emerge from training dynamics

**What we add:**
- Outlier formation may be language-biased
- HR languages may develop more robust (redundant) representations

---

## Layer 3: Foundational Mathematics

### 3.1 Information Theory (Shannon, 1948)

**Foundational concepts:**
- Entropy: H(X) = -Σ p(x) log p(x)
- Mutual Information: I(X; Y) = H(X) - H(X|Y)
- Channel Capacity: C = max_{p(x)} I(X; Y)

**What we inherit:**
- Information as the currency of representations
- Compression as information reduction
- Redundancy as excess information

**Connection to our work:**
- Redundancy R_ℓ is mutual information between pathways
- Quantization is a noisy channel with capacity ~ B bits
- Disparity arises from different source entropies

---

### 3.2 Statistical Learning Theory

**Foundational concepts:**
- Bias-variance trade-off
- Generalization bounds
- VC dimension

**The key equation (generalization bound):**

$$\mathbb{E}[L_{test}] \leq L_{train} + \sqrt{\frac{d \log(n/d) + \log(1/\delta)}{n}}$$

**What we inherit:**
- Framework for understanding model capacity
- Trade-offs in learning

**Connection to our work:**
- LR languages have fewer training examples (lower n)
- Models may have lower "effective capacity" for LR
- This manifests as less redundancy

---

### 3.3 Convex Optimization

**Foundational concepts:**
- Convexity and global optima
- Lagrangian duality
- KKT conditions

**Connection to our work:**
- ACIQ's α* derivation uses convex optimization
- The MSE function is convex in α
- Optimal α* satisfies KKT conditions

**LA-ACIQ extension:**
- Per-language optimization is still convex
- Each α*(λ) can be found independently
- Joint optimization over languages is more complex

---

### 3.4 Measure Theory and Probability

**Foundational concepts:**
- Probability spaces (Ω, F, P)
- Random variables as measurable functions
- Expectations and moments

**Connection to our work:**
- Weight distributions are probability measures
- Kurtosis is the 4th central moment
- Quantization error is an expectation

---

## Layer 4: Mathematical Bedrock

### 4.1 Real Analysis
- Continuity, limits, convergence
- Taylor expansions (for MSE approximations)
- Integral calculus (for expectations)

### 4.2 Linear Algebra
- Matrix operations (for weight analysis)
- Eigendecomposition (for SVD-based analysis)
- Norms (for measuring error)

### 4.3 Topology
- Metric spaces (for representation distances)
- Compactness (for existence of optima)

---

## The Full Stack

```
Mathematical Bedrock
├── Real Analysis
├── Linear Algebra
├── Measure Theory
└── Topology
        ↓
Foundational Mathematics
├── Information Theory (Shannon 1948)
├── Statistical Learning Theory
├── Convex Optimization
└── Probability Theory
        ↓
Deeper Foundations
├── Rate-Distortion Theory (Shannon 1959)
├── Optimal Transport
├── Heavy-Tailed Statistics
└── Implicit Bias (Soudry 2018)
        ↓
Immediate Foundations
├── ACIQ (Banner 2019)
├── Information Bottleneck (Tishby 1999)
└── Probing Classifiers (Belinkov 2019)
        ↓
Our Contributions
├── LA-ACIQ (Language-Aware Clipping)
├── Gateway-Bottleneck Model
├── Redundancy Hypothesis
└── Fair-Efficiency Score
```

---

## What's Rigorous vs What's Heuristic

### Rigorous (Inherited)
| Foundation | Rigor | Our Use |
|------------|-------|---------|
| Rate-distortion theory | Proven | Disparity bound intuition |
| ACIQ optimal clipping | Proven | α* formula |
| Shannon entropy | Axiomatic | Redundancy concept |
| Convex optimization | Proven | Existence of optimal α* |

### Semi-Rigorous (Our Extensions)
| Claim | Status | Gap |
|-------|--------|-----|
| κ_eff(λ) formula | Well-defined | Approximation via activation |
| Disparity ∝ 1/redundancy | Plausible | R_ℓ not directly measured |
| Gateway necessity | Empirical | No proof of optimality |
| Causal SCM | Proposed | No interventional validation |

### Heuristic (Needs Formalization)
| Claim | Status | Needed |
|-------|--------|--------|
| L9 is "bottleneck" | Descriptive | Information-theoretic proof |
| Redundancy = robustness | Intuitive | Formal definition of redundancy |
| Training creates outliers | Observed | Dynamical systems analysis |

---

## Key Assumptions (Often Implicit)

### Assumption 1: Independence of Layers
We treat layers as independent for MSE analysis.
**Reality:** Layers interact through residual stream.
**Impact:** May underestimate correlated errors.

### Assumption 2: Gaussian Approximation
ACIQ assumes weights are approximately Gaussian.
**Reality:** BLOOM has κ = 164 in some layers (very non-Gaussian).
**Impact:** Our κ_eff correction addresses this.

### Assumption 3: Static Activation Patterns
We assume activation patterns are fixed per language.
**Reality:** Activation depends on specific input.
**Impact:** κ_eff is an average, not per-input.

### Assumption 4: Uniform Quantization
We analyze uniform symmetric quantization.
**Reality:** Some methods use non-uniform or asymmetric.
**Impact:** Theory may not transfer to all methods.

### Assumption 5: Tokenization is Exogenous
We treat tokenization quality as given.
**Reality:** Tokenization could be changed.
**Impact:** Root cause intervention possible.

---

## Open Theoretical Questions

### Q1: Why Do Outlier Layers Form?
- BLOOM has κ = 164, XGLM has κ = 1.9
- Different training? Architecture? Data?
- Need: Dynamical systems analysis of training

### Q2: Is Gateway-Bottleneck Universal?
- We found L0, L9, L11 for GPT-2 class
- Does this generalize to Llama, Mamba?
- Need: Cross-architecture analysis

### Q3: What is the Optimal Disparity-Efficiency Trade-off?
- We defined FES = √(efficiency × fairness)
- Is this the right objective?
- Need: Axiomatic justification

### Q4: Can Training Prevent Disparity?
- If outliers cause disparity, prevent outliers?
- Smooth-SwiGLU prevents outliers (Chmiel 2025)
- Need: Multilingual training intervention

---

## Summary: Where We Stand

| Layer | Solidity |
|-------|----------|
| Mathematical bedrock | Rock solid |
| Shannon/rate-distortion | Proven, foundational |
| ACIQ framework | Published, peer-reviewed |
| Our κ_eff extension | Sound, empirically validated |
| Gateway-Bottleneck | Supported, needs formalization |
| Redundancy hypothesis | Plausible, needs measurement |
| Causal claims | Weakest link, needs intervention |

**Bottom line:** We have a strong theoretical lineage. Our immediate foundations (ACIQ) are peer-reviewed and solid. Our extensions are sound but need more formal development. The deepest foundations (information theory, rate-distortion) are unassailable.

---

*Deep foundations traced: 2026-01-11*
