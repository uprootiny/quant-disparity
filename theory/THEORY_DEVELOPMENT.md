# Theory Development: From Experiments to Proofs

*Elevating empirical findings to theoretical contributions*

---

## The Gap

| What We Have | What They Want |
|--------------|----------------|
| r = −0.998 correlation | Causal proof |
| 4.24× observed disparity | Derived bound |
| Gateway layers identified | Optimality proof |
| 217 experiments | 3 theorems |

---

## Theoretical Contributions We Can Make

### Theory 1: Information-Theoretic Disparity Bound

**Intuition:** Quantization is lossy compression. Languages with less redundancy lose more.

**Formalization:**

Let $H(X_\ell)$ be the entropy of language $\ell$'s representation in layer $L$.
Let $R_\ell$ be the redundancy: $R_\ell = H_{max} - H(X_\ell)$

**Claim:** Quantization error $\epsilon_\ell$ is bounded by:

$$\epsilon_\ell \leq \frac{C}{R_\ell + \delta}$$

where $C$ depends on bit-width and $\delta > 0$ prevents division by zero.

**Proof sketch:**
1. Quantization removes $\log_2(2^{32}/2^b)$ bits of precision
2. Redundant representations can absorb this loss
3. Low-redundancy representations cannot

**Testable prediction:** Plot $\epsilon$ vs $1/R$ — should be linear.

**Connection to our data:**
- HR languages: high $R$ → low $\epsilon$
- LR languages: low $R$ → high $\epsilon$
- The 4.24× ratio should be predictable from $R$ ratio

---

### Theory 2: Optimal Language-Aware Clipping (LA-ACIQ)

**Background:** ACIQ (Banner et al., 2019) derives optimal clipping threshold $\alpha^*$ for quantization based on weight distribution.

$$\alpha^* = \arg\min_\alpha \mathbb{E}[(Q_\alpha(w) - w)^2]$$

For Gaussian weights: $\alpha^* \approx 2.83\sigma$ (for INT4).

**Our Extension:**

Weights are activated differently by different languages. Define:

$$w_\ell^{eff} = w \odot a_\ell$$

where $a_\ell$ is the activation pattern for language $\ell$.

**Claim:** The optimal clipping threshold is language-dependent:

$$\alpha^*_\ell = f(\kappa(w_\ell^{eff}), \sigma(w_\ell^{eff}))$$

where $\kappa$ is kurtosis.

**Derivation:**
1. ACIQ shows $\alpha^* \propto \sigma \cdot g(\kappa)$ where $g$ is a correction for non-Gaussianity
2. Effective distribution $(w_\ell^{eff})$ has language-specific $\sigma_\ell, \kappa_\ell$
3. Therefore $\alpha^*_\ell \neq \alpha^*_{\ell'}$ for $\ell \neq \ell'$

**Theorem (LA-ACIQ):**
For a multilingual model with weight matrix $W$ and language activation patterns $\{a_\ell\}$:

$$\text{Disparity}(\ell_1, \ell_2) \propto \left| \frac{\kappa_{\ell_1}}{\kappa_{\ell_2}} - 1 \right|$$

**Proof:** See appendix (to be written).

**Validation:** Our empirical r = +0.838 between kurtosis ratio and disparity supports this.

---

### Theory 3: Gateway Layer Optimality

**Observation:** Protecting layers L0, L9, L11 reduces disparity by 41%.

**Question:** Is this optimal? Can we prove it?

**Formalization:**

Let $\mathcal{L} = \{0, 1, ..., L-1\}$ be the set of layers.
Let $S \subseteq \mathcal{L}$ be the set of protected layers.
Let $D(S)$ be the disparity when protecting $S$.
Let $E(S)$ be the efficiency (compression ratio) when protecting $S$.

**Optimization problem:**
$$\min_S D(S) \quad \text{s.t.} \quad E(S) \geq E_{min}$$

**Claim:** Under certain conditions, the optimal $S^*$ consists of:
1. The input layer (L0) — gateway
2. The layer with maximum cross-lingual variance — bottleneck
3. The output layer (L-1) — gateway

**Proof sketch:**
1. Information bottleneck theory: mid-layers compress
2. LR languages have less redundancy at compression point
3. Protecting the bottleneck preserves LR information
4. Input/output layers handle language-specific features

**Testable prediction:** Grid search over all $\binom{12}{3} = 220$ layer combinations should find L0+L9+L11 near-optimal.

---

### Theory 4: Tokenization-Disparity Causal Graph

**Formalization as Structural Causal Model (SCM):**

```
T (tokenization quality) → A (alignment score) → R (redundancy) → D (disparity)
```

**Variables:**
- $T$: Tokenization quality (fertility, alignment)
- $A$: Cross-lingual alignment in representation space
- $R$: Redundancy (number of pathways encoding same info)
- $D$: Disparity under quantization

**Structural equations:**
$$A = f_1(T) + \epsilon_A$$
$$R = f_2(A) + \epsilon_R$$
$$D = f_3(R) + \epsilon_D$$

**Causal claim:** $T \to D$ is mediated by $A$ and $R$.

**Testable via do-calculus:**
- $P(D | do(T))$ — intervening on tokenization
- $P(D | do(A))$ — intervening on alignment
- $P(D | do(R))$ — intervening on redundancy

**Our evidence:**
- B-011: Tokenization mediates 42% of effect
- Within-language: r = −0.998 (T → D direct)
- Cross-language: confounded (need do-calculus)

---

## Statistical Rigor Improvements

### Current State
| Aspect | Status |
|--------|--------|
| Sample size | 12 languages |
| Confidence intervals | Some |
| Power analysis | None |
| Multiple comparison correction | None |
| Pre-registration | Partial |

### Required Improvements

**1. Bootstrap Confidence Intervals for All Claims**

```python
def bootstrap_ci(statistic, data, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap confidence interval."""
    estimates = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        estimates.append(statistic(sample))
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    return lower, upper
```

**2. Power Analysis**

For our main finding (r = −0.924 with n = 12):
- Statistical power: ~0.85 for detecting |r| > 0.7
- Need n = 30 for power > 0.95

**3. Multiple Comparison Correction**

With 217 experiments, need Bonferroni or FDR correction:
- Bonferroni: α' = 0.05/217 = 0.00023
- FDR (Benjamini-Hochberg): less conservative

**4. Effect Size Reporting**

| Finding | Effect Size | Interpretation |
|---------|-------------|----------------|
| 4.24× disparity | Cohen's d = 2.1 | Very large |
| r = −0.924 | r² = 0.85 | 85% variance explained |
| 41% reduction | Cohen's d = 0.9 | Large |

---

## Proof Sketches to Develop

### Theorem 1: Disparity Lower Bound

**Statement:** For any quantization scheme $Q$ with bit-width $b$, there exists a constant $c_b$ such that:

$$\frac{\epsilon_{LR}}{\epsilon_{HR}} \geq c_b \cdot \frac{R_{HR}}{R_{LR}}$$

**Proof strategy:**
1. Use rate-distortion theory
2. Quantization is a rate-limited channel
3. Lower redundancy → higher distortion at fixed rate

### Theorem 2: Gateway Layer Necessity

**Statement:** Any disparity-minimizing protection scheme must include the first and last layers.

**Proof strategy:**
1. First layer: language-specific input processing
2. Last layer: language-specific output generation
3. If unprotected, language signal is corrupted at entry/exit

### Theorem 3: Alignment-Disparity Monotonicity

**Statement:** If $A_{\ell_1} > A_{\ell_2}$, then $D_{\ell_1} < D_{\ell_2}$ under any uniform quantization.

**Proof strategy:**
1. Higher alignment → closer to English distribution
2. English distribution is what the model optimizes for
3. Quantization preserves English-optimal regions

---

## Implementation Plan

### Phase 1: Formalization (This Week)

1. Write full proofs for Theorems 1-3
2. State assumptions explicitly
3. Identify gaps requiring GPU validation

### Phase 2: Validation (Next Week)

1. Run grid search for gateway layer optimality
2. Compute redundancy metrics empirically
3. Test predictions against held-out languages

### Phase 3: Extension (Following Week)

1. Generalize to other compression methods
2. Derive practical algorithm (LA-ACIQ)
3. Prove convergence/optimality of algorithm

---

## What This Gets Us

| Before | After |
|--------|-------|
| "We observed 4.24× disparity" | "We prove disparity ≥ 4× given redundancy ratio" |
| "L0+L9+L11 works" | "L0+L9+L11 is optimal under conditions X" |
| "Tokenization correlates" | "Tokenization causes via SCM with testable predictions" |
| "217 experiments" | "3 theorems + extensive empirical validation" |

**For Soudry:** LA-ACIQ extends Banner et al. with closed-form solution.
**For Goldberg:** Rigorous causal analysis with proper statistics.
**For Belinkov:** Mechanistic theory of circuit failure.

---

*This document outlines the theoretical development needed to meet top-tier standards.*
