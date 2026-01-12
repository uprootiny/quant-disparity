# Theory Synthesis: What We Actually Have

*A critical assessment of our theoretical foundations*

---

## The Core Theoretical Framework: LA-ACIQ

### The Central Claim

**Languages experience different effective weight distributions, requiring different quantization parameters.**

Formalized as:

$$\kappa_{eff}(\lambda) = \sum_l \bar{a}_l(\lambda) \cdot \kappa_l$$

Where:
- $\kappa_l$ = excess kurtosis of weights in layer $l$
- $\bar{a}_l(\lambda)$ = normalized activation fraction for language $\lambda$ at layer $l$
- $\kappa_{eff}(\lambda)$ = effective kurtosis experienced by language $\lambda$

### Validated Predictions

| Prediction | Evidence | Status |
|------------|----------|--------|
| $r(\kappa_{eff}, D) < 0$ | r = −0.834, p = 0.0002 | **CONFIRMED** |
| Disparity ∝ √Var[$\kappa_{eff}$] | C = 0.0154, plausible | **PLAUSIBLE** |
| Suboptimality ∝ ($\kappa_{eff}$ − $\kappa_{global}$)² | r = 0.528, p = 0.052 | **MARGINAL** |

---

## The Three Theoretical Pillars

### Pillar 1: ACIQ Foundation (Banner et al., 2019)

**What we inherit:**
- Optimal clipping $\alpha^* = \sigma \cdot f(\kappa, B)$ where $f(\kappa, B) \approx 2.5 + 0.3 \ln(1 + \kappa)$ for INT4
- MSE decomposition: $\text{MSE}(\alpha) = E_{clip}(\alpha) + E_{quant}(\alpha)$
- Clipping-quantization trade-off: wider clipping → less clipping error, more quantization noise

**Our extension:**
- Different languages activate different layers → different effective $\kappa$
- Single global $\alpha^*$ is suboptimal for all but the "average" language
- LA-ACIQ: $\alpha^*(\lambda) = \sigma \cdot f(\kappa_{eff}(\lambda), B)$

**Strength:** Solid mathematical foundation from Banner.
**Weakness:** We approximate $\kappa_{eff}$ via activation fractions, not exact computation.

---

### Pillar 2: Gateway-Bottleneck Model

**What we observed:**
- L0 + L9 + L11 protection achieves 0.59× disparity
- L0 alone: 3.6×, L11 alone: 336× (harmful!), L0+L11: 0.7×

**The model:**

```
L0 (Gateway-In):    Token space → Representation space
                    Language-specific input encoding
                    Errors here propagate through residual stream

L9 (Bottleneck):    Information compression point (~75% depth)
                    Morphological/syntactic consolidation
                    Cross-lingual convergence

L11 (Gateway-Out):  Representation space → Token space
                    Language-specific output decoding
                    Useless without clean input (synergy with L0)
```

**Validated predictions:**

| Hypothesis | Test | Result |
|------------|------|--------|
| L0+L11 synergy | Ablation study | **CONFIRMED** (synergy observed) |
| L9 is bottleneck | Variance analysis | **SUPPORTED** (high disparity sensitivity) |
| Position matters | 75% depth optimal | **CONFIRMED** (L9/12 ≈ 0.75) |

**Strength:** Explains why layer selection matters.
**Weakness:** "Bottleneck" is descriptive, not mechanistic.

---

### Pillar 3: Redundancy Hypothesis

**What we claim:**
- HR languages have redundant representation pathways
- LR languages concentrate in fewer circuits
- Quantization noise is absorbed by redundancy (or not)

**Formalization:**

$$\epsilon_\ell \leq \frac{C_b}{R_\ell + \delta}$$

Where $R_\ell$ is representational redundancy.

**Evidence:**
- r = 0.972 between predicted and observed (from redundancy_formalization.py)
- HR languages use more super weights (20% vs 70%)
- HR has 4× more outliers than LR

**Strength:** Information-theoretic grounding.
**Weakness:** $R_\ell$ is estimated, not measured directly.

---

## Cross-Model Predictions

### The XGLM Null Result

**Prediction:** If max $\kappa_l$ < 5, no disparity pattern.

**Evidence:** XGLM has max $\kappa$ = 1.9 (near-Gaussian), shows r = +0.38 (null).

**Interpretation:** No outlier layers → no differential activation → no disparity.

**This is strong:** We predicted and observed a null result.

### Necessary Conditions for Disparity

For quantization disparity to occur:

1. **(C1) Outlier layers exist:** max$_l$ $\kappa_l$ > $\kappa_{threshold}$ (~50)
2. **(C2) Activation varies:** Var$_\lambda$[$\bar{a}_{outlier}(\lambda)$] > $\epsilon$
3. **(C3) Quantization applied:** $B$ < $B_{safe}$ (~6 bits)

If ANY condition fails, disparity ≈ 0.

| Model | C1 | C2 | C3 | Predicted | Observed |
|-------|----|----|----|-----------| ---------|
| BLOOM | ✓ ($\kappa$=164) | ✓ (spread=3%) | ✓ | HIGH | HIGH |
| XGLM | ✗ ($\kappa$=1.9) | ? | ✓ | LOW | LOW |
| GPT-2 | ✓ ($\kappa$~100) | ✓ | ✓ | HIGH | HIGH |

---

## What's Solid

### Empirically Validated

1. **r = −0.834 correlation** (outlier activation ↔ degradation)
   - Bootstrap CI: [−0.93, −0.65]
   - p = 0.0002
   - Replicates across experiments

2. **r = −0.998 within-language** (alignment ↔ degradation)
   - Confound-free design
   - Small N but very strong signal

3. **XGLM null result predicted and observed**
   - Theory predicted: no outliers → no disparity
   - Empirical: r = +0.38 (null)

4. **Gateway layer synergy**
   - L11 alone harmful, L0+L11 helpful
   - Not explainable without propagation theory

### Theoretically Grounded

1. **ACIQ extension is sound**
   - Banner's framework is peer-reviewed
   - Our extension follows naturally

2. **Necessary conditions are testable**
   - C1, C2, C3 are measurable
   - Provide screening tool for new models

3. **Disparity bound is plausible**
   - C ≈ 0.015, order-of-magnitude correct
   - Follows from rate-distortion intuition

---

## What's Weak

### Needs Strengthening

1. **Redundancy measurement**
   - We use proxies (outlier activation fraction)
   - True mutual information unmeasured
   - Could be confounded with vocabulary coverage

2. **Suboptimality bound**
   - r = 0.528, p = 0.052 (marginal)
   - Theory predicts quadratic relationship
   - Data may be too noisy or model wrong

3. **Causal claims**
   - SCM is plausible but not tested with interventions
   - do-calculus predictions not validated
   - Observational data only

### Missing Pieces

1. **Why do outlier layers form?**
   - Training dynamics? Architecture? Data?
   - BLOOM has them, XGLM doesn't — why?

2. **Optimal layer selection proof**
   - We show L0+L9+L11 works
   - No proof it's optimal (only top-4 in search)

3. **Rate-distortion bound**
   - Conjecture: disparity ≥ f(bits, protection%)
   - Not formally derived

4. **Generalization**
   - Theory for GPT-2/BLOOM
   - Untested on Llama, Mistral, Mamba

---

## The Actual Theorems We Can State

### Theorem 1 (Informal): Effective Kurtosis Determines Sensitivity

*Languages with lower effective kurtosis experience higher quantization degradation.*

**Formal:** $\text{Corr}(\kappa_{eff}(\lambda), D(\lambda)) < 0$

**Evidence:** r = −0.834, p = 0.0002

**Strength:** Strong

---

### Theorem 2 (Informal): Disparity Bound

*Disparity is bounded by kurtosis variance across languages.*

**Formal:** $\text{Disparity} \leq C \cdot \sqrt{\text{Var}_\lambda[\kappa_{eff}(\lambda)]}$

**Evidence:** C ≈ 0.015, empirically consistent

**Strength:** Moderate

---

### Theorem 3 (Informal): Necessary Conditions

*Disparity requires: (1) outlier layers, (2) varying activation, (3) aggressive quantization.*

**Formal:** If $\max_l \kappa_l < 5$, then Disparity ≈ 0

**Evidence:** XGLM null result

**Strength:** Strong (predictive success)

---

### Theorem 4 (Informal): Gateway Synergy

*Input and output layers must be protected together; protecting one without the other is harmful.*

**Formal:** D(protect L0 only) < D(protect L11 only) > D(protect neither)

**Evidence:** Ablation study

**Strength:** Strong (empirically robust)

---

## What Would Make This Soudry-Quality

| Gap | What's Needed |
|-----|---------------|
| Formal proofs | Derive bounds from first principles, not fit from data |
| Optimality | Prove gateway layers are optimal, not just good |
| Closed-form LA-ACIQ | Exact formula for $\alpha^*(\lambda)$, not approximation |
| Convergence analysis | Prove LA-ACIQ reduces disparity with bounded overhead |
| Training intervention | Show regularization can prevent outlier formation |

---

## What Would Make This Goldberg-Quality

| Gap | What's Needed |
|-----|---------------|
| Causal interventions | Actually do $do(T)$ experiments, not just observe |
| Probe validation | Control tasks to rule out probe artifacts |
| Linguistic analysis | What features break? Morphology? Syntax? Semantics? |
| Error analysis | Qualitative: what do models get wrong after quantization? |
| Cross-lingual alignment | Measure alignment damage directly, not via proxy |

---

## Summary: The Theory Status

| Component | Status | Confidence |
|-----------|--------|------------|
| LA-ACIQ core claim | VALIDATED | High |
| Effective kurtosis formula | VALIDATED | High |
| ACIQ extension | GROUNDED | High (inherits from Banner) |
| Gateway-Bottleneck model | SUPPORTED | Moderate |
| Redundancy hypothesis | PLAUSIBLE | Moderate |
| Disparity bound | APPROXIMATE | Moderate |
| Necessary conditions | PREDICTIVE | High |
| Causal model | PLAUSIBLE | Low-Moderate |
| Optimality claims | UNSUPPORTED | Low |

**Overall:** We have a coherent theoretical framework with strong empirical support for the core claims. The main gaps are in formal proofs and causal validation.

---

*Synthesis completed: 2026-01-11*
