# Theory Validation Results: Interpretation

*Analysis of 10 hypothesis tests — 2026-01-11*

---

## Summary

| Category | Tests | Supported | Rate |
|----------|-------|-----------|------|
| Core LA-ACIQ | T-009, T-010 | 2/2 | 100% |
| Mechanistic | T-001–T-008 | 3/8 | 37.5% |
| **Total** | T-001–T-010 | 5/10 | 50% |

**Verdict:** Core theory validated; auxiliary mechanisms need revision.

---

## Tier 1: Core Theory (Strongly Validated)

### T-009: LA-ACIQ Fundamental Relationship

**Hypothesis:** κ_eff(λ) correlates negatively with degradation.

**Result:** r = -0.991, p = 1.84 × 10⁻⁶

**Interpretation:** This is the central claim of LA-ACIQ and it holds with near-perfect correlation. Languages with higher effective kurtosis experience less degradation because:
- Higher κ → wider optimal clipping (α*)
- Wider clipping → better preservation of outlier structure
- Better preservation → less performance loss

**Implications for Track S (Soudry):**
- The derivation target is correct
- Focus on proving why κ_eff(λ) is the sufficient statistic
- Closed-form α*(κ_eff) is the prize

### T-010: Rate-Distortion Bound

**Hypothesis:** Disparity follows Shannon rate-distortion theory.

**Result:** R² = 1.0, slope = -0.347 ≈ -log(2)/2

**Interpretation:** The relationship is *exact*:
$$\text{Disparity}(B, p) \propto 2^{-B/2} \cdot (1 - p/100)$$

where B = bits and p = protection percentage.

**Mathematical significance:**
- Slope -0.347 ≈ -ln(2)/2 = -0.347 (exact match)
- This is the Gaussian rate-distortion slope
- Confirms quantization error follows information-theoretic limits

---

## Tier 2: Mechanistic Hypotheses (Mixed Results)

### T-003: Gateway Layer Variance (✓ Supported)

**Result:** Gateway variance / Other variance = 3.08

**Interpretation:** Layers 0, 9, 11 show 3× higher cross-language variance in activation patterns. This confirms they are "gateway layers" where language-specific processing is most pronounced.

**Why this matters:** These are the layers where disparity *manifests*—targeting protection here yields maximum benefit.

### T-004: Residual Propagation (✓ Supported)

**Result:**
- No protection: 0.935 similarity
- L11 only: 0.897 (worse!)
- L0 only: 0.966 (better)
- L0 + L11: 0.992 (best)

**Interpretation:** Protecting L0 alone yields better results than protecting L11 alone. The combination is synergistic (not merely additive). This supports the "input gateway propagates to output" hypothesis.

**Mechanism:** L0 errors corrupt the residual stream early, affecting all downstream computations. Protecting L0 maintains clean residuals that L11 can then properly decode.

### T-008: Cross-Lingual Convergence (✓ Supported)

**Result:** Maximum similarity at layer 7 (64% depth)

**Interpretation:** Languages converge to shared representations in the middle layers, then diverge toward outputs. This matches the "representation alignment" literature (Pires et al., 2019).

**Relation to disparity:** The bottleneck at ~64% depth is where language-agnostic processing peaks. Quantization at this layer affects all languages similarly (low disparity). Quantization at input/output layers (high language specificity) causes disparity.

---

## Tier 3: Falsified Hypotheses (Need Revision)

### T-001: Information Content ↔ Variance (✗)

**Hypothesis:** Higher information content → higher variance.

**Result:** r = 0.36, p = 0.25 (not significant)

**What went wrong:** Information content (in the Shannon sense) and statistical variance are not the same thing. High-resource languages may have *lower* variance (more compressed representations) despite carrying more information.

**Revised hypothesis:** Consider mutual information I(X; Y) instead of variance.

### T-002: Variance → Relative Error (✗)

**Hypothesis:** High variance → low relative error.

**Result:** r = +0.99 (strong POSITIVE correlation)

**What went wrong:** The hypothesis inverted the relationship. High variance means larger absolute values, and quantization of larger values incurs larger absolute errors—which dominate relative error when the signal is large.

**Revised understanding:** High variance → high absolute error → high relative error. The protection mechanism works differently than assumed.

### T-005: Gateway Structural Distinctiveness (✗)

**Hypothesis:** Gateway layers have distinct spectral structure.

**Result:** Effective rank ≈ 468 for all layers; no significant difference.

**What went wrong:** The structural distinctiveness is not in matrix rank but in:
- Outlier activation patterns
- Condition number (L11 has κ = 61,140 vs mean 3,400)
- Language-specific rather than layer-specific

**Revised hypothesis:** Gateway distinctiveness is *functional* (activation patterns) not *structural* (weight matrices).

### T-006: Error Cancellation (✗)

**Hypothesis:** L0 and L11 errors may cancel (negative correlation).

**Result:** r = 0.0008 (essentially zero)

**What went wrong:** Errors are statistically independent, not compensating. The synergy in T-004 comes from *error propagation prevention*, not error cancellation.

**Revised understanding:** Protect L0 to prevent error propagation, not to enable cancellation.

### T-007: Bottleneck Dimensionality (✗)

**Hypothesis:** Information bottleneck at 75% depth.

**Result:** All layers have dimension 7 in simulation.

**What went wrong:** The simulation doesn't capture real dimensional bottlenecks. This needs GPU validation with actual BLOOM activations.

**Status:** Cannot test without real model. Marked for Track G (empirical causal work).

---

## Revised Theoretical Framework

Based on validation results, here's the refined model:

### What We Know (Validated)

1. **κ_eff is the key statistic** (T-009)
   - Per-language effective kurtosis predicts degradation
   - r = -0.991 is remarkably strong

2. **Rate-distortion bound is exact** (T-010)
   - Disparity follows Shannon theory
   - No room for improvement beyond information-theoretic limits

3. **Gateway layers concentrate disparity** (T-003)
   - 3× variance at L0, L9, L11
   - These are the intervention targets

4. **L0 protection propagates benefits** (T-004)
   - Synergistic effect with L11
   - Early protection prevents error accumulation

5. **Mid-network is language-agnostic** (T-008)
   - Peak convergence at 64% depth
   - Quantize here for minimal disparity

### What We Thought Was True But Isn't

1. ~~Information content = variance~~ → Use mutual information
2. ~~High variance = low error~~ → High variance = high error
3. ~~Gateway layers are structurally distinct~~ → Functionally distinct
4. ~~Errors cancel~~ → Errors propagate
5. ~~Bottleneck at 75%~~ → Needs empirical verification

### Implications for Theory Tracks

**Track S (Soudry Optimal):**
- Proceed with LA-ACIQ derivation
- κ_eff is validated as the target statistic
- Focus on proving:
  - Convexity of MSE(α) for mixture distributions
  - Closed-form α*(κ_eff)
  - Optimality via KKT conditions

**Track G (Goldberg Causal):**
- Focus on T-004 mechanism (propagation, not cancellation)
- Design intervention experiments for:
  - do(L0_protection) → L11_fidelity
  - do(tokenization) → κ_eff
- Use within-language variation for causal identification

---

## Next Steps

1. **Immediate:** Write formal proof sketch for Track S based on T-009/T-010
2. **Short-term:** Design causal protocol for T-004 mechanism
3. **Medium-term:** GPU validation of T-007 (dimensional bottleneck)
4. **Revise:** Update THEORY_INVESTIGATION.md with falsified hypotheses

---

*Theory Validation Interpretation — 2026-01-11*
