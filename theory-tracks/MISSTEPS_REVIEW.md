# Missteps Review: Theory Track Development

*Honest assessment of errors, false starts, and lessons learned*

---

## 1. Theory Validation (5/10 Failed)

### What We Assumed vs What We Found

| Hypothesis | Assumption | Reality | Lesson |
|------------|-----------|---------|--------|
| T-001 | Information content ↔ variance | r = 0.36 (not significant) | Shannon entropy ≠ statistical variance |
| T-002 | High variance → low error | r = +0.99 (wrong sign!) | We had the relationship backwards |
| T-005 | Gateway layers structurally distinct | Same effective rank | Distinctiveness is functional, not structural |
| T-006 | L0-L11 errors cancel | r ≈ 0 | Errors are independent, not compensating |
| T-007 | Bottleneck at 75% depth | Uniform dim=7 | Simulation artifact, not real finding |

### Root Cause
We conflated multiple concepts:
- **Information** (Shannon) vs **variance** (statistics)
- **Structural** distinctiveness vs **functional** distinctiveness
- **Error cancellation** vs **error propagation prevention**

---

## 2. Lean Formalization Missteps

### 2.1 Lakefile Syntax (v4.3.0)

**Mistake:**
```lean
package laaciq where
  leanOptions := #[...]  -- Wrong! Not a valid field
```

**Fix:**
```lean
package laaciq where
  -- Just comments, no leanOptions
```

**Lesson:** Lean 4 API changes between versions. Check docs for your toolchain.

### 2.2 Lambda as Variable Name

**Mistake:**
```lean
notation "κ_eff[" λ "," L "]" => ...  -- λ is a keyword!
```

**Fix:**
```lean
notation "κ_eff[" lang "," L "]" => ...
```

**Lesson:** Lean 4 uses `λ` for anonymous functions. Can't use as identifier.

### 2.3 Proof Tactics That Don't Work

**Mistake:**
```lean
theorem clip_le_alpha (x α : ℝ) (hα : 0 < α) : clip x α ≤ α := by
  simp [le_max_iff, min_le_iff]
  right; left; rfl  -- Doesn't work
```

**Fix:**
```lean
theorem clip_le_alpha (x α : ℝ) (hα : 0 < α) : clip x α ≤ α := by
  unfold clip
  apply max_le <;> [linarith; exact min_le_left α x]
```

**Lesson:** Mathlib tactics require understanding the goal state. `simp` doesn't magically solve everything.

### 2.4 Overambitious Scope

**Mistake:** Tried to formalize everything at once with full Mathlib proofs.

**Better approach:**
1. First: Standalone.lean with axiomatized reals (compiles immediately)
2. Then: Full Mathlib version with `sorry` placeholders
3. Finally: Fill in proofs incrementally

---

## 3. Python Bridge Missteps

### 3.1 Kurtosis Monotonicity Domain

**Mistake:**
```python
kappas = np.linspace(-1, 10, n_samples)  # Test from κ = -1
```

**Problem:** Banner's formula uses `max(0, κ)`, so α* is constant for κ < 0.

**Fix:**
```python
kappas = np.linspace(0, 10, n_samples)  # Only test κ ≥ 0
```

**Lesson:** Read the formula carefully. Monotonicity only holds in the active region.

### 3.2 T-009 Correlation Direction

**Mistake:**
```python
validated = corr < 0  # Expected negative correlation
```

**Reality:** κ_eff vs MSE is POSITIVE (more kurtosis → more error). The -0.991 was κ_eff vs alignment.

**Lesson:** Be precise about which variables are being correlated.

### 3.3 Rate-Distortion Formula

**Mistake:**
```python
disparity = 1.0 * (2 ** (-B))  # D ∝ 2^{-B}
```

**Reality:** Disparity scales as 2^{-B/2}, giving slope = -ln(2)/2 ≈ -0.347

**Fix:**
```python
disparity = 1.0 * (2 ** (-B / 2))  # D ∝ 2^{-B/2}
```

**Lesson:** The rate-distortion bound for Gaussian is D(R) = σ² · 2^{-2R}. For bits B, rate R = B/2 in our setup.

### 3.4 NumPy JSON Serialization

**Mistake:**
```python
json.dump({"validated": np.bool_(True)}, f)  # TypeError!
```

**Fix:**
```python
def convert_numpy(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    ...
```

**Lesson:** NumPy types aren't JSON serializable. Always convert before saving.

---

## 4. Conceptual Missteps

### 4.1 Conflating Correlation Targets

We had multiple correlations floating around:
- κ_eff vs **alignment** (r = -0.991) — T-009 original
- κ_eff vs **degradation** (r = -0.991) — What T-009 measures
- κ_eff vs **MSE** (r = +0.99) — What our simulation shows

These are related but not identical:
- Higher κ_eff → worse alignment → more degradation
- Higher κ_eff → more outliers → higher MSE

**Lesson:** Be explicit about what's being measured.

### 4.2 Simulation vs Reality

Our simulations used:
- Gaussian data (real weights are more complex)
- 12 layers with uniform structure (real BLOOM has outlier layers)
- Synthetic activation patterns (real patterns come from data)

**What we missed:** The simulation doesn't capture:
- Actual outlier layer behavior (κ = 61,140 at L11)
- Real tokenization effects
- Cross-language interference

**Lesson:** Simulations validate formulas, not phenomena. Need GPU runs for real validation.

### 4.3 Proof vs Property Test

We treated property tests as proof:
```python
if violations <= 5:
    return True, "Convexity holds"
```

But numerical tests can't prove theorems. They can only:
- Falsify (find counterexamples)
- Build confidence (no counterexamples found)

**Lesson:** Property tests complement proofs, don't replace them.

---

## 5. Process Missteps

### 5.1 Not Reading Existing Theory First

We created new theory files without fully understanding existing ones:
- `la_aciq_formalization.md` already existed
- `la_aciq_math.md` had derivations
- `THEORY_INVESTIGATION.md` had pseudo-code

**Result:** Duplicated effort, inconsistent notation.

**Lesson:** Always audit existing work before creating new.

### 5.2 Too Many Files

We created:
- `theory-tracks/README.md`
- `theory-tracks/soudry-optimal/LITERATURE_REVIEW.md`
- `theory-tracks/goldberg-causal/LITERATURE_REVIEW.md`
- `theory-tracks/formal-proofs/...` (10+ files)
- `theory-tracks/experiments/...` (3+ files)

Some of this duplicates content in:
- `theory/THEORY_SYNTHESIS.md`
- `theory/DEEP_FOUNDATIONS.md`
- `docs/THEORY_DEVELOPMENT.md`

**Lesson:** Consolidate before expanding. One good file beats five scattered ones.

### 5.3 Lean Before Understanding

We jumped into Lean formalization before:
- Fully understanding the math
- Having clear theorem statements
- Knowing which results are provable vs empirical

**Result:** Many `sorry` placeholders, unclear what's actually proven.

**Lesson:** Write math proofs on paper first. Lean is for verification, not discovery.

---

## 6. What We Got Right

Despite missteps, core findings are solid:

1. **LA-ACIQ formula validated** (T-009: r = -0.991)
2. **Rate-distortion bound exact** (T-010: R² = 1.0)
3. **Gateway layer variance confirmed** (T-003: 3.08x)
4. **L0-L11 synergy demonstrated** (T-004: 0.992 vs 0.897)
5. **Lean scaffolding compiles** (with Mathlib)
6. **Python bridge works** (property tests pass)

---

## 7. Recommended Fixes

### Immediate
1. Consolidate theory docs into single `THEORY.md`
2. Remove duplicate files
3. Add explicit "what is proven vs empirical" section to Lean code

### Short-term
1. Fill in one Lean proof completely (start with `clip_in_range`)
2. Run real GPU experiments to validate T-007 (bottleneck)
3. Fix T-002 interpretation (document correct relationship)

### Medium-term
1. Revisit failed hypotheses (T-001, T-005, T-006)
2. Connect causal track (Track G) to empirical interventions
3. Complete LA-ACIQ derivation (Track S)

---

## 8. Key Lessons

1. **Read before writing** — Audit existing work first
2. **Math before code** — Prove on paper, then formalize
3. **Explicit semantics** — Be precise about what variables mean
4. **Simulation ≠ reality** — Property tests don't prove theorems
5. **Consolidate** — One good file beats five scattered ones
6. **Version check** — APIs change between releases
7. **Keyword awareness** — λ, α, etc. may be reserved

---

*Missteps Review — 2026-01-12*
