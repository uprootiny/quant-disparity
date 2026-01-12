/-
  Laaciq.Probability.Kurtosis

  Definitions and properties of kurtosis (excess kurtosis).

  Kurtosis measures the "tailedness" of a distribution.
  - Gaussian has κ = 0
  - Heavy-tailed distributions have κ > 0
  - Light-tailed distributions have κ < 0 (but κ ≥ -2)
-/

import Mathlib.Probability.Variance
import Mathlib.MeasureTheory.Integral.Bochner

namespace Laaciq.Probability

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
variable (X : Ω → ℝ) (hX : Measurable X) (hX2 : Memℒp X 2 μ) (hX4 : Memℒp X 4 μ)

/-- The k-th raw moment: E[X^k] -/
noncomputable def rawMoment (k : ℕ) : ℝ :=
  ∫ ω, (X ω)^k ∂μ

/-- The k-th central moment: E[(X - μ)^k] -/
noncomputable def centralMoment (k : ℕ) : ℝ :=
  ∫ ω, (X ω - ∫ ω', X ω' ∂μ)^k ∂μ

/-- Mean: E[X] -/
noncomputable def mean : ℝ := rawMoment μ X 1

/-- Variance: E[(X - μ)²] -/
noncomputable def variance' : ℝ := centralMoment μ X 2

/-- Standard deviation: σ = √Var(X) -/
noncomputable def stdDev : ℝ := Real.sqrt (variance' μ X)

/-- Fourth central moment: E[(X - μ)⁴] -/
noncomputable def fourthCentralMoment : ℝ := centralMoment μ X 4

/-- Excess kurtosis: κ = E[(X - μ)⁴]/σ⁴ - 3

The -3 normalizes so that Gaussian has κ = 0.
-/
noncomputable def excessKurtosis : ℝ :=
  fourthCentralMoment μ X / (variance' μ X)^2 - 3

notation "κ[" X "]" => excessKurtosis _ X

/-!
## Properties of Kurtosis
-/

/-- Kurtosis is shift-invariant -/
theorem kurtosis_shift_invariant (c : ℝ) :
    excessKurtosis μ (fun ω => X ω + c) = excessKurtosis μ X := by
  sorry  -- Central moments are shift-invariant

/-- Kurtosis is scale-invariant -/
theorem kurtosis_scale_invariant (c : ℝ) (hc : c ≠ 0) :
    excessKurtosis μ (fun ω => c * X ω) = excessKurtosis μ X := by
  sorry  -- Both numerator and denominator scale by c^4

/-- Lower bound: κ ≥ -2 (always) -/
theorem kurtosis_lower_bound :
    excessKurtosis μ X ≥ -2 := by
  sorry  -- From Cauchy-Schwarz inequality

/-!
## Standard Distributions
-/

/-- Kurtosis of Gaussian is 0 -/
theorem kurtosis_gaussian (σ : ℝ) (hσ : 0 < σ) :
    ∀ μ_gauss : Measure ℝ, -- Gaussian measure with variance σ²
    excessKurtosis μ_gauss id = 0 := by
  sorry  -- E[Z⁴] = 3σ⁴ for Gaussian

/-- Kurtosis of Laplace is 3 -/
theorem kurtosis_laplace :
    ∀ μ_lap : Measure ℝ, -- Laplace measure
    excessKurtosis μ_lap id = 3 := by
  sorry  -- E[|Z|⁴] = 24b⁴, Var = 2b²

end Laaciq.Probability
