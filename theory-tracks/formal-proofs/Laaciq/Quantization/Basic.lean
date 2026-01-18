/-
  Laaciq.Quantization.Basic

  Basic definitions for uniform symmetric quantization with clipping.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Real.Basic

namespace Laaciq.Quantization

/-- Bit-width for quantization. Common values: 2, 3, 4, 8, 16 -/
abbrev BitWidth := ℕ

/-- Number of quantization levels for B bits -/
def numLevels (B : BitWidth) : ℕ := 2^B

/-- Step size for uniform quantization with range [-α, α] -/
noncomputable def stepSize (α : ℝ) (B : BitWidth) : ℝ :=
  2 * α / (numLevels B - 1)

/-- Clipping function: clip(x, -α, α) -/
noncomputable def clip (x α : ℝ) : ℝ :=
  max (-α) (min α x)

/-- Uniform rounding to nearest quantization level -/
noncomputable def uniformRound (x : ℝ) (Δ : ℝ) : ℝ :=
  Δ * ⌊x / Δ + 0.5⌋

/-- The quantization operator Q_α,B -/
noncomputable def quantize (x α : ℝ) (B : BitWidth) : ℝ :=
  let clipped := clip x α
  let Δ := stepSize α B
  uniformRound clipped Δ

/-- Quantization error -/
noncomputable def quantError (x α : ℝ) (B : BitWidth) : ℝ :=
  x - quantize x α B

-- Basic properties

theorem clip_le_alpha (x α : ℝ) (hα : 0 < α) : clip x α ≤ α := by
  unfold clip
  apply max_le <;> [linarith; exact min_le_left α x]

theorem neg_alpha_le_clip (x α : ℝ) (_hα : 0 < α) : -α ≤ clip x α := by
  unfold clip
  exact le_max_left (-α) (min α x)

theorem clip_in_range (x α : ℝ) (hα : 0 < α) : -α ≤ clip x α ∧ clip x α ≤ α :=
  ⟨neg_alpha_le_clip x α hα, clip_le_alpha x α hα⟩

/-- Clipping is idempotent for values already in range -/
theorem clip_of_in_range (x α : ℝ) (hx : -α ≤ x ∧ x ≤ α) : clip x α = x := by
  unfold clip
  rw [min_eq_right hx.2, max_eq_right hx.1]

/-- Clipping is idempotent: clip(clip(x, α), α) = clip(x, α) -/
theorem clip_idempotent (x α : ℝ) (hα : 0 < α) : clip (clip x α) α = clip x α := by
  apply clip_of_in_range
  exact clip_in_range x α hα

/-- Absolute value of clipped value is bounded by α -/
theorem clip_abs_le (x α : ℝ) (hα : 0 < α) : |clip x α| ≤ α := by
  rw [abs_le]
  constructor
  · have h := neg_alpha_le_clip x α hα
    linarith
  · exact clip_le_alpha x α hα

/-- Clip is monotone in x for fixed α -/
theorem clip_mono_x (x y α : ℝ) (hxy : x ≤ y) : clip x α ≤ clip y α := by
  unfold clip
  apply max_le_max_left
  apply min_le_min_left
  exact hxy

/-- Clip with larger bound contains result of clip with smaller bound -/
theorem clip_mono_alpha (x α β : ℝ) (hαβ : α ≤ β) (hα : 0 < α) :
    -β ≤ clip x α ∧ clip x α ≤ β := by
  have h := clip_in_range x α hα
  constructor
  · linarith [h.1]
  · linarith [h.2]

/-- Clipping preserves non-negativity when x ≥ 0 and α > 0 -/
theorem clip_nonneg (x α : ℝ) (hx : 0 ≤ x) (hα : 0 < α) : 0 ≤ clip x α := by
  unfold clip
  apply le_max_of_le_right
  apply le_min (le_of_lt hα) hx

end Laaciq.Quantization
