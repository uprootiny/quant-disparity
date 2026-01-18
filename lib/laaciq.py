"""
LA-ACIQ: Language-Aware Analytical Clipping for Integer Quantization

Canonical implementations of all LA-ACIQ formulas.
Each function links to its formal specification in spec.json
and corresponding Lean theorem (if proved).

Usage:
    from lib.laaciq import clip, banner_approximation, effective_kurtosis
"""
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
from scipy import stats as sp_stats

# Constants from spec.json
BANNER_C4 = 2.5       # Base clipping ratio for 4-bit
BANNER_D4 = 0.3       # Kurtosis adjustment coefficient
KURTOSIS_LOWER_BOUND = -2  # Theoretical minimum (uniform distribution)
GAUSSIAN_KURTOSIS = 0      # Excess kurtosis of Gaussian
LAPLACE_KURTOSIS = 3       # Excess kurtosis of Laplace
RATE_DISTORTION_SLOPE = -math.log(2) / 2  # ≈ -0.347


@dataclass(frozen=True)
class MixtureComponent:
    """A component in a Gaussian mixture model."""
    weight: float    # w_i, must sum to 1 across components
    mean: float      # μ_i
    variance: float  # σ_i²
    kurtosis: float  # κ_i (excess kurtosis)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# =============================================================================
# CLIPPING (Matches Lean: Quantization/Basic.lean - 9 theorems PROVED)
# =============================================================================

def clip(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Symmetric clipping: clip(x, α) = max(-α, min(α, x))

    Lean theorem: clip_in_range (PROVED)
        -α ≤ clip(x, α) ≤ α

    Lean theorem: clip_idempotent (PROVED)
        clip(clip(x, α), α) = clip(x, α)
    """
    return np.clip(x, -alpha, alpha)


def clip_scalar(x: float, alpha: float) -> float:
    """Scalar version of clip."""
    return max(-alpha, min(alpha, x))


# =============================================================================
# STEP SIZE & QUANTIZATION
# =============================================================================

def step_size(alpha: float, bits: int) -> float:
    """
    Quantization step size for uniform symmetric quantization.

    Formula: Δ = 2α / (2^B - 1)

    spec.json: formulas.step_size
    """
    return 2 * alpha / (2**bits - 1)


def quantize(x: np.ndarray, alpha: float, bits: int) -> np.ndarray:
    """
    Uniform symmetric quantization.

    Q(x) = round(clip(x, α) / Δ) * Δ
    """
    delta = step_size(alpha, bits)
    clipped = clip(x, alpha)
    return np.round(clipped / delta) * delta


# =============================================================================
# OPTIMAL ALPHA (Banner approximation)
# =============================================================================

def banner_approximation(sigma: float, kappa: float, bits: int = 4) -> float:
    """
    Optimal clipping threshold from Banner et al. (2019).

    Formula: α* ≈ σ · (C + D·ln(1 + max(0, κ)))

    For 4-bit: C=2.5, D=0.3
    For 3-bit: C=2.0, D=0.3
    For 8-bit: C=4.0, D=0.3

    spec.json: formulas.banner_approximation
    Lean theorem: laaciq_formula (sorry - scaffolded)
    """
    base = {3: 2.0, 4: BANNER_C4, 8: 4.0}.get(bits, BANNER_C4)
    adjustment = BANNER_D4 * math.log(1 + max(0, kappa))
    return sigma * (base + adjustment)


# =============================================================================
# EFFECTIVE KURTOSIS (Mixture statistics)
# =============================================================================

def mixture_variance(components: List[MixtureComponent]) -> float:
    """
    Variance of a Gaussian mixture.

    σ²_eff = Σ wᵢσᵢ² + Σ wᵢδᵢ²

    where δᵢ = μᵢ - μ_mix (deviation from mixture mean)

    spec.json: formulas.mixture_variance
    """
    if not components:
        return 0.0

    # Mixture mean
    mu_mix = sum(c.weight * c.mean for c in components)

    # Within-component variance + between-component variance
    within = sum(c.weight * c.variance for c in components)
    between = sum(c.weight * (c.mean - mu_mix)**2 for c in components)

    return within + between


def effective_kurtosis(components: List[MixtureComponent]) -> float:
    """
    Excess kurtosis of a Gaussian mixture.

    κ_eff = [Σ wᵢ(κᵢ+3)σᵢ⁴ + 6Σ wᵢσᵢ²δᵢ² + Σ wᵢδᵢ⁴] / σ⁴_eff - 3

    This is the full mixture kurtosis formula.

    spec.json: formulas.effective_kurtosis
    Lean theorem: effective_kurtosis_formula (sorry - scaffolded)
    """
    if not components:
        return GAUSSIAN_KURTOSIS

    sigma_eff_sq = mixture_variance(components)
    if sigma_eff_sq <= 0:
        return GAUSSIAN_KURTOSIS

    mu_mix = sum(c.weight * c.mean for c in components)

    # Fourth moment components
    term1 = sum(c.weight * (c.kurtosis + 3) * c.variance**2 for c in components)
    term2 = 6 * sum(c.weight * c.variance * (c.mean - mu_mix)**2 for c in components)
    term3 = sum(c.weight * (c.mean - mu_mix)**4 for c in components)

    fourth_moment_ratio = (term1 + term2 + term3) / (sigma_eff_sq**2)

    return fourth_moment_ratio - 3


def kurtosis_from_weights(weights: np.ndarray) -> float:
    """
    Compute excess kurtosis from weight array using scipy.
    """
    return float(sp_stats.kurtosis(weights.flatten(), fisher=True))


# =============================================================================
# MSE DECOMPOSITION
# =============================================================================

def clipping_error(x: np.ndarray, alpha: float) -> float:
    """
    Clipping error component of MSE.

    E_clip = E[(|X| - α)² · 𝟙_{|X| > α}]

    Lean theorem: clippingError_convex (sorry - scaffolded)
    """
    abs_x = np.abs(x)
    mask = abs_x > alpha
    if not np.any(mask):
        return 0.0
    return float(np.mean((abs_x[mask] - alpha)**2) * np.mean(mask))


def quantization_noise(alpha: float, bits: int, prob_in_range: float = 1.0) -> float:
    """
    Quantization noise component of MSE.

    E_quant = Δ²/12 · P(|X| ≤ α)

    For uniform quantization, the rounding error is uniformly
    distributed on [-Δ/2, Δ/2], giving variance Δ²/12.

    Lean theorem: quantizationNoise_convex (sorry - scaffolded)
    """
    delta = step_size(alpha, bits)
    return (delta**2 / 12) * prob_in_range


def mse_decomposition(x: np.ndarray, alpha: float, bits: int) -> Tuple[float, float]:
    """
    Decompose MSE into clipping and quantization components.

    MSE = E_clip + E_quant

    spec.json: theorems.T1_mse_decomposition
    Lean theorem: mse_decomposition (sorry - scaffolded)

    Returns: (clipping_error, quantization_noise)
    """
    e_clip = clipping_error(x, alpha)
    prob_in = float(np.mean(np.abs(x) <= alpha))
    e_quant = quantization_noise(alpha, bits, prob_in)
    return e_clip, e_quant


def mse(x: np.ndarray, alpha: float, bits: int) -> float:
    """
    Total MSE for symmetric uniform quantization.

    Lean theorem: mse_convex (sorry - scaffolded)
    """
    e_clip, e_quant = mse_decomposition(x, alpha, bits)
    return e_clip + e_quant


# =============================================================================
# LA-ACIQ: Language-Aware Optimal Alpha
# =============================================================================

def laaciq_optimal_alpha(
    weights: np.ndarray,
    components: List[MixtureComponent],
    bits: int = 4
) -> float:
    """
    Compute LA-ACIQ optimal clipping threshold.

    This is the main contribution: use per-language effective kurtosis
    to compute language-specific optimal alpha.

    1. Compute σ_eff from mixture variance
    2. Compute κ_eff from mixture kurtosis
    3. Apply Banner formula with these statistics

    spec.json: theorems (combined)
    """
    sigma_eff = math.sqrt(mixture_variance(components))
    kappa_eff = effective_kurtosis(components)
    return banner_approximation(sigma_eff, kappa_eff, bits)


# =============================================================================
# DISPARITY BOUNDS
# =============================================================================

def disparity_bound(kurtosis_variance: float, bits: int, C: float = 0.01) -> float:
    """
    Theoretical upper bound on language disparity.

    Disparity ≤ C · √Var[κ_eff] · 2^{-B/2}

    spec.json: theorems.T5_disparity_bound
    Lean theorem: disparity_bound (sorry - scaffolded)
    """
    return C * math.sqrt(kurtosis_variance) * (2 ** (-bits / 2))


def rate_distortion_degradation(baseline: float, bits_from: int, bits_to: int) -> float:
    """
    Predict degradation from rate-distortion theory.

    D(B) ∝ 2^{-B/2}

    slope ≈ -ln(2)/2 ≈ -0.347

    spec.json: theorems.T6_rate_distortion (axiom - empirically validated)
    """
    # Degradation ratio between bit widths
    ratio = 2 ** ((bits_from - bits_to) / 2)
    return baseline * ratio
