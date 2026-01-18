"""
LA-ACIQ Library

Canonical implementations for quantization disparity research.

Modules:
    laaciq: Core LA-ACIQ formulas (kurtosis, MSE, optimal alpha)
    metrics: Evaluation metrics (perplexity, fertility, layer stats)

Usage:
    from lib.laaciq import banner_approximation, effective_kurtosis, mse
    from lib.metrics import perplexity, fertility
"""

from lib.laaciq import (
    # Constants
    BANNER_C4,
    BANNER_D4,
    GAUSSIAN_KURTOSIS,
    LAPLACE_KURTOSIS,
    RATE_DISTORTION_SLOPE,

    # Data classes
    MixtureComponent,

    # Clipping (9 theorems PROVED in Lean)
    clip,
    clip_scalar,

    # Quantization
    step_size,
    quantize,

    # Optimal alpha
    banner_approximation,

    # Kurtosis
    mixture_variance,
    effective_kurtosis,
    kurtosis_from_weights,

    # MSE
    clipping_error,
    quantization_noise,
    mse_decomposition,
    mse,

    # LA-ACIQ
    laaciq_optimal_alpha,

    # Bounds
    disparity_bound,
    rate_distortion_degradation,
)

__all__ = [
    # Constants
    "BANNER_C4",
    "BANNER_D4",
    "GAUSSIAN_KURTOSIS",
    "LAPLACE_KURTOSIS",
    "RATE_DISTORTION_SLOPE",

    # Data classes
    "MixtureComponent",

    # Core functions
    "clip",
    "clip_scalar",
    "step_size",
    "quantize",
    "banner_approximation",
    "mixture_variance",
    "effective_kurtosis",
    "kurtosis_from_weights",
    "clipping_error",
    "quantization_noise",
    "mse_decomposition",
    "mse",
    "laaciq_optimal_alpha",
    "disparity_bound",
    "rate_distortion_degradation",
]
