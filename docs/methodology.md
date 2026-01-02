# Methodology: Banner et al. Framework

## Optimal Clipping Theory

From Banner et al. (2019) "Post-Training 4-bit Quantization of Convolution Networks for Rapid Deployment":

### Error Decomposition

Total quantization error = clipping error + quantization noise

```
E[error²] = E[(X - clip(X, α))²] + E[(Q(X) - X)²]
```

Where:
- `α` = clipping threshold
- `clip(X, α)` = max(-α, min(α, X))
- `Q(X)` = quantized value

### Optimal Threshold

For Gaussian weights N(0, σ²):

```
α* = argmin_α { 2σ²[(α/σ)Φ̄(α/σ) + φ(α/σ)] + (2α/(2^b-1))²/12 }
```

Approximate solution: **α*/σ ≈ 2.5** for 4-bit quantization.

### Our Extension

For non-Gaussian (heavy-tailed) distributions:

```
α*_adj = α*_gaussian × (1 + 0.1 × excess_kurtosis)
```

Rationale: Heavy tails have more outliers. Clipping them removes more information. Larger threshold preserves more semantic content.

## Kurtosis as Diagnostic

From Chmiel et al. (2025):

> "Instabilities emerge only at scale (>1T tokens)... Monitor kurtosis of activations per layer. High kurtosis → outlier-sensitive → quantization-sensitive."

We extend this to per-language analysis:
- Languages activating high-kurtosis neurons → more sensitive to quantization
- This explains the Marchisio observation mechanistically

## Implementation

```python
def optimal_clipping(sigma, bits=4):
    """Banner Sec. 3.1"""
    def total_error(alpha):
        clip = gaussian_clip_error(alpha, sigma)
        quant = (2*alpha / (2**bits - 1))**2 / 12
        return clip + quant

    return scipy.optimize.minimize_scalar(
        total_error,
        bounds=(0.5*sigma, 6*sigma)
    ).x
```

## References

1. Banner, R., Nahshan, Y., & Soudry, D. (2019). Post-training 4-bit quantization of convolution networks for rapid deployment.
2. Chmiel, B., et al. (2025). Accurate LoRA-Finetuning Quantization of LLMs via Information Retention.
