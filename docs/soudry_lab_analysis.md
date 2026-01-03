# Soudry Lab: Methodology & Technology Transfer Analysis

## Key Papers

### 1. Banner et al. (2019) — ACIQ
**"Post-training 4-bit quantization of convolution networks for rapid-deployment"**

[arXiv](https://arxiv.org/abs/1810.05723) | [NeurIPS 2019](https://proceedings.neurips.cc/paper/2019)

**Core Contribution:**
First practical post-training 4-bit quantization WITHOUT fine-tuning.

**Methodology — ACIQ (Analytical Clipping for Integer Quantization):**

```
Problem: Quantization introduces two types of error
  1. Clipping error: Values outside [-α, α] are lost
  2. Quantization noise: Rounding error within range

Insight: There's an optimal α that minimizes total MSE

For Gaussian distribution:
  α* = argmin_α { E[(|X|-α)²·1_{|X|>α}] + Δ²/12 }

Closed-form solution:
  α*/σ ≈ 2.5 (for 4-bit)
  α*/σ ≈ 3.5 (for 8-bit)
```

**Key Results:**
- 40% accuracy improvement over naive quantization (VGG16-BN, 4-bit)
- 4000x faster than KL-divergence calibration
- Enables deployment without access to full training data

**Limitations Noted:**
- Assumes Gaussian/Laplace distributions
- Hardware deployment of channel-wise quantization is challenging
- Performance degrades at very low bit-widths (2-3 bit)

---

### 2. Chmiel et al. (2025) — FP8 Training at Scale
**"Scaling FP8 training to trillion-token LLMs"**

[arXiv](https://arxiv.org/abs/2409.12517) | [ICLR 2025 Spotlight](https://openreview.net/forum?id=E1EHO0imOb)

**Core Contribution:**
First FP8 training at 2 trillion tokens (20x previous limit).

**Methodology — Outlier Amplification Analysis:**

```
Discovery: SwiGLU activation causes outlier amplification over long training

Mechanism:
  1. SwiGLU: f(x) = x · σ(x) where σ is sigmoid
  2. During training, weights align in a way that amplifies outliers
  3. This only manifests after ~200B tokens
  4. FP8's limited dynamic range can't handle these outliers

Solution: Smooth-SwiGLU
  - Modification that prevents weight alignment pathology
  - Maintains same function behavior
  - Enables stable FP8 training
```

**Key Results:**
- 7B model trained on 256 Intel Gaudi2 accelerators
- Matches BF16 baseline accuracy
- 34% throughput improvement
- First FP8 quantization of Adam optimizer moments

---

## Technology Transfer Pipeline

```
Academic Research → Industry Collaboration → Production Deployment

Banner et al. (2019)                Chmiel et al. (2025)
     │                                    │
     ▼                                    ▼
Intel Neural Compressor          Intel Habana Gaudi2
(INT4/INT8 quantization)         (FP8 training support)
     │                                    │
     ▼                                    ▼
Edge deployment                  Cloud LLM training
(mobile, IoT)                    (enterprise scale)
```

### Confirmed Industry Connections

| Partner | Relationship | Output |
|---------|-------------|--------|
| Intel | 5 research grants | Neural Compressor integration |
| Habana Labs (Intel) | Co-authored papers | Gaudi2 FP8 support |
| ERC | A-B-C-Deep grant | Theoretical foundations |

### Lab Members in Industry

- **Brian Chmiel**: PhD student → AI Research Scientist at Intel-Habana Labs
- **Ron Banner**: Co-author on both papers, Intel affiliation

---

## Methodology Deep Dive: What We Can Learn

### From ACIQ (Banner 2019)

**Their approach:**
1. Assume a distribution (Gaussian/Laplace)
2. Derive optimal clipping analytically
3. Validate on standard benchmarks (ImageNet)

**Our extension:**
1. We DON'T assume uniform distribution across languages
2. We show different languages have different effective distributions
3. This means a single α is suboptimal for multilingual models

```
Their finding:   α* depends on distribution shape (kurtosis)
Our finding:     Languages have different effective kurtosis
Our prediction:  Language-specific α* would reduce disparity
```

### From FP8 Training (Chmiel 2025)

**Their approach:**
1. Train at scale, observe failures
2. Trace to root cause (SwiGLU outliers)
3. Propose minimal intervention (Smooth-SwiGLU)
4. Validate fix at scale

**Relevance to us:**
- They found outliers cause quantization problems
- We found outlier LAYERS correlate with degradation
- The mechanism may be related

```
Their finding:   Training creates outlier weights in SwiGLU
Our finding:     BLOOM has outlier weights in specific layers
Question:        Is BLOOM's layer pattern related to SwiGLU?
                 (BLOOM uses GeLU, not SwiGLU, but similar dynamics?)
```

---

## Gap Analysis: Our Work vs. Theirs

| Aspect | Soudry Lab | Our Work |
|--------|-----------|----------|
| Scale | 7B+ models, trillions of tokens | 560M model, thousands of samples |
| Focus | Training efficiency | Inference fairness |
| Distribution | Assumed uniform | Language-dependent |
| Intervention | Smooth-SwiGLU | None yet (diagnostic only) |
| Hardware | Intel Gaudi2 | CPU only |
| Validation | Production-scale | Statistical correlation |

---

## What Would Make Our Work Complementary

### Option 1: Extend ACIQ to Multilingual

**Proposal:** Language-Aware ACIQ (LA-ACIQ)

```python
# Standard ACIQ: single α for all
α = optimal_clip(weights, bits=4)

# LA-ACIQ: per-language α based on activation pattern
def language_aware_clip(weights, lang, bits=4):
    effective_kurtosis = compute_effective_kurtosis(weights, lang)
    α = optimal_clip_for_kurtosis(effective_kurtosis, bits)
    return α
```

**Value:** Direct extension of their framework to multilingual setting.

### Option 2: Training Intervention

**Proposal:** Prevent outlier layer formation during multilingual pretraining

Using their Smooth-SwiGLU insight:
- Identify which BLOOM architectural choices cause outlier layers
- Propose training modifications
- Validate on smaller model, scale up

### Option 3: Diagnostic Tool

**Proposal:** Pre-deployment language risk assessment

```python
def assess_quantization_risk(model, languages):
    """Predict which languages will degrade under quantization."""
    for lang in languages:
        outlier_activation = compute_outlier_activation(model, lang)
        predicted_degradation = banner_framework(outlier_activation)
        if predicted_degradation > threshold:
            warn(f"{lang}: High risk under INT4")
```

**Value:** Immediate practical utility, builds on their theoretical framework.

---

## Recommended Pitch to Soudry Lab

> **Subject: Extending ACIQ to Multilingual Quantization Fairness**
>
> We've identified that the optimal clipping threshold (α*) from your
> ACIQ framework should vary by language in multilingual LLMs.
>
> **Our finding:** r = -0.834 correlation between outlier layer activation
> and quantization degradation across 14 languages in BLOOM.
>
> **Theoretical grounding:** Using your Banner et al. framework, we show
> that effective kurtosis (weighted by activation pattern) predicts
> degradation at r = +0.838.
>
> **Proposed collaboration:**
> 1. Formalize Language-Aware ACIQ
> 2. Test on BLOOM-7B/176B with your Intel resources
> 3. Develop practical calibration method
>
> This extends your efficiency work to the fairness dimension—ensuring
> quantized models work equitably across languages.

---

## Sources

- [Post-training 4-bit quantization (arXiv)](https://arxiv.org/abs/1810.05723)
- [Scaling FP8 training (arXiv)](https://arxiv.org/abs/2409.12517)
- [Daniel Soudry's homepage](https://soudry.github.io/)
- [Intel Neural Compressor](https://intellabs.github.io/distiller/algo_quantization.html)
- [A Survey of Quantization Methods (arXiv)](https://arxiv.org/pdf/2103.13630)
