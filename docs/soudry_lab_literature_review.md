# Soudry Lab: Comprehensive Literature Review

## Overview

Daniel Soudry leads the Machine Learning and Neural Networks Lab at Technion's
Department of Electrical & Computer Engineering. His research spans neural
network compression, training dynamics, and theoretical foundations of deep
learning.

---

## Publication Catalog

### Quantization & Compression Track

| Paper | Year | Venue | Core Contribution |
|-------|------|-------|-------------------|
| Binarized Neural Networks | 2016 | NeurIPS | 1-bit weights/activations training |
| Quantized Neural Networks | 2017 | arXiv | 1-bit W, 2-bit A (51% ImageNet) |
| Scalable 8-bit Training | 2018 | NeurIPS | First full 8-bit training pipeline |
| ACIQ (Post-training 4-bit) | 2019 | NeurIPS | Analytical clipping optimization |
| Data-free Compression | 2020 | CVPR | Compression without training data |
| Low Bit-Width Accumulators | 2024 | ICLR | Reduce accumulator precision |
| FP8 Training at Scale | 2025 | ICLR Spotlight | 2T tokens, Smooth-SwiGLU |
| FP4 All the Way | 2025 | preprint | Fully 4-bit training |

### Training Dynamics Track

| Paper | Year | Venue | Core Contribution |
|-------|------|-------|-------------------|
| Train Longer, Generalize Better | 2017 | NeurIPS Oral | Batch size vs updates tradeoff |
| Implicit Bias of GD | 2018 | JMLR | GD converges to max-margin |
| Multiclass Implicit Bias | 2019 | arXiv | Extension to multiclass |
| Optimization Geometry | 2020+ | ongoing | Characterizing implicit bias |

---

## Methodology Patterns

### Pattern 1: Theory-First, Then Validate

```
Theoretical Analysis → Mathematical Insight → Empirical Validation → Practical Method
```

**Example: ACIQ (Banner 2019)**
1. Formalize quantization error as MSE
2. Derive: Total MSE = clip_error + quant_noise
3. Solve analytically: optimal α depends on distribution kurtosis
4. Validate: 40% improvement over naive, 4000x faster than KL

**Example: Implicit Bias (Soudry 2018)**
1. Analyze GD dynamics on separable data
2. Prove: predictor converges to max-margin direction
3. Show: convergence is logarithmically slow
4. Explain: why training beyond zero error helps

### Pattern 2: Identify Root Cause, Minimal Intervention

```
Observe Failure → Trace to Mechanism → Propose Minimal Fix → Validate at Scale
```

**Example: FP8 Training (Chmiel 2025)**
1. Observe: FP8 fails after ~200B tokens
2. Trace: SwiGLU causes outlier amplification over training
3. Fix: Smooth-SwiGLU (minimal architectural change)
4. Validate: 2T tokens, matches BF16, 34% throughput gain

**Example: Train Longer (Hoffer 2017)**
1. Observe: Large batch → generalization gap
2. Trace: Fewer updates, not batch size itself
3. Fix: Ghost Batch Normalization + matched steps
4. Validate: Gap eliminated on ImageNet

### Pattern 3: Statistical Characterization

```
Empirical Observation → Statistical Model → Predictive Framework → Actionable Threshold
```

**Example: 8-bit Training (Banner 2018)**
1. Observe: Which operations fail at 8-bit
2. Model: Gradient noise tolerance varies by operation
3. Predict: Only final gradient step needs higher precision
4. Threshold: Range BN enables 8-bit throughout

### Pattern 4: Industry Collaboration Pipeline

```
Academic Research → Intel/Habana Collaboration → Hardware Integration → Production Deployment
```

Confirmed partners:
- Intel Neural Compressor (ACIQ integration)
- Habana Labs/Intel Gaudi2 (FP8 training support)
- ERC A-B-C-Deep grant (theoretical foundations)

Lab members → Industry:
- Brian Chmiel: PhD → Intel-Habana Labs
- Ron Banner: Co-author, Intel affiliation

---

## Key Theoretical Frameworks

### ACIQ Quantization Error (Banner 2019)

```
Total MSE = E[(X - Q(X))²] = clip_error + quant_noise

For symmetric uniform quantization:
  quant_noise = Δ²/12 where Δ = 2α/(2^B - 1)
  clip_error = E[(|X| - α)² | |X| > α] × P(|X| > α)

Optimal α depends on distribution:
  - Gaussian (kurtosis=0): α*/σ ≈ 2.5 (4-bit)
  - Heavy-tailed (high kurtosis): α*/σ increases
```

### Implicit Bias (Soudry 2018)

```
For unregularized logistic regression on separable data:
  w(t) → w_∞ / ||w_∞|| as t → ∞

where w_∞ is the max-margin (SVM) solution.

Convergence rate: O(log(t)) — very slow.
```

### Outlier Amplification (Chmiel 2025)

```
SwiGLU: f(x) = x · σ(x)

During long training:
  1. Weight alignment creates outlier dimensions
  2. Outliers grow over training (~200B tokens)
  3. FP8's limited dynamic range clips outliers
  4. Loss spike / training failure

Solution: Smooth-SwiGLU prevents alignment
```

---

## Research Gaps in Their Work

| Gap | Description | Opportunity |
|-----|-------------|-------------|
| Multilingual | All work assumes language-agnostic | Our work |
| Fairness | Focus on efficiency, not equity | Our work |
| Post-training + Multilingual | ACIQ is language-blind | LA-ACIQ |
| Small models | Recent focus on 7B+ | Our 560M work still valid |
| Language-specific calibration | Single α for all | Per-language α* |

---

## Our Research: Natural Extension

### How We Extend Their Framework

**Starting Point: ACIQ (Banner 2019)**
```
Their finding:   α* depends on distribution shape (kurtosis)
Their assumption: Single α for entire model
Their scope:     Monolingual (English) models
```

**Our Extension:**
```
Our finding:    Languages have different effective kurtosis
Our mechanism:  Outlier layer activation varies by language (r=-0.834)
Our prediction: Language-specific α* would reduce disparity
Our scope:      Multilingual models (BLOOM, XGLM)
```

### Theoretical Grounding in Their Work

Using Banner et al. framework:
1. Effective kurtosis = Σ(activation[i] × kurtosis[i]) / Σ(activation[i])
2. Languages with different activation patterns → different effective kurtosis
3. Single α optimized for English → suboptimal for other languages
4. Prediction: effective kurtosis correlates with degradation

**Our Result: r = +0.838 (p < 0.001)**

This validates that Banner's framework explains multilingual disparity.

### Connection to Chmiel 2025

**Their Finding:**
- SwiGLU causes outlier amplification during training
- Outliers cause FP8 failure at scale

**Our Finding:**
- BLOOM has outlier weights in specific layers (5, 21, 22)
- Languages activating these layers less → more degradation

**Possible Link:**
- BLOOM uses GeLU (similar to SwiGLU dynamics?)
- Outlier formation during training creates language-dependent sensitivity
- Low-resource languages may not have developed representations in outlier layers

---

## Proposed Collaboration Pitch

### Value Proposition

> We've extended your ACIQ framework to explain multilingual quantization
> disparity. With your expertise, we can:
>
> 1. **Formalize LA-ACIQ** (Language-Aware ACIQ)
> 2. **Test at scale** (7B-176B models with your Intel resources)
> 3. **Develop practical calibration** (per-language or adaptive)
>
> This extends your efficiency work to the fairness dimension.

### Deliverables

| Deliverable | Contribution | Resource Need |
|-------------|--------------|---------------|
| Theory paper | Formalize LA-ACIQ | Optimization expertise |
| Method paper | Calibration algorithm | GPU access |
| Open-source tool | Diagnostic + calibration | Engineering |

### Alignment with Lab Themes

| Their Theme | Our Alignment |
|-------------|---------------|
| Optimal quantization (ACIQ) | We USE and EXTEND this |
| Outlier dynamics (Chmiel) | We FOUND outlier layers |
| Training dynamics | We EXPLAIN via training data |
| Hardware-aware | Bit-width recommendations |
| Industry transfer | Deployable fairness tool |

---

## Comparative Analysis: Methods

| Aspect | Soudry Lab | Our Work | Gap/Opportunity |
|--------|-----------|----------|-----------------|
| Scale | 7B+ models, trillions tokens | 560M, thousands samples | Need scale-up |
| Focus | Training efficiency | Inference fairness | Complementary |
| Distribution assumption | Language-agnostic | Language-dependent | Our novelty |
| Intervention | Smooth-SwiGLU, Range BN | None yet | Need method |
| Hardware | Intel Gaudi2 | CPU only | Need GPU |
| Validation | Production-scale | Statistical correlation | Need causal proof |

---

## Reading List for Deep Dive

### Must-Read (directly relevant)

1. **Banner et al. (2019)** — ACIQ
   - arXiv: https://arxiv.org/abs/1810.05723
   - Our theoretical foundation

2. **Chmiel et al. (2025)** — FP8 Training
   - arXiv: https://arxiv.org/abs/2409.12517
   - Outlier amplification mechanism

3. **Banner et al. (2018)** — 8-bit Training
   - arXiv: https://arxiv.org/abs/1805.11046
   - Full training pipeline quantization

### Recommended (theoretical depth)

4. **Soudry et al. (2018)** — Implicit Bias
   - JMLR: https://jmlr.org/papers/v19/18-188.html
   - Gradient descent theory

5. **Hoffer et al. (2017)** — Train Longer
   - arXiv: https://arxiv.org/abs/1705.08741
   - Batch size dynamics

### Optional (broader context)

6. **Hubara et al. (2016)** — Quantized Neural Networks
   - arXiv: https://arxiv.org/abs/1609.07061
   - Binary/low-bit foundations

---

## Next Steps

1. **Formalize LA-ACIQ**: Write mathematical framework
2. **Bit-width sweep**: EXP-009 on GPU (~$0.32)
3. **Scale validation**: BLOOM-7B experiments
4. **Draft pitch**: Email to soudry@technion.ac.il

---

*Document created: 2026-01-03*
*Based on literature review of 12+ Soudry Lab publications*
