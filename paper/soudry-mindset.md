---
layout: text
title: Learning from Soudry - Research Methodology and Next Experiments
permalink: /soudry-mindset/
---

# Learning from Soudry: A Research Methodology Analysis

*Extracting patterns from 15+ years of quantization research to guide our next experiments*

---

## Part I: The Soudry Lab Methodology

### Core Pattern: Analytical Solutions Over Heuristics

Across [ACIQ](https://arxiv.org/abs/1810.05723), [FP8 Training](https://arxiv.org/abs/2409.12517), and [Lognormal Gradients](https://arxiv.org/abs/2006.08173), a consistent pattern emerges:

| Paper | Heuristic Approach | Soudry's Approach |
|-------|-------------------|-------------------|
| ACIQ (2019) | Grid search for clipping threshold | **Closed-form optimal α** |
| FP8 (2025) | Manual hyperparameter tuning | **Theoretical bound (√3 noise)** |
| Gradients (2020) | Empirical sparsity thresholds | **Analytical threshold from lognormal** |

**The Move:** When others use trial-and-error, find the underlying distribution and derive the optimal solution mathematically.

---

### Pattern 1: Distribution First, Algorithm Second

From the [Neural Gradients paper](https://arxiv.org/abs/2006.08173):

> "Most previous works focus on weights and activations. These methods are often not applicable to neural gradients, which have **very different statistical properties**."

**The Insight:** They first characterized the distribution (near-lognormal), then derived methods from it.

**Application to Our Work:**

We know:
- Weight variances differ across layers
- Activation patterns differ across languages
- Critical layers have distinct kurtosis (L11: 48.2)

**What we haven't done:** Characterized the *per-language activation distribution* and derived optimal protection from it.

---

### Pattern 2: Find the Phase Transition

From [Mean Field Theory](https://arxiv.org/abs/1906.00771):

> "L_max ∝ N^1.82 — the maximal trainable depth given N quantization levels"

From [FP8 Training](https://arxiv.org/abs/2409.12517):

> "When gradient_norm < √3 × quantization_noise, training fails"

**The Move:** Find the critical threshold where behavior changes qualitatively.

**Application to Our Work:**

We found:
- Adding L0+L11 takes disparity from 160x to 0.92x (phase transition!)
- But we don't have a formula predicting *where* this transition occurs

**What we should derive:** The critical protection percentage where disparity drops below 1.0x.

---

### Pattern 3: Decompose the Problem Mathematically

From [ACIQ](https://papers.nips.cc/paper/9008-post-training-4-bit-quantization-of-convolutional-networks-for-rapid-deployment):

> "Quantization introduces two types of error:
> 1. Clipping error (values outside [-α, α])
> 2. Quantization noise (rounding within range)
>
> There's an optimal α that minimizes total MSE."

**The Move:** Break the problem into independent, analyzable components.

**Application to Our Work:**

Disparity comes from:
1. **Clipping damage** (outlier weights clipped)
2. **Quantization noise** (rounding errors)
3. **Error propagation** (how errors compound through layers)
4. **Language-specific activation** (which paths are used)

We've studied (1), (2), (3) somewhat. **We haven't fully characterized (4).**

---

### Pattern 4: Validate at Scale, Then Explain Theoretically

From [FP8 Training](https://arxiv.org/abs/2409.12517):

> "By scaling FP8 training to 2 trillion tokens—significantly beyond the previous limit—they encountered issues not observable in earlier works."

**The Move:** Push to extreme scales to reveal hidden phenomena, then explain.

**Limitation for Us:** We can't scale (GPU-constrained). But we can:
- Push to extreme *bit widths* (INT2)
- Push to extreme *language diversity* (50+ languages)
- Push to extreme *text lengths*

---

## Part II: What Soudry Would Ask About Our Work

### Question 1: "What's the distribution?"

**What we know:**
- Layer weights are approximately Gaussian (some heavy-tailed)
- L11 has kurtosis 48.2 (extremely non-Gaussian)

**What we don't know:**
- Per-language activation distribution
- How distribution shape varies with language resource level
- Whether critical layers have universal or language-specific distributions

**Experiment S-001: Per-Language Activation Distribution**

```python
def characterize_activation_distribution(model, tokenizer, texts):
    """
    For each layer and language:
    1. Collect activations
    2. Fit multiple distributions (Gaussian, Laplace, Lognormal)
    3. Compute best-fit and parameters
    4. Compare across languages
    """
    for layer in range(num_layers):
        for lang, text in texts.items():
            acts = get_activations(model, tokenizer, text, layer)

            # Fit distributions
            gauss_params = fit_gaussian(acts)
            laplace_params = fit_laplace(acts)
            lognorm_params = fit_lognormal(abs(acts) + eps)

            # BIC for model selection
            best_dist = select_best_distribution(acts, [gauss, laplace, lognorm])

            # Record
            results[layer][lang] = {
                'best_dist': best_dist,
                'kurtosis': kurtosis(acts),
                'params': params,
            }
```

**Epistemic Value:** HIGH — This is exactly what Soudry would do first.

---

### Question 2: "What's the optimal solution?"

**ACIQ Insight:** For Gaussian weights, optimal clipping α* ≈ 2.5σ for 4-bit.

**Our Analog:** What's the optimal *layer protection* given language distribution?

**Experiment S-002: Derive Optimal Protection Formula**

```python
def derive_optimal_protection(model, tokenizer, texts):
    """
    Goal: Find closed-form for which layers to protect given:
    - Layer variance
    - Layer kurtosis
    - Language activation pattern
    - Desired disparity target

    Method: Fit parametric model to our 91 experiments
    """
    # Collect features from all experiments
    X = []  # [variance, kurtosis, position, lang_activation_entropy]
    y = []  # [disparity achieved]

    # Fit: disparity = f(features)
    # Then invert: given target_disparity, which layers to protect?

    # Analytical approximation
    # disparity ≈ exp(a + b*variance + c*kurtosis + d*position)
    # Solve for protection threshold
```

**Epistemic Value:** VERY HIGH — A formula is publishable; a heuristic is not.

---

### Question 3: "What's the phase transition?"

**From our data:**
- 0% protection: 160x disparity
- 11.5% protection (L0+L11): 0.92x disparity
- 17% protection (L0+L9+L11): 0.59x disparity

**The transition happens somewhere around 5-15%.**

**Experiment S-003: Fine-Grained Protection Sweep**

```python
def find_phase_transition(model, tokenizer, texts):
    """
    Sweep protection percentage from 0% to 30% in 1% increments.
    Find where disparity drops below 1.0x.
    """
    results = []

    for pct in range(0, 31):
        # Select layers to protect (by criticality ranking)
        num_layers_to_protect = int(12 * pct / 100)
        protect = select_top_n_critical(num_layers_to_protect)

        disparity = measure_disparity(model, tokenizer, texts, protect)
        results.append((pct, disparity))

    # Find transition point
    transition = find_inflection_point(results)
    print(f"Phase transition at {transition}% protection")
```

**Epistemic Value:** HIGH — Phase transitions are theoretically interesting.

---

### Question 4: "What's the mechanism?"

Soudry's FP8 paper traced instability to SwiGLU activation weight alignment.

**Our analog:** Why does L0+L11 have synergy?

**Experiment S-004: Residual Stream Analysis**

```python
def analyze_residual_stream(model, tokenizer, texts):
    """
    Hypothesis: L0 errors propagate through residual stream.
    L11 can't compensate without clean L0 output.

    Test: Measure how activations at each layer correlate with final output.
    """
    # Get activations at all layers
    acts = get_all_activations(model, tokenizer, texts)

    # Measure "contribution" to final output
    for layer in range(num_layers):
        # Ablate residual at this layer
        output_with = forward_with_residual(model, texts)
        output_without = forward_without_residual(model, texts, layer)

        contribution = difference(output_with, output_without)

        # Critical layers should have high contribution
        # AND high language-dependent contribution
```

**Epistemic Value:** MEDIUM-HIGH — Mechanistic understanding strengthens the paper.

---

## Part III: High-Value Experiments Without GPU

### Tier 1: Distribution Characterization (Soudry's First Move)

| Exp | Description | Time | Value |
|-----|-------------|------|-------|
| **S-001** | Per-language activation distribution | 2h | VERY HIGH |
| **S-005** | Cross-layer activation correlation | 1h | HIGH |
| **S-006** | Distribution shift: FP32 vs INT4 | 1h | HIGH |

### Tier 2: Optimal Solution Derivation

| Exp | Description | Time | Value |
|-----|-------------|------|-------|
| **S-002** | Closed-form protection formula | 2h | VERY HIGH |
| **S-003** | Phase transition mapping | 3h | HIGH |
| **S-007** | Language-specific optimal α | 2h | HIGH |

### Tier 3: Mechanistic Understanding

| Exp | Description | Time | Value |
|-----|-------------|------|-------|
| **S-004** | Residual stream contribution | 2h | MEDIUM-HIGH |
| **S-008** | Layer gradient magnitude analysis | 1h | MEDIUM |
| **S-009** | Attention pattern per language | 2h | MEDIUM |

### Tier 4: Theoretical Extensions

| Exp | Description | Time | Value |
|-----|-------------|------|-------|
| **S-010** | Information bottleneck at L9 | 2h | HIGH |
| **S-011** | Rate-distortion bound derivation | 3h | VERY HIGH |
| **S-012** | Cross-architecture prediction | 2h | HIGH |

---

## Part IV: The Next 10 Experiments

Based on Soudry's methodology, here are the highest-value experiments we can run without GPU:

### 1. S-001: Per-Language Activation Distribution

**Why (Soudry would say):** "You can't optimize what you haven't characterized."

**Method:** Collect activations, fit distributions, compare across languages.

**Expected insight:** Low-resource languages may have different effective kurtosis, explaining why they're more sensitive.

---

### 2. S-002: Closed-Form Protection Formula

**Why:** This transforms our work from "we found something" to "here's how to apply it."

**Method:** Use our 91 experiments to fit:
```
protect_layers = f(target_disparity, model_stats)
```

**Expected insight:** A formula that works for any model.

---

### 3. S-003: Phase Transition Mapping

**Why:** Phase transitions reveal fundamental structure.

**Method:** Fine-grained sweep from 0-30% protection.

**Expected insight:** Sharp transition around 10-15%, matching L0+L11 threshold.

---

### 4. S-007: Language-Specific Optimal α

**Why (ACIQ extension):** If languages have different effective distributions, they need different clipping thresholds.

**Method:**
```python
for lang in languages:
    # Compute effective distribution when this language is processed
    eff_dist = compute_effective_distribution(model, lang)

    # Derive ACIQ-optimal α for this distribution
    alpha_star = aciq_optimal_alpha(eff_dist, bits=4)

    # Compare with global α
    print(f"{lang}: α* = {alpha_star} vs global α = {global_alpha}")
```

**Expected insight:** Hebrew/Arabic may need different α than English.

---

### 5. S-010: Information Bottleneck at L9

**Why:** L9 is at 75% depth. Is this an information bottleneck?

**Method:**
```python
# Compute mutual information between:
# I(input; layer_output) for each layer

# If L9 is a bottleneck, I should be lower there
# This would explain why protecting it helps
```

**Expected insight:** L9 compresses multilingual representations.

---

### 6. S-011: Rate-Distortion Bound

**Why:** This would give us a fundamental limit on achievable disparity.

**Method:**
```python
# Theory: disparity ≥ f(bits, language_entropy, protection)

# Fit from data:
# - Multiple bit widths (2, 3, 4, 6, 8)
# - Multiple protection levels (0%, 10%, 20%, 30%)
# - Derive bound
```

**Expected insight:** "You can't achieve disparity < X without protecting at least Y% of weights."

---

### 7. S-005: Cross-Layer Activation Correlation

**Why:** Synergy between L0 and L11 suggests correlated processing.

**Method:**
```python
# Compute correlation matrix of activations across layers
# corr[i][j] = correlation(acts[i], acts[j])

# If L0 and L11 are synergistic, they may have high correlation
```

**Expected insight:** L0-L11 form a "communication channel" through the network.

---

### 8. S-006: Distribution Shift Under Quantization

**Why:** Quantization changes the effective distribution. How much?

**Method:**
```python
# Compare activation distributions before/after quantization
# Measure KL divergence

# Critical layers may show larger KL shift for LR languages
```

**Expected insight:** Quantization "damages" the distribution more for some languages.

---

### 9. S-008: Layer Gradient Magnitude

**Why (from lognormal paper):** Gradient magnitude may predict importance.

**Method:**
```python
# Run backward pass, collect gradient magnitudes per layer
# Compare with our criticality ranking
```

**Expected insight:** Critical layers may have higher gradient magnitude.

---

### 10. S-012: Cross-Architecture Prediction

**Why:** A theory should predict, not just describe.

**Method:**
```python
# From GPT-2 analysis, derive prediction for OPT
# Test on OPT-125M (which we have data for)
# If it works, predict for Llama (testable on GPU later)
```

**Expected insight:** "Critical layers are at positions X% regardless of architecture."

---

## Part V: The Soudry Pitch

If we complete these experiments, here's how we'd pitch to Soudry Lab:

> **Subject: Language-Aware Quantization: Extending ACIQ to Multilingual**
>
> We've discovered that multilingual LLMs require different quantization
> strategies per language. Building on your ACIQ framework:
>
> **Finding 1:** Languages have different effective activation distributions.
> ACIQ's optimal α should vary by language.
>
> **Finding 2:** Critical "gateway" layers (input/output) have extreme
> kurtosis, matching your observation that outliers matter.
>
> **Finding 3:** We derive a closed-form for layer protection:
> `protect = f(variance, kurtosis, position)`
>
> **Proposed collaboration:**
> 1. Validate on Llama-7B with your resources
> 2. Develop LA-ACIQ (Language-Aware ACIQ)
> 3. Integration into production quantization pipelines

---

## References

- [Post-training 4-bit Quantization (NeurIPS 2019)](https://papers.nips.cc/paper/9008-post-training-4-bit-quantization-of-convolutional-networks-for-rapid-deployment)
- [Scaling FP8 Training (ICLR 2025 Spotlight)](https://arxiv.org/abs/2409.12517)
- [Neural Gradients are Lognormal (ICLR 2021)](https://arxiv.org/abs/2006.08173)
- [Mean Field Theory of Quantization (NeurIPS 2019)](https://arxiv.org/abs/1906.00771)
- [Accurate PTQ with Small Calibration Sets (ICML 2021)](https://proceedings.mlr.press/v139/hubara21a.html)
- [ACIQ: Analytical Clipping (ICLR 2019)](https://arxiv.org/abs/1810.05723)

---

*Analysis date: 2026-01-09*
*91 experiments completed | 10 proposed*
