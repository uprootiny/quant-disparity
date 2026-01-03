# Internal Synthesis: Our Work ↔ Soudry Lab Program

*For personal use. Traces connections, assumptions, horizons.*

---

## Our Finding (Compact)

```
In BLOOM-560M:
  - Layers 5, 21, 22 have outlier weights (max|W| > 2.5, κ > 100)
  - Languages differ in activation of these layers (17-21%)
  - Correlation with degradation: r = -0.834

Interpretation:
  - Well-resourced languages → use outlier layers → redundant repr → robust
  - Low-resource languages → avoid outlier layers → fragile → degrade more

Limitation:
  - BLOOM-specific (XGLM shows no pattern)
```

---

## Soudry Lab: Paper-by-Paper Mapping

### Paper 1: ACIQ (Banner 2019)

**Their claim:**
```
Optimal clipping α* minimizes MSE = clip_error + quant_noise
For distribution with kurtosis κ: α*/σ ≈ 2.5 + 0.3·log(1+κ)
```

**Their assumption:**
```
Single α for entire model
Distribution is language-agnostic
```

**Their horizon:**
```
Fast post-training quantization (no fine-tuning)
Deploy CNNs to edge devices
```

**Their payoff:**
```
40% accuracy improvement over naive
4000x faster than KL-divergence calibration
Intel Neural Compressor integration
```

**Our connection:**
```
We EXTEND their framework:
  α* should be α*(λ) — per-language

We USE their theory:
  Effective kurtosis κ_eff(λ) predicts degradation (r=+0.84)

We CHALLENGE their assumption:
  Single α is suboptimal for multilingual
```

**Gap we fill:**
```
ACIQ + multilingual = LA-ACIQ
They optimize efficiency; we add equity dimension
```

---

### Paper 2: 8-bit Training (Banner 2018)

**Their claim:**
```
Full 8-bit training is possible if:
  - Most operations use 8-bit
  - Only final gradient step needs higher precision
```

**Their assumption:**
```
Training dynamics are uniform across operations
Model doesn't distinguish between languages
```

**Their horizon:**
```
Faster training, less memory
Enable training on cheaper hardware
```

**Their payoff:**
```
State-of-the-art ImageNet with 8-bit throughout
Range Batch Normalization technique
```

**Our connection:**
```
We operate in INFERENCE, they operate in TRAINING
But: training creates the weight patterns we observe
```

**Potential link:**
```
Could Range BN prevent outlier formation?
If training didn't create outliers → no disparity?
```

---

### Paper 3: FP8 at Scale (Chmiel 2025)

**Their claim:**
```
FP8 training fails at ~200B tokens due to outlier amplification
SwiGLU activation causes weight alignment → outliers grow
Smooth-SwiGLU prevents this
```

**Their assumption:**
```
Outlier formation is architecture-dependent (SwiGLU)
Training scale reveals hidden instabilities
```

**Their horizon:**
```
Trillion-token training in FP8
Match BF16 quality at lower precision
```

**Their payoff:**
```
First 2T token FP8 training
34% throughput improvement
ICLR 2025 Spotlight
```

**Our connection:**
```
DIRECT LINK to our finding:
  - They found: training creates outliers (in activations)
  - We found: training creates outliers (in weights)
  - They found: outliers cause failure at scale
  - We found: outliers cause disparity across languages
```

**Key question:**
```
BLOOM uses GeLU, not SwiGLU
But similar dynamics may apply
Did BLOOM's training create outliers through similar mechanism?
```

**Potential collaboration:**
```
Apply Smooth-SwiGLU insight to prevent weight outliers
Train model with regularization → test if disparity disappears
```

---

### Paper 4: Implicit Bias (Soudry 2018)

**Their claim:**
```
Gradient descent on separable data converges to max-margin solution
Convergence is logarithmically slow
No explicit regularization needed — it's implicit
```

**Their assumption:**
```
Data is linearly separable
Loss has infimum at infinity
```

**Their horizon:**
```
Explain why overparameterized models generalize
Understand optimization geometry
```

**Their payoff:**
```
Foundational theory paper (JMLR)
Sparked wave of implicit bias research
```

**Our connection:**
```
INDIRECT but relevant:
  - They study what GD finds implicitly
  - We study what training creates implicitly (outliers)
  - Implicit bias → implicit disparity?
```

**Speculative link:**
```
If GD implicitly finds max-margin for well-represented classes,
what does it implicitly find for under-represented languages?
Maybe: suboptimal representations that don't use outlier layers?
```

---

### Paper 5: Train Longer (Hoffer 2017)

**Their claim:**
```
Generalization gap from large batches is due to fewer updates, not batch size
Fix: match total updates, use Ghost Batch Normalization
```

**Their assumption:**
```
Update count matters more than batch composition
All data contributes equally
```

**Their horizon:**
```
Enable large-batch training without generalization loss
Faster wall-clock training
```

**Their payoff:**
```
Closed generalization gap on ImageNet
NeurIPS Oral presentation
```

**Our connection:**
```
BLOOM was trained with specific batch/update regime
Under-represented languages may have had:
  - Fewer effective updates
  - Less influence on weight formation
  → Didn't "claim" the outlier layers
```

**Potential experiment:**
```
Analyze BLOOM training logs (if available)
How many updates per language?
Correlate with outlier activation?
```

---

## Synthesis: Their Research Program

### Core Theme

```
EFFICIENCY through understanding optimization dynamics
```

### Sub-themes

```
1. Quantization: How low can we go? (1-bit → 4-bit → 8-bit → FP8)
2. Training dynamics: What does optimization implicitly do?
3. Scale: What breaks at scale, and why?
```

### Assumptions Across Papers

| Assumption | Papers | Status |
|------------|--------|--------|
| Single precision for all data | ACIQ, 8-bit | We challenge (multilingual) |
| Language-agnostic | All | We challenge |
| Outliers are bad | FP8 | We complicate (outliers help some) |
| Optimization is uniform | Implicit bias | We question (language-dependent?) |

### Their Blind Spots

```
1. Multilingual models (all work is English-centric)
2. Fairness/equity (focus is efficiency)
3. Inference disparity (focus is training)
4. Model-specific patterns (BLOOM vs XGLM)
```

---

## Our Niche: The Intersection

```
         Soudry Lab                    Our Work
         ──────────                    ────────
         Efficiency                    Equity
         Training                      Inference
         English-centric               Multilingual
         Outliers = problem            Outliers = symptom

                    ↓ INTERSECTION ↓

         Quantization of Multilingual Models
         Understanding Language-Dependent Effects
         Extending ACIQ to Fairness
```

---

## Strategic Positioning

### Option A: Pure Extension

```
"We extend ACIQ to multilingual setting"

Pitch: Your framework predicts our finding.
       Languages have different effective kurtosis.
       Per-language α* would help.

Payoff: Clean theoretical contribution
        Fits their existing program
        Low risk, moderate impact
```

### Option B: Challenge + Extend

```
"Your assumptions don't hold for multilingual"

Pitch: Single α is demonstrably suboptimal.
       BLOOM shows this clearly (r=-0.834).
       We need language-aware methods.

Payoff: Stronger claim
        Opens new research direction
        Higher risk (they might disagree)
```

### Option C: Training Intervention

```
"Prevent the problem at training time"

Pitch: Chmiel shows outliers form during training.
       We show outliers create disparity.
       Smooth-SwiGLU for weights, not just activations?

Payoff: Connects to their cutting-edge work
        Potential for high impact
        Requires collaboration (compute)
```

---

## What They Would Want From Us

### Data/Findings

```
- Correlation analysis across languages (we have)
- Model-specific patterns (BLOOM vs XGLM, we have)
- Theoretical predictions (r=+0.84, we have)
```

### Gaps We Fill

```
- Multilingual perspective (they lack)
- Fairness framing (they lack)
- Specific failure mode (BLOOM outliers)
```

### What We Need From Them

```
- Theoretical depth (optimization, bounds)
- Scale (GPU access, larger models)
- Industry connection (Intel deployment)
```

---

## Open Questions for Them

```
1. Did you observe language-dependent effects in FP8 training?
   (Chmiel et al. trained on "2T tokens" — which languages?)

2. Does Smooth-SwiGLU prevent outlier WEIGHTS, not just activations?
   (Could test: train BLOOM-like model with Smooth-SwiGLU)

3. Why does XGLM have Gaussian weights but BLOOM doesn't?
   (Training data? Architecture? Optimizer settings?)

4. Is there a theoretical bound on disparity from kurtosis variance?
   (We conjectured: Disparity ≤ C·√Var[κ_eff])
```

---

## Internal Research Agenda

### Phase 1: Document (Now)

```
- Write up BLOOM finding clearly
- Map to their papers explicitly
- Identify specific extension points
```

### Phase 2: Theorize (CPU)

```
- Prove disparity bound formally
- Analyze when LA-ACIQ helps
- Predict which models will show pattern
```

### Phase 3: Validate (GPU)

```
- Bit-width sweep (EXP-009)
- Larger models (BLOOM-7B)
- Other architectures (Llama, Mistral)
```

### Phase 4: Intervene (Collaboration)

```
- Training with anti-outlier regularization
- Per-language calibration method
- Deployment tool
```

---

## The Pitch (Internal Draft)

```
To: Daniel Soudry
Subject: Extending ACIQ to Multilingual Fairness

Your ACIQ paper shows optimal α depends on distribution kurtosis.
We found that in multilingual models, different languages experience
different effective kurtosis due to layer activation patterns.

Specifically, in BLOOM-560M:
  - Outlier weights concentrate in layers 5, 21, 22
  - Well-resourced languages (English) activate these more
  - Under-represented languages (Arabic) activate these less
  - Correlation with quantization degradation: r = -0.834

Using your framework, we derived Language-Aware ACIQ (LA-ACIQ),
which predicts per-language optimal clipping thresholds.

This extends your efficiency work to the equity dimension:
ensuring quantized models work fairly across languages.

We seek collaboration to:
  1. Formalize LA-ACIQ theoretically
  2. Test at scale (BLOOM-7B/176B)
  3. Connect to your FP8 outlier work (Chmiel et al.)

Our data and code: [link]
```

---

*Internal document. Last updated: 2026-01-03*
