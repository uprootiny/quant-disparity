# Research Fit Analysis: Our Work × Soudry Lab

## Executive Summary

Our research on multilingual quantization disparity is a **natural extension**
of the Soudry Lab's quantization efficiency work to the **fairness dimension**.

```
Soudry Lab Focus:     How to quantize neural networks efficiently
Our Extension:        How quantization affects languages differently
Combined Impact:      Efficient AND equitable quantization
```

---

## The Niche We Fill

### Gap in Existing Literature

| Existing Work | Missing Piece |
|---------------|---------------|
| ACIQ (Banner 2019) | Assumes language-agnostic distributions |
| FP8 Training (Chmiel 2025) | English-centric evaluation |
| All quantization surveys | Treat multilingual as monolithic |
| Marchisio et al. (2023) | Documented disparity, no mechanism |

### Our Unique Contribution

```
We provide the MECHANISM:

Observation:  Multilingual LLMs degrade non-uniformly (Marchisio 2023)
Mechanism:    Outlier layer activation pattern (r = -0.834, p = 0.0002)
Theory:       Effective kurtosis differs by language (r = +0.838)
Grounding:    Extends Banner et al. ACIQ framework
```

---

## Methodology Alignment

### We Follow Their Pattern

**Pattern 1: Theory-First**
```
Soudry Lab:
  Theoretical Analysis → Mathematical Insight → Validation

Our Application:
  1. Take Banner's ACIQ framework (α* depends on kurtosis)
  2. Insight: Languages have different effective kurtosis
  3. Validate: r = +0.838 correlation with actual degradation
```

**Pattern 2: Root Cause, Minimal Intervention**
```
Soudry Lab:
  Observe Failure → Trace Mechanism → Minimal Fix

Our Application:
  1. Observe: Non-uniform degradation across languages
  2. Trace: Outlier layer activation (layers 5, 21, 22)
  3. Fix: Language-aware calibration (proposed)
```

**Pattern 3: Statistical Characterization**
```
Soudry Lab:
  Empirical Observation → Statistical Model → Actionable Threshold

Our Application:
  1. Observe: r = -0.834 between activation and degradation
  2. Model: Effective kurtosis predicts quantization error
  3. Threshold: Per-language α* (to be computed)
```

---

## What We Bring to Collaboration

### Concrete Results

| Metric | Value | Significance |
|--------|-------|--------------|
| Primary correlation | r = -0.834 | p = 0.0002 |
| Bootstrap CI | [-0.93, -0.65] | Excludes zero |
| Permutation p | 0.0001 | Highly significant |
| Theoretical prediction | r = +0.838 | Validates mechanism |

### Reproducible Infrastructure

```
quant-disparity/
├── experiments/
│   ├── phase-1-extraction/    # Weight analysis
│   ├── phase-2-corpus/        # Corpus tools
│   └── PROTOCOL.md            # Pre-registered hypotheses
├── docs/
│   ├── soudry_lab_analysis.md # Paper analysis
│   └── research_fit_analysis.md
└── VISION.md                  # Research roadmap
```

### Novel Angle

- **Multilingual focus**: Less explored in quantization literature
- **Fairness framing**: Extends efficiency to equity
- **Practical impact**: Low-resource languages are underserved

---

## What We Need from Collaboration

### Technical Resources

| Need | Their Capability |
|------|------------------|
| GPU access | Intel Gaudi2 access |
| Scale experiments | 7B-176B model expertise |
| Optimization theory | ACIQ, implicit bias depth |
| Industry connections | Intel, Meta, cloud providers |

### Intellectual Contributions

| Need | Their Expertise |
|------|-----------------|
| Formalize LA-ACIQ | Optimal quantization theory |
| Prove optimality | Convex optimization |
| Training interventions | Smooth-SwiGLU methodology |

---

## Proposed Research Agenda

### Phase 1: Validation (1-2 months)

```
Goal: Confirm mechanism at larger scale

Experiments:
  - EXP-009: Bit-width sweep (INT8 → INT4 → INT3)
  - BLOOM-7B replication
  - BLOOM-176B sampling

Deliverable: Robust cross-scale correlation
```

### Phase 2: Theory (2-3 months)

```
Goal: Formalize Language-Aware ACIQ

Components:
  - Prove: optimal α*(lang) depends on effective kurtosis
  - Derive: closed-form solution for per-language clipping
  - Analyze: computational overhead of calibration

Deliverable: Theory paper (ICML/NeurIPS format)
```

### Phase 3: Method (3-4 months)

```
Goal: Practical calibration algorithm

Options:
  A. Per-language calibration (simple, O(|languages|) overhead)
  B. Adaptive calibration (input-dependent, runtime overhead)
  C. Mixed-precision (per-layer bit-width, compile-time)

Deliverable: Method paper + open-source tool
```

### Phase 4: Deployment (ongoing)

```
Goal: Industry adoption

Targets:
  - Intel Neural Compressor integration
  - HuggingFace Optimum support
  - Cloud provider awareness

Deliverable: Tech transfer, real-world impact
```

---

## Comparative Advantage

### Why This Collaboration?

| Alternative | Limitation |
|-------------|------------|
| Solo work | Lack GPU, scale, theory depth |
| Other quant labs | Don't have ACIQ framework |
| NLP fairness labs | Don't have quantization expertise |
| Industry labs | Slower, less academic freedom |

### Why Soudry Lab Specifically?

1. **ACIQ is our theoretical foundation** — direct extension
2. **Outlier work (Chmiel)** — related mechanism
3. **Intel connections** — deployment path
4. **Academic + industry** — both publication and impact

---

## Risk Analysis

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Correlation ≠ causation | Intervention experiments |
| BLOOM-specific | Test on Llama, Mistral |
| Small effect size | May still be meaningful for fairness |

### Collaboration Risks

| Risk | Mitigation |
|------|------------|
| Not interested | Pivot to alternative labs |
| Too busy | Offer to do most work |
| IP concerns | Open-source commitment |

---

## Draft Email Pitch

```
Subject: Extending ACIQ to Multilingual Quantization Fairness

Dear Prof. Soudry,

I'm researching why multilingual LLMs degrade non-uniformly under
quantization. Using your ACIQ framework (Banner et al. 2019), I've
identified a mechanism:

Finding: r = -0.834 correlation between outlier layer activation
         and quantization degradation across 14 languages in BLOOM.

Theory:  Effective kurtosis (weighted by language-specific activation)
         predicts degradation at r = +0.838, as your framework predicts.

This suggests language-aware clipping thresholds (α* per language)
could reduce disparity — a direct extension of ACIQ to the multilingual
setting.

I'm interested in PhD opportunities to develop this into:
1. Theoretical formalization (Language-Aware ACIQ)
2. Practical calibration methods
3. Deployable tools for equitable quantization

My work: [GitHub link]
My background: [Brief relevant experience]

Would you be open to a brief discussion?

Best regards,
[Name]
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| PhD admission | Accepted to Technion |
| First paper | Submitted within 12 months |
| Correlation at scale | r > 0.7 on BLOOM-7B+ |
| Industry adoption | One major integration |
| Citation impact | >50 citations within 3 years |

---

*Analysis completed: 2026-01-03*
