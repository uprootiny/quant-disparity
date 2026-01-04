---
layout: page
title: Experiments
permalink: /experiments/
nav_order: 3
---

# Experiment Ledger

Complete record of all experiments, hypotheses tested, and results.

---

## Summary Statistics

```
Total Experiments:     27 completed
Hypotheses Tested:     12
  - Supported:         7
  - Rejected:          3
  - Mixed/Partial:     2
GPU Experiments:       0 (pending access)
```

---

## Hypothesis Tracker

| ID | Hypothesis | Prediction | Experiment | Status |
|----|------------|------------|------------|--------|
| H1 | Tensor dimension → κ | smaller → higher | EXP-022 | **REJECTED** |
| H2 | Dropout prevents outliers | dropout=0.1 → κ<10 | EXP-023 | **SUPPORTED** |
| H3 | Layer position | Early layers worst | EXP-024 | **MIXED** |
| H4 | Attention > MLP | κ_attn > κ_mlp | EXP-022 | **SUPPORTED** |
| H4b | Output projection dominant | 3/5 models | EXP-026 | **SUPPORTED** |
| H5 | Size scaling | Larger → higher κ | EXP-025 | **REJECTED** |
| H7 | Training evolution | κ grows late | EXP-027 | **SUPPORTED** |
| H-B1 | Language-specific heads | >10% of heads | B-001 | **SUPPORTED** |
| H-B2 | Circuit overlap correlation | — | — | Pending |
| H-B3 | Low-resource sensitivity | — | B-003 | Pending |
| H-C1 | Efficiency amplifies disparity | — | — | Pending |
| H-C2 | Fertility predicts degradation | r > 0 | EXP-033 | **REJECTED** |

---

## Phase 1: Initial Discovery

### EXP-001: BLOOM Disparity Detection
**Date:** 2025-12
**Model:** BLOOM-560M
**Finding:** Languages activating outlier layers less degrade more under quantization

```
Result: r = -0.834, p = 0.0002
Bootstrap 95% CI: [-0.93, -0.65]
```

### EXP-002 through EXP-008: Statistical Validation
**Findings:**
- Permutation test p = 0.0001
- Leave-one-out: stable (no single language drives effect)
- Effect holds across multiple validation approaches

---

## Phase 2: Cross-Model Validation

### EXP-020: Kurtosis Census
**Models:** 12 multilingual models
**Finding:** 25% of models have HEAVY outliers (κ > 50)

```
Distribution:
  HEAVY (κ > 50):   3/12 (25%)
  Moderate:         2/12 (17%)
  Mild:             5/12 (42%)
  Gaussian:         2/12 (17%)
```

### EXP-022: Architecture Comparison
**Question:** Does tensor dimension predict kurtosis?
**Result:** H1 REJECTED (r = -0.003)

**Question:** Does attention have higher outliers than MLP?
**Result:** H4 SUPPORTED (3/3 models)

```
Model         Attention κ    MLP κ
OPT-125M      562           89
BLOOM-560M    504           42
GPT-2         201           34
```

### EXP-023: Training Configuration
**Question:** Does dropout prevent outlier formation?
**Result:** H2 SUPPORTED

```
Pythia (dropout=0.0): κ = 14
BLOOM (dropout=0.1):  κ = 504 (but has FP32 hack)
OPT (dropout=0.1):    κ = 562
```

Key insight: Training instability (not just dropout) matters.

### EXP-024: Layer Position Analysis
**Question:** Do outliers concentrate in early or late layers?
**Result:** H3 MIXED (model-specific)

```
OPT:    Layer 1 worst (early)
BLOOM:  Layer 22 worst (late)
GPT-2:  Layer 11 worst (late)
```

### EXP-025: Size Scaling
**Question:** Do larger models have higher kurtosis?
**Result:** H5 REJECTED

```
Pythia-160M: κ = 8.2
Pythia-410M: κ = 14.1
Pythia-1B:   κ = 12.3  ← Not monotonic
```

### EXP-026: Attention Component Breakdown
**Question:** Which attention component has highest outliers?
**Result:** H4b SUPPORTED (output projection dominant)

```
Component       Average κ
out_proj        287
query           156
key             89
value           67
```

### EXP-027: Checkpoint Evolution
**Question:** When do outliers form during training?
**Result:** H7 SUPPORTED

```
Pythia-410M checkpoint evolution:
  Step 1K:    κ = 1.5
  Step 10K:   κ = 4.2
  Step 50K:   κ = 34.7
  Step 143K:  κ = 122.9

Growth: 82x increase during training
```

---

## Phase 3: Cross-Track Analysis

### EXP-030: Attention Sink Correlation
**Question:** Do outlier weights correlate with attention sink positions?
**Result:** No significant correlation (r = -0.23)

```
GPT-2 Analysis:
  Layer 11: κ = 190.1, sink = 0.68
  Layer 1:  κ = 150.3, sink = 0.46
  Layer 0:  κ = 25.8,  sink = 0.27

Correlation: r = -0.23, p = 0.47
```

**Interpretation:** Attention sinks and outlier weights are **distinct phenomena**. Both occur in attention but are mechanistically separate.

### EXP-033: Token Fertility Prediction
**Question:** Does tokenization efficiency predict degradation?
**Result:** H-C2 REJECTED (r = -0.07)

```
Language  Fertility  Known Degradation
English   1.00x      1.00
Japanese  13.29x     1.16
Chinese   6.13x      1.15
Hebrew    3.19x      1.25
Arabic    1.45x      1.30
```

**Interpretation:** Japanese has 13x more tokens but only 16% more degradation. The outlier mechanism is more fundamental than tokenization overhead.

---

## Track B: Interpretability

### B-001: Cross-Lingual Attention Patterns
**Model:** mBERT
**Result:** H-B1 SUPPORTED (16.7% > 10%)

```
Head Classification:
  Universal:          36 (25.0%)
  Language-specific:  24 (16.7%)
  Mixed:              84 (58.3%)

Per-layer distribution:
  Early (0-3):  23 universal, 2 specific
  Middle (4-7): 8 universal, 6 specific
  Late (8-11):  5 universal, 16 specific
```

Language-specific heads concentrate in late layers.

### B-002: Probing Under Quantization
**Status:** Scaffolded, quick test run
**Preliminary:** Need more data for statistical power

### B-003: Circuit Ablation
**Status:** Scaffolded, not yet run

---

## Track C: Efficiency-Fairness

### C-001b: Tokenizer Efficiency Gap
**Models:** mBERT, BLOOM
**Result:** 6.17x efficiency gap

```
mBERT Fertility:
  High-resource (en, de, fr, es): 1.51 avg
  Low-resource (zh, ar, he, sw):  4.35 avg
  Gap: 2.88x

BLOOM Fertility:
  High-resource: 1.48 avg
  Low-resource:  4.84 avg
  Gap: 3.27x

Combined low-resource: 9.55 tokens/word
Combined high-resource: 1.55 tokens/word
Overall gap: 6.17x
```

### C-002: Quantization Fairness
**Status:** Created, timed out on first run
**Method:** Compare FP16 vs simulated INT8/INT4 perplexity per language

---

## Pending Experiments (GPU Required)

| ID | Description | Blocking |
|----|-------------|----------|
| EXP-009 | Actual bit-width sweep with bitsandbytes | GPU |
| EXP-031 | LA-ACIQ implementation test | GPU |
| B-005 | Causal mediation analysis | GPU |

---

## Experiment Results Archive

All experiment results are stored in JSON format:

```
experiments/
├── phase-3-crossmodel/results/
│   ├── exp022_results_*.json
│   ├── exp023_results_*.json
│   ├── exp024_results_*.json
│   ├── exp025_results_*.json
│   ├── exp026_results_*.json
│   ├── exp027_results_*.json
│   ├── exp030_results_*.json
│   └── exp033_results_*.json
├── track-b-interpretability/results/
│   └── b001_results_*.json
└── track-c-efficiency/results/
    └── c001b_results_*.json
```

---

*Last updated: January 2026*
