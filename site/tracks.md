---
layout: page
title: Research Tracks
permalink: /tracks/
nav_order: 2
---

# Research Tracks

We organize our investigation into four parallel tracks, each targeting a different Israeli AI lab and addressing a complementary research question.

---

## Track A: Quantization Mechanics {#track-a}

**Target Lab:** [Soudry Lab](https://eladsoudry.github.io/) — Technion
**Status:** Primary track, most advanced

### Research Question
> Why do outlier weights cause disproportionate harm to low-resource languages?

### Core Finding

```
BLOOM-560M: r = -0.834 (p = 0.0002)
Languages activating outlier layers LESS degrade MORE

Bootstrap 95% CI: [-0.93, -0.65]
Leave-one-out: Stable (no single language drives effect)
```

### Hypotheses Tested

| ID | Hypothesis | Prediction | Result |
|----|------------|------------|--------|
| H1 | Tensor dimension → κ | Smaller dims → higher κ | **REJECTED** (r=-0.003) |
| H2 | Dropout prevents outliers | dropout=0.1 → κ<10 | **SUPPORTED** |
| H4 | Attention > MLP for outliers | 3/3 models | **SUPPORTED** |
| H4b | Output projection dominant | 3/5 models | **SUPPORTED** |
| H5 | Larger models → higher κ | Monotonic increase | **REJECTED** |
| H7 | Outliers grow during training | κ increases late | **SUPPORTED** (82x) |

### Key Experiments

| Experiment | Finding |
|------------|---------|
| EXP-001 | Initial disparity detection in BLOOM |
| EXP-020 | Cross-model kurtosis census |
| EXP-022 | Architecture comparison (H1, H4) |
| EXP-023 | Training config analysis (H2) |
| EXP-027 | Checkpoint evolution (H7) |
| EXP-030 | Attention sinks ≠ outliers |

### Model Taxonomy

| Model | Max κ | Location | Class |
|-------|-------|----------|-------|
| OPT-125M | 562 | Layer 1, attn.out_proj | HEAVY |
| BLOOM-560M | 504 | Layer 22, attn.query | HEAVY |
| GPT-2-small | 201 | Layer 11, attn.c_proj | HEAVY |
| mT5-small | 45 | Decoder 2-5 | Moderate |
| Pythia-410M | 14 | — | Mild |
| XLM-R-base | 10 | — | Mild |
| XGLM-564M | 2 | — | Gaussian |

### Proposed Solution: LA-ACIQ

Language-Aware Analytical Clipping for Integer Quantization:
- Identify super weights via magnitude + gradient analysis
- Apply per-language clipping thresholds
- Preserve attention projection outliers
- Theoretical validation: r = +0.84

---

## Track B: Circuit Interpretability {#track-b}

**Target Lab:** [Belinkov Lab](https://belinkov.com/) — Technion
**Status:** Initial experiments complete

### Research Question
> What circuits handle multilingual processing, and how does quantization affect them?

### Core Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H-B1 | Different languages activate different circuit subsets | **SUPPORTED** (16.7%) |
| H-B2 | Quantization damage correlates with circuit overlap | Pending |
| H-B3 | Low-resource languages rely on fewer, more critical circuits | Pending |

### Key Finding: Language-Specific Heads

From B-001 (mBERT attention analysis):

```
Head Classification:
  Universal heads:          36 (25.0%)
  Language-specific heads:  24 (16.7%)  ← Exceeds 10% threshold
  Mixed heads:              84 (58.3%)
```

Language-specific heads concentrate in **late layers (8-11)**.

### Cross-Lingual Similarity

Language pairs ranked by attention pattern similarity (JS divergence):

| Most Similar | Least Similar |
|--------------|---------------|
| French-Arabic (0.26) | English-Spanish (0.42) |
| German-French (0.29) | Chinese-Hebrew (0.40) |
| Arabic-Hebrew (0.30) | German-Chinese (0.39) |

### Experiments

| Experiment | Status | Finding |
|------------|--------|---------|
| B-001 | Complete | 16.7% language-specific heads |
| B-002 | Scaffolded | Probing under quantization |
| B-003 | Scaffolded | Circuit ablation by language |
| B-004 | Planned | Gradient-based circuit discovery |
| B-005 | Planned | Causal mediation for disparity |

### Connection to Track A

Track A tells us **where** outliers are (attention projections).
Track B tells us **which** attention heads are language-critical.
Combined insight: Outlier weights may be in language-specific circuits.

---

## Track C: Efficiency & Fairness {#track-c}

**Target Lab:** [Schwartz Lab](https://schwartz-lab-huji.github.io/) — Hebrew University
**Status:** Key finding achieved

### Research Question
> Do efficient models systematically disadvantage low-resource languages?

### Core Hypothesis

**H-C1:** Efficiency techniques (quantization, distillation, pruning) disproportionately harm low-resource languages.

### Key Finding: Tokenization Efficiency Gap

From C-001b:

```
Tokenizer Fertility (tokens per word):
  High-resource average: 1.55
  Low-resource average:  9.55
  Efficiency gap:        6.17x
```

### Unexpected Result: Fertility ≠ Degradation

From EXP-033:

```
Correlation (fertility vs known degradation): r = -0.07
```

Japanese has 10-17x more tokens than English but only ~16% more degradation. This means:
- Models compensate for token overhead
- The bottleneck is **attention outliers**, not tokenization
- Our Track A mechanism is more fundamental

### Per-Language Fertility (BLOOM)

| Language | Fertility | Normalized |
|----------|-----------|------------|
| English | 1.30 | 1.00x |
| French | 1.35 | 1.03x |
| Spanish | 1.27 | 0.97x |
| Arabic | 1.89 | 1.45x |
| Hebrew | 4.16 | 3.19x |
| Chinese | 8.00 | 6.13x |
| Japanese | 17.33 | 13.29x |

### Experiments

| Experiment | Status | Finding |
|------------|--------|---------|
| C-001b | Complete | 6.17x efficiency gap |
| C-002 | Created | Quantization fairness (pending) |
| C-003 | Planned | Pruning effect on languages |
| C-004 | Planned | Carbon cost per language |

---

## Track D: Syntax & Morphology {#track-d}

**Target Lab:** [Goldberg Lab (BIU-NLP)](https://u.cs.biu.ac.il/~yogo/) — Bar-Ilan University
**Status:** Research plan complete, experiments pending

### Research Question
> How does quantization affect syntactic and morphological processing in morphologically rich languages (MRLs)?

### Motivation

Morphologically rich languages (Hebrew, Arabic, Turkish, Finnish) require **joint morpho-syntactic processing**. Standard NLP pipelines fail on these languages due to morphology-syntax interaction.

Key insight from BIU-NLP:
- AlephBERT shows morphological awareness improves all downstream Hebrew NLP tasks
- YAP parser demonstrates joint modeling superiority over pipeline approaches

### Core Hypotheses

| ID | Hypothesis |
|----|------------|
| H-D1 | Quantization disproportionately degrades morphological disambiguation in MRLs |
| H-D2 | Morphological complexity correlates with quantization sensitivity |
| H-D3 | Joint morpho-syntactic models are more robust than pipeline models |
| H-D4 | Subword tokenization compounds quantization damage in MRLs |

### Planned Experiments

| Experiment | Question |
|------------|----------|
| D-001 | Morphological disambiguation under quantization |
| D-002 | Syntactic parsing degradation by language |
| D-003 | Subword-morpheme alignment analysis |
| D-004 | Joint vs pipeline robustness |
| D-005 | Morphological feature circuits |

### Connection to Other Tracks

| Track | Connection |
|-------|------------|
| Track A | Outlier weights may be in morphology-processing circuits |
| Track B | Morphological probes complement POS probing |
| Track C | Tokenization efficiency directly affects morphological segmentation |

---

## Cross-Track Synthesis

```
                    Training Dynamics
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
   Attention Sinks                    Outlier Weights
   (softmax sum-to-1)                 (magnitude extremes)
        │                                   │
        │         NOT directly causal       │
        └──────────────×───────────────────┘
                                           │
                                           ▼
                          Language-Specific Heads (16.7%)
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
                   High-Resource                      Low-Resource
                   (distributed)                      (sparse)
                          │                                 │
                          ▼                                 ▼
                   Redundant paths                  Critical paths
                          │                                 │
                          ▼                                 ▼
                   Moderate damage                  Severe damage
                          │                                 │
                          └────────────────┬────────────────┘
                                           ▼
                                    r = -0.834
```

---

*Last updated: January 2026*
