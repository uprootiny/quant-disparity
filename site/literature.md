---
layout: page
title: Literature
permalink: /literature/
nav_order: 4
---

# Literature Review

Comprehensive mapping of related research, paper digests, and how prior work connects to our findings.

---

## Paper Digests

### Directly Relevant

#### How Does Quantization Affect Multilingual LLMs?
**Venue:** EMNLP 2024 Findings
**Link:** [arXiv:2407.03211](https://arxiv.org/html/2407.03211v1)

**Key Findings:**
- Non-Latin scripts suffer 1.2-3x more degradation than Latin scripts
- Automatic metrics **underestimate** damage by 10-15%
- Smaller models are more sensitive
- Mathematical reasoning is most vulnerable

**Methods:** W8, W4-g (GPTQ), W8A8, SmoothQuant on Command R, Aya 23

**Models:** 8B to 103B parameters, 23 languages

**Our Connection:** Validates our H1 (language disparity exists) but doesn't explain WHY. Our Track A provides the mechanistic explanation they're missing.

---

#### The Super Weight in Large Language Models
**Venue:** arXiv 2024
**Link:** [arXiv:2411.07191](https://arxiv.org/pdf/2411.07191)

**Key Findings:**
- 0.01% of weights are critical ("super weights")
- A single scalar weight can matter more than 7,000 other outliers
- Super weights are in attention projections (Q, K, V, output)
- Pruning the super weight → orders of magnitude perplexity increase

**Our Connection:** Directly supports our finding that attention projections have the highest kurtosis. LA-ACIQ should identify and preserve these super weights.

---

#### When Attention Sink Emerges in Language Models
**Venue:** ICLR 2025
**Link:** [arXiv:2410.10781](https://arxiv.org/html/2410.10781v1)

**Key Findings:**
- Attention sinks emerge between training steps 1k-2k
- Softmax's sum-to-one constraint is the root cause
- Larger weight decay → more sinks
- Sigmoid attention eliminates sinks up to 1B parameters

**Our Connection:**
- Matches our H7 finding (outliers grow during training)
- But EXP-030 shows sinks ≠ outliers (r=-0.23)
- They're related but distinct phenomena

---

#### Softpick: No Attention Sink, No Massive Activations
**Venue:** arXiv 2025
**Link:** [arXiv:2504.20966](https://arxiv.org/html/2504.20966v1)

**Key Findings:**
- Softpick (rectified softmax) eliminates attention sinks and massive activations
- Achieves 0% sink rate with performance parity
- 95%+ attention sparsity
- Better quantization at 2-bit and 3-bit

**Our Connection:** Potential architectural fix for future models. If sinks → outliers (partially), Softpick could reduce disparity.

---

#### Tokenization Disparities as Infrastructure Bias
**Venue:** arXiv 2025
**Link:** [arXiv:2510.12389](https://arxiv.org/abs/2510.12389)

**Key Findings:**
- 3-5x token inflation for morphologically rich languages
- Latin scripts consistently more efficient
- Creates computational inequity (cost, context length)

**Our Connection:** Validates C-001b (6.17x efficiency gap). But EXP-033 shows fertility doesn't predict degradation—our outlier mechanism is more fundamental.

---

### Background Papers

#### Massive Activations in Large Language Models
**Authors:** Sun et al. 2024

**Key Findings:**
- Coined "massive activations" term
- Linked to attention sink phenomenon
- Problematic for quantization

**Our Connection:** Theoretical foundation for why outlier weights matter.

---

#### SmoothQuant
**Authors:** Xiao et al. 2023

**Key Findings:**
- Migrate quantization difficulty from activations to weights
- Channel-wise scaling
- Enables W8A8 quantization

**Our Connection:** One of the quantization methods we should test LA-ACIQ against.

---

#### GPTQ: Accurate Quantization for Generative Pre-trained Transformers
**Authors:** Frantar et al. 2023

**Key Findings:**
- One-shot weight quantization
- Hessian-based importance scoring
- Works at 4-bit with minimal degradation (on English)

**Our Connection:** Baseline quantization method; doesn't account for language disparity.

---

### Morphology and Multilingual NLP

#### AlephBERT
**Lab:** BIU-NLP (Bar-Ilan University)
**Link:** [cris.biu.ac.il](https://cris.biu.ac.il/en/publications/alephbert-a-hebrew-large-pre-trained-language-model-to-start-off-)

**Key Findings:**
- Hebrew-specific PLM
- Morphological awareness improves all downstream tasks
- Segmentation, POS, NER, sentiment

**Our Connection:** Track D foundation—morphological processing may be a vulnerability point.

---

#### YAP: Joint Morpho-Syntactic Processing
**Lab:** ONLP Lab (Bar-Ilan University)
**Link:** [nlp.biu.ac.il](https://nlp.biu.ac.il/~rtsarfaty/onlp/hebrew/about)

**Key Findings:**
- Joint modeling > pipeline for MRLs
- Hebrew morphology requires syntactic context
- Standard pipelines fail on Semitic languages

**Our Connection:** H-D3 hypothesis—joint models may be more quantization-robust.

---

## Research Gaps We Fill

| Gap in Literature | Our Contribution |
|-------------------|------------------|
| No mechanistic explanation for disparity | Track A: Outlier-disparity correlation (r=-0.834) |
| No language-aware quantization | LA-ACIQ framework |
| No circuit-level analysis | Track B: Language-specific heads (16.7%) |
| No efficiency-fairness connection | Track C: Tokenization gap + negative result |
| No morphology-quantization study | Track D: MRL processing under quantization |

---

## Key Insights from Literature

### 1. The Softmax Problem

From ICLR 2025 and Softpick papers:

```
Softmax sum-to-one constraint
         ↓
To implement "no-op" (skip token), model drives logits to -∞
         ↓
Creates massive activations as byproduct
         ↓
Massive activations → outlier weights
         ↓
Quantization clips outliers
         ↓
Disproportionate damage
```

### 2. The Training Dynamics Link

From attention sink and our H7/H2 findings:

- Outliers **form during training** (1k-143k steps, 82x growth)
- **Weight decay** and **learning rate** affect formation
- **Dropout** may be protective (but complex interaction with other factors)
- **Training instability** (BLOOM's FP32 hack) correlates with outliers

### 3. The Tokenization Red Herring

From our EXP-033:

- Literature suggested tokenization → disparity
- But fertility ≠ degradation (r=-0.07)
- Japanese: 13x tokens but only 16% more degradation
- Models compensate for token overhead
- The **bottleneck is attention outliers**, not tokenization

---

## Citation Map

```
Core Papers
    │
    ├── "Massive Activations" (Sun 2024)
    │       │
    │       ├── "Attention Sink Emerges" (ICLR 2025)
    │       │       │
    │       │       └── "Softpick" (2025)
    │       │
    │       └── "Super Weight" (2024)
    │               │
    │               └── Our Track A findings
    │
    ├── "Quantization Affects Multilingual" (EMNLP 2024)
    │       │
    │       └── Our Track A (mechanistic explanation)
    │
    └── "Tokenization Disparities" (2025)
            │
            └── Our Track C (negative result: not the cause)
```

---

## Recommended Reading Order

For someone new to this research:

1. **Start:** "How Does Quantization Affect Multilingual LLMs?" — establishes the problem
2. **Then:** "Massive Activations" + "Super Weight" — understand outlier phenomenon
3. **Then:** "When Attention Sink Emerges" — training dynamics
4. **Finally:** Our research — mechanistic explanation and cross-track synthesis

---

*Last updated: January 2026*
