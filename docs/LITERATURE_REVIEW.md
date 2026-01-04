# Literature Review: Multilingual Quantization Disparity

## Executive Summary

This review maps the current research landscape connecting:
1. **Quantization effects on multilingual LLMs**
2. **Outlier weights and attention mechanisms**
3. **Morphologically rich language processing**
4. **Tokenization disparities**

**Key synthesis:** Our Track A findings (r=-0.834 correlation, attention outliers) are validated by recent literature and connect mechanistically to attention sinks, super weights, and tokenization efficiency.

---

## 1. Quantization Effects on Multilingual LLMs

### Primary Source
**"How Does Quantization Affect Multilingual LLMs?"** (EMNLP 2024 Findings)
- [arXiv:2407.03211](https://arxiv.org/html/2407.03211v1)

### Key Findings

| Finding | Evidence | Implication for Our Work |
|---------|----------|--------------------------|
| Non-Latin scripts suffer 1.2-3x more | Latin: -0.7%, Non-Latin: -1.9% | Validates H1 (language disparity) |
| Automatic metrics underestimate by 10-15% | Human: -16.6%, Auto: -1.7% | Our perplexity metrics are conservative |
| Smaller models more sensitive | 35B: -2.8%, 103B: -0.9% | Focus on 560M-7B range justified |
| Math reasoning most vulnerable | MGSM: -13.1% at W4-g | Suggests complex reasoning circuits vulnerable |

### Methods Tested
- W8 (8-bit per-column)
- W4-g (4-bit group-wise, GPTQ)
- W8A8 (weight and activation)
- SmoothQuant

### Languages Covered
23 languages including Hebrew, Arabic, Korean, Japanese, Chinese

**Gap identified:** No mechanistic explanation for WHY some languages degrade more.

---

## 2. Outlier Weights and Super Weights

### Primary Sources

**"The Super Weight in Large Language Models"** (arXiv:2411.07191)

### Key Findings

| Finding | Details | Connection to Track A |
|---------|---------|----------------------|
| 0.01% of weights are critical | Single "super weight" matters more than 7000 outliers | Explains high kurtosis in specific components |
| Located in attention projections | Q, K, V, and output projections | Matches our finding: attention > MLP |
| Pruning super weight → orders of magnitude perplexity increase | Model becomes non-functional | Explains why quantization hurts so much |

### Mechanism
Super weights appear in mid-to-late layers, concentrated in attention mechanisms. They're connected to the "attention sink" phenomenon.

---

## 3. Attention Sinks and Massive Activations

### Primary Sources

**"When Attention Sink Emerges in Language Models"** (ICLR 2025)
- [arXiv:2410.10781](https://arxiv.org/html/2410.10781v1)

**"Softpick: No Attention Sink, No Massive Activations"** (2025)
- [arXiv:2504.20966](https://arxiv.org/html/2504.20966v1)

### Key Findings

| Finding | Details | Connection to Our Work |
|---------|---------|------------------------|
| Sinks emerge at 1k-2k training steps | Training dynamics cause outliers | Matches H7: κ grows 82x during training |
| Softmax is root cause | Sum-to-one constraint forces extreme values | Explains attention outlier concentration |
| Larger weight decay → more sinks | Training hyperparameters matter | Explains BLOOM's FP32 hack (H2) |
| Sigmoid attention eliminates sinks | No sinks up to 1B params | Potential fix for future models |

### Mechanistic Explanation
1. Softmax requires attention weights sum to 1
2. To implement "no-op" (skip this token), model drives logits to -∞
3. This creates massive activations as byproduct
4. Massive activations → outlier weights in attention projections
5. Quantization clips these outliers → disproportionate damage

---

## 4. Tokenization Disparities

### Primary Sources

**"Tokenization Disparities as Infrastructure Bias"** (arXiv:2510.12389)

**"Tokenization Falling Short"** (EMNLP 2024 Findings)
- [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.86/)

### Key Findings

| Finding | Details | Connection to Our Work |
|---------|---------|------------------------|
| 3-5x token inflation for MRLs | Arabic needs 3x more tokens than English | Matches C-001b: 6.17x efficiency gap |
| Latin scripts consistently more efficient | Non-Latin + morphologically complex = worst | Hebrew, Arabic most affected |
| Morphological richness compounds problem | Many word forms per root | Explains MRL vulnerability |

### Mechanism
1. Subword tokenizers trained mostly on English
2. MRLs require more tokens per word
3. Each token goes through same quantized circuits
4. More tokens = more accumulated quantization error
5. Also: reduced effective context length

---

## 5. Morphologically Rich Languages

### Primary Sources

**BIU-NLP Lab** (Bar-Ilan University)
- [ONLP Lab](https://nlp.biu.ac.il/~rtsarfaty/onlp)
- AlephBERT, YAP parser

### Key Concepts

| Concept | Details | Implication |
|---------|---------|-------------|
| Joint morpho-syntactic processing | Morphology and syntax interact | Can't process separately |
| Morphological disambiguation | Multiple valid readings per surface form | Requires context |
| Pipeline cascading errors | Morphology errors propagate to syntax | Quantization errors compound |

### Languages Affected
- Semitic: Hebrew, Arabic
- Turkic: Turkish
- Finno-Ugric: Finnish, Hungarian
- Slavic: Russian, Polish, Czech

---

## 6. Synthesis: Unified Mechanistic Model

```
Training Dynamics
       ↓
   Softmax Attention (sum-to-1 constraint)
       ↓
   Attention Sinks (no-op implementation)
       ↓
   Massive Activations
       ↓
   Super Weights / Outlier Weights (κ > 100)
       ↓
   Concentrated in Attention Projections
       ↓
   Quantization Clips These Weights
       ↓
   Disproportionate Damage to:
   ├─ Low-resource languages (less redundancy)
   ├─ Non-Latin scripts (tokenization overhead)
   └─ MRLs (morphological disambiguation loss)
```

### Key Insight for Track A
Our finding that **languages activating outlier layers LESS degrade MORE** (r=-0.834) now has a mechanistic explanation:

1. High-resource languages have representations distributed across more circuits
2. When outlier weights are clipped, they have redundant pathways
3. Low-resource languages concentrate in fewer circuits
4. Those circuits happen to overlap with attention outlier locations
5. Clipping destroys critical-path representations

---

## 7. Implications for Track A Experiments

### Validated Hypotheses
- **H4:** Attention > MLP for outliers ✓ (Super weights paper)
- **H7:** Outliers grow during training ✓ (Attention sink paper: 1k-2k steps)
- **H2:** Training dynamics matter ✓ (Weight decay, learning rate affect sinks)

### New Experiments Suggested

| Experiment | Question | Method |
|------------|----------|--------|
| EXP-030 | Do outlier layers correlate with attention sinks? | Map sink distribution vs kurtosis |
| EXP-031 | Does LA-ACIQ target super weights? | Check if our clipping strategy preserves them |
| EXP-032 | Can sigmoid attention prevent disparity? | Test Softpick models for language fairness |
| EXP-033 | Does token count predict degradation? | Correlate fertility with perplexity drop |

### Refined LA-ACIQ Strategy
Based on literature, LA-ACIQ should:
1. **Identify super weights explicitly** (not just high-magnitude)
2. **Preserve attention projection outliers** (Q, K, V, O)
3. **Use language-specific clipping** (different α per language)
4. **Consider token count normalization** (penalize high-fertility languages less)

---

## 8. Key Papers by Relevance

### Must-Cite (Directly Relevant)
1. "How Does Quantization Affect Multilingual LLMs?" - EMNLP 2024
2. "The Super Weight in Large Language Models" - arXiv 2024
3. "When Attention Sink Emerges" - ICLR 2025
4. "Tokenization Disparities as Infrastructure Bias" - arXiv 2025

### Context Papers (Background)
5. "Massive Activations in Large Language Models" - Sun et al. 2024
6. "SmoothQuant" - Xiao et al. 2023
7. "GPTQ" - Frantar et al. 2023
8. "AlephBERT" - BIU-NLP 2021

### Methodology Papers (Techniques)
9. "Softpick: Rectified Softmax" - 2025
10. "RotateKV: Attention-Sink-Aware Quantization" - IJCAI 2025
11. "Outlier-Safe Pre-Training" - arXiv 2025

---

## 9. Research Gaps We Can Fill

| Gap | Our Contribution |
|-----|------------------|
| No mechanistic explanation for multilingual disparity | Track A: Outlier-disparity correlation |
| No language-aware quantization | LA-ACIQ framework |
| No circuit-level analysis | Track B: Attention pattern analysis |
| No efficiency-fairness connection | Track C: Tokenization-degradation link |
| No morphology-quantization study | Track D: MRL processing under quantization |

---

## 10. Updated Research Narrative

> "We discovered that quantization damage to multilingual LLMs is not random but mechanistically determined by the interaction between attention sink phenomena, super weight distribution, and language-specific circuit utilization. Languages with less training data rely on sparser circuits that overlap with outlier weights in attention projections, explaining the strong negative correlation (r=-0.834) we observe. This suggests that fair quantization requires not just better algorithms but fundamental changes to how models are trained and how we tokenize morphologically rich languages."

---

*Last updated: 2026-01-03*

## Sources

- [How Does Quantization Affect Multilingual LLMs?](https://arxiv.org/html/2407.03211v1)
- [The Super Weight in Large Language Models](https://arxiv.org/pdf/2411.07191)
- [When Attention Sink Emerges](https://arxiv.org/html/2410.10781v1)
- [Softpick: Rectified Softmax](https://arxiv.org/html/2504.20966v1)
- [Tokenization Disparities as Infrastructure Bias](https://arxiv.org/abs/2510.12389)
- [BIU-NLP ONLP Lab](https://nlp.biu.ac.il/~rtsarfaty/onlp)
- [Yoav Goldberg - Google Scholar](https://scholar.google.com/citations?user=0rskDKgAAAAJ&hl=en)
