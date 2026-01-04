# Track D: Syntax and Morphology in Quantized Models

## Target Lab
**Yoav Goldberg Lab** — Bar-Ilan University (BIU-NLP)

## Research Question
> How does quantization affect syntactic and morphological processing, especially in morphologically rich languages (MRLs)?

## Motivation

From Goldberg/BIU-NLP research:
- Morphologically rich languages (Hebrew, Arabic, Turkish) require joint morpho-syntactic processing
- Standard NLP pipelines fail on MRLs due to morphology-syntax interaction
- AlephBERT shows morphological awareness improves all downstream Hebrew NLP tasks
- YAP parser demonstrates joint modeling superiority over pipeline approaches

**Gap:** No analysis of how quantization affects morphological processing circuits.

---

## Core Hypotheses

**H-D1:** Quantization disproportionately degrades morphological disambiguation in MRLs.

**H-D2:** Morphological complexity correlates with quantization sensitivity.

**H-D3:** Joint morpho-syntactic models are more robust to quantization than pipeline models.

**H-D4:** Subword tokenization compounds quantization damage in MRLs.

---

## Experiment Series

### D-001: Morphological Disambiguation Under Quantization

**Question:** Does quantization hurt morphological disambiguation accuracy?

**Method:**
1. Load mBERT/AlephBERT at FP16, INT8, INT4
2. Evaluate on Hebrew/Arabic morphological disambiguation tasks
3. Measure accuracy drop by morphological feature (gender, number, tense)
4. Compare MRL vs non-MRL languages

**Datasets:** Hebrew UD Treebank, Arabic UD Treebank

**Prediction:** MRLs show 2-3x higher accuracy drop than English.

---

### D-002: Syntactic Parsing Degradation

**Question:** How does quantization affect dependency parsing?

**Method:**
1. Run dependency parsing on UD treebanks at various bit widths
2. Measure UAS/LAS drop per language
3. Correlate with morphological complexity index
4. Analyze error patterns (attachment vs label errors)

**Prediction:** MRLs have more attachment errors due to morphological ambiguity propagation.

---

### D-003: Subword-Morpheme Alignment Analysis

**Question:** Do quantization errors correlate with poor subword-morpheme alignment?

**Method:**
1. Compute subword-to-morpheme mapping quality per language
2. Correlate alignment quality with quantization degradation
3. Test hypothesis: poor alignment → more quantization damage

**Connects to:** C-001b tokenizer efficiency findings (6.17x gap)

---

### D-004: Joint vs Pipeline Robustness

**Question:** Are joint morpho-syntactic models more quantization-robust?

**Method:**
1. Compare YAP-style joint model vs pipeline approach
2. Quantize both at INT8/INT4
3. Measure relative performance drop
4. Analyze where errors compound in pipeline

**Prediction:** Pipeline shows cascading errors under quantization.

---

### D-005: Morphological Feature Circuits

**Question:** Which attention heads encode morphological features?

**Method:**
1. Use probing classifiers for morphological features (gender, number, case)
2. Identify heads with high feature selectivity
3. Measure quantization impact on these specific heads
4. Connect to Track A outlier location findings

**Connects to:** B-001 attention patterns, A-* outlier analysis

---

## Connection to Other Tracks

| Track | Connection |
|-------|------------|
| Track A | Outlier weights may be in morphology-processing circuits |
| Track B | Morphological probes complement POS probing |
| Track C | Tokenization efficiency directly affects morphological segmentation |

---

## Key Literature

### Goldberg Lab
- AlephBERT: Hebrew PLM with morphological awareness
- YAP: Joint morpho-syntactic parsing
- Deverbal noun analysis

### Quantization + Multilingual
- "How Does Quantization Affect Multilingual LLMs?" (EMNLP 2024)
- "Super Weight" paper (arXiv:2411.07191)
- "When Attention Sink Emerges" (ICLR 2025)

### Tokenization
- "Tokenization Disparities as Infrastructure Bias" (2025)
- SIGMORPHON 2024 morphological segmentation

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| MRL accuracy drop ratio | > 2x vs non-MRL |
| Morphological feature correlation | r > 0.5 with quantization damage |
| Joint model advantage | > 10% less degradation than pipeline |

---

*Created: 2026-01-03*
