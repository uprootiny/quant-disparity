# Experimental Protocols

## From Literature

### Protocol 1: Quantization Degradation Measurement
*Source: "How Does Quantization Affect Multilingual LLMs?" (EMNLP 2024)*

1. Select representative texts per language (50-100 tokens each)
2. Compute baseline perplexity in FP32/FP16
3. Apply quantization (INT8, INT4, etc.)
4. Compute post-quantization perplexity
5. Calculate degradation: `(PPL_quant - PPL_base) / PPL_base * 100`

**Key metrics:**
- Absolute perplexity
- Relative degradation (%)
- Cross-language variance

### Protocol 2: Outlier Detection
*Source: "Super Weight in Large Language Models" (arXiv:2411.07191)*

1. Load model weights
2. For each weight tensor:
   - Compute magnitude distribution
   - Identify top-k% by absolute value
   - Compute sparsity index: `||top-k||_2 / ||all||_2`
3. Classify as SPARSE (>0.5) or DENSE (<0.2)

**Key metrics:**
- Super Weight Ratio (SWR): `max(|w|) / mean(|w|)`
- Magnitude concentration in top 0.01%, 0.1%, 1%

### Protocol 3: Language-Specific Layer Analysis
*Source: "When Attention Sink Emerges in LLMs" (ICLR 2025)*

1. For each language in test set:
   - Forward pass with hooks on all layers
   - Record activation magnitudes per layer
2. Identify critical layers: top-k by activation variance
3. Compare critical layers across languages

**Key metrics:**
- Per-language layer importance ranking
- Cross-language layer divergence

### Protocol 4: Preservation Intervention
*Source: Our methodology*

1. Baseline: FP32 perplexity per language
2. Establish disparity ratio at k=0% preservation
3. For k in [1, 5, 10, 20, 50]:
   - Preserve top-k% weights by magnitude
   - Quantize remaining to INT4
   - Measure perplexity
   - Compute disparity ratio
4. Analyze correlation(k, disparity)

**Success criteria:**
- Negative correlation: preservation helps
- Positive correlation: preservation hurts
- No correlation: inconclusive

---

## Experimental Constants

### Languages Tested

| Code | Language | Resource Level | Script | Token Fertility |
|------|----------|----------------|--------|-----------------|
| en | English | 1.00 | Latin | 1.0x |
| de | German | 0.85 | Latin | 1.2x |
| fr | French | 0.80 | Latin | 1.15x |
| zh | Chinese | 0.50 | Han | 2.1x |
| he | Hebrew | 0.15 | Hebrew | 3.5x |
| ar | Arabic | 0.25 | Arabic | 3.2x |
| ru | Russian | 0.40 | Cyrillic | 2.0x |

### Resource Level Classification

- **High Resource (HR)**: level > 0.5 (en, de, fr)
- **Low Resource (LR)**: level <= 0.5 (zh, he, ar, ru)

### Standard Test Texts

```
en: "The quick brown fox jumps over the lazy dog and runs through the forest."
de: "Der schnelle braune Fuchs springt über den faulen Hund und rennt durch den Wald."
fr: "Le renard brun rapide saute par-dessus le chien paresseux et court dans la forêt."
zh: "敏捷的棕色狐狸跳过懒狗，穿过森林寻找食物。"
he: "השועל החום המהיר קופץ מעל הכלב העצלן ורץ ביער בחיפוש אחר אוכל."
ar: "الثعلب البني السريع يقفز فوق الكلب الكسول ويجري عبر الغابة."
```

### Quantization Parameters

| Format | Bits | Range | Scale Formula |
|--------|------|-------|---------------|
| INT8 | 8 | [-128, 127] | abs_max / 127 |
| INT4 | 4 | [-8, 7] | abs_max / 7 |
| INT3 | 3 | [-4, 3] | abs_max / 3 |

---

## Replication Checklist

For any experiment to be considered valid:

- [ ] Model loaded fresh (or state restored)
- [ ] Same tokenizer settings across languages
- [ ] Same max_length for all texts
- [ ] Temperature = 0 (deterministic)
- [ ] Results saved with timestamp and random seed
- [ ] At least 3 repetitions for statistical validity

---

*Last updated: 2026-01-04*
