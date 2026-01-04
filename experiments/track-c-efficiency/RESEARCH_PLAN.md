# Track C: Efficiency-Fairness Tradeoffs

## Target Lab
**Roy Schwartz Lab** — Hebrew University of Jerusalem (HUJI)

## Research Question
> Do efficient NLP techniques (distillation, pruning, quantization) amplify language disparities?

## Motivation

From Green AI (Schwartz et al., 2020):
- Computations doubling every few months
- Carbon footprint is environmentally unfriendly
- Financial cost excludes researchers from emerging economies

**Gap:** Efficiency research focuses on English; fairness implications for other languages unexplored.

---

## Core Hypothesis

**H-C1:** Efficiency techniques disproportionately harm low-resource languages.

**Prediction:** Performance gap (high-resource vs low-resource) increases after:
- Distillation
- Pruning
- Quantization

---

## Experiment Series

### C-001: Distillation Disparity

**Question:** Does distillation amplify language gaps?

**Method:**
1. Compare BERT-base-multilingual vs DistilmBERT on:
   - High-resource: English, German, French
   - Low-resource: Swahili, Yoruba, Amharic
2. Measure: accuracy drop per language
3. Compute: disparity ratio = (drop_low / drop_high)

**Prediction:** disparity_ratio > 1.5

**Data:** XNLI, WikiANN NER, or similar multilingual benchmarks

---

### C-002: Pruning Effect on Language Performance

**Question:** Does structured pruning hurt low-resource languages more?

**Method:**
1. Apply magnitude pruning to mBERT at 30%, 50%, 70% sparsity
2. Evaluate on same language split as C-001
3. Plot: sparsity vs accuracy for each language group

**Prediction:** Low-resource languages hit performance floor earlier

---

### C-003: Quantization Fairness (Connects to Track A)

**Question:** Does INT8/INT4 quantization widen language gaps?

**Method:**
1. Quantize BLOOM-560M to INT8, INT4
2. Measure perplexity degradation per language
3. Correlate with: resource level, script complexity, tokenizer fertility

**Prediction:** Degradation correlates with resource level (r > 0.5)

**Note:** This connects to our Track A findings on outlier weights.

---

### C-004: Carbon Cost Per Language

**Question:** What is the compute cost to achieve parity across languages?

**Method:**
1. For each language, find minimum model size for threshold accuracy
2. Compute FLOPs required
3. Calculate: carbon_cost_per_language

**Output:** Fairness-adjusted efficiency metric

---

### C-005: Vocabulary Efficiency Gap

**Question:** Do tokenizer inefficiencies compound with model compression?

**Method:**
1. Measure fertility (tokens/word) across languages
2. Apply compression (distill/prune/quantize)
3. Test if high-fertility languages suffer more

**Connects to:** Schwartz Lab's "Vocab Diet" research

---

## Metrics

| Metric | Definition |
|--------|------------|
| Disparity Ratio | performance_drop_low / performance_drop_high |
| Fairness Gap | max(accuracy) - min(accuracy) across languages |
| Efficiency-Fairness Score | accuracy / (FLOPs × disparity_ratio) |
| Carbon Per Parity | CO2 to achieve equal performance across languages |

---

## Datasets

| Dataset | Task | Languages |
|---------|------|-----------|
| XNLI | NLI | 15 languages |
| WikiANN | NER | 100+ languages |
| FLORES | Translation | 100+ languages |
| Tatoeba | Retrieval | 100+ languages |

---

## Tools

- HuggingFace Transformers (distillation, quantization)
- Neural Network Intelligence (pruning)
- CodeCarbon (carbon tracking)

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Disparity ratio significant | > 1.5 with p < 0.05 |
| Correlation with resources | r > 0.5 |
| Novel metric proposed | Efficiency-Fairness Score |
| Actionable finding | Specific technique recommendations |

---

## Timeline (CPU-feasible)

```
Week 1: C-001 (DistilBERT vs BERT)
Week 2: C-002 (Pruning sweep)
Week 3: C-003 (Quantization, connects to Track A)
Week 4: C-004, C-005 (Carbon, vocabulary)
Week 5: Analysis and writeup
```

---

## Publication Target

**Venue:** EMNLP 2027 (Green NLP track) or ACL 2027

**Angle:** "The Hidden Cost of Efficiency: How Model Compression Amplifies Language Disparities"

---

## Connection to Track A

Track A (Soudry): WHY quantization hurts some languages (outlier weights)
Track C (Schwartz): HOW MUCH and policy implications

Combined story: Mechanism + Impact + Solution

---

*Created: 2026-01-03*
