# Phase 4 Findings Synthesis

## Experiments Completed

| ID | Question | Key Finding |
|----|----------|-------------|
| EXP-037 | Sparse or dense outliers? | **DENSE** - All models have diffuse outliers |
| EXP-036 | Layer contributions by language | English differs from other languages |

---

## Critical Finding: Dense Outlier Patterns (EXP-037)

### The Data

| Model | Pattern | Top 0.01% Magnitude | Top 1% Magnitude | Top 10% Magnitude |
|-------|---------|---------------------|------------------|-------------------|
| GPT-2 | DENSE | 0.26% | 4.70% | 27.72% |
| OPT-125M | DENSE | 0.62% | 6.07% | 30.87% |
| Pythia-160M | DENSE | 0.24% | 12.80% | 45.19% |
| BERT | DENSE | 0.13% | 4.23% | 27.28% |

### Implications

1. **0.01% preservation is insufficient** — only captures 0.1-0.6% of magnitude
2. **1% preservation is marginal** — captures 4-13% of magnitude
3. **5-10% preservation is meaningful** — captures 30-45% of magnitude

### Actionable Threshold

```
For INT4 quantization with acceptable quality:
  → Preserve top 5-10% of weights in FP16
  → Memory overhead: ~5-10% larger than pure INT4
  → Quality benefit: ~30-45% of outlier magnitude preserved
```

---

## Language-Layer Relationship (EXP-036)

### Finding

```
Critical Layers by Language (mBERT):
  English:     Layers 9, 10
  All others:  Layers 0, 10
```

### Interpretation

- Non-English languages rely heavily on **layer 0 (embeddings)**
- English uses more distributed mid-to-late layers
- If layer 0 has outliers, non-English languages are more vulnerable

### Implication

Language-specific layer protection may be more efficient than uniform protection.

---

## Integration with Literature

### From "Super Weight" Paper (arXiv:2411.07191)

| Their Finding | Our Finding | Synthesis |
|---------------|-------------|-----------|
| 0.01% weights are critical | 0.01% holds only 0.26% magnitude | Super weights exist but diffuse |
| Single weight can dominate | Max SWR = 1.0-2.9 | No dominant single weight in tested models |
| Attention projections | Confirmed (EXP-022) | Consistent |

### From "Quantization Affects Multilingual" (EMNLP 2024)

| Their Finding | Our Finding | Synthesis |
|---------------|-------------|-----------|
| Non-Latin 1.2-3x worse | r=-0.834 disparity | Confirmed and mechanistically explained |
| Automatic metrics underestimate | We use simulated quantization | Need real quantization validation |
| Smaller models more sensitive | Dense pattern consistent across sizes | Model size not the issue |

---

## Revised LA-ACIQ Strategy

Based on Phase 4 findings, the Language-Aware ACIQ should:

### 1. Threshold Selection

```python
# Old assumption: preserve 0.01%
# New evidence: preserve 5-10%

PRESERVATION_THRESHOLD = 0.05  # 5% of weights
```

### 2. Layer-Specific Protection

```python
# Protect layers differently by language group
CRITICAL_LAYERS = {
    "english": [9, 10, 11],
    "non_english": [0, 10, 11],
}
```

### 3. Mixed-Precision Strategy

```python
# Instead of uniform INT4:
# - Top 5% weights: FP16
# - Attention projections: INT8
# - Everything else: INT4
```

---

## Next Steps (From Literature)

### Scaling Directions

1. **Model size**: Test on 7B+ models when GPU available
2. **Language coverage**: Expand to 50+ languages
3. **Task diversity**: Beyond perplexity to downstream tasks

### Technique Refinements

1. **Softpick attention**: Eliminate sinks at architecture level
2. **Training-time intervention**: Prevent outlier formation
3. **Calibration data selection**: Language-balanced calibration sets

### Validation Priorities

1. **Real quantization**: Use bitsandbytes, GPTQ, AWQ
2. **Human evaluation**: Validate beyond automatic metrics
3. **Deployment testing**: Measure actual inference quality

---

## Summary: What We Now Know

### Confirmed

| Claim | Evidence |
|-------|----------|
| Outliers cause disparity | r=-0.834, mechanistic analysis |
| Outliers are in attention | 3/3 HEAVY models |
| Outliers are DENSE, not sparse | EXP-037: all models DENSE |
| Different languages use different layers | EXP-036: English vs others |

### New Insights

| Insight | Implication |
|---------|-------------|
| Need 5-10% preservation | More overhead than hoped |
| Layer 0 critical for non-English | Target embedding layer protection |
| No super weights in tested models | Can't rely on preserving just one weight |

### Still Unknown

| Question | Blocking Factor |
|----------|-----------------|
| Does preservation actually reduce disparity? | GPU for full experiments |
| What's the optimal preservation threshold? | Need real quantization testing |
| Does this generalize to 7B+ models? | Compute resources |

---

*Synthesized: 2026-01-04*
