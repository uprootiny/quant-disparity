# Model Taxonomy: Outlier Classification

## Full Census (Updated 2026-01-03)

| Model | Size | Type | Max κ | Location | Component | Class | Disparity Risk |
|-------|------|------|-------|----------|-----------|-------|----------------|
| OPT-125M | 125M | Decoder | **562** | Layer 1 | attn.out_proj | HEAVY | HIGH |
| BLOOM-560M | 560M | Decoder | **504** | Layer 22 | attn.query | HEAVY | HIGH (confirmed r=-0.834) |
| GPT-2-small | 124M | Decoder | **201** | Layer 11 | attn.c_proj | HEAVY | HIGH |
| mT5-small | 300M | Enc-Dec | 45 | Decoder 2-5 | attn.out | Moderate | MEDIUM |
| BERT-Tiny | 4M | Encoder | 30 | — | — | Moderate | MEDIUM |
| Pythia-410M | 410M | Decoder | 14 | — | — | Mild | LOW |
| Pythia-70M | 70M | Decoder | 12 | — | — | Mild | LOW |
| DistilmBERT | 135M | Encoder | 11 | — | — | Mild | LOW |
| XLM-R-base | 270M | Encoder | 10 | — | — | Mild | LOW |
| Pythia-160M | 160M | Decoder | 9 | — | — | Mild | LOW |
| XGLM-564M | 564M | Decoder | 2 | — | — | Gaussian | NONE |
| Tiny-GPT2 | 2M | Decoder | 1 | — | — | Gaussian | NONE |

---

## Classification Thresholds

```
HEAVY:     max κ > 50    → HIGH disparity risk
Moderate:  max κ 15-50   → MEDIUM disparity risk
Mild:      max κ 5-15    → LOW disparity risk
Gaussian:  max κ < 5     → NO disparity mechanism
```

---

## By Category

### HEAVY Outliers (Disparity Expected)

| Model | Max κ | Notable |
|-------|-------|---------|
| OPT-125M | 372.0 | Highest observed |
| BLOOM-560M | 164.3 | Confirmed disparity |
| GPT-2-small | 92.3 | Monolingual baseline |

**Common traits:**
- Decoder-only architecture
- Likely minimal dropout during training
- OPT and GPT-2 are Meta/OpenAI models (different from BLOOM)

**Implication:** Outliers are NOT BLOOM-specific. Multiple model families affected.

### Moderate Outliers (Disparity Possible)

| Model | Max κ | Notable |
|-------|-------|---------|
| mT5-small | 44.7 | Encoder-decoder, decoder has outliers |
| BERT-Tiny | 30.0 | Very small encoder |

**Common traits:**
- Mixed architectures
- May show weaker disparity effects

### Mild Outliers (Disparity Unlikely)

| Model | Max κ | Notable |
|-------|-------|---------|
| Pythia-410M | 13.8 | EleutherAI, well-regularized |
| Pythia-70M | 11.6 | Smaller Pythia |
| DistilmBERT | 10.8 | Distilled from BERT |
| XLM-R-base | 9.8 | Meta, multilingual |
| Pythia-160M | 8.6 | Middle Pythia |

**Common traits:**
- Often have dropout (Pythia: 0.0, but different training)
- Encoder architectures common
- Well-studied training procedures

### Gaussian (No Disparity Mechanism)

| Model | Max κ | Notable |
|-------|-------|---------|
| XGLM-564M | 1.9 | Same size as BLOOM, no outliers |
| Tiny-GPT2 | 1.3 | Very small, no outliers |

**Common traits:**
- Dropout present (XGLM: 0.1)
- Stable training (no FP32 hacks)
- Near-perfect Gaussian weight distributions

---

## Outlier Location by Model (Refined EXP-022/024)

| Model | Max κ | Outlier Layers | Component | Pattern |
|-------|-------|----------------|-----------|---------|
| OPT-125M | 562 | 1 (early), 11 (late) | attn.out_proj | BIMODAL |
| BLOOM-560M | 504 | 22, 21, 23 (late) | attn.query | LATE-HEAVY |
| GPT-2-small | 201 | 11 (late), 1 (early) | attn.c_proj | BIMODAL |
| mT5-small | 45 | Decoder 2-5 | attn.out | MID-LAYER |

**Key findings (EXP-022, EXP-024):**

1. **Attention dominates:** All HEAVY models have worst outliers in attention
   - OPT: attn.out_proj (κ=562)
   - BLOOM: attn.query (κ=504)
   - GPT-2: attn.c_proj (κ=201)

2. **Layer position varies:**
   - OPT: Early layers worst (layer 1)
   - BLOOM/GPT-2: Late layers worst

3. **No universal pattern:** Outlier location is model-specific

## Key Findings

### 1. Outliers Are Common

```
HEAVY outliers: 3/12 models (25%)
Some outliers:  6/12 models (50%)
Gaussian:       3/12 models (25%)
```

### 2. Architecture Matters Less Than Training

```
Decoder-only with outliers:  BLOOM, GPT-2, OPT
Decoder-only without:        XGLM, Pythia

Same architecture, different outcome → Training matters
```

### 3. OPT Has Extreme Outliers

```
OPT-125M:  κ = 372 (2.3x BLOOM's 164!)
OPT likely has SEVERE disparity issues if multilingual
```

### 4. Pythia Is Well-Behaved

```
All Pythia models: κ < 15
EleutherAI's training recipe prevents outliers
Worth studying: what did they do differently?
```

---

## Predictions

### Models Expected to Show Disparity (r < -0.5)

1. **OPT-125M, OPT-350M** — If used multilingually
2. **GPT-2** — English-only, but pattern exists
3. **BLOOM family** — Confirmed

### Models Expected to Show NO Disparity (|r| < 0.3)

1. **XGLM** — Confirmed null
2. **Pythia family** — Well-regularized
3. **Tiny models** — No capacity for outliers

### Models Needing Testing

1. **mT5** — Has outliers, but encoder-decoder
2. **XLM-R** — Mild outliers, encoder-only
3. **OPT-1.3B+** — Do outliers scale with size?

---

## Next Steps

1. **Test OPT multilingual degradation** — Expect high disparity
2. **Compare Pythia vs GPT-2 training** — Why different?
3. **Test mT5 with decoder-specific analysis** — Outliers in decoder only

---

*Taxonomy v1.0 — 2026-01-03*
*Based on analysis of 12 models*
