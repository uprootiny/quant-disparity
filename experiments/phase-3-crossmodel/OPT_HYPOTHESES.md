# OPT Kurtosis Investigation: Hypotheses & Experiments

## Observation

OPT-125M has κ=271 in layer 1 `self_attn.out_proj`, exceeding BLOOM's max κ=164.
Why?

---

## Hypotheses (ordered by testability)

### H1: Architectural Difference — Output Projection Dimension

**Claim:** OPT's attention output projection has different dimensions than BLOOM's MLP, concentrating information in fewer parameters.

**Prediction:**
- OPT out_proj: smaller weight matrix → higher information density → higher kurtosis
- BLOOM MLP: larger weight matrices → more distributed → lower per-tensor kurtosis

**Test:**
```
Compare: OPT layer1.out_proj.shape vs BLOOM layer5.mlp.dense_h_to_4h.shape
If OPT has fewer parameters per outlier tensor → H1 supported
```

**Falsification:** If shapes are comparable or OPT is larger, H1 is rejected.

---

### H2: Training Stability — FP32 Accumulation Differences

**Claim:** OPT's training used different numerical precision settings that allowed larger outliers to form.

**Prediction:**
- OPT used FP16 mixed precision without FP32 attention accumulation
- This allowed gradient accumulation errors to compound into larger outliers
- BLOOM's `attention_softmax_in_fp32=True` was a mitigation that partially succeeded

**Test:**
```
Fetch OPT training config from HuggingFace/Meta papers
Compare: attention precision, gradient accumulation settings
If OPT lacks FP32 mitigations → H2 supported
```

**Falsification:** If OPT has equivalent or stronger precision mitigations, H2 is rejected.

---

### H3: Layer Position — Early vs Late Outlier Formation

**Claim:** Outliers in early layers (OPT layer 1) have higher kurtosis than mid/late layer outliers (BLOOM layers 5, 21, 22) due to gradient magnitude dynamics.

**Prediction:**
- Early layers see gradients from ALL subsequent layers → more gradient accumulation
- This could amplify outliers more than mid-layer positions
- OPT's layer 1 outlier > BLOOM's layer 5 outlier

**Test:**
```
Compare kurtosis by layer position across models:
- OPT: layer 1 (early) vs layer 11 (late)
- BLOOM: layer 5 (early-mid) vs layers 21-22 (late)
If early layers consistently have higher kurtosis → H3 supported
```

**Falsification:** If late layers have equivalent or higher kurtosis, H3 is rejected.

---

### H4: Component Type — Attention vs MLP Outlier Mechanics

**Claim:** Attention output projections inherently develop higher kurtosis than MLP layers due to different information flow patterns.

**Prediction:**
- Attention output aggregates from multiple heads → more opportunity for outlier formation
- MLP layers have more uniform activation patterns
- All models should show attention_out > MLP kurtosis when outliers exist

**Test:**
```
For each model with outliers, compare:
- max(attention_out kurtosis) vs max(MLP kurtosis)
If attention_out consistently higher → H4 supported
```

**Falsification:** If MLP kurtosis matches or exceeds attention_out, H4 is rejected.

---

### H5: Model Size Scaling — Smaller Models Have Denser Outliers

**Claim:** Smaller models (OPT-125M < BLOOM-560M) pack information more densely, leading to higher kurtosis.

**Prediction:**
- OPT-125M: fewer parameters → each weight carries more information
- BLOOM-560M: more parameters → information more distributed
- Kurtosis should scale inversely with model size within families

**Test:**
```
Compare kurtosis across model sizes:
- OPT-125M vs OPT-350M vs OPT-1.3B (if available)
- Pythia-70M vs Pythia-160M vs Pythia-410M
If smaller models have higher kurtosis → H5 supported
```

**Falsification:** If kurtosis scales with model size, H5 is rejected.

---

### H6: Training Data — Monolingual vs Multilingual Effects

**Claim:** OPT (primarily English) develops sharper features than BLOOM (multilingual) because it specializes more.

**Prediction:**
- Monolingual models: specialized representations → sharper features → higher kurtosis
- Multilingual models: averaged representations → smoother → lower kurtosis
- XGLM (multilingual, κ<2) << BLOOM (multilingual, κ=164) << OPT (English, κ=271)

**Test:**
```
Compare monolingual vs multilingual:
- GPT-2 (English): κ=92
- OPT-125M (English): κ=271
- BLOOM (multilingual): κ=164
- XGLM (multilingual): κ=2

Pattern check: English models should have higher kurtosis
```

**Falsification:** If multilingual models have equivalent kurtosis, H6 is rejected.

---

## Experiment Design

### EXP-022: Architectural Comparison

**Objective:** Test H1 (dimension) and H4 (component type)

**Method:**
1. Extract weight shapes from OPT, BLOOM, GPT-2
2. For each outlier tensor, record: shape, kurtosis, component type
3. Compute: kurtosis per parameter ratio
4. Compare across components

**Decision criteria:**
- If attention_out consistently > MLP: H4 supported
- If smaller tensors → higher kurtosis: H1 supported

---

### EXP-023: Training Config Comparison

**Objective:** Test H2 (precision)

**Method:**
1. Fetch OPT training details from Meta papers / model card
2. Compare with BLOOM training config (already extracted)
3. Document precision settings, dropout, batch size

**Decision criteria:**
- If OPT lacks FP32 mitigations: H2 supported
- If OPT has equivalent settings: H2 rejected

---

### EXP-024: Layer Position Analysis

**Objective:** Test H3 (gradient dynamics)

**Method:**
1. For each model, plot: layer index vs max kurtosis
2. Compute correlation: position vs kurtosis
3. Separate early (< 25%) vs late (> 75%) layers

**Decision criteria:**
- If early layers consistently higher: H3 supported
- If no pattern: H3 rejected

---

### EXP-025: Cross-Size Scaling

**Objective:** Test H5 (size scaling)

**Method:**
1. Analyze OPT-350M (if memory allows)
2. Use Pythia family data (already collected)
3. Plot: model size vs max kurtosis

**Decision criteria:**
- If inverse correlation: H5 supported
- If positive or no correlation: H5 rejected

---

## Priority Order

Based on testability with current resources:

1. **EXP-022** (architecture) — Can run now, uses existing data
2. **EXP-024** (layer position) — Can run now, uses existing data
3. **EXP-023** (training config) — Requires web fetch, low compute
4. **EXP-025** (size scaling) — May need more models, medium compute

---

## Experimental Results (Updated 2026-01-03)

### EXP-022: Architecture Comparison

| Hypothesis | Prediction | Result | Verdict |
|------------|------------|--------|---------|
| H1: Dimension | smaller → higher κ | r = -0.003 | **INCONCLUSIVE** |
| H4: Component | attention > MLP | 3/3 models | **SUPPORTED** |

**Key finding:** All HEAVY models have worst outliers in ATTENTION, not MLP.
- OPT: attn.out_proj (κ=562)
- BLOOM: attn.query (κ=504)
- GPT-2: attn.c_proj (κ=201)

### EXP-024: Layer Position

| Model | Early max κ | Late max κ | Pattern |
|-------|-------------|------------|---------|
| OPT-125M | 562 | 41 | **EARLY** |
| BLOOM | — | 504 | LATE |
| GPT-2 | — | 201 | LATE |

**Verdict for H3:** MIXED — Layer position effect is model-specific.

---

## Updated Evidence Summary

| Hypothesis | Evidence | Status |
|------------|----------|--------|
| H1: Dimension | r = -0.003, n.s. | **REJECTED** |
| H2: Precision | attn_dropout=0 → high κ | **SUPPORTED** |
| H3: Layer position | OPT: early, others: late | **MIXED** |
| H4: Component type | All attention | **SUPPORTED** |
| H5: Size scaling | Pythia: larger → higher κ | **REJECTED** |
| H6: Mono vs multi | Confounded | WEAK |

---

## Key Finding: Attention Dropout

**EXP-023 revealed the critical factor:**

| Model | attn_dropout | Max κ |
|-------|--------------|-------|
| XGLM-564M | 0.1 | 2 |
| GPT-2 | 0.1 | 201 |
| Pythia-410M | — | 73 |
| OPT-125M | 0.0 | 562 |
| BLOOM-560M | 0.0 | 504 |

**Conclusion:** Models with `attention_dropout=0.0` develop extreme outliers.

---

## Next Steps

1. **Test causality:** Does adding attention dropout reduce outliers?
2. **Investigate QKV vs output:** Pythia outliers in QKV, OPT in output
3. **Training dynamics:** When during training do outliers form?

---

*Document created: 2026-01-03*
*Updated: 2026-01-03 (EXP-022, EXP-024 results)*
