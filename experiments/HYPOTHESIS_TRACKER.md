# Hypothesis Tracker

## Active Hypotheses

| ID | Hypothesis | Prediction | Experiment | Status |
|----|------------|------------|------------|--------|
| H1 | Tensor dimension → κ | smaller → higher | EXP-022 | **REJECTED** |
| H2 | Dropout → low κ | dropout=0.1 → κ<10 | EXP-023 | **SUPPORTED** |
| H3 | Layer position → κ | early → higher | EXP-024 | MIXED |
| H4 | Attention > MLP | attn max > mlp max | EXP-022 | **SUPPORTED** |
| H4b | Attention component varies | model-specific | EXP-026 | PENDING |
| H5 | Size scaling | smaller → higher κ | EXP-025 | **REJECTED** |
| H6 | Mono vs multilingual | mono → higher κ | — | WEAK |
| H7 | Training evolution | κ grows late | EXP-027 | PENDING |

---

## Confirmed Findings

### F1: Attention Dropout Prevents Outliers
```
attn_dropout = 0.0  →  κ = 500+  (OPT, BLOOM)
attn_dropout = 0.1  →  κ = 2-201 (XGLM, GPT-2)
```
**Mechanism:** Dropout regularizes attention weights during training.

### F2: Outliers Concentrate in Attention
All HEAVY-outlier models (κ > 100) have worst outliers in attention projections:
- OPT: `attn.out_proj`
- BLOOM: `attn.query`
- GPT-2: `attn.c_proj`

### F3: Disparity Mechanism (BLOOM)
```
r = -0.834 (p = 0.0002)
Languages activating outlier layers LESS degrade MORE
```

---

## Rejected Hypotheses

### H1: Tensor Dimension
- **Prediction:** Smaller tensors → higher kurtosis
- **Result:** r = -0.003 (no correlation)
- **Conclusion:** Outlier formation not related to tensor size

### H5: Model Size Scaling
- **Prediction:** Smaller models → higher kurtosis
- **Result:** Pythia-410M (73) > Pythia-70M (45)
- **Conclusion:** Larger models can have higher kurtosis

---

## Hypothesis Revision Log

| Date | Change |
|------|--------|
| 2026-01-02 | H1-H6 formulated |
| 2026-01-03 | H1 rejected (EXP-022) |
| 2026-01-03 | H2 supported (EXP-023) |
| 2026-01-03 | H4 supported (EXP-022) |
| 2026-01-03 | H5 rejected (EXP-025) |
| 2026-01-03 | H4b, H7 added |

---

## Next Hypotheses to Test

1. **H4b:** Which attention component develops outliers depends on architecture
2. **H7:** Outliers emerge during late training (after representations stabilize)
3. **H8:** Outliers correlate with training loss instability

---

*Tracker updated: 2026-01-03*
