# Research Status: What We Know

## Core Finding (BLOOM)

```
BLOOM-560M: r = -0.834 (p = 0.0002)
            Languages activating outlier layers LESS degrade MORE

Bootstrap CI: [-0.93, -0.65]
Permutation p: 0.0001
Leave-one-out: Stable (no single language drives effect)
```

**This is statistically robust within BLOOM.**

---

## Cross-Model Census (Updated 2026-01-03)

| Model | Max κ | Location | Component | Class |
|-------|-------|----------|-----------|-------|
| OPT-125M | **562** | Layer 1 | attn.out_proj | HEAVY |
| BLOOM-560M | **504** | Layer 22 | attn.query | HEAVY |
| GPT-2-small | **201** | Layer 11 | attn.c_proj | HEAVY |
| mT5-small | 45 | Decoder 2-5 | attn.out | Moderate |
| Pythia-410M | 14 | — | — | Mild |
| XLM-R-base | 10 | — | — | Mild |
| XGLM-564M | 2 | — | — | Gaussian |

**Key insight:** 25% of models have HEAVY outliers. Outliers concentrate in **attention projections**.

---

## New Findings (EXP-020 through EXP-024)

### 1. Outliers Are Common
```
HEAVY (κ > 50):   3/12 models (25%)
Moderate:         2/12 models (17%)
Mild:             5/12 models (42%)
Gaussian:         2/12 models (17%)
```

### 2. Attention Dominates (EXP-022)
All HEAVY-outlier models have worst outliers in **attention**, not MLP:
- OPT: `attn.out_proj` (κ=562)
- BLOOM: `attn.query` (κ=504)
- GPT-2: `attn.c_proj` (κ=201)

### 3. Layer Position is Model-Specific (EXP-024)
- OPT: Early layers worst (layer 1)
- BLOOM/GPT-2: Late layers worst

### 4. OPT Has Higher Kurtosis Than BLOOM
OPT-125M (κ=562) > BLOOM-560M (κ=504) despite being smaller.

---

## Hypothesis Status

| Hypothesis | Evidence | Status |
|------------|----------|--------|
| H1: Dimension → κ | r = -0.003 | **REJECTED** |
| H2: Training precision | BLOOM has FP32 hack | TESTABLE |
| H3: Layer position | Model-specific | **MIXED** |
| H4: Attention > MLP | 3/3 models | **SUPPORTED** |
| H5: Size scaling | Need more data | TESTABLE |

---

## What We Can Claim

### Strong Claims

1. **Outlier weights are common** — 25% of tested models have HEAVY outliers
2. **Outliers concentrate in attention** — All HEAVY models show this
3. **BLOOM disparity mechanism confirmed** — r=-0.834, robust
4. **Training dynamics matter** — BLOOM's FP32 hack suggests instability

### Moderate Claims

5. **OPT likely has similar disparity** — Higher κ, same pattern
6. **Pythia/XGLM are robust** — Near-Gaussian weights
7. **LA-ACIQ would help** — Theory validated (r=+0.84)

### Weak Claims

8. **Universal fix exists** — Outlier location varies by model
9. **Causation** — Still correlation only

---

## Research Directions

### Path A: OPT Disparity Test (Next Priority)
```
OPT has κ=562 (higher than BLOOM's 504)
→ If multilingual OPT exists, test degradation
→ Should show similar r<0 pattern
→ Validates cross-model generalization
```

### Path B: Attention Mechanism Deep Dive
```
Why attention projections?
→ Information aggregation from multiple heads
→ Training dynamics in attention layers
→ Potential fix: attention-specific quantization
```

### Path C: Training Recipe Investigation
```
Why does Pythia avoid outliers?
→ Compare training configs
→ Identify protective factors
→ Practical recommendations for training
```

---

## Immediate Next Steps

1. **Test H2:** Compare OPT training config with BLOOM
2. **Test H5:** Use Pythia family for size scaling analysis
3. **Investigate attention:** Why do all HEAVY models have attention outliers?
4. **GPU when available:** EXP-009 bit-width sweep

---

## Files Reference

| Category | Key Files |
|----------|-----------|
| Theory | `theory/la_aciq_v2.md`, `theory/la_aciq_math.md` |
| Census | `phase-3-crossmodel/MODEL_TAXONOMY.md` |
| Hypotheses | `phase-3-crossmodel/OPT_HYPOTHESES.md` |
| Protocol | `experiments/PROTOCOL.md` |
| PhD | `docs/israeli_labs_narratives.md` |

---

*Status as of: 2026-01-03*
