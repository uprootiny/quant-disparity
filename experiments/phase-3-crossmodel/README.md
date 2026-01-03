# Phase 3: Cross-Model Analysis

## Objective

Determine whether BLOOM's outlier-driven disparity mechanism generalizes to other models.

## Key Findings

### Model Census (12 models analyzed)

| Class | Models | Max κ | Disparity Risk |
|-------|--------|-------|----------------|
| HEAVY | OPT-125M, BLOOM-560M, GPT-2 | 200-562 | HIGH |
| Moderate | mT5-small, BERT-Tiny | 30-45 | MEDIUM |
| Mild | Pythia family, XLM-R | 8-14 | LOW |
| Gaussian | XGLM-564M, Tiny-GPT2 | 1-2 | NONE |

### Attention Dominates (EXP-022)

All HEAVY-outlier models have worst outliers in **attention projections**:
- OPT: `attn.out_proj` (κ=562)
- BLOOM: `attn.query` (κ=504)
- GPT-2: `attn.c_proj` (κ=201)

### Layer Position Varies (EXP-024)

- OPT: Early layers (layer 1)
- BLOOM/GPT-2: Late layers

## Experiments

| ID | Hypothesis | Result |
|----|------------|--------|
| EXP-020 | Cross-model census | 25% HEAVY |
| EXP-021 | OPT investigation | κ=562 in attn.out_proj |
| EXP-022 | H1 (dimension), H4 (component) | H4 supported |
| EXP-024 | H3 (layer position) | Model-specific |

## Files

### Documentation
- `MODEL_TAXONOMY.md` — Full census with classification
- `OPT_HYPOTHESES.md` — Hypothesis testing framework
- `config_analysis.md` — BLOOM vs XGLM training comparison

### Scripts
- `exp022_architecture.py` — Tests H1, H4
- `exp024_layer_position.py` — Tests H3
- `investigate_opt.py` — OPT deep dive
- `investigate_gpt2.py` — GPT-2 deep dive

### Data
- `exp022_results.json` — Architecture comparison results
- `exp024_results.json` — Layer position results
- `opt_outliers.json` — OPT outlier details
- `gpt2_analysis.json` — GPT-2 analysis

## Next Steps

1. **H2:** Compare OPT training config with BLOOM
2. **H5:** Analyze Pythia family for size scaling
3. **Attention mechanism:** Why do outliers concentrate there?

---

*Phase 3 started: 2026-01-03*
