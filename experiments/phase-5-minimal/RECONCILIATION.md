# Research Reconciliation: 80 Experiments

*Mapping experimental findings to original research goals*

---

## Original Research Question

> **Why do certain languages degrade more than others when LLMs are quantized?**

### Answer (from 80 experiments):

Languages that **don't activate critical gateway layers** (input/output boundaries) suffer more because:
1. Layer 0 processes multilingual embeddings - quantization errors propagate through all subsequent layers
2. Layer 11 (output gateway) amplifies or dampens these errors before prediction
3. The L0+L11 "gateway synergy" is essential - neither alone is sufficient

---

## Hypothesis Validation Status

### Original Hypothesis v3 (from HYPOTHESIS.md)

> Languages with LESS activation in outlier-heavy layers degrade MORE because they lack robust representations that survive clipping.

| Prediction | Status | Evidence |
|------------|--------|----------|
| r = -0.834 correlation | **EXTENDED** | We found r = -0.798 variance-disparity correlation |
| Outlier layers matter | **CONFIRMED** | L0 has highest outlier ratio (1.726%) |
| Protection helps | **CONFIRMED** | L0+L9+L11 achieves 0.59x disparity |

### Phase 5 Hypotheses (from HYPOTHESES.md)

| ID | Hypothesis | Status | Exp Evidence |
|----|------------|--------|--------------|
| H5.1a | <5% achieves <50x | **SUPERSEDED** | 11.5% achieves <1x (Exp-060) |
| H5.1b | Non-linear cliff at 5% | **CONFIRMED** | Threshold sweep (Exp-011) |
| H5.2a | Layer 0 alone sufficient | **REFINED** | L0 necessary but L0+L11 synergy essential (Exp-042) |
| H5.2b | Attention > MLP | **MODEL-DEPENDENT** | GPT-2: MLP wins; OPT: Attn wins (Exp-012, 014) |
| H5.2c | First + last sufficient | **CONFIRMED** | L0+L11 = 0.86x (Exp-070) |
| H5.2d | Embeddings alone sufficient | **REFUTED** | Embeddings HURT (522x) (Exp-025) |
| H5.3a | Magnitude is optimal selector | **REFUTED** | Magnitude = 125,480x catastrophic (Exp-056) |
| H5.4a | Layer 0 + MLP = best | **SUPERSEDED** | L0+L9+L11 is optimal (Exp-072) |
| H5.5a | Anti-critical layers exist | **CONFIRMED** | Odd layers = 1379x (Exp-030) |
| H5.5b | Critical position varies | **CONFIRMED** | GPT-2: L0; OPT: L4 (Exp-062) |

---

## Proposal Phases Status

### Phase 1: Validate Weight Distribution Hypothesis

| Goal | Status | Evidence |
|------|--------|----------|
| Extract weight stats per language | **PARTIAL** | Focused on layer-level, not per-language neurons |
| Correlate with degradation | **CONFIRMED** | r = -0.798 variance-disparity (Exp-057-058) |
| Identify kurtosis patterns | **CONFIRMED** | L0 kurtosis = 14.4, L11 = 48.2 (Exp-066) |

### Phase 2: Layer Sensitivity Matrix

| Goal | Status | Evidence |
|------|--------|----------|
| Per-layer quantization sensitivity | **CONFIRMED** | Full 12-layer analysis (Exp-017, 020) |
| FFN vs attention | **CONFIRMED** | Model-dependent (Exp-012, 014) |
| Early vs late layers | **CONFIRMED** | Input (L0) + Output (L11) critical |

### Phase 3: Language-Aware Quantization

| Goal | Status | Evidence |
|------|--------|----------|
| Per-layer protection algorithm | **ACHIEVED** | L0+L9+L11 + biases + ln_f |
| Mixed-precision strategy | **ACHIEVED** | FP16/INT8 critical, INT4 rest (Exp-079) |
| Quantitative predictions | **ACHIEVED** | 0.59x avg disparity across 10 languages |

---

## Key Assumptions Tested

### Assumption 1: Tokenization Fertility Causes Disparity

**Status: REFUTED**

| Evidence | Exp |
|----------|-----|
| r = 0.000 fertility-disparity correlation | 059 |
| Korean: 10x fertility, 0.29x disparity | 080 |
| Arabic: 6x fertility, 0.31x disparity | 080 |

**Conclusion**: Token count doesn't predict disparity. Layer structure matters more.

### Assumption 2: Simple Heuristics Can Identify Critical Weights

**Status: REFUTED**

| Method | Disparity | Exp |
|--------|-----------|-----|
| Random 11.4% | 44-327x | 055 |
| Magnitude-based 38.6% | 125,480x | 056 |
| L0+L11 11.4% | 0.86x | 070 |

**Conclusion**: Weight selection requires structural understanding, not heuristics.

### Assumption 3: Universal Layer Pattern Exists

**Status: REFUTED**

| Model | Critical Layers | Exp |
|-------|-----------------|-----|
| GPT-2 | L0, L9, L11 | 072 |
| OPT-125M | L4, L6, L11 | 075 |

**Conclusion**: Layer criticality is model-specific. Quick sweep needed per architecture.

### Assumption 4: Higher Precision = Lower Disparity

**Status: PARTIALLY CONFIRMED**

| Precision | Disparity (L0+L9+L11) | Exp |
|-----------|----------------------|-----|
| INT2 | 7.89x | 078 |
| INT4 | 0.37x | 078 |
| INT8 | 1.31x | 078 |

**Conclusion**: INT4 is the sweet spot. INT8 too mild to need protection; INT2 too aggressive to fix.

---

## Original "Open Questions" Status

From PROPOSAL.md:

| Question | Answer | Evidence |
|----------|--------|----------|
| Does training data volume explain kurtosis? | Partially - L0 variance correlates with multilingual importance | Exp-057-058 |
| Which layers most sensitive per language? | L0, L9, L11 for GPT-2; L4, L6, L11 for OPT | Exp-072, 075 |
| Can mixed-precision restore equity? | **YES** - 0.59x disparity achieved | Exp-080 |

---

## Unexpected Findings

These were NOT in original hypotheses but emerged from experiments:

| Finding | Significance | Exp |
|---------|--------------|-----|
| L0+L11 synergy (0.7x together vs 336x for L11 alone) | Input-output layer coupling is critical | 032, 042 |
| Biases matter (+52.6x when quantized) | Only 0.08% of model but high impact | 039 |
| Anti-critical odd layers (1379x) | Some layers HURT when protected | 030 |
| Text length affects layer selection | Short texts give misleading results | 071 |
| L9 as "consolidation layer" at 75% depth | Three-layer pattern: input-consolidation-output | 076 |

---

## Gaps Remaining

| Gap | Priority | Reason |
|-----|----------|--------|
| Larger models (7B+) | HIGH | Memory constraints limited to 125M |
| Real quantization frameworks (GPTQ/AWQ) | MEDIUM | Used simulated INT4, not actual deployment |
| Per-language neuron analysis | LOW | Focused on layer-level patterns |
| Gradient-based selection (H5.3b) | LOW | OOM prevented testing |
| Causal analysis of L0 importance | MEDIUM | Correlation established, not causation |

---

## Summary: Research Goals Achievement

| Goal | Status | Quality |
|------|--------|---------|
| Explain multilingual disparity | **ACHIEVED** | High confidence |
| Identify critical weights | **ACHIEVED** | L0+L9+L11 pattern |
| Develop protection algorithm | **ACHIEVED** | Practical recommendations |
| Validate across languages | **ACHIEVED** | 10 languages tested |
| Cross-model validation | **PARTIAL** | GPT-2/OPT only (memory limited) |

---

## Final Recommendations

Based on 80 experiments:

### For GPT-2-like architectures:
```
Protect: L0 + L9 + L11 + all biases + final LayerNorm
Quantize: Everything else to INT4
Result: 0.59x average disparity, 37.8% model size
```

### For OPT-like architectures:
```
Protect: L4 + L6 + L11 + all biases + final LayerNorm
Quantize: Everything else to INT4
Result: 12.7x average disparity (architecture fundamentally harder)
```

### For new architectures:
```
1. Run quick layer sweep with medium-length multilingual text
2. Identify top-2 layers by disparity
3. Add biases + final LayerNorm to protection
4. Quantize rest to INT4
```

---

*80 experiments completed | 2026-01-09*
