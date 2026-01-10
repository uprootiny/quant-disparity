# Track C: Efficiency-Fairness Tradeoffs

*Target: Roy Schwartz Lab (Hebrew University of Jerusalem)*

---

## Research Problem

**Central Question:** Do efficient NLP techniques (distillation, pruning, quantization) systematically amplify language disparities, and what is the true cost of multilingual fairness?

**Scope:** We measure disparity across the "efficiency trifecta" and propose metrics that account for fairness costs.

**Gap:** Green AI research focuses on English; fairness implications for other languages are unexplored.

---

## Contextual Knowledge: Schwartz Lab

### Key Publications & Insights

| Paper | Key Insight | Our Application |
|-------|-------------|-----------------|
| **Schwartz et al. (2020)** "Green AI" | Efficiency metrics hide societal costs | We show they also hide fairness costs |
| **Schwartz & Stanovsky (2022)** "On the Limitations of Dataset Biases" | Datasets reflect power imbalances | LR underrepresentation → fragile representations |
| **Dodge et al. (2019)** "Show Your Work: Improved Reporting of Experimental Results" | Reporting standards matter | We propose Fair-Efficiency Score |

### Green AI Framework

From Schwartz (2020):
> "We propose to report: results as a function of computation, taking efficiency into account when making research decisions, and incentivizing efficient research."

**Our extension:**
> "Efficiency metrics must also account for FAIRNESS. An 'efficient' model that only works for English is not truly efficient."

### Lab's Core Arguments → Our Extensions

| Their Argument | Our Extension |
|----------------|---------------|
| "Report computation alongside accuracy" | "Report disparity alongside efficiency" |
| "Carbon cost matters" | "Carbon cost differs by language" |
| "Efficiency enables accessibility" | "Efficiency without fairness excludes speakers of LR languages" |

---

## Hypotheses

### H-C1: The Efficiency Trifecta
**Statement:** ALL major efficiency techniques (quantization, distillation, pruning) disproportionately harm low-resource languages.

**Rationale:** LR languages have less redundancy in representations. Any compression removes what little signal exists.

**Testable Prediction:** Disparity ratio > 2.0x for each technique.

**Result:** ✓ CONFIRMED
- Quantization: 4.24x
- Distillation: 3.02x
- Pruning: 3.04x

---

### H-C2: Fertility ≠ Degradation
**Statement:** Token count (fertility) does NOT predict quantization degradation.

**Rationale:** Error accumulation across tokens is a plausible but incorrect mechanism. The real mechanism is structural (alignment).

**Testable Prediction:** Correlation between fertility and degradation is weak (r < 0.3).

**Result:** ✓ CONFIRMED — r = -0.07 (no correlation). Track D found the real mechanism: alignment (r = -0.956).

---

### H-C3: Sparsity Tolerance Differs
**Statement:** LR languages hit performance thresholds at lower sparsity levels than HR languages.

**Rationale:** Pruning removes weights by magnitude; LR-specific weights tend to be lower magnitude (less training signal) and get pruned first.

**Testable Prediction:** LR languages become unusable at 30% sparsity while HR remain usable at 70%.

**Result:** ✓ CONFIRMED — English usable at 70%, Hebrew breaks at 30%.

---

### H-C4: Carbon Cost of Fairness
**Statement:** Achieving equivalent performance across languages requires disproportionate compute for LR languages, creating a measurable carbon cost of fairness.

**Rationale:** LR languages need larger models to achieve the same perplexity threshold.

**Testable Prediction:** Compute disparity > 10x between English and Hebrew for equivalent PPL.

**Result:** ✓ CONFIRMED — 56x compute disparity (Hebrew needs Llama-7B, English needs GPT-2 small).

---

## Experiment Sequence

### Phase 1: Tokenization Baseline

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| C-001b | Tokenizer efficiency | Fertility measurement | Baseline | ✓ DONE | 6.17x gap |
| C-005 | Fertility vs degradation | Correlation analysis | H-C2 | ✓ DONE | r = -0.07 |

---

### Phase 2: Efficiency Trifecta

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| C-001 | Distillation disparity | mBERT vs DistilmBERT | H-C1 | ✓ DONE | 3.02x |
| C-002 | Pruning disparity | Magnitude pruning sweep | H-C1, H-C3 | ✓ DONE | 3.04x |
| C-003 | Quantization disparity | (Merged with Track A) | H-C1 | ✓ DONE | 4.24x |

---

### Phase 3: Policy Implications

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| C-004 | Carbon cost | Model scaling analysis | H-C4 | ✓ DONE | 56x compute disparity |
| C-004b | Fair-Efficiency metric | Novel metric proposal | — | ✓ DONE | FE = throughput/disparity |

---

## Evidence Summary

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| H-C1 (Trifecta) | Quant 4.24x, Distill 3.02x, Prune 3.04x | **CONFIRMED** |
| H-C2 (Fertility ≠ degradation) | r = -0.07 | **CONFIRMED** |
| H-C3 (Sparsity tolerance) | EN: 70%, HE: 30% | **CONFIRMED** |
| H-C4 (Carbon cost) | 56x compute disparity | **CONFIRMED** |

---

## Novel Metrics Proposed

### Fair-Efficiency Score
```
Fair-Efficiency = throughput / disparity_ratio
```

| Model | Throughput | Disparity | Fair-Eff |
|-------|------------|-----------|----------|
| mBERT FP32 | 1.0x | 1.0x | 1.00 |
| DistilmBERT | 2.4x | 3.0x | 0.80 |
| mBERT INT4 | 3.2x | 4.2x | 0.76 |
| mBERT 50% sparse | 2.0x | 3.0x | 0.67 |
| **mBERT + L0+L9+L11** | 2.6x | 0.59x | **4.41** |

**Key insight:** When accounting for fairness, naive efficiency gains disappear. But smart protection (L0+L9+L11) achieves BOTH efficiency AND fairness.

---

## Cross-Track Synthesis

| Track | Finding | Connection to Track C |
|-------|---------|----------------------|
| **A** | L0+L9+L11 achieves 0.59x disparity | C shows this is carbon-efficient |
| **B** | 3.3x representation damage | C explains THIS is why trifecta holds |
| **D** | Alignment r=-0.956 | C-005 falsified fertility, D found truth |

---

## Policy Implications

1. **"Efficient" ≠ "Fair"**
   - Current efficiency metrics hide fairness costs
   - Deploying quantized models may violate fairness principles

2. **Carbon cost is real**
   - 56x compute disparity for equivalent performance
   - Extra 2,409 tonnes CO2/year per LR language at scale

3. **Protection is carbon-efficient**
   - L0+L9+L11 uses only 2% of adaptive approach carbon
   - Reconciles Green AI with Inclusive AI

---

## Publication Contribution

**Novel findings:**
1. Efficiency trifecta: ALL techniques cause disparity
2. Fair-Efficiency Score: new evaluation metric
3. Falsification: fertility ≠ degradation

**Policy relevance:** Carbon cost quantified; Green AI and fairness reconciled.

**Venue:** EMNLP Green NLP track, ACL Social Good theme

**Title:** "The Hidden Cost of Efficiency: How Compression Techniques Amplify Language Disparities"

---

*Last updated: 2026-01-10*
