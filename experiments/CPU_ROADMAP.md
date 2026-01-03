# CPU-Only Research Directions

## What Requires GPU
```
✗ Model inference (perplexity computation)
✗ Activation extraction during forward pass
✗ Bit-width sweep experiments
✗ Large model loading (7B+)
```

## What We CAN Do on CPU

### Direction 1: More Models (Weight Statistics Only)

Extract weight distributions from additional multilingual models:

| Model | Size | Can Load? | Value |
|-------|------|-----------|-------|
| BLOOM-560M | 560M | ✓ | Already done |
| XGLM-564M | 564M | ✓ | Already done |
| mGPT | 1.3B | ✓ (swap) | Russian-focused, different training |
| XGLM-1.7B | 1.7B | ✓ (swap) | Larger XGLM for comparison |
| mT5-base | 580M | ✓ | Encoder-decoder architecture |
| XLM-RoBERTa | 560M | ✓ | Masked LM, different objective |

**Question:** Does the outlier layer pattern appear in other architectures?

### Direction 2: More Languages

Expand from 14 to 30+ languages using Marchisio et al. data:

```
Current (14): ara, deu, eng, fin, fra, heb, hin, jpn, kor, rus, tha, tur, vie, zho

Potential additions from FLORES-200 / Wikipedia:
  - Romance: ita, spa, por, ron, cat
  - Slavic: pol, ces, ukr, bul
  - Germanic: nld, swe, nor, dan
  - Other: ind, msa, swh, yor, amh
```

**Question:** Does correlation hold with larger language sample?

### Direction 3: Theoretical Formalization

Write mathematical framework for LA-ACIQ without compute:

```
1. Define: Effective kurtosis κ_eff(lang) = Σ_l a_l(lang) × κ_l
2. Derive: Optimal α*(lang) as function of κ_eff
3. Prove: Disparity bound in terms of κ_eff variance
4. Analyze: Computational overhead of per-language calibration
```

**Deliverable:** Theory section of paper draft

### Direction 4: Corpus Infrastructure

Build robust multilingual corpus for future experiments:

```
Current: 60MB across 6 languages
Target: 500MB across 30 languages

Sources:
  - Wikipedia (all languages)
  - CC-100 (Common Crawl, 100 languages)
  - OPUS (parallel corpora)
```

**Deliverable:** Reusable corpus collection pipeline

### Direction 5: Cross-Architecture Analysis

Compare weight patterns without inference:

| Analysis | BLOOM | XGLM | mGPT | XLM-R |
|----------|-------|------|------|-------|
| Layer kurtosis distribution | ✓ | ✓ | ? | ? |
| Outlier layer identification | ✓ | ✓ | ? | ? |
| Weight magnitude patterns | ✓ | ✓ | ? | ? |
| Embedding statistics | ? | ? | ? | ? |

**Question:** Is outlier pattern BLOOM-specific or architecture-specific?

### Direction 6: Tokenizer Deep Dive

Analyze tokenization without model inference:

```python
# Can compute on CPU:
- Fertility per language (tokens/word)
- Vocabulary coverage (% known tokens)
- Subword fragmentation patterns
- Cross-lingual token sharing
- Byte fallback frequency
```

**Question:** Does tokenization correlate with outlier activation?

### Direction 7: Statistical Refinements

More sophisticated analysis of existing data:

```
- Partial correlation (control for confounds)
- Mediation analysis (tokenization → activation → degradation)
- Bayesian inference (posterior on correlation)
- Causal discovery (PC algorithm, etc.)
```

**Deliverable:** Stronger statistical claims

### Direction 8: Paper Draft

Write up current findings:

```
Sections:
1. Introduction (disparity problem)
2. Related Work (Marchisio, Banner, Chmiel)
3. Method (weight extraction, activation analysis)
4. Results (r=-0.834, bootstrap validation)
5. Theory (ACIQ extension, effective kurtosis)
6. Discussion (limitations, future work)
```

**Deliverable:** Workshop/short paper ready for submission

---

## Recommended Priority

| Priority | Direction | Effort | Impact |
|----------|-----------|--------|--------|
| 1 | More models (mGPT, XLM-R) | Medium | High — generalization |
| 2 | Theoretical formalization | Medium | High — novelty |
| 3 | Paper draft | High | High — deliverable |
| 4 | More languages | Low | Medium — robustness |
| 5 | Tokenizer analysis | Low | Medium — mechanism |
| 6 | Corpus expansion | Low | Low (for later) |

---

## Immediate Next Steps (No GPU)

```
EXP-010: mGPT weight extraction
  - Load model on CPU
  - Extract per-layer kurtosis
  - Compare with BLOOM/XGLM pattern

EXP-011: XLM-RoBERTa weight extraction
  - Different architecture (encoder-only)
  - Does outlier pattern appear?

EXP-012: Theoretical bounds
  - Formalize LA-ACIQ
  - Derive disparity bound

EXP-013: Extended language analysis
  - Add 10+ languages from Marchisio data
  - Recompute correlation
```
