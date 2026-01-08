# Research Gaps and Validation Status

*Last updated: 2026-01-08*

## Executive Summary

**Best Result Achieved**: 3.8x disparity with Layer 0 protection (5.7% overhead)
- Validated across 6 languages (en, de, fr, zh, ar, he)
- Hebrew improved from 214x to 5.9x (36x better)
- Arabic improved from 82x to 1.3x (63x better)

---

## Evidence Status

### Strong Evidence (Multiple Experiments, Multiple Models)

| Finding | Evidence | Experiments |
|---------|----------|-------------|
| ✓ Disparity exists (78-214x baseline) | CONFIRMED | 001-010, 014 |
| ✓ Hebrew most affected (971,648% deg) | CONFIRMED | 001-010, 022 |
| ✓ 5% magnitude preservation optimal | CONFIRMED | 011 |
| ✓ Layer 0 protection effective (3.8x) | CONFIRMED | 012, 021, 022 |
| ✓ Statistical reproducibility (0% CV) | CONFIRMED | 016 |
| ✓ Text length: medium/long reliable | CONFIRMED | 015 |

### Confirmed with Caveats

| Finding | Status | Details |
|---------|--------|---------|
| ⚠ Component criticality | MODEL-DEPENDENT | GPT-2: MLP wins; OPT: Attention wins |
| ⚠ Per-layer patterns | MODEL-DEPENDENT | GPT-2: L0 best; OPT: L11 best |
| ⚠ Anti-critical layers exist | CONFIRMED | GPT-2 L1, OPT L7 increase disparity |

### Validated Across Models

| Finding | GPT-2 | OPT-125M | Pythia-160M | BLOOM-560M |
|---------|-------|----------|-------------|------------|
| Disparity exists | ✓ 214x | ✓ 153x | ⚠ (tokenization) | - (RAM) |
| Layer 0 helps | ✓ 5.9x | ✓ 45x | - | - |
| Best MLP layer | L0 (139x) | L11 (92x) | - | - |
| Best Attn layer | L0 (54x) | L0 (101x) | - | - |
| Anti-critical | L1 (381x) | L7 (245x) | - | - |

---

## Completed Experiments (22 total)

| Exp | Name | Key Result |
|-----|------|------------|
| 001-010 | Baseline series | 78-214x disparity confirmed |
| 011 | Threshold sweep | 5% optimal (45x) |
| 012 | Layer-specific (GPT-2) | MLP > Attention |
| 013 | Hybrid strategy | Layer 0 + MLP = 1.4x |
| 014 | OPT validation | Pattern is model-dependent |
| 015 | Text length | Medium/long texts reliable |
| 016 | Robustness | 0% CV (deterministic) |
| 017 | Per-layer MLP | L0 best (GPT-2), L11 best (OPT) |
| 018 | BLOOM | Memory exceeded |
| 019 | Pythia | Tokenization issues with Hebrew |
| 020 | Per-layer attention | L0 best (both models) |
| 021 | Combined critical | 4.8x with avoid-anti strategy |
| 022 | Multi-language | 3.8x across 6 languages |

---

## Remaining Gaps

### High Priority (Blocking Publication)

| Gap | Status | Blocker |
|-----|--------|---------|
| Real quantization (GPTQ/AWQ) | UNTESTED | Needs bitsandbytes setup |
| Downstream task metrics | UNTESTED | Need eval datasets |
| 7B+ model validation | UNTESTED | GPU required |

### Medium Priority (Would Strengthen Paper)

| Gap | Status | Notes |
|-----|--------|-------|
| Longer text evaluation | PARTIAL | Exp-015 used medium texts |
| BLOOM validation | BLOCKED | Memory constraints |
| Gradient-based selection | UNTESTED | Alternative to magnitude |
| Language-activation analysis | UNTESTED | Explain anti-critical layers |

### Low Priority (Future Work)

| Gap | Status | Notes |
|-----|--------|-------|
| 50+ language coverage | UNTESTED | Need test texts |
| Production deployment | UNTESTED | Need inference benchmarks |
| Training-time interventions | UNTESTED | Different research direction |

---

## Hypotheses Status

| ID | Hypothesis | Status |
|----|------------|--------|
| H5.1a | <5% achieves <50x | ✓ CONFIRMED (45x at 5%) |
| H5.1b | Non-linear cliff at ~5% | ✓ CONFIRMED |
| H5.2a | Layer 0 alone sufficient | ✓ CONFIRMED (3.8x) |
| H5.2b | Attention > MLP | ⚠ MODEL-DEPENDENT |
| H5.2d | Embeddings alone sufficient | ✗ REFUTED (1216x, worse) |
| H5.3a | Magnitude optimal | ⚠ PARTIAL (non-monotonic) |
| H5.4a | Layer 0 + MLP = best | ✓ CONFIRMED (1.4x) |
| H5.5a | Anti-critical layers exist | ✓ CONFIRMED |
| H5.5b | Critical position varies | ✓ CONFIRMED |
| H5.5c | Anti-critical = English-specific | ? HYPOTHESIS |

---

## Next Steps

### Immediate (No New Resources)
1. Exp-023: Longer text validation (paragraph-length)
2. Exp-024: Arabic/Chinese focus (non-Hebrew LR languages)
3. Update TECHNICAL_WRITEUP for publication draft

### Requires Setup
1. Install bitsandbytes for real INT4 quantization
2. Set up evaluation harness for downstream tasks
3. Access GPU for 7B+ models

---

## Publication Readiness

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | ✓ Complete | Updated with 3.8x result |
| Introduction | ✓ Complete | - |
| Methodology | ✓ Complete | - |
| Results | ✓ Complete | All 22 experiments documented |
| Analysis | ✓ Complete | Per-layer findings added |
| Recommendations | ✓ Complete | Updated with multi-lang results |
| Limitations | ⚠ Update | Add Pythia tokenization note |
| Future Work | ⚠ Update | Prioritize based on gaps |

---

*Repository: github.com/uprootiny/quant-disparity*
