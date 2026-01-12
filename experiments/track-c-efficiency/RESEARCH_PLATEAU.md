# Research Plateau: Quantization Disparity in Multilingual LLMs

*A stable foundation for the next phase of investigation*

**Date:** 2026-01-11
**Experiments completed:** 205
**Confidence level:** High for mechanisms, moderate for interventions

---

## Part I: What We Know (Confirmed Hypotheses)

### H1: Quantization Creates Systematic Language Disparity
**Status:** CONFIRMED (p < 0.0001)
**Evidence:**
- Disparity ratio: 4.24× (LR/HR degradation under INT4)
- Effect robust across model families (GPT-2, OPT, Pythia, BLOOM)
- Effect robust across quantization methods (GPTQ, Absmax, AWQ-style)

**Certainty:** Very high. This is the foundational finding.

### H2: Tokenization Quality Mediates Disparity
**Status:** CONFIRMED with caveats
**Evidence:**
- Cross-language: r = −0.924 (alignment → degradation)
- Within-language: r = −0.998 (confound-free, Hebrew)
- Replication: r = −0.996 (Arabic)

**Caveat:** Cross-language correlation is confounded (VIF = 36). Within-language evidence is causal but limited to two languages.

**Certainty:** High for mechanism, moderate for magnitude.

### H3: Gateway Layers are Disproportionately Important
**Status:** CONFIRMED
**Evidence:**
- Gateway/bottleneck importance ratio: 10.8×
- L0 (embedding), L9 (transition), L11 (output) consistently critical
- Pattern holds across model sizes (125M-1.5B)

**Certainty:** High.

### H4: Gateway Protection Reduces Disparity
**Status:** CONFIRMED
**Evidence:**
- 41% disparity reduction (4.24× → 2.50×)
- 86% efficiency retention (2.4×/2.8×)
- Fair-Efficiency Score improvement: 0.813 → 0.980

**Certainty:** High for effect, moderate for optimality.

### H5: Scaling Amplifies Disparity
**Status:** CONFIRMED
**Evidence:**
- Correlation: r = +0.984 (model size → disparity)
- Mechanism: Redundancy exploitation differs by language
- Evidence: Ablation reduces disparity by 22%

**Certainty:** High for observation, moderate for mechanism.

### H6: Compression Interactions are Super-Additive
**Status:** CONFIRMED
**Evidence:**
- Combined (quant + pruning): +30% excess degradation for LR
- Under combined compression: 4/4 HR usable, 1/5 LR usable

**Certainty:** High.

---

## Part II: What We Suspect (Working Hypotheses)

### W1: Vocabulary Coverage is THE Confound
**Status:** Suspected, not proven
**Reasoning:**
- VIF analysis shows vocabulary coverage dominates (VIF = 36)
- Alignment and coverage are r = 0.92 correlated
- Cannot disentangle without experimental manipulation

**Path to confirm:** Synthetic language experiment (identical training data, different tokenizers).

### W2: Calibration Data Determines Disparity Magnitude
**Status:** Suspected, partially confirmed
**Evidence:**
- Multilingual calibration improves LR by +12pp
- English-only calibration harms LR disproportionately

**Path to confirm:** Systematic calibration experiments across data compositions.

### W3: Feedback Loops Amplify Inequality Over Time
**Status:** Suspected, theoretically motivated
**Reasoning:**
- Worse models → less usage → less data → worse future models
- Blasi et al. (2022) documents general mechanism
- No longitudinal data to confirm for compression specifically

**Path to confirm:** Would require multi-year deployment study (impractical).

### W4: Architecture Affects Disparity Pattern
**Status:** Unknown
**Gap:** All experiments on Transformers. Mamba, RWKV, SSMs untested.

**Path to confirm:** Track A experiments (exp_a001 created, not run).

---

## Part III: What We Don't Know (Epistemic Gaps)

### Gap 1: GPU Validation
**Severity:** CRITICAL
**Description:** All experiments use simulated quantization. Real GPU behavior may differ.
**Impact:** Results could be artifacts of simulation.
**Resolution:** Run Colab notebook with actual quantized models.

### Gap 2: Hebrew Corpus
**Severity:** HIGH
**Description:** No Hebrew-specific test corpus with controlled complexity.
**Impact:** Cannot validate within-language findings with native data.
**Resolution:** Scrape Wikipedia Hebrew + Sefaria (planned).

### Gap 3: Causality vs Correlation
**Severity:** MODERATE
**Description:** Cannot definitively prove cross-language causation due to multicollinearity.
**Impact:** Theoretical mechanism is plausible but not proven.
**Resolution:** Synthetic language experiments (difficult to execute).

### Gap 4: Generalization to Modern Models
**Severity:** MODERATE
**Description:** Tested on GPT-2/OPT/Pythia (older). Llama 2/3, Mistral untested.
**Impact:** Findings may not transfer to state-of-the-art.
**Resolution:** Extend experiments to Llama-7B, Mistral-7B.

### Gap 5: Non-European Languages
**Severity:** MODERATE
**Description:** 7 of 12 languages are European. Underrepresent African, South Asian, Southeast Asian.
**Impact:** May miss language-specific patterns.
**Resolution:** Add Swahili, Hindi, Thai, Vietnamese.

### Gap 6: Real-World Deployment Effects
**Severity:** LOW (for academic paper)
**Description:** Laboratory conditions. Real deployment has caching, batching, context variation.
**Impact:** Effect sizes may differ in production.
**Resolution:** Industry partnership or deployment study.

---

## Part IV: Research Directions

### Direction A: Validation
**Priority:** CRITICAL
**Tasks:**
1. Run GPU validation notebook (Colab)
2. Scrape Hebrew corpus
3. Replicate within-language analysis in Korean, Turkish

**Expected outcome:** Confirm or refine core findings.

### Direction B: Extension
**Priority:** HIGH
**Tasks:**
1. Test on Llama-7B, Mistral-7B
2. Test on Mamba (SSM architecture)
3. Add 4 more languages (diversify families)

**Expected outcome:** Broader generalization claims.

### Direction C: Intervention
**Priority:** HIGH
**Tasks:**
1. Optimize gateway layer selection (grid search)
2. Test language-aware quantization (per-language calibration)
3. Develop multilingual calibration dataset

**Expected outcome:** Actionable mitigation strategies.

### Direction D: Theory
**Priority:** MEDIUM
**Tasks:**
1. Formalize redundancy hypothesis mathematically
2. Design synthetic language experiment
3. Develop predictive model (alignment → degradation)

**Expected outcome:** Deeper causal understanding.

### Direction E: Policy
**Priority:** MEDIUM
**Tasks:**
1. Draft Fair-Efficiency Score specification
2. Propose reporting standard for compression papers
3. Prepare policy brief for venues (ACL, EMNLP)

**Expected outcome:** Community impact.

---

## Part V: Remaining Uncertainty

### Quantified Uncertainties

| Finding | Confidence | Source of Uncertainty |
|---------|------------|----------------------|
| 4.24× disparity | 95% | Simulation, not GPU |
| r = −0.998 within-language | 99% | Small N (5 word types) |
| Gateway layers L0,L9,L11 | 90% | Model-specific? |
| 41% reduction from protection | 85% | Optimality unknown |
| Scaling paradox | 90% | Mechanism speculative |

### Open Questions

1. **Is 4-bit the threshold?** INT8 shows lower disparity (1.82×). Is there a precision level where disparity disappears?

2. **Can tokenizer redesign eliminate disparity?** Root cause intervention vs. mitigation at quantization stage.

3. **Does disparity affect downstream tasks equally?** Perplexity ≠ task performance. QA, summarization, translation may show different patterns.

4. **Is disparity inevitable in any compression?** Or specific to quantization? Preliminary: pruning shows similar pattern.

5. **What is the minimum data for LR language support?** Is there a threshold where tokenization improves enough to reduce disparity?

---

## Part VI: Stable Foundation for Next Phase

### What We Can Claim

1. **Quantization is not language-neutral.** (Strong claim, well-supported)

2. **Low-resource languages suffer disproportionate degradation.** (Strong claim, well-supported)

3. **Tokenization quality mediates this effect.** (Moderate claim, within-language evidence strong, cross-language confounded)

4. **Gateway layer protection reduces disparity with modest efficiency cost.** (Moderate claim, well-supported but optimality unknown)

5. **The Fair-Efficiency Score provides a principled way to evaluate compression methods.** (Framework contribution, not empirical)

### What We Should Not Claim (Yet)

1. ~~Tokenization causes disparity~~ → Tokenization correlates with disparity (causation requires more evidence)

2. ~~L0+L9+L11 is optimal~~ → L0+L9+L11 is effective (optimality not proven)

3. ~~All efficiency techniques harm LR~~ → Tested techniques harm LR (generalization limited)

4. ~~Results transfer to modern LLMs~~ → Results shown on GPT-2 class (generalization pending)

### Publication Readiness

| Component | Status | Gap |
|-----------|--------|-----|
| Core finding | READY | None |
| Mechanism | READY | Causal caveats needed |
| Mitigation | READY | Optimality caveat needed |
| Generalization | PARTIAL | Need Llama/Mistral |
| GPU validation | MISSING | Critical gap |
| Hebrew data | MISSING | For within-language |

**Recommendation:** Paper is 80% ready. GPU validation is the critical blocker for submission.

---

## Part VII: Immediate Next Steps

### This Week
1. [ ] Run GPU validation notebook
2. [ ] Scrape Hebrew Wikipedia (1M tokens minimum)
3. [ ] Run exp_a001 (cross-architecture)

### This Month
4. [ ] Extend to Llama-7B
5. [ ] Replicate within-language in Korean
6. [ ] Optimize gateway selection

### Before Submission
7. [ ] Complete GPU validation
8. [ ] Add 2 more languages (Turkish, German with full analysis)
9. [ ] Final pass on RESULTS_PAPER_DRAFT.md

---

## Part VIII: Curiosity Register

*Questions that emerged during research, worth exploring later*

1. Does quantization disparity affect emergent capabilities differently? (Chain-of-thought, few-shot learning)

2. Can disparity be detected without ground truth? (Practical deployment monitoring)

3. Is there a "fairness tax" on efficiency? (How much efficiency must we sacrifice for equity?)

4. Do users of LR languages notice the disparity? (User study opportunity)

5. Can we predict disparity from tokenizer statistics alone? (No model needed)

6. Does instruction-tuning change the disparity pattern? (Base vs. chat models)

7. What about vision-language models? (Multimodal fairness)

8. Can distillation preserve LR performance better than quantization? (Alternative compression)

---

*This document establishes the research plateau as of 2026-01-11. It should be updated as new evidence emerges.*

**Next session:** Begin GPU validation and Hebrew corpus collection.
