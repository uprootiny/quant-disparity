---
layout: page
title: Roadmap
permalink: /roadmap/
nav_order: 6
---

# Research Roadmap

What's done, what's next, and where we're headed.

---

## Current Progress

```
Overall:              [████████░░] 80%

Track A (Soudry):     [█████████░] 90%
Track B (Belinkov):   [████░░░░░░] 40%
Track C (Schwartz):   [█████░░░░░] 50%
Track D (Goldberg):   [██░░░░░░░░] 20%

GPU Experiments:      [░░░░░░░░░░] 0% (blocked)
Publication Draft:    [░░░░░░░░░░] 0%
```

---

## Phase 1: Discovery (Complete)

- [x] Detect disparity in BLOOM-560M
- [x] Establish correlation: r = -0.834
- [x] Statistical validation (bootstrap, permutation, LOO)
- [x] Identify outlier weights as mechanism
- [x] Locate outliers in attention projections

**Outcome:** Core finding validated.

---

## Phase 2: Cross-Model Validation (Complete)

- [x] Test 12 multilingual models
- [x] Classify by kurtosis (HEAVY/Moderate/Mild/Gaussian)
- [x] Confirm attention > MLP pattern (3/3 models)
- [x] Test training dynamics (dropout, evolution)
- [x] Rule out size scaling as explanation

**Outcome:** Mechanism generalizes across architectures.

---

## Phase 3: Multi-Track Expansion (In Progress)

### Track A: Remaining

- [x] EXP-030: Attention sink correlation
- [x] EXP-033: Token fertility prediction
- [ ] EXP-034: Language-specific head overlap with outliers
- [ ] EXP-009: Real quantization sweep (GPU)
- [ ] EXP-031: LA-ACIQ implementation test (GPU)

### Track B: Interpretability

- [x] B-001: Attention pattern analysis
- [ ] B-002: Full probing experiment
- [ ] B-003: Circuit ablation by language
- [ ] B-004: Gradient-based circuit discovery
- [ ] B-005: Causal mediation (GPU)

### Track C: Efficiency-Fairness

- [x] C-001b: Tokenizer efficiency gap
- [ ] C-002: Quantization fairness (needs debugging)
- [ ] C-003: Pruning effect on languages
- [ ] C-004: Carbon cost per language

### Track D: Syntax-Morphology

- [x] Research plan created
- [ ] D-001: Morphological disambiguation under quantization
- [ ] D-002: Syntactic parsing degradation
- [ ] D-003: Subword-morpheme alignment
- [ ] D-004: Joint vs pipeline robustness
- [ ] D-005: Morphological feature circuits

---

## Phase 4: Causal Intervention (Blocked on GPU)

**Required:** GPU access for activation patching and real quantization.

| Experiment | Description | Status |
|------------|-------------|--------|
| EXP-009 | Bit-width sweep with bitsandbytes | Blocked |
| EXP-031 | LA-ACIQ implementation | Blocked |
| B-005 | Causal mediation analysis | Blocked |
| REAL-001 | End-to-end quantization + eval | Blocked |

**Timeline:** Unblock when GPU access available.

---

## Phase 5: Publication Preparation

### Target Venues

| Venue | Deadline | Track Focus |
|-------|----------|-------------|
| EMNLP 2027 | ~June 2027 | Track A + B |
| ACL 2027 | ~Feb 2027 | Track A + C |
| COLM 2027 | TBD | Track A mechanistic |

### Paper Structure (Draft)

1. **Abstract:** Quantization hurts low-resource languages; we explain why
2. **Introduction:** The fairness problem in model compression
3. **Background:** Outlier weights, attention sinks, multilingual LLMs
4. **Method:** Cross-model kurtosis analysis, disparity measurement
5. **Results:** r=-0.834, attention dominance, training dynamics
6. **Analysis:** Unified mechanistic model
7. **Discussion:** LA-ACIQ proposal, limitations
8. **Related Work:** EMNLP 2024 paper, super weights, etc.

### Artifacts to Prepare

- [ ] Main paper (8 pages)
- [ ] Appendix with all experiments
- [ ] Code release (this repo)
- [ ] Figures: kurtosis heatmaps, correlation plots
- [ ] Reproducibility checklist

---

## Immediate Next Steps

### This Week (No GPU)

1. **Track D kickoff:** Run D-001 morphological disambiguation
2. **Cross-track analysis:** Correlate language-specific heads with outlier locations
3. **Debug C-002:** Fix timeout in quantization fairness experiment
4. **Documentation:** Complete this GitHub Pages site

### When GPU Available

1. **EXP-009:** Real quantization sweep (bitsandbytes)
2. **LA-ACIQ:** Implement and test on BLOOM
3. **Causal intervention:** Activation patching for B-005

### Before Submission

1. **Scale up:** Test on 7B+ models
2. **More languages:** Expand to 50+
3. **Human evaluation:** Validate beyond automatic metrics
4. **Ablations:** Robustness checks

---

## Lab Engagement Strategy

### Soudry Lab (Track A)

**Pitch:** "We found the mechanism behind multilingual quantization disparity—it's in the attention outliers you've studied."

**Materials:**
- 1-page summary of r=-0.834 finding
- LA-ACIQ proposal
- Connection to their quantization work

### Belinkov Lab (Track B)

**Pitch:** "We can connect your probing/circuit work to a practical fairness problem in model compression."

**Materials:**
- Language-specific head findings (16.7%)
- Connection to causal mediation methods

### Schwartz Lab (Track C)

**Pitch:** "Efficiency techniques create language disparities—here's the data and a path forward."

**Materials:**
- Tokenization efficiency gap (6.17x)
- Negative result: fertility ≠ degradation
- Carbon cost framing

### Goldberg Lab (Track D)

**Pitch:** "Hebrew and Arabic may be especially vulnerable to quantization due to morphological complexity."

**Materials:**
- Track D research plan
- Connection to AlephBERT/YAP

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| GPU never available | Low | Cloud credits, collaboration |
| LA-ACIQ doesn't work | Medium | Publish negative result, iterate |
| Effect doesn't scale to large models | Medium | Focus on mechanistic insight |

### Research Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Similar work published first | Medium | Move fast, focus on mechanism |
| Reviewers want causal evidence | High | GPU experiments priority |
| Too many tracks, no depth | Medium | Prioritize Track A |

---

## Long-Term Vision

### If Everything Works

1. **LA-ACIQ adopted:** Language-aware quantization becomes standard
2. **Training recommendations:** How to train models that quantize fairly
3. **Benchmarking standard:** Multilingual quantization evaluation suite
4. **Follow-up research:** Apply to pruning, distillation, other compression

### Minimum Viable Contribution

Even if GPU experiments fail:
- **Mechanistic insight:** Attention outliers explain disparity
- **Negative result:** Tokenization isn't the bottleneck
- **Methodology:** Cross-model, multi-track research framework

---

## Milestones

| Date | Milestone |
|------|-----------|
| Jan 2026 | Track A core complete ✓ |
| Feb 2026 | GPU experiments (if access) |
| Mar 2026 | All tracks complete |
| Apr 2026 | Paper draft |
| May 2026 | Internal review |
| Jun 2026 | Submission (EMNLP 2027) |

---

*Last updated: January 2026*
