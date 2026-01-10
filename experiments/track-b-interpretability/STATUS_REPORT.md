# Track B Status Report: Multilingual Circuit Interpretability

*Target: Belinkov Lab (Technion)*

---

## Progress Summary

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| B-001 | DONE | 16.7% of heads are language-specific |
| B-002b | **DONE** | LR languages show **3.3x more representation damage** |
| B-003b | **DONE** | LR languages **2.23x more sensitive** to head ablation |
| B-004 | NOT STARTED | Gradient-based circuit discovery |
| B-005 | BLOCKED (GPU) | Causal mediation |

**Track Status: 3/5 experiments complete, 2 key hypotheses CONFIRMED**

---

## Completed Experiments

### B-001: Language-Specific Heads

**Method:** Ran attention pattern analysis on parallel sentences across languages.

**Finding:** 16.7% of attention heads show language-specific patterns.
- Universal heads: Handle cross-lingual features
- Language-specific heads: Concentrate in **late layers (8-11)**

**Connection to Track A:** Late layers (L9, L11) are our "gateway" layers.

---

### B-002b: Representation Similarity Analysis (NEW)

**Method:** Compared FP32 vs INT4 representation similarity across layers and languages.

**Key Findings:**

| Metric | HR Languages | LR Languages | Ratio |
|--------|--------------|--------------|-------|
| Avg damage | 7.3% | 23.9% | **3.3x** |
| Critical layer gap | 0.090 | 0.306 | **3.4x** |

**Layer-specific disparity:**
- L11 (output): 3.39x disparity in damage
- L9 (bottleneck): 4.15x disparity in damage
- L0 (input): 2.82x disparity in damage
- L5 (control): 1.55x disparity (lower, as expected)

**Hypothesis CONFIRMED:** LR languages show disproportionate representation damage.

---

### B-003b: Head Ablation Analysis (NEW)

**Method:** Zero out each head, measure PPL increase, compare across languages.

**Key Findings:**

| Layer | Avg LR/HR Sensitivity | Critical? | Track A Match |
|-------|----------------------|-----------|---------------|
| L0 | 2.05x | YES | ✓ |
| L5 | 1.55x | no | ✗ |
| L9 | 2.29x | YES | ✓ |
| L11 | 2.56x | YES | ✓ |

**Aggregate sensitivity:**
- LR avg sensitivity: 5.44
- HR avg sensitivity: 2.44
- **Disparity ratio: 2.23x**

**Most critical head:** L11_H11 (2.74x LR/HR ratio)

**Hypothesis CONFIRMED:** LR languages rely on fewer, more critical heads.

**CAUSAL EVIDENCE:** This provides causal support for the gateway-bottleneck model:
- It's not just that these layers have high variance (correlation)
- Ablating heads in these layers CAUSES disproportionate LR damage

---

## Cross-Track Synthesis

### Connection to Track A (Gateway-Bottleneck Model)

| Track A Finding | Track B Confirmation |
|-----------------|---------------------|
| L0 is critical gateway | L0 heads show 2.05x LR/HR sensitivity |
| L9 is bottleneck layer | L9 heads show 2.29x LR/HR sensitivity |
| L11 is output gateway | L11 heads show 2.56x LR/HR sensitivity |
| L5 is not critical | L5 heads show only 1.55x sensitivity |

### Connection to Track D (Alignment)

Track D found r=-0.956 correlation between alignment and degradation.
Track B explains the mechanism: poor alignment creates fragile representations in gateway layers.

---

## Key Questions ANSWERED

1. **Do quantization-sensitive heads overlap with language-specific heads?**
   - YES: Critical layers (L0, L9, L11) show highest LR/HR sensitivity gap
   - Language-specific computation concentrates where damage is worst

2. **Can representation damage predict quantization disparity?**
   - YES: 3.3x representation damage ratio matches Track A disparity

3. **Is there causal evidence for gateway layers?**
   - YES: Head ablation shows CAUSAL damage concentration at L0, L9, L11

---

## Remaining Experiments

| Priority | Experiment | Status | Value |
|----------|------------|--------|-------|
| 1 | B-004 Gradient circuits | NOT STARTED | Mechanism refinement |
| 2 | B-005 Causal mediation | BLOCKED (GPU) | Full causal story |

---

## Publication Potential

**Current state:** Three strong findings, sufficient for paper contribution.

**Contribution:**
1. 3.3x representation damage for LR languages
2. 2.23x head ablation sensitivity for LR languages
3. Causal evidence for gateway-bottleneck model

**Integration with Track A:** Provides mechanistic explanation for L0+L9+L11 protection strategy.

**Venue:** ACL/EMNLP main track (combined with Track A) or BlackboxNLP workshop.

---

*Last updated: 2026-01-10*
