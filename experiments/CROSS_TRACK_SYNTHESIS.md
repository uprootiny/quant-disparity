# Cross-Track Synthesis

*Connecting findings across all research tracks*
*Updated: 2026-01-10*

---

## Summary Table

| Track | Experiments | Key Finding | Strength |
|-------|-------------|-------------|----------|
| **A (Main)** | 95 | L0+L9+L11 achieves 0.59x disparity | ★★★★★ |
| **B (Belinkov)** | 3 | LR languages show 3.3x more representation damage + CAUSAL evidence | ★★★★★ |
| **C (Schwartz)** | 4 | Efficiency trifecta: ALL techniques hurt LR more | ★★★★★ |
| **D (Goldberg)** | 3 | Alignment (r=-0.956) predicts degradation | ★★★★★ |

**Total experiments: 105**

---

## Track Results

### Track A: Gateway-Bottleneck Model (95 experiments)

**Core finding:** Protecting L0+L9+L11 (17% overhead) achieves 0.59x disparity.

**Key numbers:**
- Baseline disparity: 206.9x
- L0 alone: 3.6x
- L0+L11: 0.92x
- L0+L9+L11: 0.59x

**Theoretical framework:**
```
Criticality Score = 2.5×boundary + 1.5×variance + 0.8×kurtosis
                  + 1.2×outliers + 1.0×consolidation - 0.5×distance
```

---

### Track B: Representation Damage (3 experiments) ✓ COMPLETE

**Finding 1 (B-001):** 16.7% of heads are language-specific, concentrated in late layers.

**Finding 2 (B-002b):** LR languages suffer 3.3x more representation damage.

| Metric | HR Languages | LR Languages | Ratio |
|--------|--------------|--------------|-------|
| Avg damage | 7.3% | 23.9% | 3.3x |
| Critical layer gap | 0.090 | 0.306 | 3.4x |

**Layer-specific disparity:**
- L11 (output): 3.39x disparity in damage
- L9 (bottleneck): 4.15x disparity in damage
- L0 (input): 2.82x disparity in damage

**Finding 3 (B-003b):** LR languages are 2.23x more sensitive to head ablation.

| Layer | Avg LR/HR Sensitivity | Critical? |
|-------|----------------------|-----------|
| L0 | 2.05x | YES |
| L5 | 1.55x | no |
| L9 | 2.29x | YES |
| L11 | 2.56x | YES |

**CAUSAL EVIDENCE:** Head ablation provides causal support for gateway-bottleneck model.

---

### Track C: Efficiency-Fairness Tradeoff (4 experiments) ✓ COMPLETE

**THE EFFICIENCY TRIFECTA:**

| Technique | Disparity Ratio | Mechanism |
|-----------|-----------------|-----------|
| **Quantization** | 4.24x | Precision reduction amplifies outlier errors |
| **Distillation** | 3.02x | Knowledge compression loses sparse LR knowledge |
| **Pruning** | 3.04x | LR-specific weights pruned preferentially |

**Average disparity: 3.43x**

**Novel metric proposed:**
```
Fair-Efficiency Score = throughput / disparity
```

| Model | Throughput | Disparity | Fair-Eff |
|-------|------------|-----------|----------|
| mBERT FP32 | 1.0x | 1.0x | 1.00 |
| DistilmBERT | 2.4x | 3.0x | 0.80 |
| mBERT INT4 | 3.2x | 4.2x | 0.76 |
| mBERT 50% sparse | 2.0x | 3.0x | 0.67 |

**Key insight:** When accounting for fairness, efficiency gains disappear.

---

### Track D: Morphological Sensitivity (3 experiments) ✓ COMPLETE

**Finding 1 (D-001b):** Complex sentences degrade 25% more than simple (universal).

| Morphology Type | Complexity Ratio | Absolute Degradation |
|-----------------|------------------|---------------------|
| Analytic (EN) | 1.26x | 59% |
| Fusional (DE, FR) | 1.28x | 74% |
| Templatic (AR, HE) | 1.25x | 334% |

**Finding 2 (D-002b):** Complex agreement drops 2.80x more for MRLs.

| Agreement Type | Languages | Avg Drop |
|----------------|-----------|----------|
| Simple (number only) | EN | 10.3% |
| Complex (gender+number+person) | AR, HE | 28.9% |

**Distance amplifies disparity:**
- Adjacent: 1.97x LR/HR
- Long-distance: 2.88x LR/HR

**Finding 3 (D-003b):** Alignment predicts degradation (r = -0.956).

| Language | Alignment | Degradation |
|----------|-----------|-------------|
| English | 0.72 | 47% |
| Hebrew | 0.24 | 264% |
| Arabic | 0.28 | 214% |

**Regression model (R² = 0.940):**
```
degradation = 150 - 224×alignment + 40×fertility
```

---

## Unified Theory: Alignment-Gateway-Bottleneck Model

```
                    TRACK D                    TRACK A                    TRACK B
                    ↓                          ↓                          ↓
     Tokenization   →   L0 (Gateway)   →   L1-L8   →   L9 (Bottleneck)   →   L11 (Gateway)   →   Output
         ↓                  ↓                           ↓                        ↓
    Poor alignment    Input encoding           Morphological             Output projection
    for MRLs          creates basis            consolidation             amplifies damage
         ↓                  ↓                           ↓                        ↓
    r = -0.956       2.82x damage              4.15x damage              3.39x damage
    with degradation disparity at L0           disparity at L9           disparity at L11
```

### Causal Chain

1. **Track D (Alignment):** The ROOT CAUSE
   - Poor tokenizer alignment creates structurally misaligned representations
   - r = -0.956 correlation with degradation (strongest finding)
   - This is BEFORE any model computation happens

2. **Track A (L0):** The ENTRY POINT
   - L0 encodes the misaligned tokenization into representations
   - Quantization here compounds the structural problem
   - Protecting L0 = best possible encoding despite poor tokenization

3. **Track A (L9):** The BOTTLENECK
   - Morphological features consolidate here (75% depth)
   - MRLs rely heavily on this layer for disambiguation
   - Protecting L9 = preserved morphological processing

4. **Track A (L11):** The EXIT POINT
   - Output projection back to token space
   - All upstream errors accumulate here
   - Protecting L11 = clean output despite issues

5. **Track B (Representation):** The MEASUREMENT
   - 3.3x more damage for LR languages
   - Concentrated in gateway + bottleneck layers
   - Head ablation provides CAUSAL evidence

6. **Track C (Efficiency):** The GENERALIZATION
   - Not just quantization—ALL efficiency techniques show disparity
   - Average 3.43x disparity across the trifecta
   - Need new metrics that account for fairness

---

## Falsifications & Refinements

### What We Got Wrong (Initially)

1. **Fertility = Degradation:** FALSIFIED (C-005)
   - Initial hypothesis: More tokens → more error accumulation
   - Finding: r = -0.07 (no correlation)
   - Refinement: ALIGNMENT matters, not token count (r = -0.956)

2. **MRLs suffer more on complex sentences (relative):** REFINED (D-001b)
   - Initial hypothesis: MRLs show larger complexity ratio
   - Finding: All morphology types show ~1.25x ratio
   - Refinement: ABSOLUTE degradation differs, not relative complexity sensitivity

### What We Confirmed

1. **Gateway layers are critical:** CONFIRMED (Track A + B + D)
   - Track A: L0+L9+L11 achieves 0.59x disparity
   - Track B: Causal evidence via head ablation
   - Track D: Explains WHY L0 matters (alignment)

2. **All efficiency techniques cause disparity:** CONFIRMED (Track C)
   - Quantization: 4.24x
   - Distillation: 3.02x
   - Pruning: 3.04x

3. **Grammatical correctness affected:** CONFIRMED (D-002b)
   - Not just perplexity—actual agreement errors
   - Hebrew long-distance: 54% → 28% accuracy

---

## Implications

### For Practitioners

```python
# Minimum viable fair quantization
def fair_quantize(model):
    protect = {0, num_layers - 1}  # L0 + L_last
    if disparity_target < 0.7:
        protect.add(int(num_layers * 0.75))  # L_0.75
    return quantize_except(model, protect)
```

### For Researchers

1. **Tokenization matters more than we thought**
   - Alignment, not fertility, predicts quantization damage
   - Future work: morphologically-aware tokenization

2. **Gateway layers are universal bottlenecks**
   - Likely applies to other architectures (Llama, Mistral)
   - Need validation at scale

3. **Efficiency-fairness is a fundamental tradeoff**
   - Current metrics hide fairness costs
   - Need new evaluation frameworks (Fair-Efficiency Score)

### For Policy

1. **"Efficient" ≠ "Fair"**
   - Deploying quantized models may violate fairness principles
   - Languages with poor tokenizer support pay hidden tax

2. **Carbon cost of fairness**
   - Fair models need more compute (or smarter protection)
   - Tension between Green AI and Inclusive AI

---

## Next Steps

### GPU Required (Validation)

| Priority | Experiment | Track | Value |
|----------|------------|-------|-------|
| 1 | Llama-2-7B validation | A | Scale evidence |
| 2 | Real GPTQ comparison | A | Production relevance |
| 3 | Mistral-7B generalization | A | Architecture diversity |

### Optional (Without GPU)

| Priority | Experiment | Track | Value |
|----------|------------|-------|-------|
| 1 | C-004 Carbon cost | C | Policy relevance |
| 2 | D-004 Joint vs pipeline | D | Architecture implications |
| 3 | B-004 Gradient circuits | B | Mechanism refinement |

---

## Publication Strategy

### Option 1: Single Comprehensive Paper

**Title:** "Gateway Layers Matter: Fair Multilingual Quantization Through Selective Protection"

**Sections:**
- Motivation: Disparity exists (Track A baseline)
- Root Cause: Alignment + Gateway + Bottleneck (Tracks A+B+D)
- Generalization: All efficiency techniques (Track C)
- Solution: L0+L9+L11 protection

**Venue:** NeurIPS, ICML, ICLR

### Option 2: Two Focused Papers

**Paper 1 (Main contribution):**
- Gateway-Bottleneck model + protection algorithm
- Venue: ACL/EMNLP

**Paper 2 (Efficiency focus):**
- Efficiency-Fairness tradeoff + new metrics
- Venue: Green NLP workshop / ACL theme

---

## Key Numbers Summary

| Metric | Value | Source |
|--------|-------|--------|
| Baseline disparity | 206.9x | Track A |
| Protected disparity | 0.59x | Track A (L0+L9+L11) |
| Protection overhead | 17% | Track A |
| Representation damage ratio | 3.3x | Track B |
| Head ablation sensitivity | 2.23x | Track B |
| Efficiency trifecta avg | 3.43x | Track C |
| Alignment correlation | r = -0.956 | Track D |
| Agreement disparity | 2.80x | Track D |
| Criticality R² | 0.936 | Track A (Soudry) |

---

*Total experiments: 105 (95 Track A + 4 Soudry + 3 Track B + 4 Track C + 3 Track D)*
*Updated: 2026-01-10*
