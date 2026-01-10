# Theory Derivation Gap: What We Know vs. What We Can Prove

*The gap between empirical findings and theoretical understanding*

---

## Current State

### What We HAVE (Empirical)

| Finding | Evidence | Confidence |
|---------|----------|------------|
| L0+L9+L11 reduces disparity to 0.59x | 80 experiments | HIGH |
| Variance predicts criticality (R²=0.936) | Exp-087b | HIGH |
| MRLs benefit 9.7x more from protection | Exp-089 | HIGH |
| √3 threshold mechanism differs | Exp-082 | HIGH |
| Token fertility ≠ degradation | Exp-059 | HIGH |

### What We DON'T HAVE (Theoretical)

| Gap | Why It Matters |
|-----|----------------|
| **Why variance predicts criticality** | Correlation ≠ causation |
| **Why L0 and L11 are synergistic** | Empirical finding, no derivation |
| **Why L9 is the consolidation point** | Position at 75% is arbitrary |
| **Connection to information theory** | No bits/entropy analysis |
| **Formal disparity bound** | Can't predict minimum achievable disparity |

---

## The Theory Gap Explained

### 1. Variance-Criticality Relationship

**What we found:**
```
log(disparity) = 4.76 - 2.22 × norm(variance) + ...
R² = 0.936
```

**What we can't explain:**
- Why does higher variance → lower disparity?
- Is this because high-variance layers encode more information?
- Or because high-variance layers are more robust to noise?
- Or because of some interaction with quantization scale selection?

**Relevant Soudry work:**
- [ACIQ](https://arxiv.org/abs/1810.05723): Optimal clipping depends on distribution
- Insight: High-variance weights have larger α*, which means less relative error

**Needed derivation:**
```
Given: weight distribution W ~ N(0, σ²)
       quantization to b bits
       clipping at α = 2.5σ

Show: E[disparity] ∝ f(σ, language_activation_pattern)
```

---

### 2. L0+L11 Synergy

**What we found:**
- L0 alone: 3.6x disparity
- L11 alone: 336x disparity (HARMFUL)
- L0+L11 together: 0.7x disparity (synergistic)

**What we can't explain:**
- Why is L11 harmful alone but beneficial with L0?
- What information flows from L0 to L11?
- Is this related to residual connections?

**Relevant theory:**
- [Signal propagation in transformers](https://arxiv.org/abs/2403.02579)
- Residual stream acts as "memory" across layers
- First and last layers are "gateways" to/from the stream

**Needed derivation:**
```
Model: x_{L11} = f(x_L0, residual_stream)

If x_L0 is quantized: errors propagate to x_{L11}
If only x_{L11} is protected: x_L0 errors still damage output

Show: Protecting L0 "stabilizes" the residual stream,
      making L11 protection effective
```

---

### 3. Why 75% Depth (L9)?

**What we found:**
- L9 is at 75% depth (9/12 layers)
- Adding L9 to L0+L11 improves from 0.7x to 0.59x

**What we can't explain:**
- Why 75%? Why not 50% or 90%?
- Is this related to information consolidation?
- Does morphological disambiguation happen at L9?

**Relevant theory:**
- [Simplifying Transformer Blocks](https://proceedings.iclr.cc/paper_files/paper/2024/file/24fd58f52ff8d0496add8da3991644e9-Paper-Conference.pdf) (ICLR 2024)
- Mean field analysis shows different behavior at different depths
- Late layers may consolidate before final projection

**Needed derivation:**
```
Hypothesis: Layer L9 performs representation consolidation

Evidence needed:
1. Activation similarity analysis (languages converge at L9)
2. Information bottleneck analysis (compression at L9)
3. Morphological feature probing (disambiguation at L9)
```

---

### 4. Information-Theoretic Bound

**What we DON'T have:**
- Minimum achievable disparity given model/quantization
- Fundamental limit on fairness

**Relevant theory:**
- Rate-distortion theory for quantization
- [Mean Field Theory of Quantization](https://arxiv.org/abs/1906.00771): L_max ∝ N^1.82

**Needed derivation:**
```
Define: Mutual information I(X; Y) between:
  X = language-specific features
  Y = quantized representations

Show: disparity ≥ f(I(X;Y), bits, language_distribution)

This would give: "You cannot achieve disparity < k without protecting
                  at least n% of weights"
```

---

## Research Directions to Close Gap

### Direction 1: Information Flow Analysis

**Approach:** Track information content through layers per language.

**Method:**
```python
# Compute entropy/MI at each layer per language
for layer in range(num_layers):
    for lang in languages:
        activations = get_activations(model, text[lang], layer)
        entropy[layer][lang] = compute_entropy(activations)

# Find where languages diverge/converge
# Connect to L0/L9/L11 criticality
```

**Expected insight:** L0 diverges languages, L9 re-converges, L11 projects.

---

### Direction 2: Causal Tracing

**Approach:** Use activation patching to find causal paths.

**Method:**
```python
# Patch activations from FP32 to INT4 model
# Find which patches restore performance
# This reveals CAUSAL layer importance
```

**Expected insight:** Formal causal structure, not just correlation.

---

### Direction 3: Mean Field Extension

**Approach:** Extend Soudry's mean field theory to multilingual setting.

**Key modification:**
- Standard: All tokens/languages assumed equal
- Multilingual: Different effective distributions per language

**Expected insight:** Formal derivation of why languages differ.

---

### Direction 4: Connect to Attention Sink Theory

**Recent work:**
- [When Attention Sink Emerges](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1b04face60081b689ba740d39ea8f37-Paper-Conference.pdf) (ICLR 2025)
- [Attention Sinks and Outlier Features](https://arxiv.org/abs/2502.00919): "Catch, Tag, and Release"

**Key insight:** Attention sinks are caused by low-rank structures in attention weights. These structures enable dynamic segmentation.

**Connection:** Do L0/L11 have specific attention sink patterns that enable multilingual processing?

---

## What We Can Do Now (CPU)

1. **Information flow analysis** using cached activations
2. **Entropy computation** per layer per language
3. **Theoretical derivation** on paper
4. **Literature synthesis** connecting to existing theory

## What Needs GPU

1. **Causal tracing** (requires gradient computation)
2. **Full activation patching** (memory intensive)
3. **Large model validation** (>3GB)

---

## Publication Strategy

### Option A: Empirical Paper
- Present findings as-is
- Acknowledge theoretical gap
- Position as "discovery" paper

### Option B: Theory + Empirical
- Derive theoretical framework first
- Use experiments as validation
- Stronger contribution, harder to execute

### Option C: Workshop + Full Paper
- Submit empirical findings to workshop
- Develop theory for full paper later
- Lower risk, staged approach

---

## Key References for Theory Development

1. **Soudry Lab:**
   - [ACIQ](https://arxiv.org/abs/1810.05723) - Optimal clipping theory
   - [Mean Field Theory](https://arxiv.org/abs/1906.00771) - Quantization-depth tradeoff
   - [FP4 Training](https://arxiv.org/abs/2505.19115) - √3 threshold

2. **Signal Propagation:**
   - [Geometric Dynamics](https://arxiv.org/abs/2403.02579) - Trainability prediction
   - [Simplifying Transformers](https://iclr.cc/paper/2024) - Block analysis

3. **Attention Mechanisms:**
   - [Attention Sink](https://arxiv.org/abs/2309.17453) - Why first token matters
   - [Catch Tag Release](https://arxiv.org/abs/2502.00919) - Low-rank structures

4. **Multilingual:**
   - [How Quantization Affects Multilingual LLMs](https://arxiv.org/abs/2407.03211) - EMNLP 2024
   - [Layer Relevance Propagation](https://arxiv.org/abs/2401.11243) - Mixed precision

---

*Created: 2026-01-09*
