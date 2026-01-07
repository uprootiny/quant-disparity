# Phase 5 Hypotheses: Peeling the Bulk

## Core Premise

The disparity problem can be solved by protecting a **minimal subset** of weights. The question is: how small can this subset be, and how do we identify it?

---

## Hypothesis Tree

```
H5: Minimal intervention exists
├── H5.1: Threshold hypotheses (how much?)
│   ├── H5.1a: <5% preservation achieves <50x disparity
│   ├── H5.1b: <2% preservation achieves <20x disparity
│   └── H5.1c: <1% preservation achieves <10x disparity
│
├── H5.2: Location hypotheses (where?)
│   ├── H5.2a: Layer 0 alone is sufficient
│   ├── H5.2b: Attention layers are more critical than MLP
│   ├── H5.2c: First + last layers are sufficient
│   └── H5.2d: Token embeddings alone are sufficient
│
├── H5.3: Selection hypotheses (which weights?)
│   ├── H5.3a: Magnitude is optimal selector
│   ├── H5.3b: Gradient sensitivity is better than magnitude
│   ├── H5.3c: Language-activation correlation identifies critical weights
│   └── H5.3d: Attention pattern entropy identifies critical heads
│
└── H5.4: Combination hypotheses
    ├── H5.4a: Layer 0 + top 1% = 5% magnitude
    ├── H5.4b: Embeddings + attention outliers < 3%
    └── H5.4c: Script-specific heads < 0.5%
```

---

## Detailed Hypotheses

### H5.1: Threshold Hypotheses

**H5.1a**: Preserving less than 5% of weights can achieve disparity below 50x.

| Preservation | Expected Disparity | Rationale |
|--------------|-------------------|-----------|
| 5% | 45-129x | Confirmed |
| 4% | ~50x | Interpolation |
| 3% | ~60x | Interpolation |
| 2% | ~80x | Approaching baseline |
| 1% | ~100x | Near baseline |

**Test**: Sweep k ∈ [1, 2, 3, 4] and measure disparity.

---

**H5.1b**: The relationship between preservation % and disparity is non-linear with a "cliff" somewhere between 3-5%.

```
Disparity
    │
200 ├────────────────────────────●─── 10% (worse!)
    │                           ╱
100 ├──────────────●───────────╱──── 0% baseline
    │             ╱
 50 ├────────────●─────────────────── 5% (current best)
    │           │
    │           │ ← cliff?
 20 ├───────────│────────────────────
    │           │
    └───────────┴────────────────────
        1%  2%  3%  4%  5%  10%
```

**Test**: Fine-grained sweep to find the cliff.

---

### H5.2: Location Hypotheses

**H5.2a**: Protecting Layer 0 (embeddings + first transformer block) alone achieves similar disparity reduction as 5% magnitude preservation.

| Layer 0 Components | Weights | % of Total |
|-------------------|---------|------------|
| Token embeddings | 38.6M | 31.0% |
| First attention | 2.4M | 1.9% |
| First MLP | 4.7M | 3.8% |
| First LayerNorm | 0.004M | 0.003% |
| **Total Layer 0** | **45.7M** | **36.8%** |

**Problem**: Layer 0 is 37% of model - larger than 5%!

**Refined hypothesis**: Only token embeddings (31%) or only first attention (2%) is critical.

---

**H5.2b**: Attention layers are more critical than MLP layers.

| Component | % of Model | Hypothesis |
|-----------|------------|------------|
| All Attention | 22.8% | More critical |
| All MLP | 46.2% | Less critical |

**Test**: Quantize only MLP vs only Attention and compare disparity.

---

### H5.3: Selection Hypotheses

**H5.3a**: Magnitude-based selection is near-optimal for minimizing disparity.

Current approach: Select top-k% by |weight|.

**H5.3b**: Gradient-based selection outperforms magnitude.

Alternative: Select weights with highest |∂L/∂w| on multilingual calibration set.

**H5.3c**: Language-activation correlation identifies script-critical weights.

Alternative: Select weights where activation differs most between languages.

---

### H5.4: Combination Hypotheses

**H5.4a**: Protecting embeddings (31%) + top 1% of remaining weights achieves better results than top 5% globally.

**H5.4b**: A hybrid strategy of layer-specific + magnitude-based protection achieves <3% overhead with <20x disparity.

---

## Priority Ranking

| Hypothesis | Priority | Difficulty | Expected Impact |
|------------|----------|------------|-----------------|
| H5.1a (threshold sweep) | HIGH | LOW | Immediate actionable |
| H5.2a (layer 0 test) | HIGH | LOW | Architecture insight |
| H5.2b (attn vs MLP) | MEDIUM | LOW | Component insight |
| H5.3a (magnitude optimal?) | MEDIUM | MEDIUM | Algorithm insight |
| H5.4b (hybrid strategy) | HIGH | MEDIUM | Practical solution |

---

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|--------------|
| Disparity ratio | <10x | 45x (5% preservation) |
| Memory overhead | <3% | 5% |
| Compute overhead | <5% | ~5% (threshold computation) |

---

## Experiment Mapping

| Hypothesis | Experiment | Est. Duration |
|------------|------------|---------------|
| H5.1a | Exp-011: Fine threshold sweep | 10 min |
| H5.2a | Exp-012: Layer 0 only | 5 min |
| H5.2b | Exp-013: Attention vs MLP | 10 min |
| H5.3a | Exp-014: Selection comparison | 15 min |
| H5.4b | Exp-015: Hybrid strategy | 10 min |

---

*Formulated: 2026-01-05*
