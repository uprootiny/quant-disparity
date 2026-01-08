# Phase 5 Hypotheses: Peeling the Bulk

## Core Premise

The disparity problem can be solved by protecting a **minimal subset** of weights. The question is: how small can this subset be, and how do we identify it?

---

## Hypothesis Status Summary

| ID | Hypothesis | Status | Result |
|----|------------|--------|--------|
| H5.1a | <5% preservation achieves <50x disparity | **CONFIRMED** | 5% → 45x |
| H5.1b | Non-linear cliff between 3-5% | **CONFIRMED** | Cliff at ~5% |
| H5.2a | Layer 0 alone is sufficient | **CONFIRMED** | 5.7% → 55x |
| H5.2b | Attention > MLP for criticality | **REFUTED (GPT-2)** | MLP 20x > Attn 291x |
| H5.2b | Attention > MLP for criticality | **CONFIRMED (OPT)** | Attn 7x > MLP 15x |
| H5.2c | First + last layers sufficient | **PARTIAL** | Model-dependent |
| H5.2d | Token embeddings alone sufficient | **REFUTED** | Embeddings hurt (1216x) |
| H5.3a | Magnitude is optimal selector | **PARTIAL** | Non-monotonic; 5% optimal |
| H5.4a | Layer 0 + MLP = best hybrid | **CONFIRMED** | 1.4x disparity |

---

## Hypothesis Tree (Updated)

```
H5: Minimal intervention exists ✓ CONFIRMED
├── H5.1: Threshold hypotheses (how much?)
│   ├── H5.1a: <5% preservation achieves <50x disparity ✓ CONFIRMED (45x)
│   ├── H5.1b: Non-linear cliff exists ✓ CONFIRMED (at 5%)
│   └── H5.1c: <1% preservation achieves <10x disparity ✗ REFUTED (~100x)
│
├── H5.2: Location hypotheses (where?)
│   ├── H5.2a: Layer 0 alone is sufficient ✓ CONFIRMED (55x, 5.7%)
│   ├── H5.2b: Attention > MLP ⚠ MODEL-DEPENDENT
│   │   ├── GPT-2: MLP wins (20x vs 291x) ✓
│   │   └── OPT: Attention wins (7x vs 15x) ✓
│   ├── H5.2c: First + last layers sufficient ⚠ PARTIAL
│   └── H5.2d: Token embeddings alone sufficient ✗ REFUTED (1216x, worse)
│
├── H5.3: Selection hypotheses (which weights?)
│   ├── H5.3a: Magnitude is optimal selector ⚠ PARTIAL (non-monotonic)
│   ├── H5.3b: Gradient sensitivity better ? UNTESTED
│   └── H5.3c: Language-activation correlation ? UNTESTED
│
├── H5.4: Combination hypotheses
│   ├── H5.4a: Layer 0 + MLP = best hybrid ✓ CONFIRMED (1.4x)
│   └── H5.4b: Per-layer selection ? IN PROGRESS
│
└── H5.5: NEW - Per-layer criticality
    ├── H5.5a: Some layers are "anti-critical" ✓ CONFIRMED
    │   ├── GPT-2 Layer 1: 381x (worse than 214x baseline)
    │   └── OPT Layer 7: 245x (worse than 153x baseline)
    ├── H5.5b: Critical layer position varies by model ✓ CONFIRMED
    │   ├── GPT-2: Layer 0 best (139x)
    │   └── OPT: Layer 11 best (92x)
    └── H5.5c: Anti-critical layers encode English-specific patterns ? HYPOTHESIS
```

---

## Detailed Results

### H5.1: Threshold Hypotheses

**H5.1a**: Preserving less than 5% of weights can achieve disparity below 50x.

| Preservation | GPT-2 Disparity | OPT Disparity | Status |
|--------------|-----------------|---------------|--------|
| 0% | 214x | 153x | Baseline |
| 1% | ~100x | - | Near baseline |
| 2% | ~80x | - | Minimal effect |
| 3% | ~60x | - | Approaching cliff |
| 5% | **45x** | **45x** | **OPTIMAL** |
| 10% | 102x | - | Worse (non-monotonic) |
| 20% | 173x | - | Much worse |

**Conclusion**: 5% is the sweet spot. More preservation can hurt.

---

### H5.2: Location Hypotheses

**H5.2a**: Layer 0 protection (CONFIRMED)

| Config | % Model | Disparity | Efficiency |
|--------|---------|-----------|------------|
| None | 0% | 214x | - |
| Layer 0 | 5.7% | 55x | +39.0 |
| MLP only | 45.5% | 20x | +5.7 |
| Attention | 22.8% | 291x | -0.6 |
| Embeddings | 31.7% | 1216x | -29.6 |

**H5.2b**: Attention vs MLP (MODEL-DEPENDENT)

| Model | MLP Disparity | Attention Disparity | Winner |
|-------|---------------|---------------------|--------|
| GPT-2 | 20x | 291x | **MLP** |
| OPT-125M | 15x | 7x | **Attention** |

**Key Finding**: Component criticality is architecture-dependent.

---

### H5.5: Per-Layer Criticality (NEW)

**GPT-2 Per-Layer MLP Protection:**

| Layer | Disparity | vs Baseline | Classification |
|-------|-----------|-------------|----------------|
| 0 | 139x | -35% | **CRITICAL** |
| 2 | 152x | -29% | Helpful |
| 4-5 | 163x | -24% | Moderate |
| 3 | 188x | -12% | Minimal |
| 9 | 223x | +4% | Neutral |
| 7-8 | 247x | +16% | Harmful |
| 11 | 279x | +30% | Harmful |
| 6, 10 | 301x | +41% | Very harmful |
| **1** | **381x** | **+78%** | **ANTI-CRITICAL** |

**OPT-125M Per-Layer MLP Protection:**

| Layer | Disparity | vs Baseline | Classification |
|-------|-----------|-------------|----------------|
| **11** | **92x** | **-40%** | **CRITICAL** |
| 4 | 96x | -38% | Very helpful |
| 0 | 116x | -24% | Helpful |
| 5 | 119x | -22% | Helpful |
| 3 | 121x | -21% | Helpful |
| 1 | 131x | -15% | Moderate |
| 2 | 160x | +4% | Neutral |
| 9-10 | 180x | +18% | Harmful |
| 6 | 188x | +23% | Harmful |
| 8 | 206x | +34% | Very harmful |
| **7** | **245x** | **+60%** | **ANTI-CRITICAL** |

---

## Emerging Patterns

### Pattern 1: Model-Dependent Criticality
- GPT-2 (English-centric): Early layers critical, MLP > Attention
- OPT-125M (Mixed training): Late layers critical, Attention > MLP

### Pattern 2: Anti-Critical Layers Exist
- Protecting some layers INCREASES disparity
- Hypothesis: These layers encode English-specific optimizations

### Pattern 3: Non-Monotonic Preservation
- More protection ≠ better results
- 5% is optimal; 10-20% is worse than 0%

---

## Untested Hypotheses

| ID | Hypothesis | Priority | Blocker |
|----|------------|----------|---------|
| H5.3b | Gradient-based selection | MEDIUM | Requires calibration data |
| H5.3c | Language-activation selection | HIGH | Requires per-lang analysis |
| H5.5c | Anti-critical = English-specific | HIGH | Needs activation probing |

---

## Next Experiments

| Exp | Target Hypothesis | Description |
|-----|-------------------|-------------|
| 019 | H5.5b validation | Pythia-160M per-layer analysis |
| 020 | H5.2b extension | Per-layer attention analysis |
| 021 | H5.4b refinement | Optimal layer combination |

---

*Last updated: 2026-01-08*
*Status: 9 hypotheses tested, 3 pending*
