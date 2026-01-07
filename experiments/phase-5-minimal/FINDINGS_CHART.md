# Phase 5 Findings: Visual Summary

## The Disparity Landscape

```
         Disparity (log scale)
         │
   1000+ │  ●─────────────────────────  Embeddings only (1216x) ❌
         │
    500+ │
         │
    300+ │  ●─────────────────────────  Attention only (291x) ❌
    277  │  ●─────────────────────────  Baseline (no protection)
         │
    200+ │  ●─────────────────────────  1% magnitude (215x)
         │
    100+ │  ●─────────────────────────  Layer 0+embed (121x)
         │
     50+ │  ●─────────────────────────  Layer 0 only (55x) ✓ EFFICIENT
         │  ●─────────────────────────  First 3 layers (52x)
         │
     20+ │  ●─────────────────────────  MLP only (20x) ✓ BEST
         │
     10+ │  ═══════════════════════════  TARGET: <10x
         │
      1+ │
         └──────────────────────────────────────────────────────
              0%        20%       40%       60%       80%    100%
                          Weights Protected
```

## Efficiency Chart (Disparity Reduction per % Protected)

```
Efficiency Score
(higher = better)

     40 │  ██████████████████████████████████████  Layer 0 (39.0)
        │
     30 │
        │
     20 │
        │
     13 │  █████████████████  First 3 Layers (13.2)
        │
      6 │  ████████  MLP Only (5.7)
      4 │  █████  Layer 0+Embed (4.2)
      0 ├────────────────────────────────────────────────────
        │
    -1  │  ▓  Attention Only (-0.6) ❌
        │
   -30  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Embeddings (-29.6) ❌
        │
```

## The Surprise Finding

```
    EXPECTED                         ACTUAL
    ────────                         ──────

    Embeddings ─┐                    MLP ─────────────┐
                │                                     │ Most
    Attention ──┼── Critical         Layer 0 ────────┼── Critical
                │                                     │
    MLP ────────┘                    Embeddings ─────┘ Least

    We thought embeddings           MLP layers actually
    and attention were              determine multilingual
    the key to multilingual         fairness under
    representation                  quantization
```

## Binary Diff Size Analysis

```
    Full Model (FP32)                    496 MB
    ════════════════════════════════════════════

    Pure INT4                             62 MB
    ════════════════

    Layer 0 Protected (5.7%)              76 MB  ← Most efficient
    ════════════════════

    MLP Protected (45.5%)                118 MB  ← Best disparity
    ════════════════════════════════════

    Target: <10x disparity               ??? MB  ← Need hybrid approach
```

## Strategy Recommendation

```
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMENDED APPROACH: Hybrid Layer 0 + Selective MLP          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Protect Layer 0 completely (5.7%)                          │
│     → Gives 55x disparity                                      │
│                                                                 │
│  2. Add top-k% MLP weights by magnitude                        │
│     → Target: <10% total protected                             │
│                                                                 │
│  3. Expected result: ~30-40x disparity at ~10% overhead        │
│                                                                 │
│  4. Further: Identify which MLP neurons matter most            │
│     → Language-specific activation analysis                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Next Experiments Needed

1. **Exp-013**: Hybrid Layer 0 + top-k% MLP
2. **Exp-014**: MLP neuron-level analysis
3. **Exp-015**: Validate on OPT-125M

---

*Visualized: 2026-01-05*
