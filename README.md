# quant-disparity

**Why do multilingual LLMs degrade non-uniformly under quantization?**

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Weight distribution kurtosis predicts quantization sensitivity.           │
│  High-kurtosis languages need larger clipping thresholds.                  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Status

| Phase | Experiments | Status |
|-------|-------------|--------|
| Phase 0: Validation | EXP-001, EXP-002 | COMPLETE |
| Phase 1: Analysis | EXP-003 to EXP-008 | COMPLETE |
| Phase 2: Corpus | EXP-009b (theory) | COMPLETE |
| Phase 3: Cross-model | EXP-020 (census) | IN PROGRESS |
| Phase 4: Algorithm | LA-ACIQ | FUTURE (needs GPU) |

See [STATUS.md](STATUS.md) for full details.

| Experiment | Finding | r | Status |
|------------|---------|---|--------|
| EXP-003 | Layer activation × kurtosis | -0.77 | Significant |
| EXP-007 | Outlier layer activation | **-0.83** | Confirmed |
| EXP-008 | Bootstrap validation | CI [-0.93,-0.65] | Robust |

**Key Finding:** Languages that activate outlier layers (5,21,22) LESS degrade MORE.

```
           BLOOM               XGLM
           ─────               ────
Layer kurt: 0.96–164.30       0.2–1.9 (near Gaussian)
W.kurt spread: 5.5            0.06 (uniform)
Correlation: r=-0.77 **       r=+0.38 n.s.
```

**Interpretation:** BLOOM's heavy-tailed weights create language-dependent patterns. XGLM's Gaussian weights don't differentiate.

## Quick Start

```bash
# Run the weight distribution analysis
cd experiments/phase-0-validation
python3 distrib_analysis.py

# Output:
#   kurtosis      r=+0.916  p<0.0001  [significant]
#   outlier_ratio r=+0.908  p<0.0001  [significant]
```

## Core Finding

Languages with **high-kurtosis weight distributions** (more outliers) degrade more under quantization because standard clipping (α*/σ ≈ 3.5) removes critical information.

| Regime | Languages | Kurtosis | Optimal α*/σ |
|--------|-----------|----------|--------------|
| Low | eng, fra, deu, vie | < 1.0 | ~3.5 |
| High | ara, heb, jpn, zho, kor, hin, tha | ≥ 1.5 | ~4.3 |

## Structure

```
quant-disparity/
├── README.md                 # This file
├── PROPOSAL.md               # Research proposal (Soudry Lab)
├── experiments/
│   ├── phase-0-validation/   # Mock data validation [current]
│   ├── phase-1-extraction/   # Real weight extraction [next]
│   └── phase-2-sensitivity/  # Layer sensitivity matrix [future]
├── data/
│   ├── degradation.json      # Marchisio et al. degradation values
│   ├── samples/              # Per-language text samples
│   └── weights/              # Extracted weight statistics [future]
├── docs/
│   ├── methodology.md        # Banner et al. framework
│   ├── papers.md             # Literature review
│   └── friction.md           # Blockers and workarounds
├── ledgers/
│   └── decisions.md          # Decision log
└── src/
    └── analysis.py           # Reusable analysis functions
```

## Theory

From Banner et al. (2019):

```
Total error = clipping_error + quantization_noise
α* = argmin_α { E[(X - clip(X,α))²] + Δ²/12 }

For Gaussian:  α*/σ ≈ 2.5 (4-bit)
For heavy-tail: α*/σ must increase with kurtosis
```

Our extension: **Languages activate different neuron subsets with different weight distributions.**

## Next Steps

1. **[DONE]** Extract real weight statistics from BLOOM-560M
2. **[DONE]** Establish activation-outlier mechanism (r=-0.83)
3. **[CURRENT]** EXP-009: Bit-width sweep (test threshold predictions)
4. **[NEXT]** Build layer×language sensitivity matrix
5. **[FUTURE]** Develop language-aware quantization

## References

- Banner et al. 2019 — Post-Training 4-bit Quantization
- Chmiel et al. 2025 — FP8 Training at Scale
- Marchisio et al. 2024 — Multilingual Quantization Disparity

## License

MIT
