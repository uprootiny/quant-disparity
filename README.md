# quant-disparity

**Why do multilingual LLMs degrade non-uniformly under quantization?**

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Weight distribution kurtosis predicts quantization sensitivity.           │
│  High-kurtosis languages need larger clipping thresholds.                  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Status

| Hypothesis | Predictor | r | p | Status |
|------------|-----------|---|---|--------|
| Tokenization | fertility | +0.34 | >0.05 | falsified |
| Per-language weight kurtosis | kurtosis | +0.92 | <0.001 | mock only |
| **Layer activation pattern** | **weighted kurtosis** | **-0.77** | **0.001** | **significant** |

**Key Finding:** Languages with LOWER activation-weighted kurtosis degrade MORE (r=-0.77).

```
eng/fra: w.kurt=43 → degradation=0.005-0.007  (low)
ara/hin: w.kurt=38 → degradation=0.021-0.025  (high)
```

**Interpretation:** Under-represented languages rely more on early layers, which may be more fragile.

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

1. **[NEXT]** Extract real weight statistics from BLOOM-560M
2. **[WAIT]** Build layer×language sensitivity matrix
3. **[WAIT]** Develop language-aware quantization

## References

- Banner et al. 2019 — Post-Training 4-bit Quantization
- Chmiel et al. 2025 — FP8 Training at Scale
- Marchisio et al. 2024 — Multilingual Quantization Disparity

## License

MIT
