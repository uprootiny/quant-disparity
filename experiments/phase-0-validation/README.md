# Phase 0: Validation

**Status:** Complete (mock data)
**Next:** Phase 1 (real weight extraction)

## Purpose

Validate the weight distribution hypothesis using mock data before investing compute in real extraction.

## Results

```
kurtosis      r=+0.916  p<0.0001  [significant]
outlier_ratio r=+0.908  p<0.0001  [significant]
alpha/sigma   r=+0.916  p<0.0001  [significant]
```

## Files

| File | Purpose |
|------|---------|
| `distrib_analysis.py` | Main analysis script |
| `results.json` | Output correlations |

## Run

```bash
python3 distrib_analysis.py
```

## Caveat

Mock weight statistics are used. These were designed with expected patterns:
- Low kurtosis for Latin-script analytic languages
- High kurtosis for non-Latin morphologically rich languages

**VALIDATION NEEDED** with real model weights to confirm assumptions.
