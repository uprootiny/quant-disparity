# Literature Review

## Core Papers

### Banner et al. 2019 — Post-Training 4-bit Quantization

**Key contribution:** Analytical solution for optimal clipping threshold.

```
α* = argmin_α { clip_error(α) + quant_error(α) }
```

**Relevance:** Our theoretical foundation. We extend to multilingual case.

### Chmiel et al. 2025 — FP8 Training at Scale

**Key contribution:** Identified outlier amplification as cause of FP8 instability.

**Method:** Monitor per-layer kurtosis during training.

**Relevance:** We use kurtosis as diagnostic for quantization sensitivity.

### Marchisio et al. 2024 — Multilingual Quantization Disparity

**Key contribution:** Documented non-uniform degradation across 40+ languages.

**Finding:** Non-Latin scripts degrade 2-3x more under W4.

**Relevance:** The motivating observation for our research.

## Supporting Papers

### Hubara et al. 2017 — Quantized Neural Networks

Foundational paper on QNNs. Introduced Straight-Through Estimator (STE).

### Frantar et al. 2023 — GPTQ

Layerwise Hessian-based quantization. Practical PTQ method.

### Lin et al. 2023 — AWQ

Activation-aware weight quantization. Considers activation patterns.

### Belinkov 2022 — Probing Classifiers

Linear probes for linguistic features. Alternative neuron identification.

## Linguistic Resources

### WALS — World Atlas of Language Structures

Cross-linguistic structural features. Source for typological classification.

### Lupyan-Dale Index

Morphological complexity measure. Potential predictor variable.

## Citation Graph

```
Banner 2019
    └─→ [our work] ←─ Marchisio 2024
         ↑
    Chmiel 2025
```
