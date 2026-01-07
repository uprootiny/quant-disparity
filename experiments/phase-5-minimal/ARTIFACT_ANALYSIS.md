# Practical Artifact Analysis: Binary Diffs for Fairness

## The Question

If we're creating "language fairness patches" for quantized models, what's the actual size of the artifacts we'd be distributing?

---

## Model Size Breakdown (GPT-2 124M)

| Format | Params | Bits/Param | Size | Ratio |
|--------|--------|------------|------|-------|
| FP32 | 124M | 32 | 496 MB | 1.00x |
| FP16 | 124M | 16 | 248 MB | 0.50x |
| INT8 | 124M | 8 | 124 MB | 0.25x |
| INT4 | 124M | 4 | 62 MB | 0.125x |

---

## The "Fairness Patch" Artifact

If we protect 5% of weights in FP16 while quantizing rest to INT4:

### Components

| Component | Description | Size |
|-----------|-------------|------|
| Base INT4 model | Quantized weights | 62 MB |
| Protection mask | Bitmap of which weights to protect | 0.8 MB |
| FP16 protected values | 6.2M weights × 2 bytes | 12.4 MB |
| **Total** | | **75.2 MB** |

### Size Reduction vs Original

| Approach | Size | vs FP32 | vs FP16 |
|----------|------|---------|---------|
| Original FP32 | 496 MB | 1.00x | 2.00x |
| Original FP16 | 248 MB | 0.50x | 1.00x |
| Pure INT4 | 62 MB | 0.125x | 0.25x |
| Fair INT4 (5% FP16) | 75 MB | 0.15x | 0.30x |

**Overhead of fairness: 13 MB (21% more than pure INT4)**

---

## Distributable Artifact Options

### Option 1: Complete Fair Model
- Full INT4 model with FP16 patches embedded
- Size: 75 MB
- Pro: Self-contained
- Con: Must redistribute entire model

### Option 2: Delta Patch
- Just the protection mask + FP16 values
- Size: 13.2 MB
- Pro: Small, applies to any quantized version
- Con: Requires base model

### Option 3: PEFT-style Adapter
- Low-rank approximation of critical weights
- Size: ~2-5 MB (rank-dependent)
- Pro: Smallest
- Con: Approximation may reduce effectiveness

---

## SOTA Techniques Compendium

### Quantization Methods

| Method | Bits | Approach | Multilingual Impact |
|--------|------|----------|---------------------|
| GPTQ | 4 | Layer-wise OBQ | Unknown disparity |
| AWQ | 4 | Activation-aware | Claims better multilingual |
| bitsandbytes | 4/8 | Block-wise | Standard library |
| LLM.int8() | 8 | Outlier preservation | Some protection |
| GGML/GGUF | 2-8 | Block quantization | Popular format |
| QLoRA | 4+LoRA | Quantize + adapt | Training-time |

### Preservation/Protection Techniques

| Technique | Description | Overhead |
|-----------|-------------|----------|
| Outlier channels | Keep entire channels in FP16 | ~5-10% |
| Mixed precision | Different bits per layer | Variable |
| Activation-aware | Protect high-activation weights | ~3-5% |
| Sensitivity-based | Protect high-gradient weights | Compute |
| Block-wise | Protect specific blocks | Variable |

### Compression/Distribution Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Delta compression | Store only differences | Updates |
| LoRA/PEFT | Low-rank adaptation | Fine-tuning |
| Bitstream coding | Entropy-coded weights | Minimal size |
| Sparse matrices | Only store non-zeros | Pruned models |

---

## Grammar of Operations

### Standard Pipeline

```
Source Model (FP32/FP16)
    │
    ├─► Calibration (language-balanced dataset)
    │
    ├─► Sensitivity Analysis
    │   ├── Magnitude-based
    │   ├── Gradient-based
    │   └── Activation-based
    │
    ├─► Protection Mask Generation
    │   ├── Top-k by magnitude
    │   ├── Layer-specific
    │   └── Head-specific
    │
    ├─► Quantization
    │   ├── Protected weights → FP16
    │   └── Other weights → INT4/INT8
    │
    └─► Packaging
        ├── Monolithic (full model)
        └── Delta (patch only)
```

### Operations Grammar

```
ANALYZE(model, calibration_data) → sensitivity_scores
SELECT(sensitivity_scores, k%) → protection_mask
QUANTIZE(model, bits, protection_mask) → quantized_model
PACK(quantized_model, format) → artifact
APPLY(base_model, delta_artifact) → fair_model
```

---

## Minimal Intervention Estimates

### By Preservation Strategy

| Strategy | Weights | Size Overhead | Expected Disparity |
|----------|---------|---------------|-------------------|
| Top 1% magnitude | 1.2M | 2.4 MB | ~100x |
| Top 3% magnitude | 3.7M | 7.4 MB | ~60x |
| Top 5% magnitude | 6.2M | 12.4 MB | ~45x |
| Layer 0 only | 45.7M | 91.4 MB | Unknown |
| Embeddings only | 38.6M | 77.2 MB | Unknown |
| First attention | 2.4M | 4.8 MB | Unknown |

### Sweet Spot Hypothesis

Based on our data, the sweet spot appears to be:
- **Protection level**: 3-5% of weights
- **Size overhead**: 6-13 MB
- **Disparity reduction**: 50-70%
- **Total artifact**: 70-75 MB (vs 62 MB pure INT4)

---

## Practical Distribution Scenarios

### Scenario 1: Model Hub Integration
- HuggingFace hosts "fair quantized" variants
- Users download complete fair model
- Size: 75 MB per model

### Scenario 2: Patch Repository
- Separate repo of "fairness patches"
- Apply to any compatible quantized model
- Size: 13 MB per patch

### Scenario 3: On-demand Protection
- Quantization library computes protection at runtime
- No additional distribution needed
- Cost: Compute overhead at quantization time

---

## Research Gaps (Don't Reinvent)

### Already Solved
- ✓ Quantization algorithms (GPTQ, AWQ, etc.)
- ✓ Mixed-precision formats (bitsandbytes)
- ✓ Delta distribution (git-lfs, HF Hub)
- ✓ Sparse matrix formats (CSR, CSC)

### Our Novel Contribution
- **Language-aware weight selection**
- **Disparity-minimizing protection masks**
- **Multilingual calibration methodology**
- **Fairness-vs-overhead tradeoff analysis**

---

## Next Steps

1. **Exp-011b**: Measure actual disparity at 1%, 2%, 3%
2. **Exp-012**: Test layer-specific protection (smaller overhead?)
3. **Exp-013**: Compare magnitude vs activation-based selection
4. **Prototype**: Create actual delta patch format

---

*Created: 2026-01-05*
