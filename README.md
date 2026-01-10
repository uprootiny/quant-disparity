# Multilingual Quantization Disparity

**Why do compressed LLMs hurt low-resource languages disproportionately, and how can we fix it?**

## Key Findings (105 experiments)

| Finding | Metric | Source |
|---------|--------|--------|
| **L0+L9+L11 protection achieves 0.59x disparity** | 17% overhead | Track A |
| Alignment predicts degradation | r = -0.956 | Track D |
| Efficiency trifecta: ALL compression hurts LR | 3.43x avg | Track C |
| LR languages show 3.3x representation damage | cosine distance | Track B |
| Head ablation confirms gateway layers | 2.23x sensitivity | Track B |
| Complex agreement drops 2.80x more for MRLs | accuracy | Track D |

## The Problem

Quantized multilingual models show **206x disparity** in perplexity degradation:
- English: 30% degradation
- Hebrew: 300%+ degradation

This isn't just perplexity—**grammatical correctness suffers**:
- Hebrew long-distance agreement: 54% → 28% accuracy under INT4

## The Solution: Gateway-Bottleneck Protection

```python
def fair_quantize(model):
    """Protect critical layers for multilingual fairness."""
    protect = {0, num_layers - 1}  # L0 + L_last
    if disparity_target < 0.7:
        protect.add(int(num_layers * 0.75))  # L_0.75
    return quantize_except(model, protect)
```

**Result:** 0.59x disparity with only 17% compute overhead.

## Research Tracks

| Track | Target Lab | Key Finding |
|-------|------------|-------------|
| **A: Gateway-Bottleneck** | Soudry (Technion) | L0+L9+L11 protection algorithm |
| **B: Interpretability** | Belinkov (Technion) | 3.3x representation damage, causal evidence |
| **C: Efficiency-Fairness** | Schwartz (HUJI) | Efficiency trifecta, Fair-Efficiency metric |
| **D: Morphology** | Goldberg (BIU-NLP) | Alignment is the root cause (r=-0.956) |

## Repository Structure

```
quant-disparity/
├── experiments/
│   ├── track-a-main/           # Gateway-bottleneck experiments
│   ├── track-b-interpretability/  # Belinkov-style circuit analysis
│   ├── track-c-efficiency/     # Green AI fairness analysis
│   ├── track-d-syntax/         # Morphological sensitivity
│   └── CROSS_TRACK_SYNTHESIS.md  # Unified theory
├── paper/
│   ├── PAPER_OUTLINE.md        # Full paper structure
│   └── soudry-prerequisites.md # Math background
├── gpu-experiments/            # Colab/Kaggle notebooks (validation)
└── README.md
```

## Quick Start

```bash
# Run a cross-track experiment
cd experiments/track-d-syntax
python3 exp_d003b_alignment_analysis.py

# Key output:
#   Alignment vs Degradation: r = -0.956
#   Hebrew alignment: 0.24, degradation: 264%
#   English alignment: 0.72, degradation: 47%
```

## Theoretical Framework

### Criticality Score Formula
```
score = 2.5×boundary + 1.5×variance + 0.8×kurtosis
      + 1.2×outliers + 1.0×consolidation - 0.5×distance
```
R² = 0.936 for predicting layer importance.

### Alignment-Gateway-Bottleneck Model
```
Tokenization → L0 (Gateway) → L1-L8 → L9 (Bottleneck) → L11 (Gateway) → Output
     ↓              ↓                       ↓                  ↓
Poor alignment   2.82x damage          4.15x damage       3.39x damage
for MRLs         disparity             disparity          disparity
```

## Key Metrics

| Metric | Definition |
|--------|------------|
| Disparity Ratio | mean(LR degradation) / mean(HR degradation) |
| Fair-Efficiency | throughput / disparity |
| Alignment | BPE-to-morpheme boundary agreement |

## Citation

```bibtex
@article{quant-disparity-2026,
  title={Gateway Layers Matter: Fair Multilingual Quantization Through Selective Protection},
  author={TBD},
  year={2026}
}
```

## License

MIT
