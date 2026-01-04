---
layout: home
title: Home
nav_order: 1
---

# Quantization Disparity in Multilingual LLMs

**Why does model compression hurt some languages more than others?**

---

## The Problem

When large language models are quantized (compressed from 16-bit to 8-bit or 4-bit weights), performance degrades unevenly across languages. Low-resource languages like Swahili, Hebrew, and Yoruba suffer disproportionately compared to English and German.

This isn't just a technical curiosity—it's a fairness issue. As quantized models become the default for deployment, billions of speakers of underrepresented languages get worse AI.

---

## Our Discovery

We found a **mechanistic explanation** for this disparity:

```
Correlation: r = -0.834 (p = 0.0002)
Languages activating outlier layers LESS degrade MORE
```

The key insight: **outlier weights in attention projections** are the culprit. These extreme-magnitude weights (kurtosis > 100) form during training and concentrate in attention mechanisms. When quantization clips them, languages with sparser representations in these circuits suffer most.

---

## Key Findings

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Outliers concentrate in attention | 3/3 models show κ > 200 in `attn.out_proj` | Target attention for fixes |
| Outliers grow during training | 82x increase from step 1k to 143k | Training dynamics matter |
| 16.7% of heads are language-specific | mBERT attention analysis | Some circuits are vulnerable |
| Token fertility ≠ degradation | r = -0.07 | Tokenization isn't the cause |

---

## Research Tracks

We're pursuing four parallel research tracks, each targeting a different Israeli AI lab:

<div class="track-grid">

### [Track A: Quantization Mechanics](/quant-disparity/tracks#track-a)
**Target: Soudry Lab (Technion)**
Core findings on outlier weights and disparity correlation.

### [Track B: Circuit Interpretability](/quant-disparity/tracks#track-b)
**Target: Belinkov Lab (Technion)**
Which attention heads matter for which languages?

### [Track C: Efficiency & Fairness](/quant-disparity/tracks#track-c)
**Target: Schwartz Lab (HUJI)**
Do efficient models amplify language disparities?

### [Track D: Syntax & Morphology](/quant-disparity/tracks#track-d)
**Target: Goldberg Lab (Bar-Ilan)**
How does quantization affect morphologically rich languages?

</div>

---

## Current Status

```
Research Progress    [████████░░] 80%
Experiments Run      27 / ~35 planned
Hypotheses Tested    12 (7 supported, 3 rejected, 2 mixed)
GPU Experiments      Pending access
Publication Target   EMNLP 2027 / ACL 2027
```

---

## Quick Links

- [Experiment Ledger](/quant-disparity/experiments) — All experiments with results
- [Literature Review](/quant-disparity/literature) — Related work and paper digests
- [Methodology](/quant-disparity/methodology) — Our research approach
- [Roadmap](/quant-disparity/roadmap) — What's next

---

## The Proposed Fix: LA-ACIQ

We propose **Language-Aware Analytical Clipping for Integer Quantization** (LA-ACIQ):

1. Identify super weights in attention projections
2. Apply language-specific clipping thresholds
3. Preserve critical circuits for low-resource languages
4. Theoretical validation: r = +0.84 correlation with expected improvement

---

*Last updated: January 2026*
