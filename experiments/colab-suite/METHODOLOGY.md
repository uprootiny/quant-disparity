# Experimental Methodology

## Overview

This suite validates the LA-ACIQ theory through controlled experiments on Google Colab T4 GPU.

## Research Questions

1. **RQ1:** Does quantization degrade low-resource languages more than high-resource languages?
2. **RQ2:** Does effective kurtosis predict quantization sensitivity?
3. **RQ3:** Can per-language optimal clipping reduce disparity?

## Hypotheses

| ID | Hypothesis | Prediction | Test |
|----|------------|------------|------|
| H1 | Disparity exists | D_LR / D_HR > 1.5 | PPL ratio comparison |
| H2 | Kurtosis correlation | r(κ_eff, D) < -0.7 | Pearson correlation |
| H3 | Fertility predicts | r(fertility, D) > 0.7 | Pearson correlation |
| H4 | Rate-distortion | slope ≈ -0.347 | Linear regression on log(D) vs B |
| H5 | LA-ACIQ helps | Disparity reduction > 20% | A/B comparison |

## Experimental Design

### Independent Variables
- **Model:** BLOOM-560M (fits T4 memory)
- **Bit-width:** 4, 8 bits (INT4 via bitsandbytes, INT8)
- **Language:** 8 languages spanning resource levels

### Dependent Variables
- **Perplexity (PPL):** exp(cross-entropy loss)
- **Degradation (D):** (PPL_quant - PPL_base) / PPL_base
- **Disparity:** max(D) - min(D)

### Controls
- Same text length per language (512 tokens)
- Temperature = 0 (deterministic)
- 5 samples per language (bootstrap variance)

## Language Selection

| Language | Code | Script | Resource Level | Justification |
|----------|------|--------|----------------|---------------|
| English | en | Latin | High | Baseline, HR |
| German | de | Latin | High | HR, morphologically rich |
| French | fr | Latin | High | HR, BLOOM training lang |
| Chinese | zh | Hanzi | High | Different script, HR |
| Arabic | ar | Arabic | Medium | RTL, morphological |
| Hebrew | he | Hebrew | Low-Medium | RTL, our focus |
| Swahili | sw | Latin | Low | Under-resourced |
| Yoruba | yo | Latin | Very Low | Under-resourced |

## Corpus

- **Source:** Wikipedia first paragraphs (neutral, factual)
- **Size:** 5 passages × 8 languages = 40 samples
- **Processing:** Tokenize, truncate to 512 tokens

## Statistical Analysis

1. **Normality:** Shapiro-Wilk test on degradation values
2. **Correlation:** Pearson if normal, Spearman otherwise
3. **Significance:** α = 0.05, Bonferroni correction for multiple comparisons
4. **Effect size:** Cohen's d for group comparisons, r² for correlations

## Hardware Requirements

- **GPU:** NVIDIA T4 (16GB VRAM)
- **RAM:** 12GB system RAM
- **Storage:** ~2GB for model weights
- **Runtime:** ~30 minutes total

## Reproducibility

- Fixed random seeds (42)
- Version-pinned dependencies
- All data saved to Drive

## References

1. Ahia, O., et al. (2021). "The Low-Resource Double-Bind." ACL.
2. Banner, R., et al. (2019). "Post-Training 4-bit Quantization." NeurIPS.
3. Dettmers, T., et al. (2022). "LLM.int8()." NeurIPS.
4. Frühwirth-Schnatter, S. (2006). "Finite Mixture Models." Springer.
