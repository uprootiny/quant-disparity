---
layout: text
title: Gateway Layers and Multilingual Quantization Disparity
permalink: /quant-disparity/
---

# Gateway Layers: Why Quantization Hurts Some Languages More

*80 experiments on multilingual fairness in model compression*

<details>
<summary><strong>Annotation Layer: Reading Guide</strong></summary>

This paper has three reading modes:

| Mode | What to read | Time |
|------|--------------|------|
| **TL;DR** | Just the gray boxes | 2 min |
| **Practitioner** | Main text + tables | 15 min |
| **Researcher** | Everything including annotations | 45 min |

Annotations appear in collapsible `<details>` blocks like this one. They contain:
- Hypothesis evolution and falsified ideas
- Experimental dead-ends
- Methodological caveats
- Connections to related work

</details>

---

## TL;DR — What We Found

> **Problem:** INT4 quantization degrades Hebrew 160x more than English.
>
> **Solution:** Protect layers 0, 9, and 11. Reduces disparity to 0.59x.
>
> **Cost:** 17% of weights stay in FP16. The rest can be INT4.

| Configuration | Overhead | Average Disparity | Assessment |
|---------------|----------|-------------------|------------|
| No protection | 0% | 160x | Catastrophic |
| L0 + L11 | 11.5% | 0.92x | Good |
| **L0 + L9 + L11** | **17%** | **0.59x** | **Recommended** |

*Disparity < 1.0x means non-English languages are LESS affected than English.*

---

## The Problem: Unequal Degradation

When you compress a multilingual model from FP32 to INT4:

| Language | Script | Degradation | Relative to English |
|----------|--------|-------------|---------------------|
| English | Latin | 5,549% | 1.0x (reference) |
| German | Latin | 4,982% | 0.9x |
| French | Latin | 3,842% | 0.7x |
| Hebrew | Hebrew | 231x worse | **42x** |
| Arabic | Arabic | 111x worse | **20x** |
| Chinese | CJK | 35x worse | **6x** |
| Korean | Hangul | 29x worse | **5x** |

**This is not acceptable.** A model that works equally well for English and Hebrew before compression shouldn't become 42x worse for Hebrew after compression.

<details>
<summary><strong>Annotation: Why does this happen?</strong></summary>

Our initial hypotheses (and their fate):

| Hypothesis | Status | Why |
|------------|--------|-----|
| Token fertility (more tokens = more errors) | **Falsified** | r = 0.000 correlation (Exp-059) |
| Script complexity | **Partial** | Correlates but doesn't explain mechanism |
| Training data volume | **Likely but indirect** | Under-represented → different weight usage |
| **Outlier layer activation** | **Supported** | r = -0.834 correlation with degradation |

The current theory: Low-resource languages don't activate the "outlier-heavy" layers (layers with high-kurtosis weight distributions). When these layers are quantized, they lose information that high-resource languages weren't using anyway.

**We don't claim to have proven causation.** The correlation is strong and consistent, but the mechanism remains a hypothesis.

</details>

---

## The Solution: Gateway Layer Protection

We discovered that protecting specific "gateway" layers eliminates most disparity:

```
Layer 0  (Input gateway)   — processes raw embeddings
Layer 9  (Consolidation)   — 75% through the network
Layer 11 (Output gateway)  — final transformer block
```

### Why These Layers?

| Layer | Role | Evidence |
|-------|------|----------|
| **L0** | First to see tokens | Highest outlier ratio (1.7%), highest sparsity (10.2%) |
| **L9** | Representation consolidation | Position at 75% depth, intermediate statistics |
| **L11** | Projects to vocabulary | Highest variance (0.026), synergy with L0 |

<details>
<summary><strong>Annotation: How we found this</strong></summary>

**Experiment evolution:**

1. **Exp-011: Threshold sweep** — Found 5% protection has non-linear effect
2. **Exp-017: Per-layer analysis** — Layer 0 consistently best for GPT-2
3. **Exp-032: L0+L11 synergy** — Together achieve 0.7x, L11 alone is harmful (336x)
4. **Exp-042: L0 uniqueness** — No other layer can replace L0 (L2+L11 = 4749x)
5. **Exp-072: Triple layer search** — L0+L9+L11 beats all 55 combinations tested

**Key insight from Exp-032:** L0 and L11 have synergy. L11 protected alone makes things WORSE (336x disparity). But L0+L11 together achieve 0.7x. This suggests they form a complementary "gateway" pair.

**Dead ends we tried:**
- Random protection (44-327x) — Exp-055
- Magnitude-based selection (125,480x) — Exp-056
- Protecting embeddings (522x, harmful) — Exp-025

</details>

---

## Experimental Results

### 80 Experiments, 10 Languages

Final validation across diverse languages:

| Language | Script | Tokens | Disparity (L0+L9+L11) | Assessment |
|----------|--------|--------|----------------------|------------|
| Korean | Hangul | 67 | 0.29x | EXCELLENT |
| Arabic | Arabic | 58 | 0.31x | EXCELLENT |
| Chinese | CJK | 34 | 0.31x | EXCELLENT |
| Japanese | Mixed | 42 | 0.33x | EXCELLENT |
| French | Latin | 30 | 0.43x | EXCELLENT |
| Hebrew | Hebrew | 62 | 0.48x | EXCELLENT |
| German | Latin | 25 | 0.67x | VERY GOOD |
| English | Latin | 14 | 1.00x | Reference |
| Spanish | Latin | 30 | 1.03x | GOOD |
| Russian | Cyrillic | 66 | 1.47x | GOOD |

**Average: 0.59x** — Non-English languages are slightly LESS affected than English.

<details>
<summary><strong>Annotation: Token count doesn't predict disparity</strong></summary>

Note that Korean (67 tokens) has LOWER disparity (0.29x) than Spanish (30 tokens, 1.03x). This falsifies the "more tokens = more errors" hypothesis.

From Exp-059, the correlation between token fertility and disparity is r = 0.000.

What matters is not how many tokens, but which layers process them.

</details>

### Configuration Comparison

| Config | Model Size | Disparity | Use Case |
|--------|------------|-----------|----------|
| No protection | 25% | 160x | Don't do this |
| Biases only | 25.1% | 108x | Marginal help |
| L0 + L11 | 37.1% | 0.92x | Minimum viable |
| **L0 + L9 + L11** | **37.8%** | **0.59x** | **Recommended** |
| L0 + L8 + L9 + L11 | 43% | 0.30x | Diminishing returns |
| INT8 critical + INT4 rest | 29.2% | 0.74x | Maximum compression |

---

## What Doesn't Work

We tested several intuitive approaches that fail:

### 1. Random Weight Selection

*"Just protect 11% of weights randomly"*

| Trial | Disparity |
|-------|-----------|
| Random seed 1 | 44x |
| Random seed 2 | 327x |
| Random seed 3 | 156x |
| **L0+L11 (same 11%)** | **0.86x** |

**Conclusion:** Structure matters. Protecting random weights is almost as bad as protecting nothing.

### 2. Magnitude-Based Selection

*"Protect the largest weights"*

| Method | % Protected | Disparity |
|--------|-------------|-----------|
| Top 11% by magnitude | 11% | 284x |
| Top 20% by magnitude | 20% | 89,432x |
| Top 38% by magnitude | 38% | **125,480x** |

**Conclusion:** Magnitude-based selection is CATASTROPHIC. It protects the wrong weights (mostly embeddings and LayerNorms, which should NOT be protected).

<details>
<summary><strong>Annotation: Why magnitude fails</strong></summary>

The largest weights are in:
1. Token embeddings (31.7% of model)
2. Position embeddings
3. Final LayerNorm

But Exp-025 showed that protecting embeddings INCREASES disparity (522x vs 214x baseline). The embedding layer encodes English-centric representations, and protecting it preserves that bias.

The lesson: **criticality ≠ magnitude**.

</details>

### 3. Protecting Embeddings

*"Embeddings encode language information"*

| Config | Disparity |
|--------|-----------|
| No protection | 214x |
| Token embeddings protected | **522x** (worse!) |
| Position embeddings protected | 133x (helps slightly) |
| **Layer 0 protected** | **55x** (much better) |

**Conclusion:** Token embeddings should NOT be protected. They encode English-centric representations.

---

## Architecture Matters

**Important caveat:** Our findings are specific to GPT-2. Different architectures need different layers.

| Model | Critical Layers | Best Disparity | Key Difference |
|-------|-----------------|----------------|----------------|
| GPT-2 | L0, L9, L11 | 0.59x | Post-LayerNorm, high L0 variance |
| OPT-125M | L4, L6, L11 | 12.7x | Pre-LayerNorm, low L0 variance |

<details>
<summary><strong>Annotation: Why OPT is different</strong></summary>

From Exp-062 and Exp-075:

GPT-2 Layer 0 has:
- Variance: 0.039 (highest)
- Outlier ratio: 1.7%

OPT Layer 0 has:
- Variance: 0.006 (6x lower)
- Different architecture (pre-LayerNorm)

OPT's critical layer is L4, not L0. The "gateway layer" concept transfers, but the specific layers don't.

**Implication:** You need to run a layer sweep for each architecture. We provide a tool for this.

</details>

### The Layer Sweep Method

For any new model:

1. Protect each layer individually
2. Measure disparity on English + one LR language
3. Rank by disparity (lower = more critical)
4. Protect top-2 or top-3 layers

This takes ~30 seconds per model and identifies architecture-specific critical layers.

---

## Limitations (Honest Assessment)

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Only tested 125M models** | Findings may not scale to 7B+ | Need GPU validation |
| **Simulated INT4** | Not real GPTQ/AWQ | Transfer validation needed |
| **Perplexity only** | Not downstream task performance | Task evaluation needed |
| **2 architectures** | GPT-2 and OPT only | More models needed |
| **No gradient analysis** | OOM blocked gradient-based selection | Need more memory |

<details>
<summary><strong>Annotation: What could invalidate our findings</strong></summary>

**High risk:**
- If GPTQ's calibration-based optimization already handles disparity → our protection is redundant
- If larger models have different layer criticality patterns → specific layers don't transfer

**Medium risk:**
- If perplexity doesn't correlate with task performance → metric is wrong
- If modern architectures (Llama, Mistral) have no gateway layers → concept doesn't generalize

**Low risk:**
- Basic finding that "some layers matter more" is robust across our tests
- The quick sweep method works for both models we tested

</details>

---

## Recommendations

### For Practitioners

```python
# GPT-2 and similar architectures
protect_layers = [0, 9, 11]  # ~17% overhead
protect_biases = True        # +0.08%
protect_final_ln = True      # +0.01%
quantize_rest_to = "INT4"    # 62% of weights
```

Expected result: **0.59x average disparity** across languages.

### For Researchers

1. **Validate on larger models** — Our 125M results need 7B+ confirmation
2. **Test real quantization** — GPTQ/AWQ may behave differently
3. **Investigate mechanism** — Why do gateway layers matter?
4. **Develop theory** — Connect to transformer interpretability work

### For Framework Developers

Consider adding a `--protect-layers` flag to quantization tools:

```bash
gptq-quantize model.safetensors \
  --bits 4 \
  --protect-layers 0,9,11 \
  --protect-biases
```

---

## Experimental Log

<details>
<summary><strong>Full experiment index (80 experiments)</strong></summary>

| Exp | Name | Key Finding |
|-----|------|-------------|
| 001-010 | Baseline series | 78-214x disparity confirmed |
| 011 | Threshold sweep | 5% is optimal threshold |
| 012 | Layer-specific | MLP > Attention for GPT-2 |
| 017 | Per-layer MLP | Layer 0 best |
| 020 | Per-layer attention | Layer 0 best |
| 025 | Embeddings | Token embeddings harmful |
| 030 | Anti-critical combos | Odd layers catastrophic (1379x) |
| 032 | L0+L11 synergy | 0.7x together, 336x for L11 alone |
| 039 | Biases | +52.6x when quantized |
| 042 | Synergy test | L0 uniquely enables synergy |
| 043 | L0 necessity | Irreplaceable (L2+L11 = 4749x) |
| 055 | Random protection | 44-327x (fails) |
| 056 | Magnitude selection | 125,480x (catastrophic) |
| 057-058 | Variance analysis | r = -0.798 correlation |
| 059 | Token fertility | r = 0.000 (falsified) |
| 061-064 | OPT analysis | Different critical layers |
| 072 | Triple layer search | L0+L9+L11 optimal |
| 078 | Precision study | INT4 sweet spot |
| 079 | Mixed precision | INT8 critical viable |
| 080 | Final validation | 0.59x across 10 languages |

</details>

---

## Citation

If you use these findings:

```bibtex
@misc{quantdisparity2026,
  title={Gateway Layers: Multilingual Quantization Disparity},
  author={[Anonymous]},
  year={2026},
  note={80 experiments on GPT-2 and OPT-125M},
  url={https://github.com/uprootiny/quant-disparity}
}
```

---

## Code & Data

- **Repository:** [github.com/uprootiny/quant-disparity](https://github.com/uprootiny/quant-disparity)
- **Layer sweep tool:** `pip install quant-fairness` (coming soon)
- **Experiment scripts:** `experiments/phase-5-minimal/`

---

*80 experiments | 10 languages | January 2026*

*This research was conducted under memory constraints (~3GB) using CPU only. Findings are validated on small models (125M parameters) and await confirmation on larger architectures.*
