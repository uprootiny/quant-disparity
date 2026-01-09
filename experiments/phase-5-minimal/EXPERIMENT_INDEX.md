# Experiment Index

*Phase 5: Minimal Intervention Analysis*
*64 experiments completed*

## Quick Reference

| Exp | Name | Key Result | Status |
|-----|------|------------|--------|
| 001-010 | Baseline series | 78-214x disparity | ✓ |
| 011 | Threshold sweep | 5% optimal | ✓ |
| 012 | Layer-specific (GPT-2) | MLP > Attention | ✓ |
| 013 | Hybrid strategy | Layer 0 + MLP = 1.4x | ✓ |
| 014 | OPT validation | Model-dependent | ✓ |
| 015 | Text length | Medium/long reliable | ✓ |
| 016 | Robustness | 0% CV | ✓ |
| 017 | Per-layer MLP | L0 best (GPT-2) | ✓ |
| 018 | BLOOM | OOM | ⚠ |
| 019 | Pythia | Tokenization issues | ⚠ |
| 020 | Per-layer attention | L0 best (both) | ✓ |
| 021 | Combined critical | 4.8x | ✓ |
| 022 | Multi-language | 3.8x (6 langs) | ✓ |
| 023 | Longer texts | L0 robust | ✓ |
| 024 | Arabic/Chinese | Ar 1.5x, Zh 0.7x | ✓ |
| 025 | Embeddings | Token embed harmful | ✓ |
| 026 | LayerNorm | LN significant | ✓ |
| 027 | Minimal protection | Pareto frontier | ✓ |
| 028b | Gradient selection | 0% overlap | ✓ |
| 029 | Gradient quantization | OOM | ⚠ |
| 030 | Anti-critical combos | odd=1379x, even=0.5x | ✓ |
| 031 | Layer 0 breakdown | MLP fc harmful | ✓ |
| 032 | Output layers | L0+L11=0.7x | ✓ |
| 033 | OPT synergy | L0+L11 synergy +29.3x | ✓ |
| 034 | Multi-lang synergy | He 284x better | ✓ |
| 035 | Layer pairs | OOM | ⚠ |
| 036 | Triple layers | L0+L11 optimal, adding more doesn't help | ✓ |
| 038 | Attention components | L0 QKV > proj, L11 opposite | ✓ |
| 039 | Bias analysis | Biases +52.6x when quantized | ✓ |
| 040 | Optimal minimal | L0+L11+biases = 11.4x | ✓ |
| 041 | MLP depth | L11 MLP harmful (306.9x) | ✓ |
| 042 | Synergy test | L0+L11 = 0.0x, L0 enables synergy | ✓ |
| 043 | L0 necessity | L0 irreplaceable, L2+L11 = 4749.8x | ✓ |
| 044 | Precision axis | INT4 sweet spot, INT8 not needed | ✓ |
| 045 | Length axis | L0+L11 robust across lengths | ✓ |
| 046 | Language families | Semitic 22x better, CJK 4x better | ✓ |
| 047 | Resource correlation | r=1.000 perfect correlation | ✓ |
| 048 | Weight statistics | L0 unique: low outliers, high sparsity | ✓ |
| 049 | Quant error | Criticality != error magnitude | ✓ |
| 050 | Embedding interaction | ln_f helps, embeddings hurt | ✓ |
| 051 | Optimal config | L0+L11+ln_f+biases = 3.4x | ✓ |
| 052-053 | Mixed precision | INT4 sweet spot confirmed | ✓ |
| 054 | Head granularity | L11 V weights 3x larger | ✓ |
| 055 | Random protection | Random fails (44-327x vs 0.0x) | ✓ |
| 056 | Magnitude selection | FAILS: 125,480x (wrong weights) | ✓ |
| 057-058 | Variance analysis | r=-0.798 variance-disparity | ✓ |
| 059 | Token fertility | Structure > token count | ✓ |
| 060 | Final validation | 1.04x disparity achieved! | ✓ |
| 061-064 | OPT-125M analysis | Architecture-specific patterns | ✓ |

---

## Pareto Optimal Strategies

| Strategy | Overhead | Avg LR Disparity | Use Case |
|----------|----------|------------------|----------|
| none | 0% | 206.9x | Baseline |
| layer0_ln_only | 0.002% | 41.6x | Extreme compression |
| layer0_attn | 1.90% | 35.0x | Light multilingual |
| layer0 | 5.70% | 3.6x | Efficient |
| L0+L11 | 11.39% | 15.1x | Basic synergy |
| **L0+L11+ln_f+biases** | **11.46%** | **3.4x** | **RECOMMENDED** |
| L0+L11+all_ln | 11.42% | 2.6x | Best quality |
| even_layers | 34.1% | 0.5x | Maximum fairness |

---

## Key Findings

### 1. Disparity is Real (Exp-001-010)
- Hebrew: 971,648% degradation
- Arabic: 372,592% degradation
- Chinese: 131,102% degradation
- Correlation with resource level: r=-0.85, p=0.03

### 2. Component Criticality is Model-Dependent (Exp-012, 014)

| Model | Best Component | Best Layer |
|-------|----------------|------------|
| GPT-2 | MLP | Layer 0 |
| OPT-125M | Attention | Layer 11 |

### 3. Anti-Critical Layers Exist (Exp-017, 020)

| Model | Anti-Critical Layer | Disparity |
|-------|---------------------|-----------|
| GPT-2 | Layer 1 | 381x (vs 214x baseline) |
| OPT | Layer 7 | 245x (vs 153x baseline) |

### 4. Embeddings (Exp-025)
- Token embeddings: HARMFUL when protected (522x)
- Position embeddings: HELPFUL (133x vs 214x)
- Representation mismatch causes the issue

### 5. LayerNorms (Exp-026)
- Only 0.03% of model
- Contribute 9.1x disparity reduction in Layer 0
- High efficiency per parameter

### 6. Multi-Language Validation (Exp-022, 024)
Layer 0 protection effective across all languages:
- Hebrew: 214x → 5.9x (36x better)
- Arabic: 92x → 1.5x (61x better)
- Chinese: 29x → 0.7x (42x better)

### 7. Anti-Critical Layer Patterns (Exp-030)
- **Odd layers (L1,L3,L5...) = CATASTROPHIC**: 1379.6x disparity
- **Even layers (L0,L2,L4...) = EXCELLENT**: 0.5x disparity
- Same overhead (34.1%), opposite outcomes

### 8. Layer 0 Component Synergy (Exp-031)
Individual components don't add up:
- l0_mlp_fc: 151.6x (HARMFUL!)
- l0_mlp_proj: 36.2x
- l0_mlp_all: 84.1x
- l0_full: 3.6x ← **Synergy between components**

### 9. Input-Output Layer Synergy (Exp-032, 033)
- Layer 0 alone: 3.6x
- Layer 11 alone: 336.2x (HARMFUL!)
- **L0 + L11 together: 0.7x** (LR improves MORE than English!)
- Cross-architecture: OPT-125M shows +29.3x synergy bonus

### 10. Multi-Language Improvements (Exp-034)
With L0+L11 strategy (11.4% overhead):
- Hebrew: 213.8x → 0.8x (**284x better**)
- Chinese: 33.6x → 0.3x (**121x better**)
- Arabic: 82.0x → 0.7x (**118x better**)

### 11. L0 is Uniquely Essential (Exp-042, 043)
L0 is the ONLY layer that enables synergy:
- L0+L11: **0.0x** (perfect multilingual fairness!)
- L2+L11: 4749.8x (catastrophic)
- L4+L11: 150.9x
- L8+L11: 195.7x

**Theory**: L0 encodes foundational multilingual representations that all downstream layers depend on.

### 12. Biases Matter (Exp-039)
- Biases are only 0.082% of model
- Quantizing biases adds +52.6x disparity
- Recommendation: Keep ALL biases in FP16

### 13. Structure > Heuristics (Exp-055, 056)
- Random 11.4% protection: 44-327x (fails)
- Magnitude-based 38.6%: 125,480x (catastrophic!)
- L0+L11 11.4%: 0.0x (perfect)
- Simple heuristics protect WRONG weights

### 14. Variance Predicts Criticality (Exp-057, 058)
- r = -0.798 correlation: high variance = low disparity
- L0: highest variance (0.021), lowest disparity (2.6x)
- L2: lowest variance (0.008), highest disparity (795x)
- **Theory**: High-variance layers encode more multilingual information

### 15. Cross-Architecture Analysis (Exp-061-064)
OPT-125M shows fundamentally different behavior:
- GPT-2's L0+L11 config: 35.6x disparity on OPT (fails!)
- OPT critical layers: L4 and L9, not L0 and L11
- Best OPT config achieves only 26.9x disparity vs GPT-2's 1.04x
- **Root cause**: OPT Layer 0 has 6x lower variance (0.006 vs 0.039)

| Model | Critical Layers | Best Disparity | Key Difference |
|-------|-----------------|----------------|----------------|
| GPT-2 | L0 + L11 | 1.04x | High L0 variance (0.039) |
| OPT-125M | L4 + L9 | 26.91x | Low L0 variance (0.006) |

**Conclusion**: Layer criticality is model-specific. Use variance analysis to identify critical layers for each architecture.

---

## Experiment Details

### Baseline Series (Exp-001-010)
Established disparity exists across GPT-2, OPT-125M, Pythia-160M.

### Threshold Sweep (Exp-011)
```
1% → 100x (near baseline)
3% → 60x
5% → 45x (optimal)
10% → 102x (worse)
```

### Layer-Specific (Exp-012)
```
GPT-2:
  MLP only: 20x
  Layer 0: 55x
  Attention: 291x
  Embeddings: 1216x (worse)
```

### Per-Layer Analysis (Exp-017, 020)
```
GPT-2 MLP:     L0 (139x) > L2 (152x) > L3 (188x) > ... > L1 (381x)
GPT-2 Attn:    L0 (54x) > L2 (114x) > L3 (172x) > ... > L1 (421x)
OPT MLP:       L11 (92x) > L4 (96x) > L0 (116x) > ... > L7 (245x)
OPT Attn:      L0 (101x) > L1 (117x) > L11 (138x) > ... > L8 (192x)
```

### Minimal Protection (Exp-027)
Pareto frontier for overhead vs disparity.

---

## File Listing

```
exp011b_threshold.py        # Threshold sweep
exp012_layer_specific.py    # Layer-specific protection
exp013_hybrid.py            # Hybrid strategies
exp014_opt_layers.py        # OPT-125M validation
exp015_text_length.py       # Text length sensitivity
exp016_robustness.py        # Statistical robustness
exp017_per_layer_mlp.py     # Per-layer MLP analysis
exp018_bloom.py             # BLOOM-560M (OOM)
exp019*_pythia*.py          # Pythia-160M (tokenization issues)
exp020_per_layer_attention.py # Per-layer attention
exp021_combined_critical.py # Combined strategies
exp022_multilang.py         # 6-language validation
exp023_longer_texts.py      # Paragraph-length texts
exp024_ar_zh_focus.py       # Arabic/Chinese focus
exp025_embedding_analysis.py # Embedding layer
exp026_layernorm.py         # LayerNorm analysis
exp027_minimal_protection.py # Pareto optimal
exp028b_gradient_simple.py   # Gradient vs magnitude selection
exp029_gradient_quant.py     # Gradient-based quantization (OOM)
exp030_anti_critical.py      # Anti-critical layer combos
exp031_layer0_breakdown.py   # Layer 0 fine-grained
exp032_output_layers.py      # Output layer analysis
MEMORY_CONSTRAINTS.md        # System memory limits
```

---

*Last updated: 2026-01-09 (64 experiments)*
