# Experiment Index

*Phase 5: Minimal Intervention Analysis*
*27 experiments completed*

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

---

## Pareto Optimal Strategies

| Strategy | Overhead | Avg LR Disparity | Use Case |
|----------|----------|------------------|----------|
| none | 0% | 60.9x | Baseline |
| layer0_ln_only | 0.002% | 41.6x | Extreme compression |
| layer0_attn | 1.90% | 16.7x | Light multilingual |
| layer0_mlp | 3.79% | **4.9x** | **Recommended** |
| layer0 | 5.70% | 3.8x | Best quality |

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
```

---

*Last updated: 2026-01-08*
