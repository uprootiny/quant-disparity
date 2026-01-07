# Weight Universe Map: From Bulk to Minimal Intervention

## The Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 0: TOTAL TRAINED WEIGHT SUPERSET                            │
│  (All models ever trained, published or not)                       │
│  Estimate: ~10^18 parameters (exascale)                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: HUGGINGFACE PUBLIC MODELS                                │
│  (Downloadable, documented, accessible)                            │
│  Estimate: ~500,000 models, ~10^15 total parameters               │
│                                                                     │
│  Breakdown by size:                                                 │
│    <1B params:     ~400,000 models  (~10^14 params total)          │
│    1B-10B params:   ~80,000 models  (~4×10^14 params total)        │
│    10B-100B params: ~15,000 models  (~5×10^14 params total)        │
│    >100B params:     ~5,000 models  (~10^15 params total)          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 2: MULTILINGUAL-CAPABLE MODELS                              │
│  (Models with non-trivial multilingual performance)                │
│  Estimate: ~50,000 models, ~10^14 parameters                       │
│                                                                     │
│  Key families:                                                      │
│    - mBERT, XLM-RoBERTa family (~10^10)                            │
│    - BLOOM family (~10^11)                                          │
│    - Llama-based multilingual (~10^12)                             │
│    - GPT-based multilingual (~10^11)                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 3: ACCESSIBLE FOR STUDY (CPU/small GPU)                     │
│  (Models we can actually load and experiment with)                 │
│  Estimate: ~10,000 models, <10B params each                        │
│                                                                     │
│  Our test set:                                                      │
│    - GPT-2 (124M) ✓                                                │
│    - OPT-125M (125M) ✓                                             │
│    - Pythia-160M (162M) ✓                                          │
│    - BLOOM-560M (560M) pending                                     │
│    - mGPT-1.3B (1.3B) pending                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 4: UNIVERSE OF CONCERN                                      │
│  (Weights that matter for multilingual quality)                    │
│                                                                     │
│  Per-model breakdown (GPT-2 124M example):                         │
│    Total weights:           124,337,664 (100%)                     │
│    Attention weights:        28,311,552 (22.8%)                    │
│    Embedding weights:        38,597,376 (31.0%)                    │
│    MLP weights:              57,428,736 (46.2%)                    │
│                                                                     │
│  For multilingual concern:                                          │
│    - Embeddings: HIGH (vocabulary/script encoding)                 │
│    - Layer 0: HIGH (initial representation)                        │
│    - Attention: MEDIUM (language-specific heads exist)             │
│    - MLP: LOW (mostly language-agnostic)                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 5: DISPARITY-CRITICAL WEIGHTS                               │
│  (Weights that, when quantized, cause disparity)                   │
│                                                                     │
│  From our experiments:                                              │
│    - Top 5% by magnitude reduces disparity 50%                     │
│    - Implies: ~6.2M weights are critical (GPT-2)                   │
│    - Location: Distributed across attention + embeddings           │
│                                                                     │
│  Hypothesis: These are "outlier" weights that encode               │
│  script-specific / morphology-specific information                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 6: MINIMAL INTERVENTION SET                                 │
│  (Smallest set to protect for acceptable disparity)                │
│                                                                     │
│  Unknown - this is what we need to find:                           │
│    - Is 5% the minimum, or can we go lower?                        │
│    - Can we use location (layer, head) instead of magnitude?       │
│    - Can we identify weights by language sensitivity?              │
│                                                                     │
│  Target: <1% of weights protected → <10x disparity                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Quantitative Estimates

### HuggingFace Ecosystem (as of 2026)

| Category | Models | Total Params | Notes |
|----------|--------|--------------|-------|
| All models | ~500K | ~10^15 | Growing ~50K/month |
| Text generation | ~200K | ~10^14 | Our focus |
| Multilingual | ~50K | ~10^14 | Subset of concern |
| <1B params | ~400K | ~10^14 | Accessible |
| Open weights | ~300K | ~10^14 | Actually usable |

### Per-Model Weight Distribution (GPT-2 124M)

| Component | Weights | % of Total | Multilingual Importance |
|-----------|---------|------------|------------------------|
| Token embeddings | 38.6M | 31.0% | CRITICAL |
| Position embeddings | 0.6M | 0.5% | Low |
| Attention (Q,K,V,O) | 28.3M | 22.8% | HIGH |
| MLP (up + down) | 57.4M | 46.2% | Medium |
| LayerNorm | 0.05M | 0.04% | Low |

### Disparity-Critical Weight Estimates

| Preservation % | Weights Protected | Disparity | Efficiency |
|----------------|-------------------|-----------|------------|
| 0% | 0 | 78-214x | Baseline |
| 1% | 1.2M | ? | Unknown |
| 5% | 6.2M | 45-129x | **Current best** |
| 10% | 12.4M | 100-173x | Worse |
| 20% | 24.9M | 173x | Much worse |

## The Key Question

**What is the minimal set of weights that, when protected from quantization, reduces disparity to acceptable levels (<10x)?**

### Sub-questions:

1. **By magnitude**: Is 5% truly optimal, or is there a lower threshold?
2. **By location**: Can we protect specific layers/heads instead of global top-k?
3. **By language**: Can we identify language-specific critical weights?
4. **By function**: Are attention weights more important than embeddings?

## Intervention Techniques Under Consideration

| Technique | Description | Overhead |
|-----------|-------------|----------|
| Magnitude preservation | Keep top-k% in FP16 | k% memory |
| Layer preservation | Keep specific layers in FP16 | layer% memory |
| Head preservation | Keep specific attention heads | head% memory |
| Mixed precision | INT8 for some, INT4 for others | Variable |
| Calibration optimization | Better quantization ranges | Compute only |
| Activation caching | Cache hot-path activations | Runtime memory |

## Next Steps

1. **Exp-011**: Find minimum preservation threshold (1%, 2%, 3%, 4%)
2. **Exp-012**: Layer-specific preservation (layer 0 only)
3. **Exp-013**: Attention-only preservation
4. **Exp-014**: Embedding-only preservation
5. **Exp-015**: Language-guided weight selection

---

*Created: 2026-01-05*
