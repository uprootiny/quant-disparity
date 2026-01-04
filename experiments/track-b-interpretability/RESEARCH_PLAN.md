# Track B: Multilingual Circuit Interpretability

## Target Lab
**Yonatan Belinkov Lab** — Technion

## Research Question
> What circuits handle multilingual processing, and how does quantization affect them?

## Motivation

From Belinkov's work:
- Probing reveals what linguistic properties are encoded
- Causal mediation identifies which components matter
- Circuits are sparse subnetworks that implement specific behaviors

**Gap:** Circuit analysis focused on English; multilingual circuits unexplored.

---

## Core Hypotheses

**H-B1:** Different languages activate different circuit subsets.

**H-B2:** Quantization damage correlates with circuit overlap.

**H-B3:** Low-resource languages rely on fewer, more critical circuits.

---

## Experiment Series

### B-001: Cross-Lingual Attention Pattern Analysis

**Question:** Do attention heads specialize by language?

**Method:**
1. Run mBERT/BLOOM on parallel sentences (same meaning, different languages)
2. Extract attention patterns per head
3. Compute cross-lingual similarity: sim(attn_en, attn_de)
4. Identify language-specific vs universal heads

**Metrics:**
- Jensen-Shannon divergence between attention distributions
- Head clustering by language preference

**Prediction:** Some heads are universal, others language-specific.

---

### B-002: Probing Quantization Effects

**Question:** Does quantization affect linguistic probe accuracy?

**Method:**
1. Train probing classifiers on BLOOM FP16 representations:
   - POS tagging
   - Dependency parsing
   - NER
2. Repeat on INT8, INT4 quantized models
3. Compare accuracy drop per language

**Prediction:** Low-resource languages lose more probe accuracy under quantization.

---

### B-003: Circuit Ablation by Language

**Question:** Which circuit components are critical for each language?

**Method:**
1. Identify top-k important heads per language (via activation magnitude)
2. Ablate heads and measure perplexity increase
3. Compare ablation sensitivity across languages

**Prediction:** Low-resource languages are more sensitive to ablation (less redundancy).

---

### B-004: Gradient-Based Circuit Discovery

**Question:** What does the backward pass reveal about language processing?

**Method (inspired by "Backward Lens"):**
1. Compute gradients of loss w.r.t. layer outputs
2. Project gradients to vocabulary space
3. Analyze which tokens drive gradients per language

**Connects to:** Belinkov's "Backward Lens" (Best Paper 2024)

---

### B-005: Causal Mediation for Disparity

**Question:** Which components mediate the quantization disparity?

**Method (inspired by "Quest for Right Mediator"):**
1. Apply EAP-IG to identify causal paths
2. Compare paths for high-resource vs low-resource languages
3. Identify mediators that explain disparity

**Connects to:** Belinkov's COLM 2024 work

---

## Metrics

| Metric | Definition |
|--------|------------|
| Attention Overlap | JS divergence between language attention patterns |
| Probe Accuracy Drop | accuracy_fp16 - accuracy_quantized |
| Ablation Sensitivity | perplexity increase per ablated head |
| Circuit Sparsity | % of model used per language |

---

## Datasets

| Dataset | Task | Use |
|---------|------|-----|
| UD Treebanks | POS, Parsing | Probing |
| WikiANN | NER | Probing |
| Tatoeba | Parallel sentences | Attention analysis |
| FLORES | Translation | Circuit discovery |

---

## Tools

- TransformerLens (mechanistic interpretability)
- Baukit (activation patching)
- HuggingFace (model loading)

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Language-specific heads found | > 10% of heads |
| Probe accuracy drop correlation | r > 0.5 with resource level |
| Ablation sensitivity differs | p < 0.05 between language groups |

---

## Connection to Track A

Track A: WHY quantization hurts (outlier weights in attention)
Track B: WHERE in the circuit (which heads/layers matter)

Combined insight: Outlier weights may be in language-critical circuits.

---

## Publication Target

**Venue:** EMNLP 2027 or ACL 2027

**Angle:** "Circuit Anatomy of Multilingual Disparity: Why Quantization Hurts Some Languages More"

---

*Created: 2026-01-03*
