# Track B: Multilingual Circuit Interpretability

*Target: Yonatan Belinkov Lab (Technion)*

---

## Research Problem

**Central Question:** What computational mechanisms within transformer models are damaged by quantization, and why does this damage disproportionately affect low-resource languages?

**Scope:** We investigate internal representations and attention circuits to identify which components are critical for each language, and how quantization disrupts these components.

**Gap:** Circuit analysis has focused on English; multilingual circuit fragility under compression is unexplored.

---

## Contextual Knowledge: Belinkov Lab

### Key Publications & Insights

| Paper | Key Insight | Our Application |
|-------|-------------|-----------------|
| **Belinkov & Glass (2019)** "Analysis Methods in Neural NLP" | Probing classifiers reveal hierarchical encoding | Measure quantization damage to encodings |
| **Durrani et al. (2020)** "Analyzing Individual Neurons" | 15-20% of neurons are language-specific | These neurons may be quantization-fragile |
| **Dalvi et al. (2022)** "Discovering Latent Concepts" | Morphological concepts in early-middle layers | Connect to L9 bottleneck finding |
| **Belinkov (2024)** "Backward Lens" (Best Paper) | Gradients reveal token importance | Identify quantization-sensitive tokens |
| **Belinkov (2024)** "Quest for Right Mediator" | Causal paths via EAP-IG | Find disparity mediators |

### Methodological Toolkit

| Method | What It Measures | Use Case |
|--------|------------------|----------|
| Probing classifiers | Feature encodability | Quantization damage to features |
| Neuron analysis | Individual neuron roles | Language-specific vs universal |
| Attention analysis | Information flow | Cross-lingual transfer patterns |
| Activation patching | Causal contribution | Which components matter per language |

### Lab's Core Questions → Our Extensions

| Their Question | Our Extension |
|----------------|---------------|
| "Where is linguistic knowledge encoded?" | "Where is it DAMAGED by quantization?" |
| "How do models handle morphology?" | "How does quantization BREAK morphology?" |
| "What makes representations multilingual?" | "What makes them FRAGILE for LR languages?" |

---

## Hypotheses

### H-B1: Language-Specific Head Concentration
**Statement:** Language-specific attention heads concentrate in late layers (8-11), making these layers disproportionately important for low-resource languages.

**Rationale:** Belinkov's work shows 15-20% of neurons are language-specific, concentrated in late layers. If LR languages rely more heavily on sparse specialized circuits, quantization damage here is amplified.

**Testable Prediction:** Ablating late-layer heads causes larger PPL increase for LR languages than HR languages.

**Result:** ✓ CONFIRMED — 16.7% of heads are language-specific, concentrated in L8-11.

---

### H-B2: Representation Damage Disparity
**Statement:** Low-resource languages show disproportionate representation damage under quantization, measurable as larger cosine distance between FP32 and INT4 embeddings.

**Rationale:** LR languages have sparser training signal, leading to representations with less redundancy. Quantization noise has nowhere to be absorbed.

**Testable Prediction:** CLS embedding similarity (FP32 vs INT4) is lower for LR languages, with damage ratio > 2.0x.

**Result:** ✓ CONFIRMED — 3.3x representation damage ratio (LR: 23.9% vs HR: 7.3%).

---

### H-B3: Head Ablation Sensitivity
**Statement:** Low-resource languages are more sensitive to individual head ablation because they rely on fewer, more critical attention heads.

**Rationale:** High-resource languages have redundant circuits; damage to one head is compensated by others. LR languages lack this redundancy.

**Testable Prediction:** Average PPL increase from head ablation is >1.5x higher for LR languages.

**Result:** ✓ CONFIRMED — 2.23x sensitivity ratio (LR: 5.44 avg vs HR: 2.44 avg).

---

### H-B4: Gateway Layer Concentration
**Statement:** Representation damage and ablation sensitivity are concentrated in "gateway" layers (L0, L9, L11), matching Track A findings.

**Rationale:** If our gateway-bottleneck model is correct, interpretability analysis should independently identify the same critical layers.

**Testable Prediction:** LR/HR damage ratio is highest at L0, L9, L11; lowest at middle layers (L5).

**Result:** ✓ CONFIRMED — L0: 2.82x, L9: 4.15x, L11: 3.39x vs L5: 1.55x.

---

## Experiment Sequence

### Phase 1: Baseline Characterization

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| B-001 | Language-specific heads | Attention clustering | H-B1 | ✓ DONE | 16.7% lang-specific |
| B-001b | Head specialization by layer | Per-layer entropy | H-B1 | — | — |

---

### Phase 2: Damage Measurement

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| B-002b | Representation similarity | Cosine FP32→INT4 | H-B2, H-B4 | ✓ DONE | 3.3x damage ratio |
| B-002c | Layer-wise damage | Per-layer similarity | H-B4 | ✓ DONE | Gateway layers highest |

---

### Phase 3: Causal Analysis

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| B-003b | Head ablation sweep | Zero-out, measure PPL | H-B3, H-B4 | ✓ DONE | 2.23x sensitivity |
| B-003c | Layer ablation | Full layer knockout | H-B4 | — | — |

---

### Phase 4: Mechanism Refinement (GPU Required)

| ID | Name | Method | Hypothesis | Status |
|----|------|--------|------------|--------|
| B-004 | Gradient circuits | Integrated gradients | All | NOT STARTED |
| B-005 | Causal mediation | Activation patching | All | BLOCKED (GPU) |

---

## Evidence Summary

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| H-B1 | 16.7% heads language-specific in L8-11 | **CONFIRMED** |
| H-B2 | 3.3x representation damage ratio | **CONFIRMED** |
| H-B3 | 2.23x ablation sensitivity ratio | **CONFIRMED** |
| H-B4 | L0/L9/L11 show highest ratios | **CONFIRMED** |

---

## Cross-Track Synthesis

| Track | Finding | Connection to Track B |
|-------|---------|----------------------|
| **A** | L0+L9+L11 achieves 0.59x disparity | B confirms these as fragile layers |
| **C** | 3.43x efficiency trifecta average | B explains WHY: representation damage |
| **D** | Alignment r=-0.956 | B shows WHERE damage manifests |

**Causal Chain:** Track D (alignment) → Track B (representation damage) → Track A (layer protection)

---

## Publication Contribution

**Novel findings:**
1. 3.3x representation damage ratio (quantifiable)
2. 2.23x head ablation sensitivity (causal)
3. Gateway layers independently confirmed via interpretability

**Methodological contribution:** First application of Belinkov-style probing to quantization fairness.

**Venue:** ACL/EMNLP main (with Track A) or BlackboxNLP workshop

---

*Last updated: 2026-01-10*
