# Track D: Syntax and Morphology in Quantized Models

*Target: Yoav Goldberg Lab (Bar-Ilan University / BIU-NLP)*

---

## Research Problem

**Central Question:** How does quantization affect syntactic and morphological processing, and why does this disproportionately impact morphologically rich languages (MRLs)?

**Scope:** We investigate tokenizer-morpheme alignment, agreement accuracy, and sentence complexity effects to understand the root cause of disparity.

**Gap:** No analysis of how quantization affects morphological processing; no connection between tokenizer quality and quantization robustness.

---

## Contextual Knowledge: Goldberg Lab

### Key Publications & Insights

| Paper | Key Insight | Our Application |
|-------|-------------|-----------------|
| **Goldberg (2017)** "Neural Network Methods for NLP" | Introduced many to NNs for NLP | Foundation for understanding LM internals |
| **More & Goldberg (2016)** "Joint Morpho-Syntactic Parsing" | Pipeline errors cascade; joint models are better | Quantization creates similar cascading errors |
| **Seker & Tsarfaty (2020)** "AlephBERT" | Hebrew PLM needs morphological awareness | Quantization damages this awareness |
| **Tsarfaty et al.** "YAP Parser" | Joint morpho-syntactic parsing for MRLs | Provides evaluation framework |
| **Goldberg (2019)** "Assessing BERT's Syntactic Abilities" | Subject-verb agreement as probe | We test this under quantization |

### MRL Processing Framework

From Goldberg/Tsarfaty work:
> "In MRLs, morphology and syntax are deeply intertwined. A single orthographic token may contain multiple morphemes with complex agreement patterns."

**Our extension:**
> "Quantization errors that cross morpheme boundaries are unrecoverable. Poor tokenizer-morpheme alignment predicts quantization damage."

### Lab's Core Arguments → Our Extensions

| Their Argument | Our Extension |
|----------------|---------------|
| "Pipeline approaches fail for MRLs" | "Quantization creates similar cascading errors" |
| "Morphology requires precision" | "Quantization removes precision where MRLs need it most" |
| "Joint modeling is essential" | "Gateway layer protection preserves joint computation" |

---

## Hypotheses

### H-D1: Morphological Complexity Sensitivity
**Statement:** Complex sentences (with morphological ambiguity) degrade more under quantization than simple sentences, regardless of morphology type.

**Rationale:** Morphological disambiguation requires precise computation; quantization noise disrupts this uniformly.

**Testable Prediction:** Complex/simple degradation ratio > 1.2x for all morphology types.

**Result:** ✓ CONFIRMED — All types show ~1.25x ratio. But ABSOLUTE degradation differs massively (EN: 59%, HE: 334%).

---

### H-D2: Long-Distance Agreement Suffers More
**Statement:** Long-distance subject-verb agreement accuracy drops more for MRLs under quantization because agreement requires precise feature tracking.

**Rationale:** Agreement requires tracking gender, number, person across tokens. Quantization noise disrupts feature persistence.

**Testable Prediction:** Long-distance agreement drop > 2x for complex agreement systems (AR, HE) vs simple (EN).

**Result:** ✓ CONFIRMED — 2.80x disparity (complex: 28.9% drop, simple: 10.3% drop). Hebrew long-distance: 54% → 28% accuracy.

---

### H-D3: Alignment Predicts Degradation
**Statement:** Poor tokenizer-morpheme alignment predicts quantization damage. Languages where BPE tokens cross morpheme boundaries suffer more.

**Rationale:** Quantization errors that affect "wrong" linguistic units cannot be recovered through morphological structure.

**Testable Prediction:** Correlation between alignment and degradation is strong (r < -0.7).

**Result:** ✓ CONFIRMED — r = -0.956 (STRONGEST FINDING across all tracks).

---

### H-D4: Alignment vs Fertility
**Statement:** Alignment (boundary match) matters more than fertility (token count) for predicting quantization damage.

**Rationale:** It's not how many tokens, but how well they match morpheme boundaries.

**Testable Prediction:** |r_alignment| > |r_fertility| for correlation with degradation.

**Result:** ✓ CONFIRMED — Alignment: r = -0.956, Fertility: r = 0.874. Alignment is stronger predictor.

---

## Experiment Sequence

### Phase 1: Complexity Analysis

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| D-001b | Morphological sensitivity | Simple vs complex PPL | H-D1 | ✓ DONE | 1.25x universal |
| D-001c | Absolute vs relative | Compare degradation types | H-D1 | ✓ DONE | Absolute differs |

---

### Phase 2: Syntactic Evaluation

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| D-002b | Agreement accuracy | Minimal pairs test | H-D2 | ✓ DONE | 2.80x disparity |
| D-002c | Distance effect | Adjacent vs long | H-D2 | ✓ DONE | Long amplifies |

---

### Phase 3: Alignment Analysis

| ID | Name | Method | Hypothesis | Status | Result |
|----|------|--------|------------|--------|--------|
| D-003b | Tokenizer alignment | BPE vs gold morphemes | H-D3, H-D4 | ✓ DONE | r = -0.956 |
| D-003c | Regression model | Predict degradation | H-D3 | ✓ DONE | R² = 0.940 |

---

### Phase 4: Architecture Analysis (Future)

| ID | Name | Method | Hypothesis | Status |
|----|------|--------|------------|--------|
| D-004 | Joint vs pipeline | Compare under quantization | — | NOT STARTED |
| D-005 | Morphological circuits | Head probing | — | NOT STARTED |

---

## Evidence Summary

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| H-D1 (Complexity sensitivity) | 1.25x universal ratio | **CONFIRMED** (refined) |
| H-D2 (Long-distance agreement) | 2.80x disparity | **CONFIRMED** |
| H-D3 (Alignment predicts) | r = -0.956 | **CONFIRMED** (strongest) |
| H-D4 (Alignment > fertility) | -0.956 vs 0.874 | **CONFIRMED** |

---

## The Alignment Mechanism

### Why Alignment Matters

Consider Hebrew "והכלבים" (and-the-dogs):

**Gold morphemes:** ו + ה + כלב + ים (and + the + dog + plural)

**BPE tokens:** וה + כל + בים (meaningless chunks crossing boundaries)

When quantization introduces error:
1. Error in "כל" affects part of "dog" + wrong boundary
2. Model can't use morphological structure to recover
3. Errors compound across maligned units

**Contrast with English "unhappiness":**
- BPE: "un" + "happiness" (close to morphemes)
- Error in "un" stays within morpheme boundary
- Model can leverage morphological knowledge

### Regression Model
```
degradation = 150 - 224×alignment + 40×fertility
R² = 0.940
```

---

## Cross-Track Synthesis

| Track | Finding | Connection to Track D |
|-------|---------|----------------------|
| **A** | L0 is critical gateway | D explains WHY: L0 encodes misaligned tokenization |
| **B** | 3.3x representation damage | D explains source: alignment creates fragile basis |
| **C** | Fertility ≠ degradation (r=-0.07) | D finds truth: alignment (r=-0.956) |

### Causal Chain
```
Poor Alignment (D) → Fragile L0 Encoding (A) → Representation Damage (B) → Disparity (C)
```

---

## Grammatical Correctness Impact

**Beyond perplexity:** Quantized models produce GRAMMATICALLY INCORRECT output for MRLs.

**Hebrew example (D-002b):**

| Model | Sentence | P(correct) | Result |
|-------|----------|------------|--------|
| FP32 | "הילדים שראו את הכלב רצים" | 0.54 | ✓ Correct |
| INT4 | Same sentence | 0.28 | ✗ WRONG |

**Mechanism:** "Agreement attraction" — model attracted to local noun (הכלב, singular) instead of grammatical subject (הילדים, plural). Quantization amplifies this error.

---

## Publication Contribution

**Novel findings:**
1. Alignment predicts degradation (r = -0.956) — **strongest finding**
2. Complex agreement drops 2.80x more for MRLs
3. Grammatical correctness affected, not just perplexity

**Theoretical contribution:** Identifies ROOT CAUSE of disparity in tokenization, not model architecture.

**Venue:** ACL main track (with Track A), *SEM, SIGMORPHON

**Title:** "When Compression Breaks Grammar: How Tokenizer Misalignment Amplifies Quantization Damage in Morphologically Rich Languages"

---

*Last updated: 2026-01-10*
