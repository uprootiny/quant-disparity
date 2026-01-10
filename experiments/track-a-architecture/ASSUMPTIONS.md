# Track A Assumptions: Architecture Analysis

*What we assume about model architecture and layer importance*

**Target Lab:** Soudry Lab (Neural Network Compression)

---

## Track-Specific Assumptions

### A-1: 12-Layer Model is Representative
**Assumption:** A 12-layer transformer is sufficient to study layer importance patterns; findings will scale to larger models.

**Could be wrong if:**
- Larger models have qualitatively different layer patterns
- 12 layers is too few for full pattern emergence
- Layer importance is non-linear with depth

**Evidence:** E7 showed scaling paradox, suggesting patterns change with size

**Risk Level:** MODERATE

---

### A-2: Layer Function is Position-Determined
**Assumption:** Layer 0 handles tokenization encoding, middle layers handle syntax, late layers handle semantics.

**Could be wrong if:**
- Functions are distributed, not localized
- Language affects function distribution
- Training dynamics alter layer roles

**Evidence:** Probing literature supports this; our E1 confirms L0 contribution varies by alignment

**Risk Level:** LOW

---

### A-3: Gateway Pattern is Model-Agnostic
**Assumption:** The L0 + L_last importance pattern holds across GPT-2, Llama, Mistral, etc.

**Could be wrong if:**
- Architecture differences change patterns
- Pre-training objectives affect layer roles
- Our finding is GPT-2-specific

**Evidence:** NOT VALIDATED - only tested on GPT-2

**Risk Level:** HIGH - critical for publication

---

### A-4: Quantization Sensitivity = Weight Importance
**Assumption:** Layers that are sensitive to quantization are "important" for the task.

**Could be wrong if:**
- Sensitivity measures noise, not importance
- Importance for HR ≠ importance for LR
- Redundancy confounds sensitivity

**Evidence:** E11 shows low-redundancy layers are more sensitive (confirms assumption)

**Risk Level:** LOW

---

### A-5: Mixed Precision is Feasible
**Assumption:** We can implement different precisions per layer without prohibitive overhead.

**Could be wrong if:**
- Hardware doesn't support mixed precision efficiently
- Memory fragmentation is severe
- Inference latency is unacceptable

**Evidence:** ASSUMED - hardware validation needed

**Risk Level:** MODERATE for deployment

---

## Track A Specific Confounds

1. **Model-specific findings:** All experiments on one model family
2. **Simulated quantization:** Real hardware may behave differently
3. **Static analysis:** Dynamic behavior during inference not captured

---

## What Would Falsify Track A Claims?

1. Finding a model where middle layers are critical
2. Showing layer importance doesn't vary by language
3. Demonstrating quantization sensitivity is random
