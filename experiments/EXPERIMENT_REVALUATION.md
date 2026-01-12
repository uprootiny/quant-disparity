# Experiment Revaluation Post-Confounder Analysis

*Which planned experiments are still valuable? Which should be abandoned?*

---

## Evaluation Criteria (Post E14-E15)

An experiment is valuable if it:
1. **Tests something robust to confounds** (mechanistic, architectural)
2. **Could distinguish causation from correlation**
3. **Has practical value regardless of root cause**
4. **Doesn't just reconfirm confounded findings**

---

## Planned Experiments: Revaluation

### V1: Llama-2-7B Validation
**Original purpose:** Validate scaling paradox on real 7B model

**Post-confounder assessment:**
- Scaling paradox is ARCHITECTURAL (redundancy mechanism)
- Not confounded by training data at fixed model size
- Cross-model validation would strengthen claims

**Verdict:** ✅ STILL VALUABLE
**Reason:** Tests mechanism, not correlation; cross-model is key for publication

---

### V2: Cross-Architecture Test (Mistral)
**Original purpose:** Verify gateway pattern generalizes

**Post-confounder assessment:**
- Gateway importance is MECHANISTIC
- If it holds across architectures, not model-specific
- Different training data distributions add diversity

**Verdict:** ✅ STILL VALUABLE
**Reason:** Architectural claims should transfer; if they don't, we learn something

---

### H14: Per-Language Optimal α (LA-ACIQ)
**Original purpose:** Test if language-specific quantization helps

**Post-confounder assessment:**
- E10 showed this INCREASES disparity (HR benefits more)
- Doesn't help our fairness goal
- May still be useful for absolute performance

**Verdict:** ⚠️ DEPRIORITIZE
**Reason:** Already tested, doesn't help fairness; keep for completeness only

---

### H15: Middle Layers Truly Redundant
**Original purpose:** Strengthen gateway claim

**Post-confounder assessment:**
- E11 already confirmed (10.8x importance ratio)
- This is ARCHITECTURAL, not confounded
- Further testing has diminishing returns

**Verdict:** ✅ DONE - No further work needed

---

### H16: Tokenizer Retraining Reduces Disparity
**Original purpose:** Test intervention at source

**Post-confounder assessment:**
- E8 SIMULATED this (35% reduction)
- But confounders suggest alignment may be EFFECT not CAUSE
- If alignment is downstream of training data, tokenizer fix alone won't help
- HOWEVER: Even if correlation, fixing tokenizer might help via different path

**Verdict:** ⚠️ UNCERTAIN BUT IMPORTANT
**Reason:** This is the key INTERVENTION test; simulation isn't enough; needs real experiment

---

## NEW: Experiments That Would Actually Help

### N1: Same Training Data, Different Tokenizer
**Purpose:** Isolate tokenizer effect from training data confound

**Design:**
- Take model trained on multilingual data
- Apply TWO tokenizers: standard BPE vs morphology-aware
- Compare degradation with SAME model weights
- If morphology-aware is better, tokenizer matters independent of training

**Why valuable:** Breaks the confound; true intervention test

**Feasibility:** MODERATE - need to retokenize without retraining

---

### N2: Within-Language Variation
**Purpose:** Test alignment-degradation within single language

**Design:**
- Take Hebrew texts of varying complexity
- Some texts have words that align well, others poorly
- Test degradation per-document, not per-language
- Controls ALL language-level confounds

**Why valuable:** Same language = same training data, benchmarks, etc.

**Feasibility:** HIGH - just need to segment existing data

---

### N3: Synthetic Language with Controlled Properties
**Purpose:** Ground truth for alignment-degradation relationship

**Design:**
- Create artificial "language" with known tokenization properties
- Control alignment, frequency, complexity independently
- Test if alignment predicts degradation when other factors constant

**Why valuable:** Perfect control; no confounds possible

**Feasibility:** MODERATE - need to design synthetic data carefully

---

### N4: Parallel Corpus Degradation
**Purpose:** Same content, different languages

**Design:**
- Use FLORES or similar parallel corpus
- Exact same sentences in multiple languages
- Controls for content/domain completely
- Only language properties vary

**Why valuable:** Strongest real-data control available

**Feasibility:** HIGH - FLORES exists, just need to run experiments

---

### N5: Residualized Prediction
**Purpose:** Test if alignment adds predictive power beyond confounds

**Design:**
- Regress degradation on training data, vocab coverage, benchmark quality
- Take residuals (unexplained variance)
- Test if alignment predicts residuals
- If yes, alignment has independent effect

**Why valuable:** Statistical control for confounds

**Feasibility:** HIGH - just analysis of existing data

---

## Priority Ranking (Post-Confounder)

### Tier 1: Do Now (High Value, Feasible)
1. **N5: Residualized Prediction** - Statistical test, no new data needed
2. **N2: Within-Language Variation** - Breaks confounds, uses existing data
3. **N4: Parallel Corpus** - Strong control, FLORES exists

### Tier 2: Do If Resources Allow
4. **V1: Llama Validation** - Cross-model, needs GPU
5. **N1: Same Data Different Tokenizer** - True intervention, complex setup

### Tier 3: Deprioritize
6. **V2: Mistral** - Nice to have but similar to V1
7. **N3: Synthetic Language** - Elegant but complex
8. **H14: LA-ACIQ refinement** - Already shown not to help fairness

### Tier 4: Abandon
9. **More alignment-degradation correlations** - Confounded, won't convince skeptics
10. **More simulations of tokenizer effects** - Need real experiments

---

## What Changes in Our Story?

### Old Story (Pre-Confounder)
> "Alignment causes disparity. Fix alignment via tokenizer, disparity goes away."

### New Story (Post-Confounder)
> "Disparity exists and is predicted by alignment, though causation is unclear due to confounds with training data investment. HOWEVER:
> 1. Gateway layer importance is mechanistic and robust
> 2. Language family patterns are typological and robust
> 3. Scaling paradox is architectural and robust
> 4. Practical interventions (layer protection) work regardless of root cause"

### What We Now Emphasize
- Mechanistic findings (gateway, redundancy, scale)
- Practical interventions (protection strategies)
- Honest uncertainty about causation
- Need for intervention studies

### What We De-Emphasize
- "Alignment is THE root cause"
- Specific degradation percentages
- Simple causal narrative
