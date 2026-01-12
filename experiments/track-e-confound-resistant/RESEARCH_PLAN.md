# Track E: Confound-Resistant Evidence

*The skeptic's track: Only claims that survive rigorous confound analysis*

**Philosophy:** If we can't prove causation, prove mechanism. If we can't separate variables, find natural experiments. If correlation is confounded, test interventions.

---

## Research Problem

**Core Question:** What can we ACTUALLY claim about quantization disparity that a skeptical reviewer couldn't dismiss as confounded?

**Constraint:** Every claim must survive the critique: "But that's just because LR languages have less training data."

---

## Track E Hypotheses

### E-H1: Gateway Importance is Architectural (Not Data-Driven)
**Claim:** L0 and L11 are critical due to their POSITION, not language properties.

**Why robust:** Layer position is fixed by architecture, not training data.

**Test:** Does gateway importance hold for SYNTHETIC tokens? If yes, it's architectural.

**Falsifiable by:** Finding that gateway importance varies with training data, not position.

---

### E-H2: Redundancy Mechanism Explains Scale Effect
**Claim:** Larger models have more redundancy; HR languages leverage redundancy better.

**Why robust:** This is about model ARCHITECTURE interacting with representation DENSITY.

**Test:** Artificially reduce redundancy (prune random neurons). Does disparity decrease?

**Falsifiable by:** Showing redundancy doesn't correlate with HR advantage.

---

### E-H3: Within-Language Variation Mirrors Cross-Language Pattern
**Claim:** Texts within ONE language that have poor tokenization show same degradation pattern.

**Why robust:** Same language = same training data, benchmarks, everything. Only tokenization varies.

**Test:** Segment Hebrew corpus by alignment quality. Does low-alignment Hebrew degrade more?

**Falsifiable by:** Finding no within-language alignment effect.

---

### E-H4: Parallel Content Shows Language Effect
**Claim:** Same content in different languages degrades differently.

**Why robust:** Controls for content, domain, complexity. Only language varies.

**Test:** Use FLORES parallel corpus. Same sentences, different degradation.

**Falsifiable by:** Finding no degradation difference on parallel content.

---

### E-H5: Protection Works Regardless of Cause
**Claim:** Gateway protection reduces disparity WHETHER OR NOT alignment is causal.

**Why robust:** Practical claim; doesn't require causal story.

**Test:** Apply protection, measure disparity reduction. Report empirically.

**Falsifiable by:** Protection not helping (but E11 already shows it helps).

---

## Experiment Sequence

### Phase 1: Establish Mechanism (No Confounds Possible)

**E-EXP1: Synthetic Token Importance**
- Create random token sequences (no language)
- Quantize model
- Measure per-layer importance
- If L0/L11 still critical → architectural fact

**E-EXP2: Redundancy Ablation**
- Take trained model
- Randomly ablate X% of neurons per layer
- Measure how degradation changes
- HR should suffer less (more redundant paths)

### Phase 2: Control for Confounds

**E-EXP3: Within-Hebrew Variation**
- Segment Hebrew Wikipedia by word alignment scores
- High-alignment Hebrew paragraphs vs low-alignment
- Quantize, measure degradation per segment
- Same language, different alignment

**E-EXP4: Parallel Corpus Test**
- FLORES-200 dev set (parallel sentences)
- Measure degradation per language
- Same content = same difficulty
- Only language properties vary

### Phase 3: Residual Analysis

**E-EXP5: Residualized Alignment**
- Full regression: degradation ~ training_data + vocab_coverage + benchmark_quality
- Extract residuals
- Correlate residuals with alignment
- If r > 0.3, alignment has independent effect

**E-EXP6: Held-Out Language Prediction**
- Fit model on 10 languages
- Predict degradation for 5 held-out languages
- If alignment predicts, effect generalizes

### Phase 4: Intervention (If Resources Allow)

**E-EXP7: Protection Effectiveness**
- Apply L0+L9+L11 FP16 protection
- Measure disparity before/after
- Report pure empirical result
- No causal claim needed

---

## Success Criteria

| Experiment | Success = | Failure = |
|------------|-----------|-----------|
| E-EXP1 | L0/L11 important for random tokens | Position doesn't matter |
| E-EXP2 | HR more robust to ablation | No HR advantage |
| E-EXP3 | Within-Hebrew alignment matters | No within-language effect |
| E-EXP4 | Parallel content shows disparity | No language effect on parallel |
| E-EXP5 | Residual r > 0.3 | Alignment fully confounded |
| E-EXP6 | Held-out prediction works | Model doesn't generalize |
| E-EXP7 | Protection reduces disparity | Protection doesn't help |

---

## What This Track Gives Us

**If most experiments succeed:**
- Mechanistic story is solid (architecture matters)
- Alignment has independent effect (some causal evidence)
- Practical intervention works (regardless of theory)
- Paper is defensible against confound critiques

**If experiments fail:**
- We learn what's actually confounded
- We can honestly report negative results
- We pivot to "disparity exists but cause is training data"
- Still valuable contribution (honest science)

---

## Difference from Main Track

| Aspect | Main Track (D) | Alternate Track (E) |
|--------|----------------|---------------------|
| Focus | Alignment as cause | What survives confounds |
| Method | Correlation + mechanism | Mechanism + intervention |
| Claims | "Alignment causes disparity" | "Disparity exists, mechanism is X" |
| Risk | Confound critique | Less ambitious claims |
| Reward | Strong causal story | Unassailable findings |

---

## Integration with Main Track

Track E doesn't REPLACE Track D. It COMPLEMENTS it:

1. **Track D** pursues the full causal story (alignment → disparity)
2. **Track E** establishes what's defensible regardless

**Publication strategy:**
- Lead with Track E findings (unassailable)
- Present Track D findings with caveats
- Acknowledge confounds explicitly
- Let readers judge causal claims themselves

---

## Status

| Experiment | Status | Priority |
|------------|--------|----------|
| E-EXP1 | PLANNED | HIGH |
| E-EXP2 | PLANNED | HIGH |
| E-EXP3 | PLANNED | HIGH |
| E-EXP4 | PLANNED | HIGH |
| E-EXP5 | PLANNED | MEDIUM |
| E-EXP6 | PLANNED | MEDIUM |
| E-EXP7 | DONE (E11) | - |
