# Hypothesis Reconciliation: What Are We Missing?

*A brutally honest assessment of our claims vs potential confounders*

---

## The Core Problem

We claim: **"BPE alignment causes quantization disparity across languages"**

But we've identified that alignment is highly correlated with:
- Vocabulary coverage (r=0.966)
- Benchmark quality (r=0.987)
- Training data quantity (r=0.709)
- Domain match (r=0.968)

**Question: Is "alignment" doing any work, or is it just a proxy for "how much the model was trained on this language"?**

---

## Hypothesis-by-Hypothesis Reconciliation

### H1: Gateway Layers Are Critical

**Our claim:** L0 and L11 are disproportionately important for LR languages.

**Alternative explanations:**
1. Gateway layers are important for ALL languages equally; we're seeing noise
2. Gateway importance reflects model architecture, not language properties
3. Our importance metric is flawed

**Confound check:** Does gateway importance vary by language, or is it constant?
- E1 showed L0 contribution correlates with alignment (r=-0.794)
- This suggests language-specific effect, not just architecture

**Verdict:** ✓ LIKELY REAL - mechanism-based, varies by language

---

### H2: Alignment Predicts Degradation

**Our claim:** Lower alignment → higher degradation under quantization.

**Alternative explanations:**
1. **Training data quantity** - Less data → worse baseline → more degradation headroom
2. **Vocabulary coverage** - Poor coverage → OOV issues → degradation
3. **Benchmark quality** - Noisier benchmarks → higher variance → apparent degradation
4. **Tokenizer optimization** - Tokenizer trained on English → English tokens are better
5. **Model capacity** - Model allocated less capacity to LR during pretraining

**Confound checks:**
- E14: Partial correlations collapse for vocab coverage and benchmark quality
- E15: 3/4 confound-resistant tests pass

**Verdict:** ⚠️ UNCERTAIN - Effect exists but causation unclear

---

### H3: Language Families Cluster

**Our claim:** Related languages show similar quantization sensitivity.

**Alternative explanations:**
1. **Geographic confound** - Families correlate with region, region correlates with resources
2. **Script confound** - Families often share scripts, scripts affect tokenization
3. **Linguistic research bias** - Better benchmarks for well-studied families
4. **Training data overlap** - Related languages share training data sources

**Confound check:** Do families cluster AFTER controlling for training data?
- E15 within-family analysis: r=-0.828 even within families
- This suggests typological effect beyond just resources

**Verdict:** ✓ LIKELY REAL - clustering is robust to within-family analysis

---

### H4: Scaling Paradox (Larger Models = More Disparity)

**Our claim:** Bigger models have more redundancy, which HR leverages better.

**Alternative explanations:**
1. **Training data scaling** - Larger models trained on more data, which is HR-biased
2. **Evaluation scaling** - Benchmarks may be harder for larger models in LR
3. **Capacity allocation** - Larger models allocate even more to HR
4. **Measurement artifact** - Perplexity scales differently by language

**Confound check:** Is this testable without confounds?
- This is an architectural claim about redundancy
- Needs intervention: train same-size models with different data distributions

**Verdict:** ⚠️ PLAUSIBLE BUT UNTESTED - mechanism is coherent but not validated

---

### H5: Tokenizer Intervention Would Help

**Our claim:** Morphology-aware tokenization would reduce disparity.

**Alternative explanations:**
1. **Wishful thinking** - We haven't actually tried it
2. **Insufficient intervention** - Tokenizer is just one component
3. **Downstream effects** - New tokenizer needs new model training
4. **Alignment isn't the cause** - If confounders explain everything, tokenizer won't help

**Confound check:** This is ONLY testable via intervention.

**Verdict:** ❓ SPECULATION - No empirical evidence, only simulation

---

## Confounders We May Have Missed Entirely

### Missed Confounder 1: Evaluation Prompt Design

**The issue:** Our evaluation prompts were likely designed in English and translated. Translated prompts may be unnatural in target languages.

**How it confounds:** Unnatural prompts → higher perplexity → apparent degradation

**How to check:**
- Use native-speaker-designed prompts for each language
- Compare translated vs native prompts
- Control for prompt naturalness

**Severity:** HIGH - could explain much of our signal

---

### Missed Confounder 2: Perplexity Normalization

**The issue:** Perplexity depends on vocabulary size and token frequency distribution. Different languages have different effective vocabularies.

**How it confounds:** Languages with more rare tokens have higher baseline perplexity and potentially different degradation patterns.

**How to check:**
- Use character-level perplexity (language-agnostic)
- Normalize by language-specific baseline
- Use downstream tasks instead of perplexity

**Severity:** MODERATE - affects absolute numbers, maybe not ratios

---

### Missed Confounder 3: Tokenizer Training Distribution

**The issue:** The BPE tokenizer was trained on a corpus that's ~90% English. This isn't just "alignment" - the tokenizer's merge rules are optimized for English patterns.

**How it confounds:** It's not that Hebrew has poor alignment; it's that the tokenizer never learned Hebrew-appropriate merges.

**How to check:**
- Train tokenizer on balanced multilingual corpus
- Compare alignment before/after
- Test if alignment improvement helps

**This suggests:** "Alignment" may be effect, not cause. Real cause = tokenizer training bias.

**Severity:** HIGH - reframes our entire narrative

---

### Missed Confounder 4: Model Pretraining Distribution

**The issue:** The model was trained on ~X% English, Y% German, Z% French... and <1% Hebrew. Model capacity is allocated proportionally.

**How it confounds:** Low-resource languages have fewer dedicated neurons, less robust representations - REGARDLESS of alignment.

**How to check:**
- Find languages with same training % but different alignment
- Find languages with different training % but same alignment
- This is essentially impossible with current models

**Severity:** CRITICAL - probably explains most of our signal

---

### Missed Confounder 5: Benchmark Domain Mismatch

**The issue:** English benchmarks come from diverse domains. LR benchmarks may be narrow (religious texts, news, government documents).

**How it confounds:** Domain mismatch → higher perplexity → apparent degradation

**How to check:**
- Use parallel corpora (same content, multiple languages)
- Control for domain explicitly
- Use FLORES-200 (designed for this)

**Severity:** MODERATE - we partially addressed with FLORES but didn't control rigorously

---

### Missed Confounder 6: Annotation Quality and Consistency

**The issue:** LR benchmark annotations may be lower quality, inconsistent, or done by non-native speakers.

**How it confounds:** Noisy labels → noisy measurements → apparent effects

**How to check:**
- Inter-annotator agreement by language
- Native speaker validation
- Synthetic benchmarks with known ground truth

**Severity:** MODERATE-HIGH - hard to assess without manual review

---

### Missed Confounder 7: Hardware/Implementation Artifacts

**The issue:** We simulated quantization. Real hardware quantization may behave differently. CUDA implementations may be optimized for certain patterns.

**How it confounds:** Simulation errors → wrong conclusions

**How to check:**
- Validate on actual quantized models
- Use multiple quantization implementations
- Compare to published benchmarks

**Severity:** MODERATE - our simulations are standard, but real validation needed

---

## The Devastating Possibility

**What if our entire "alignment" story is backwards?**

Possible causal chain:
```
Training data quantity
    ↓
├── Model learns language better (more neurons allocated)
├── Tokenizer learns language better (better merges)
├── More/better benchmarks exist
└── "Alignment" is higher (EFFECT, not cause)

All of these cause lower degradation.
"Alignment" predicts degradation because it's a downstream indicator of training investment.
```

**If this is true:**
- Fixing alignment won't help (treating symptom, not disease)
- Only solution is more balanced training data
- Our "Gateway-Bottleneck" findings may still hold (they're architectural)

---

## What Evidence Would Distinguish?

### To prove alignment is causal (not just correlational):

1. **Intervention:** Retrain tokenizer with Hebrew-optimized merges. Does degradation decrease?

2. **Natural experiment:** Find two languages with:
   - Same training data quantity
   - Different alignment
   - Test if different degradation

3. **Cross-model:** If alignment is causal, effect should hold across models. If it's training-data-driven, different models (with different training distributions) should show different patterns.

4. **Synthetic languages:** Create artificial languages with controlled alignment. Test degradation.

### To prove our findings are confounded:

1. **Residual analysis:** After controlling for training data quantity, does alignment add predictive power?
   - E15 suggests YES (3/4 tests pass)
   - But we need more rigorous analysis

2. **Within-language variation:** Do different texts in the same language show alignment-degradation relationship?
   - This would control for all language-level confounds

3. **Ablation:** Train model without Hebrew. Add Hebrew data without changing tokenizer. Does "alignment" predict degradation?

---

## Honest Assessment

### What we KNOW is real:
1. Disparity exists (LR degrades more than HR under quantization)
2. Layers differ in importance (gateway > middle)
3. Languages cluster by family
4. Effect increases with model scale

### What we BELIEVE but can't prove:
1. Alignment is a cause (not just correlate)
2. Tokenizer intervention would help
3. Findings generalize across models

### What we're probably WRONG about:
1. Specific degradation percentages (benchmark noise)
2. Clean causal story (reality is more complex)
3. Completeness (we've tested one model, one quantization method)

---

## Recommended Next Steps

### Immediate (no new data needed):
1. **Residualized analysis:** Regress out training data, test if alignment still matters
2. **Bootstrap confidence intervals:** How stable are our estimates?
3. **Sensitivity analysis:** How do conclusions change with different assumptions?

### Short-term (new analysis, existing data):
1. **Cross-model test:** Run on Llama, Mistral if possible
2. **Per-document analysis:** Within-language variation
3. **Parallel corpus:** Same content, multiple languages

### Long-term (requires intervention):
1. **Tokenizer retraining:** Actually test the intervention
2. **Controlled pretraining:** Same data distribution, different tokenizer
3. **Synthetic language experiments:** Complete control over confounds

---

## Conclusion

**Our findings are PROBABLY partially real:**
- Gateway importance is mechanistic and robust
- Family clustering survives confound tests
- Some alignment effect exists beyond training data (3/4 tests)

**But our causal story is PROBABLY too simple:**
- Alignment is entangled with training investment
- Can't cleanly separate tokenizer from model from data
- Real causation is likely multi-factorial

**Recommended framing:**
> "We find that quantization disproportionately harms low-resource languages, with disparity correlated with BPE-morpheme alignment. While we cannot definitively establish causation due to confounding with training data distribution, multiple confound-resistant tests suggest alignment has an independent effect. Gateway layer protection offers a practical intervention regardless of ultimate causation."

This is honest, defensible, and still useful.
