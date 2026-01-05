# Next Steps: Scaling and Validation

## Validated So Far

1. **Disparity is massive**: 88.68x ratio (INT4 on GPT-2)
2. **5% preservation is optimal**: Reduces disparity to 45.39x (49% improvement)
3. **Non-monotonic relationship**: More preservation ≠ better fairness
4. **Dense outliers**: All tested models show diffuse patterns

## Immediate Next Steps (No GPU)

### Step 1: Statistical Validation
- [ ] Run each preservation level 3x for confidence intervals
- [ ] Use different random seeds for reproducibility
- [ ] Test statistical significance of 5% optimum

### Step 2: Text Diversity
- [ ] Test with longer texts (100+ tokens)
- [ ] Test with domain-specific texts (technical, news, creative)
- [ ] Validate across different text samples

### Step 3: Language Coverage
- [ ] Add Russian (Cyrillic script)
- [ ] Add Japanese (mixed scripts)
- [ ] Add Korean (Hangul)
- [ ] Measure per-script disparity patterns

## Medium-Term (With GPU Access)

### Step 4: Model Scaling
- [ ] OPT-125M validation
- [ ] Pythia-160M validation
- [ ] BLOOM-560M (multilingual model)
- [ ] 7B model spot-check

### Step 5: Real Quantization
- [ ] bitsandbytes INT4/INT8
- [ ] GPTQ
- [ ] AWQ
- [ ] Compare to simulated quantization

### Step 6: Layer-Specific Preservation
- [ ] Test preserving only layer 0 (embeddings)
- [ ] Test preserving attention layers only
- [ ] Find minimal preservation for 50% disparity reduction

## Long-Term (Publication Ready)

### Step 7: Technique Development
- [ ] Language-aware threshold selection
- [ ] Calibration data optimization
- [ ] Mixed-precision strategy

### Step 8: Downstream Tasks
- [ ] Translation quality (BLEU)
- [ ] Question answering (accuracy)
- [ ] Text generation (human eval)

### Step 9: Publication
- [ ] Write technical report
- [ ] Prepare replication package
- [ ] Submit to Israeli AI labs for collaboration

---

## Priority Order

1. **Statistical validation** - Ensure 5% finding is robust
2. **Language coverage** - Validate across script types
3. **BLOOM validation** - Test on truly multilingual model
4. **Real quantization** - Confirm with production methods
5. **Technique publication** - Write up LA-ACIQ method

---

*Planned: 2026-01-04*
