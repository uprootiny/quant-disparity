# Experimental Plan: Generalizability of Outlier-Disparity Finding

## Core Finding to Test

```
In BLOOM-560M:
  r(outlier_activation, degradation) = -0.834
  Languages using outlier layers LESS → degrade MORE
```

**Question:** How general is this? What are the boundary conditions?

---

## Dimension 1: Model Breadth

### 1.1 Architecture Families

| Family | Models to Test | Hypothesis |
|--------|---------------|------------|
| BLOOM-like | BLOOM-1B, BLOOM-3B | Effect scales with size |
| GPT-like | GPT-2, GPT-Neo, Pythia | Effect if outliers exist |
| OPT | OPT-125M, OPT-350M | Meta's recipe differs |
| Llama-like | TinyLlama, Llama-2-7B | Modern architecture |
| T5-like | mT5-small, mT5-base | Encoder-decoder |
| BERT-like | mBERT, XLM-R | Encoder-only |

**Experiments:**

```
EXP-020: Model Outlier Census
  For each model:
    1. Extract per-layer kurtosis
    2. Classify: Heavy (κ>50), Moderate (15-50), Mild (5-15), Gaussian (<5)
    3. Identify outlier layer positions

EXP-021: Cross-Architecture Correlation
  For models with outliers (κ>50):
    1. Extract language-wise activation patterns
    2. Compute r(outlier_activation, degradation)
    3. Compare with BLOOM's r=-0.834

  Prediction:
    - Heavy outlier models: r < -0.5
    - Gaussian models: |r| < 0.3
```

### 1.2 Model Scale

| Size | Models | Question |
|------|--------|----------|
| <100M | Tiny-GPT2, BERT-tiny | Too small for outliers? |
| 100M-500M | BLOOM-560M, GPT-2, Pythia-410M | Current evidence |
| 500M-1B | BLOOM-1B1, Pythia-1B | Does effect strengthen? |
| 1B-7B | BLOOM-3B, Llama-2-7B | Scale limits? |
| >7B | BLOOM-7B, BLOOM-176B | Production scale |

**Experiments:**

```
EXP-022: Scale Dependence
  Within BLOOM family:
    1. Measure κ_max at each scale
    2. Measure outlier layer count
    3. Measure disparity (if degradation data available)

  Prediction:
    - κ_max increases with scale (more training = more outliers)
    - Disparity increases with scale
```

---

## Dimension 2: Language Depth

### 2.1 Language Coverage

| Tier | Languages | Training Data | Expected Behavior |
|------|-----------|---------------|-------------------|
| High-resource | eng, zho, fra, deu | >10% of training | Low degradation |
| Mid-resource | ara, hin, jpn, kor | 1-10% | Moderate degradation |
| Low-resource | tha, fin, heb, tur | 0.1-1% | High degradation |
| Very low | amh, yor, swh | <0.1% | Highest degradation? |

**Experiments:**

```
EXP-023: Extended Language Correlation
  Add 10+ languages to current 14:
    - African: amh, yor, swh, hau
    - South Asian: ben, tam, tel
    - Southeast Asian: ind, msa
    - European: pol, ces, ron

  For each:
    1. Compute outlier activation fraction
    2. Get degradation from literature or measure
    3. Extend correlation analysis

  Prediction:
    - Very low-resource languages cluster at low activation, high degradation
    - Correlation strengthens with more languages

EXP-024: Training Data Correlation
  If BLOOM training stats available:
    1. Get tokens per language
    2. Correlate: tokens(λ) ↔ outlier_activation(λ)
    3. Test: is training data the cause?

  Prediction:
    - r(training_tokens, outlier_activation) > 0.7
    - Causal chain: more data → more specialization → more outlier use
```

### 2.2 Language Families

| Family | Languages | Hypothesis |
|--------|-----------|------------|
| Indo-European | eng, fra, deu, hin | Similar activation patterns? |
| Sino-Tibetan | zho, mya | Different patterns? |
| Afroasiatic | ara, heb | Script effects? |
| Japonic | jpn | Isolate, unique pattern? |
| Dravidian | tam, tel | Underrepresented family |

**Experiments:**

```
EXP-025: Language Family Effects
  1. Group languages by family
  2. Compare within-family vs between-family variance
  3. Test: does family predict activation pattern?

  Prediction:
    - Within-family variance < between-family variance
    - But resource level dominates family effects
```

---

## Dimension 3: Quantization Depth

### 3.1 Bit-Width Sweep

| Bits | Method | Expected Disparity |
|------|--------|-------------------|
| FP16 | Baseline | ~0 (no quantization) |
| INT8 | Standard PTQ | Low |
| INT6 | Intermediate | Moderate |
| INT4 | Aggressive | High (our data) |
| INT3 | Extreme | Very high? |
| INT2 | Theoretical | Maximum? |

**Experiments:**

```
EXP-026: Bit-Width Threshold
  For BLOOM-560M, each language:
    1. Quantize at 8, 6, 4, 3, 2 bits
    2. Measure degradation D(λ, B)
    3. Find threshold B*(λ) where degradation spikes

  Prediction:
    - B*(eng) < B*(ara) — English tolerates lower precision
    - Threshold correlates with outlier activation
    - r(outlier_activation, B*) > 0.5

EXP-027: Disparity vs Bit-Width
  1. Compute disparity at each bit-width
  2. Plot: Disparity(B) vs B
  3. Find critical bit-width where disparity emerges

  Prediction:
    - Disparity ≈ 0 for B ≥ 8
    - Disparity grows as B decreases
    - Critical threshold around B = 5-6 for BLOOM
```

### 3.2 Quantization Methods

| Method | Description | Hypothesis |
|--------|-------------|------------|
| Naive | Uniform, single α | Baseline (high disparity) |
| ACIQ | Optimal α per tensor | Reduced disparity? |
| GPTQ | Hessian-based | Method-dependent? |
| AWQ | Activation-aware | May help low-resource? |
| LA-ACIQ | Per-language α | Should minimize disparity |

**Experiments:**

```
EXP-028: Method Comparison
  For BLOOM-560M, INT4:
    1. Apply each quantization method
    2. Measure per-language degradation
    3. Compute disparity for each method

  Prediction:
    - Naive: highest disparity
    - ACIQ: some improvement
    - AWQ: may help (activation-aware)
    - LA-ACIQ: lowest disparity (if implemented correctly)

EXP-029: LA-ACIQ Implementation
  1. Implement per-language calibration
  2. For each language, use calibration set from that language
  3. Compare disparity: single α vs per-language α

  Prediction:
    - Disparity reduction of 30-50%
    - Validates theoretical framework
```

---

## Dimension 4: Layer Depth

### 4.1 Layer-wise Analysis

| Layer Group | BLOOM Layers | Kurtosis | Role |
|-------------|--------------|----------|------|
| Embedding | 0 | Low | Tokenization |
| Early | 1-7 | Mixed (5 is outlier) | Low-level features |
| Middle | 8-16 | Low | ? |
| Late | 17-23 | Mixed (21,22 outliers) | High-level features |
| Head | 24 | Low | Prediction |

**Experiments:**

```
EXP-030: Layer Contribution
  1. Quantize layers selectively:
     a. Only outlier layers (5, 21, 22) at INT4
     b. Only non-outlier layers at INT4
     c. Mixed precision
  2. Measure per-language degradation
  3. Decompose: which layers contribute to disparity?

  Prediction:
    - Quantizing outlier layers alone: LOW disparity (affects all equally)
    - Quantizing non-outlier layers: HIGH disparity (low-resource relies on these)
    - This would CONFIRM our mechanism

EXP-031: Component Analysis
  Within each layer, decompose:
    - Query/Key/Value weights
    - Attention output
    - MLP up-projection
    - MLP down-projection

  Question: Which component has outliers? Which causes disparity?
```

---

## Dimension 5: Mechanism Depth

### 5.1 Causal Interventions

**Experiments:**

```
EXP-032: Activation Pattern Manipulation
  Hypothesis: If we could force low-resource languages to use outlier layers,
              their degradation would decrease.

  Method (approximation):
    1. Fine-tune on low-resource language briefly
    2. Measure activation pattern change
    3. Measure degradation change

  Prediction:
    - Fine-tuning increases outlier activation
    - Degradation decreases

EXP-033: Outlier Removal
  Hypothesis: If we remove outliers, disparity disappears (but so does performance).

  Method:
    1. Clip weights to remove outliers (set max|W| = 1)
    2. Measure per-language degradation

  Prediction:
    - All languages degrade more
    - But disparity decreases (now all languages suffer equally)

EXP-034: Outlier Amplification
  Hypothesis: If we amplify outliers, disparity increases.

  Method:
    1. Scale up weights in outlier layers (×2)
    2. Measure per-language degradation

  Prediction:
    - Low-resource languages degrade more
    - High-resource languages may improve slightly
```

### 5.2 Training Interventions

**Experiments:**

```
EXP-035: Dropout Effect on Outliers
  Train small model with and without dropout:
    1. BLOOM-like architecture, small scale
    2. Variant A: dropout = 0 (like BLOOM)
    3. Variant B: dropout = 0.1 (like XGLM)

  Measure: κ_max, outlier layer count

  Prediction:
    - Variant A develops outliers
    - Variant B stays Gaussian
    - Confirms training dynamics hypothesis

EXP-036: Balanced Data Effect
  If we could retrain BLOOM with balanced data:
    1. Simulate with fine-tuning
    2. Equal samples per language
    3. Measure activation pattern change

  Prediction:
    - Activation patterns converge across languages
    - Disparity decreases
```

---

## Resource Requirements

### CPU-Only (Can Do Now)

| Experiment | Estimated Time | Status |
|------------|---------------|--------|
| EXP-020: Outlier Census | 2-4 hours | Ready |
| EXP-023: Extended Languages | 1-2 hours | Need activation data |
| EXP-031: Component Analysis | 1 hour | Ready |

### GPU Required (Need Access)

| Experiment | GPU Hours | Priority |
|------------|-----------|----------|
| EXP-021: Cross-Architecture | 4-8 | HIGH |
| EXP-026: Bit-Width Sweep | 2-4 | HIGH |
| EXP-029: LA-ACIQ Implementation | 4-8 | HIGH |
| EXP-030: Layer Contribution | 8-16 | MEDIUM |
| EXP-032: Activation Manipulation | 16-32 | LOW |
| EXP-035: Training Intervention | 100+ | Future |

### Cloud Cost Estimate

```
Immediate priorities (EXP-021, 026, 029):
  g4dn.xlarge: $0.526/hr
  Estimated hours: 15
  Total: ~$8

Full experimental plan:
  Estimated hours: 100
  Total: ~$50-100
```

---

## Execution Order

### Phase 1: Breadth (CPU)
```
Week 1:
  [x] EXP-020a: Census (BLOOM, XGLM, XLM-R, mT5) — DONE
  [ ] EXP-020b: Census (Pythia, GPT-2, OPT)
  [ ] EXP-031: Component analysis for BLOOM
```

### Phase 2: Depth (GPU)
```
Week 2-3:
  [ ] EXP-026: Bit-width sweep (BLOOM)
  [ ] EXP-027: Disparity vs bit-width
  [ ] EXP-029: LA-ACIQ prototype
```

### Phase 3: Validation (GPU)
```
Week 4-5:
  [ ] EXP-021: Cross-architecture correlation
  [ ] EXP-030: Layer contribution decomposition
  [ ] EXP-023: Extended languages
```

### Phase 4: Mechanism (GPU+Significant)
```
Week 6+:
  [ ] EXP-032: Activation manipulation
  [ ] EXP-033: Outlier removal
  [ ] EXP-035: Training intervention (if resources)
```

---

## Success Criteria

### Minimum Viable Result

```
At least ONE of:
  1. Cross-architecture replication (r < -0.5 in another heavy-outlier model)
  2. Bit-width threshold correlation (r > 0.5)
  3. LA-ACIQ reduces disparity by >20%
```

### Strong Result

```
ALL of:
  1. Replication in 2+ models
  2. Bit-width mechanism validated
  3. LA-ACIQ works
  4. Clear boundary conditions (Gaussian models show no effect)
```

### Breakthrough Result

```
PLUS:
  1. Training intervention prevents outliers
  2. Production-ready LA-ACIQ tool
  3. Explains discrepancy between BLOOM and XGLM training
```

---

## Testable Predictions Summary

| ID | Prediction | Test | Falsifiable? |
|----|------------|------|--------------|
| P1 | Heavy-outlier models show r < -0.5 | EXP-021 | Yes |
| P2 | Gaussian models show \|r\| < 0.3 | EXP-021 | Yes |
| P3 | κ_max increases with model scale | EXP-022 | Yes |
| P4 | Training tokens correlate with activation | EXP-024 | Yes |
| P5 | B*(eng) < B*(ara) | EXP-026 | Yes |
| P6 | Disparity emerges at B ≈ 5-6 | EXP-027 | Yes |
| P7 | LA-ACIQ reduces disparity | EXP-029 | Yes |
| P8 | Quantizing non-outlier layers causes disparity | EXP-030 | Yes |
| P9 | Dropout prevents outlier formation | EXP-035 | Yes |

---

*Experimental plan v1.0 — 2026-01-03*
