# Literature Foundation: Track C (Efficiency-Fairness)

*Grounding our work in the Green AI and Efficient NLP literature*

**Target: Roy Schwartz Lab (Hebrew University of Jerusalem)**

---

## Core Publications: Schwartz Lab

### 1. "Green AI" (Schwartz et al., 2020) - *Communications of the ACM*

**Key Claims:**
- AI research overly focuses on accuracy, ignoring efficiency
- Propose reporting computation alongside results
- Call for "Red AI" → "Green AI" paradigm shift

**Our Extension:**
- Efficiency metrics ALSO ignore fairness
- An "efficient" model that only works for English is not truly efficient
- Propose: Report disparity alongside efficiency

**Connection Experiments:**
- C-004 (Carbon Cost): Quantifies fairness cost in carbon terms
- C-009 (Pareto Frontier): Maps efficiency-fairness tradeoff

---

### 2. "Show Your Work" (Dodge et al., 2019) - *EMNLP*

**Key Claims:**
- Reproducibility requires reporting hyperparameters, compute, variance
- Random seeds and hardware significantly affect results
- Propose reporting standards

**Our Extension:**
- Add LANGUAGE to reporting standards
- Results vary dramatically by language (4.24x disparity)
- Single-language evaluation hides multilingual harm

**Connection Experiments:**
- C-010 (Reporting Standards): What should be reported for fair evaluation
- C-011 (Variance by Language): Language-specific variance analysis

---

### 3. Efficient NLP Survey Work

**Relevant Papers:**
- Tay et al. (2022): "Efficient Transformers: A Survey"
- Menghani (2023): "Efficient Deep Learning: A Survey"
- Wan et al. (2023): "Efficient LLMs: A Survey"

**Common Gap:**
All focus on efficiency metrics (FLOPs, memory, latency) without fairness.

**Our Contribution:**
- Fair-Efficiency Score (FES)
- Language-stratified efficiency reporting
- Disparity-aware compression selection

---

## Foundational Quantization Literature

### 4. LLM.int8() (Dettmers et al., 2022)

**Key Claims:**
- 8-bit quantization preserves quality for LLMs
- Outlier features require mixed precision
- Emergent features at scale need protection

**Our Finding:**
- "Quality preservation" is language-dependent
- Outliers may be language-specific
- LR languages may have fewer/different outliers

**Connection Experiments:**
- C-012 (Outlier Analysis): Language-specific outlier distributions
- A-001 (Architecture): Outlier patterns across architectures

---

### 5. GPTQ (Frantar et al., 2023)

**Key Claims:**
- One-shot quantization to 4-bit is practical
- Layer-wise Hessian-based optimization
- Calibration data affects quality

**Our Finding:**
- Calibration data is typically English-dominated
- Layer importance varies by language
- GPTQ may systematically disadvantage LR languages

**Connection Experiments:**
- C-013 (Calibration Bias): Effect of calibration language
- C-014 (Hessian by Language): Language-specific sensitivities

---

### 6. AWQ (Lin et al., 2023)

**Key Claims:**
- Activation-aware quantization outperforms weight-only
- Salient weights (1%) need protection
- Search-based scale selection

**Our Finding:**
- Salient weights may differ by language
- Scale selection optimizes for majority language
- Activation patterns are language-dependent

**Connection Experiments:**
- C-015 (Salient Weights): Language-specific saliency
- C-016 (Scale Selection): Cross-lingual scale optimization

---

## Fairness in NLP Literature

### 7. Multilingual Bias Work

**Relevant Papers:**
- Blasi et al. (2022): "Systematic Inequalities in Language Technology"
- Joshi et al. (2020): "State and Fate of Linguistic Diversity"
- Ruder et al. (2022): "Square One Bias in NLP"

**Key Claims:**
- NLP systematically disadvantages non-English languages
- Resource availability creates feedback loops
- Benchmarks are English-centric

**Our Extension:**
- Compression AMPLIFIES existing disparities
- Efficient deployment makes things WORSE
- Need fairness-aware compression

---

### 8. Tokenization Bias

**Relevant Papers:**
- Rust et al. (2021): "How Good is Your Tokenizer?"
- Petrov et al. (2023): "Language Model Tokenizers Introduce Unfairness"
- Ahia et al. (2023): "Do All Languages Cost the Same?"

**Key Claims:**
- BPE creates fertility disparities
- LR languages pay more per semantic unit
- Tokenizer bias propagates through model

**Our Extension:**
- Tokenizer bias → alignment → quantization sensitivity
- The causal chain is: Tokenization → Alignment → Redundancy → Disparity
- We quantify the magnitude at each step

---

## Experimental Gaps in Literature

### Gaps We Address:

| Gap | Literature Status | Our Contribution |
|-----|-------------------|------------------|
| Cross-lingual quantization | NOT STUDIED | Full characterization |
| Efficiency-fairness tradeoff | NOT MAPPED | Pareto frontier |
| Fair compression metrics | NOT DEFINED | Fair-Efficiency Score |
| Layer importance by language | NOT STUDIED | Gateway-Bottleneck Model |
| Recovery strategies for LR | NOT STUDIED | C-008 recovery analysis |
| Combined compression effects | MINIMAL | C-006 interaction study |

---

## Experiment Design Principles

Based on literature review:

1. **Ground in established metrics** (perplexity, BLEU, accuracy)
2. **Use standard compression methods** (GPTQ, pruning, distillation)
3. **Test on representative languages** (diverse typology)
4. **Report both efficiency AND fairness**
5. **Provide actionable recommendations**

---

## 12 New Experiments Derived from Literature

| ID | Name | Literature Connection | Question |
|----|------|----------------------|----------|
| C-010 | Reporting Standards | Dodge 2019 | What should fairness reports include? |
| C-011 | Variance by Language | Dodge 2019 | Is variance language-dependent? |
| C-012 | Outlier Analysis | Dettmers 2022 | Are outliers language-specific? |
| C-013 | Calibration Bias | GPTQ | Does calibration language matter? |
| C-014 | Hessian by Language | GPTQ | Is Hessian sensitivity uniform? |
| C-015 | Salient Weights | AWQ | Do salient weights differ by language? |
| C-016 | Scale Optimization | AWQ | Can scales be language-optimized? |
| C-017 | Tokenizer Cascade | Rust 2021 | Tokenizer → Alignment → Disparity chain |
| C-018 | Benchmark Coverage | Joshi 2020 | How benchmarks hide disparity |
| C-019 | Feedback Loops | Blasi 2022 | Does compression create feedback? |
| C-020 | Semantic Unit Cost | Ahia 2023 | Cost per meaning across languages |
| C-021 | Green-Fair Reconciliation | Schwartz 2020 | Can Green AI be Fair AI? |

---

*This document grounds Track C in peer-reviewed literature and identifies specific experiments to fill gaps.*
