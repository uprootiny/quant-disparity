# Experimental Constraints and Requirements

## From Literature

### Constraint Set 1: Fair Comparison
*Source: "How Does Quantization Affect Multilingual LLMs?" (EMNLP 2024)*

1. **Semantic equivalence**: Test texts must convey same meaning across languages
2. **Length normalization**: Report per-token metrics when comparing languages
3. **Calibration data**: Must include all test languages in calibration set
4. **Model selection**: Use models with documented multilingual training data

### Constraint Set 2: Quantization Validity
*Source: "Super Weight in Large Language Models" (arXiv:2411.07191)*

1. **Weight coverage**: Analyze all weight matrices, not just attention
2. **Magnitude measurement**: Use absolute value, not signed values
3. **Preservation selection**: Global threshold, not per-tensor
4. **Dequantization**: Always apply scale factor after quantization

### Constraint Set 3: Reproducibility
*Source: Standard scientific practice*

1. **Random seeds**: Document and fix for reproducibility
2. **Environment**: Record transformers, torch versions
3. **Hardware**: Note CPU vs GPU, memory constraints
4. **Timestamps**: All results must have generation timestamp

---

## Technical Requirements

### Memory Constraints

| Operation | Min RAM | Recommended |
|-----------|---------|-------------|
| GPT-2 (124M) | 2GB | 4GB |
| OPT-125M | 2GB | 4GB |
| BLOOM-560M | 4GB | 8GB |
| 7B models | 16GB | 32GB |
| 7B INT4 | 4GB | 8GB |

### Compute Constraints

| Operation | CPU Time | GPU Time |
|-----------|----------|----------|
| GPT-2 PPL (1 text) | ~0.5s | ~0.1s |
| Weight analysis (GPT-2) | ~2s | ~0.5s |
| Full pipeline (Phase A+B) | ~10min | ~2min |

---

## Validity Constraints

### Statistical Validity

1. **Minimum samples**: 3 runs for any reported metric
2. **Significance**: p < 0.05 for correlation claims
3. **Effect size**: Report Cohen's d or similar
4. **Confidence intervals**: Include for key metrics

### Experimental Validity

1. **No data leakage**: Test texts not in training data
2. **Fresh model loads**: Start from pretrained, not fine-tuned
3. **Consistent tokenization**: Same tokenizer for all languages
4. **Temperature = 0**: Deterministic generation for perplexity

---

## Boundary Conditions

### Languages

- **Minimum test set**: en, zh, he (high, mid, low resource)
- **Recommended**: en, de, fr, zh, he, ar (covers Latin, Han, Hebrew, Arabic scripts)
- **Extended**: Add ru, ja, ko for Cyrillic, Japanese, Korean

### Quantization Levels

- **Primary**: INT4 (most aggressive, clearest disparity)
- **Secondary**: INT8 (practical deployment target)
- **Tertiary**: INT3, INT2 (extreme, research only)

### Preservation Levels

- **Test range**: 0%, 1%, 5%, 10%, 20%, 50%
- **Skip**: >50% (negates quantization benefit)

---

## Ethical Constraints

1. **No bias amplification**: Report if quantization amplifies existing biases
2. **Fairness documentation**: Document disparity for all tested languages
3. **Limitations**: Clearly state what findings do NOT generalize to
4. **Dual use**: Consider deployment implications

---

## Reporting Requirements

### Minimum Report Contents

1. Model identifier (HuggingFace ID)
2. Quantization method (symmetric/asymmetric, per-tensor/per-channel)
3. Languages tested with resource classification
4. Baseline metrics per language
5. Post-quantization metrics per language
6. Disparity ratio (LR/HR)
7. Statistical significance of any correlations

### Format

```markdown
## Experiment: [ID]
- Model: [HuggingFace ID]
- Quantization: [Method] to [Bits]
- Languages: [List]

### Results
| Language | Resource | Baseline PPL | Quant PPL | Degradation |
|----------|----------|--------------|-----------|-------------|
| ... | ... | ... | ... | ... |

### Analysis
- HR avg degradation: X%
- LR avg degradation: Y%
- Disparity ratio: Z

### Conclusion
[Support/Refute hypothesis, with confidence level]
```

---

*Last updated: 2026-01-04*
