---
layout: page
title: Methodology
permalink: /methodology/
nav_order: 5
---

# Research Methodology

Our approach combines statistical rigor, mechanistic analysis, and cross-model validation.

---

## Research Philosophy

### Guiding Principles

1. **Pre-registration:** Hypotheses stated before experiments run
2. **Multiple validation:** Bootstrap, permutation, leave-one-out
3. **Cross-model testing:** Findings must generalize beyond one model
4. **Mechanistic depth:** Not just correlation, but causal understanding
5. **Negative results matter:** Document what doesn't work (e.g., fertility ≠ degradation)

### The Unix Philosophy Applied

```
Make each experiment do one thing well.
Expect the output of every experiment to become
the input to another, as yet unknown, analysis.
Write experiments to handle data streams.
```

---

## Statistical Framework

### Primary Analysis

For the core disparity finding:

```python
# Pearson correlation with bootstrapped confidence intervals
r, p = pearsonr(outlier_activation, degradation_rate)

# Bootstrap 95% CI (10,000 resamples)
bootstrap_ci = bootstrap_correlation(data, n=10000)

# Permutation test (10,000 permutations)
perm_p = permutation_test(outlier_activation, degradation_rate, n=10000)

# Leave-one-out sensitivity
loo_results = [pearsonr(...) for subset in leave_one_out(data)]
```

### Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Significance | p < 0.05 | Standard |
| Practical significance | \|r\| > 0.5 | Meaningful effect |
| Hypothesis support | p < 0.1 + direction correct | Allows weak effects |

### Kurtosis Measurement

Excess kurtosis (Fisher's definition):

```python
from scipy.stats import kurtosis
k = kurtosis(weights, fisher=True)  # Excess kurtosis

# Classification
if k > 50:   class = "HEAVY"
elif k > 10: class = "Moderate"
elif k > 3:  class = "Mild"
else:        class = "Gaussian"
```

---

## Experimental Protocol

### Standard Experiment Structure

```python
def run_experiment():
    # 1. State hypothesis with prediction
    hypothesis = "H-X: [claim]"
    prediction = "[quantitative prediction]"

    # 2. Load data/models
    model = load_model(...)

    # 3. Extract measurements
    measurements = extract_metric(model, ...)

    # 4. Statistical analysis
    result = statistical_test(measurements)

    # 5. Compare to prediction
    supported = evaluate_hypothesis(result, prediction)

    # 6. Save results
    save_json(results, f"exp{N}_results_{timestamp}.json")

    return results
```

### Reproducibility Requirements

1. **Random seeds:** Set and documented
2. **Model versions:** Exact HuggingFace identifiers
3. **Result storage:** JSON with timestamps
4. **Code versioning:** Git commits for each experiment

---

## Model Selection

### Primary Models

| Model | Parameters | Why Selected |
|-------|------------|--------------|
| BLOOM-560M | 560M | Original disparity finding |
| OPT-125M | 125M | Highest kurtosis (κ=562) |
| GPT-2-small | 124M | English-centric baseline |
| mBERT | 110M | Multilingual, well-studied |
| Pythia family | 160M-1B | Controlled size comparison |

### Model Coverage

- **Architecture:** Decoder-only, encoder-only, encoder-decoder
- **Training:** Different data mixtures, dropout settings
- **Size:** 100M to 1B parameters (CPU-feasible)
- **Multilinguality:** Monolingual to 100+ languages

---

## Measurement Techniques

### Kurtosis Extraction

```python
def compute_layer_kurtosis(model):
    """Extract kurtosis per layer."""
    layer_kurtosis = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.detach().cpu().numpy().flatten()
            k = kurtosis(weights, fisher=True)
            layer_kurtosis[name] = k

    return layer_kurtosis
```

### Attention Pattern Analysis

```python
def extract_attention_patterns(model, tokenizer, text):
    """Extract attention weights from all heads."""
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack: (num_layers, num_heads, seq_len, seq_len)
    return torch.stack(outputs.attentions)
```

### Pseudo-Perplexity for MLM

```python
def compute_pseudo_perplexity(model, tokenizer, text):
    """Pseudo-perplexity via masked prediction."""
    inputs = tokenizer(text, return_tensors="pt")
    total_loss = 0
    n_tokens = 0

    for i in range(1, len(inputs["input_ids"][0]) - 1):
        masked = inputs["input_ids"].clone()
        original = masked[0, i].item()
        masked[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked).logits

        loss = F.cross_entropy(logits[0, i:i+1],
                               torch.tensor([original]))
        total_loss += loss.item()
        n_tokens += 1

    return np.exp(total_loss / n_tokens)
```

### Simulated Quantization

```python
def simulate_quantization(tensor, bits=8):
    """Simulate quantization without specialized libraries."""
    abs_max = tensor.abs().max()
    scale = abs_max / (2 ** (bits - 1) - 1)

    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized,
                            -(2 ** (bits - 1)),
                            2 ** (bits - 1) - 1)

    return quantized * scale
```

---

## Cross-Track Integration

### Data Flow

```
Track A (Soudry)                Track B (Belinkov)
    │                               │
    ├── Kurtosis per layer          ├── Head classification
    │                               │
    └───────────┬───────────────────┘
                │
                ▼
        Cross-Track Analysis
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
Track C (Schwartz)      Track D (Goldberg)
    │                       │
Tokenization            Morphology
efficiency              processing
```

### Shared Resources

1. **Parallel corpus:** Same sentences across languages
2. **Resource level estimates:** Consistent across tracks
3. **Model loading:** Cached to avoid redundant downloads
4. **Results format:** Standardized JSON schema

---

## Validation Strategies

### Within-Experiment

1. **Bootstrap CI:** 10,000 resamples for confidence intervals
2. **Permutation test:** 10,000 permutations for p-values
3. **Leave-one-out:** Check if single observations drive effect

### Cross-Experiment

1. **Hypothesis pre-registration:** State predictions before running
2. **Effect replication:** Core findings in multiple experiments
3. **Negative controls:** Experiments expected to fail

### Cross-Model

1. **Generalization check:** Effect in 3+ architectures
2. **Size scaling:** Effect across model sizes
3. **Training variation:** Different training configurations

---

## Limitations

### Current Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| No GPU access | Can't run real quantization | Simulated quantization |
| Model size | Limited to <1B params | Focus on accessible models |
| Language coverage | Not exhaustive | Prioritize diverse sample |

### Known Threats to Validity

1. **Simulated vs real quantization:** May miss implementation-specific effects
2. **Correlation ≠ causation:** Need causal intervention (GPU required)
3. **Sample size:** Some analyses have few languages

### Planned Mitigations

- **GPU experiments:** When access available, run EXP-009, EXP-031
- **Larger models:** Cloud compute for 7B+ models
- **More languages:** Expand to 50+ with careful sampling

---

## Experiment Orchestration

### Runner Infrastructure

```python
# experiments/runner.py
EXPERIMENTS = {
    "exp022": {
        "name": "Architecture Comparison",
        "script": "phase-3-crossmodel/exp022_architecture.py",
        "hypotheses": ["H1", "H4"],
        "timeout": 300,
    },
    # ...
}

def run_parallel(exp_ids, max_workers=2):
    """Run multiple experiments in parallel."""
    with ThreadPoolExecutor(max_workers) as executor:
        futures = {
            executor.submit(run_experiment, eid): eid
            for eid in exp_ids
        }
        return {eid: f.result() for f, eid in futures.items()}
```

### Timeout Handling

- Default: 5 minutes per experiment
- Long-running: 10 minutes max
- Background mode: For overnight runs

---

## Reporting Standards

### Experiment Results

Each experiment produces:
1. Console output with summary
2. JSON file with full results
3. Hypothesis verdict (SUPPORTED/REJECTED/MIXED)

### Visualization (Planned)

- Kurtosis heatmaps per model
- Language degradation scatter plots
- Attention pattern visualizations

---

*Last updated: January 2026*
