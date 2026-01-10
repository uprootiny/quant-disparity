# Theory Investigation: Hypotheses and Experiments

*Closing the gap between empirical findings and theoretical understanding*

---

## The Core Mystery

We found that **L0+L9+L11 achieves 0.59x disparity** with 80 experiments.

But we can't answer: **WHY these layers?**

This document proposes hypotheses and experiments to build theoretical understanding.

---

## Gap 1: Why Does Variance Predict Criticality?

### Empirical Finding
```
log(disparity) = 4.76 - 2.22 × norm(variance) + ...
R² = 0.936

High variance → LOW disparity → MORE critical layer
```

### Competing Hypotheses

#### H1.1: Information Content Hypothesis
> High-variance layers encode MORE information, so losing precision costs more.

**Prediction:** Mutual information I(input; layer_output) correlates with variance.

**Experiment T-001:**
```python
def test_information_content(model, texts):
    """
    For each layer, estimate information content via:
    1. Activation entropy
    2. Reconstruction loss (can you decode input from activations?)
    """
    for layer in range(num_layers):
        acts = get_activations(model, texts, layer)

        # Entropy of activations
        entropy = compute_entropy(acts)

        # Reconstruction: train small decoder from acts → input
        recon_loss = train_decoder(acts, texts)

        # Compare with variance
        variance = compute_weight_variance(model, layer)

        # Test: high variance → high entropy → low recon_loss?
```

**Status:** CPU-feasible (no gradients needed for frozen decoder)

---

#### H1.2: Quantization Robustness Hypothesis
> High-variance weights have larger scale factors, so relative quantization error is smaller.

**Prediction:** Quantization MSE / signal power is lower for high-variance layers.

**Experiment T-002:**
```python
def test_relative_error(model):
    """
    For each layer:
    1. Compute INT4 quantization error
    2. Normalize by signal power
    3. Correlate with disparity
    """
    for layer in range(num_layers):
        weights = get_weights(model, layer)
        quantized = simulate_int4(weights)

        # Absolute error
        mse = ((weights - quantized) ** 2).mean()

        # Relative error (normalized)
        signal_power = (weights ** 2).mean()
        relative_error = mse / signal_power

        # Test: high variance → low relative_error?
```

**Status:** CPU-feasible

---

#### H1.3: Language Distribution Hypothesis
> High-variance layers are activated differently by different languages; quantization disrupts language-specific pathways.

**Prediction:** Activation variance differs more across languages in critical layers.

**Experiment T-003:**
```python
def test_language_activation_variance(model, tokenizer, texts):
    """
    For each layer and language:
    1. Get activations
    2. Compute activation variance
    3. Test if critical layers show MORE variance across languages
    """
    for layer in range(num_layers):
        lang_variances = {}
        for lang, text in texts.items():
            acts = get_activations(model, tokenizer, text, layer)
            lang_variances[lang] = acts.var()

        # Across-language variance
        cross_lang_var = np.var(list(lang_variances.values()))

        # Test: critical layers have higher cross_lang_var?
```

**Status:** CPU-feasible with activation caching

---

## Gap 2: Why L0+L11 Synergy?

### Empirical Finding
```
L0 alone: 3.6x disparity
L11 alone: 336x disparity (HARMFUL!)
L0+L11 together: 0.7x disparity (SYNERGISTIC)
```

### Competing Hypotheses

#### H2.1: Residual Stream Theory
> L0 errors propagate through the residual stream. Without L0 protection, L11 protection is useless because it receives corrupted input.

**Prediction:** Activations at L11 are more similar (FP32 vs INT4) when L0 is protected.

**Experiment T-004:**
```python
def test_residual_propagation(model, texts):
    """
    Compare activation similarity at L11 under different protection schemes.
    """
    state = save_state(model)

    # Config 1: Only L11 protected
    restore(model, state)
    quantize_except(model, protect=[11])
    acts_l11_only = get_activations(model, texts, layer=11)

    # Config 2: L0+L11 protected
    restore(model, state)
    quantize_except(model, protect=[0, 11])
    acts_both = get_activations(model, texts, layer=11)

    # Config 3: FP32 baseline
    restore(model, state)
    acts_fp32 = get_activations(model, texts, layer=11)

    # Similarity
    sim_l11_only = cosine_sim(acts_l11_only, acts_fp32)
    sim_both = cosine_sim(acts_both, acts_fp32)

    # Test: sim_both > sim_l11_only?
```

**Status:** CPU-feasible

---

#### H2.2: Input-Output Gateway Theory
> L0 and L11 are "gateways" - L0 encodes input into the model's representation space, L11 decodes back to output space. Both transformations are critical.

**Prediction:** L0 and L11 have unique weight structure compared to middle layers.

**Experiment T-005:**
```python
def test_gateway_structure(model):
    """
    Analyze weight matrix structure:
    - Rank (SVD analysis)
    - Condition number
    - Eigenvalue distribution
    """
    for layer in range(num_layers):
        weights = get_attention_weights(model, layer)

        # SVD analysis
        U, S, V = torch.svd(weights)
        effective_rank = (S > S.max() * 0.01).sum()
        condition = S.max() / S.min()

        # Test: L0 and L11 have distinct structure?
```

**Status:** CPU-feasible

---

#### H2.3: Error Cancellation Theory
> Errors at L0 and L11 may partially cancel when both are quantized together, but protecting only L11 amplifies L0 errors.

**Prediction:** Error covariance between L0 and L11 is negative.

**Experiment T-006:**
```python
def test_error_correlation(model, texts):
    """
    Measure how quantization errors at different layers correlate.
    """
    state = save_state(model)

    # Get FP32 outputs
    restore(model, state)
    out_fp32 = model_forward(model, texts)

    # Get outputs with only L0 quantized
    restore(model, state)
    quantize_only_layer(model, 0)
    out_l0_quant = model_forward(model, texts)
    error_l0 = out_l0_quant - out_fp32

    # Get outputs with only L11 quantized
    restore(model, state)
    quantize_only_layer(model, 11)
    out_l11_quant = model_forward(model, texts)
    error_l11 = out_l11_quant - out_fp32

    # Correlation
    correlation = torch.corrcoef(error_l0.flatten(), error_l11.flatten())

    # Test: negative correlation means errors cancel?
```

**Status:** CPU-feasible

---

## Gap 3: Why 75% Depth (L9)?

### Empirical Finding
```
L0+L11: 0.92x
L0+L9+L11: 0.59x (adding L9 helps significantly)
L9 is at position 9/12 = 75%
```

### Competing Hypotheses

#### H3.1: Information Bottleneck Hypothesis
> L9 is an "information bottleneck" where representations compress before the final layers expand toward output.

**Prediction:** Activation dimensionality (effective rank) is lowest at ~75% depth.

**Experiment T-007:**
```python
def test_bottleneck_structure(model, texts):
    """
    Measure effective dimensionality at each layer.
    """
    for layer in range(num_layers):
        acts = get_activations(model, texts, layer)

        # PCA on activations
        pca = PCA().fit(acts)
        explained_var = pca.explained_variance_ratio_

        # Effective dimensionality (where 95% variance explained)
        cumsum = np.cumsum(explained_var)
        eff_dim = np.searchsorted(cumsum, 0.95) + 1

        # Test: eff_dim lowest at ~75% depth?
```

**Status:** CPU-feasible

---

#### H3.2: Morphological Consolidation Hypothesis
> L9 is where morphological features are consolidated for MRLs, explaining why it helps Hebrew/Arabic most.

**Prediction:** Probing accuracy for morphological features peaks at L9.

**Experiment T-008:**
```python
def test_morphological_probing(model, tokenizer):
    """
    Probe for morphological features at each layer.
    Requires annotated data for Hebrew/Arabic.
    """
    # This requires morphological annotations
    # Simplified: use agreement prediction as proxy

    agreement_texts = {
        'correct': 'הילדים רצים',  # The boys run (correct agreement)
        'wrong': 'הילדים רץ',  # The boys runs (wrong agreement)
    }

    for layer in range(num_layers):
        # Get probability of correct vs wrong
        prob_correct = get_continuation_prob(model, agreement_texts['correct'])
        prob_wrong = get_continuation_prob(model, agreement_texts['wrong'])

        # Ratio indicates morphological encoding
        agreement_score = prob_correct / (prob_correct + prob_wrong)

        # Test: agreement_score peaks at L9?
```

**Status:** CPU-feasible (but needs careful text construction)

---

#### H3.3: Representation Convergence Hypothesis
> L9 is where multilingual representations converge before language-specific output processing.

**Prediction:** Cross-lingual activation similarity is highest at L9.

**Experiment T-009:**
```python
def test_representation_convergence(model, tokenizer):
    """
    Measure cross-lingual similarity at each layer.
    Use parallel sentences (same meaning, different languages).
    """
    parallel = {
        'en': 'The dog runs.',
        'de': 'Der Hund läuft.',
        'he': 'הכלב רץ.',
    }

    for layer in range(num_layers):
        acts = {}
        for lang, text in parallel.items():
            acts[lang] = get_activations(model, tokenizer, text, layer)

        # Pairwise similarity
        sim_en_de = cosine_sim(acts['en'], acts['de'])
        sim_en_he = cosine_sim(acts['en'], acts['he'])
        avg_sim = (sim_en_de + sim_en_he) / 2

        # Test: avg_sim peaks at L9?
```

**Status:** CPU-feasible

---

## Gap 4: Information-Theoretic Bound

### Missing Piece
We have no formal bound on minimum achievable disparity.

### Hypothesis H4.1: Rate-Distortion Bound
> There exists a fundamental trade-off: disparity ≥ f(bits, language_entropy, layer_protection)

**Experiment T-010:**
```python
def test_rate_distortion(model, tokenizer, texts):
    """
    Vary bit-width and protection %, measure disparity.
    Fit rate-distortion curve.
    """
    results = []

    for bits in [2, 3, 4, 6, 8]:
        for protect_pct in [0, 10, 20, 30, 50]:
            disparity = measure_with_config(model, texts, bits, protect_pct)
            results.append((bits, protect_pct, disparity))

    # Fit: disparity = a * 2^(-b*bits) * (1 - c*protect_pct)
    # This gives theoretical relationship
```

**Status:** CPU-feasible (multiple runs)

---

## Experiment Priority Matrix

| Exp | Hypothesis | CPU | GPU | Time | Value |
|-----|------------|-----|-----|------|-------|
| T-002 | Relative error | ✓ | | 30m | HIGH |
| T-003 | Language activation | ✓ | | 1h | HIGH |
| T-004 | Residual propagation | ✓ | | 1h | HIGH |
| T-009 | Representation convergence | ✓ | | 1h | HIGH |
| T-005 | Gateway structure | ✓ | | 30m | MEDIUM |
| T-007 | Bottleneck | ✓ | | 1h | MEDIUM |
| T-001 | Information content | ✓ | | 2h | MEDIUM |
| T-006 | Error correlation | ✓ | | 1h | MEDIUM |
| T-008 | Morphological probing | ✓ | | 2h | HIGH |
| T-010 | Rate-distortion | ✓ | | 3h | HIGH |

---

## Theoretical Framework Sketch

If experiments support our hypotheses:

```
PROPOSED THEORY: Gateway-Bottleneck Model of Multilingual Quantization

1. GATEWAY LAYERS (L0, L_last):
   - Transform between token space and representation space
   - High variance = high information content
   - Quantization here disrupts fundamental encoding/decoding

2. BOTTLENECK LAYER (L_0.75):
   - Compresses representations before output expansion
   - Cross-lingual features converge here
   - Quantization here disrupts shared multilingual processing

3. SYNERGY:
   - L0 errors propagate through residual stream
   - L_last cannot compensate without clean input
   - Protecting both creates "error containment"

4. MRL EFFECT:
   - Morphologically rich languages use bottleneck layer more heavily
   - More complex morphology → more reliance on consolidation
   - Explains disproportionate benefit from L9 protection

FORMAL STATEMENT:
  disparity(L) ∝ exp(-α × variance(L) - β × position_boundary(L) - γ × bottleneck(L))

Where:
  - variance(L) = weight variance of layer L
  - position_boundary(L) = proximity to input/output
  - bottleneck(L) = information compression at layer L
```

---

## Implementation Plan

### Phase 1: Quick Tests (Today)
1. T-002: Relative error analysis
2. T-004: Residual propagation test
3. T-009: Representation convergence

### Phase 2: Deeper Analysis (This Week)
4. T-003: Language activation variance
5. T-007: Bottleneck structure
6. T-005: Gateway structure

### Phase 3: Theoretical Synthesis
7. T-010: Rate-distortion curve
8. Fit formal model to data
9. Derive predictions for new architectures

---

## Expected Outcomes

### If Hypotheses Supported:
- Theoretical paper with formal framework
- Predictions for Llama-2, Mistral (testable on GPU)
- Novel contribution: First theory of multilingual quantization disparity

### If Hypotheses Refuted:
- Revised understanding
- New hypotheses based on data
- Still valuable: rules out explanations

---

*Created: 2026-01-09*
