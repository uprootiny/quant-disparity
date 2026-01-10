# Soudry Mindset: High-Value CPU Experiments

*What can we learn without GPU, applying Soudry's methodology?*

---

## Soudry Lab Methodology Patterns

From analyzing their papers:

| Pattern | Example | Application to Our Work |
|---------|---------|-------------------------|
| **Theoretical grounding** | L_max ∝ N^1.82 from mean field | Derive why L0+L11 works mathematically |
| **Find thresholds** | √3 × quantization noise | Per-layer disparity threshold |
| **Distribution analysis** | Gradients are lognormal | Per-language activation distributions |
| **Minimal intervention** | Smooth-SwiGLU | Our L0+L11 is already minimal |
| **Closed-form solutions** | ACIQ optimal α | Can we derive optimal protection? |

---

## High-Value Experiments (No GPU Required)

### Tier 1: Theoretical Grounding

#### Exp-081: Mean Field Analysis of Gateway Layers

**Question:** Why do L0 and L11 form a synergistic pair?

**Soudry approach:** Use mean field theory to analyze signal propagation.

**Method:**
```python
# Compute signal-to-noise ratio through network
# For each layer, track how quantization noise propagates
# L0 noise propagates through ALL subsequent layers
# L11 noise directly affects output

def signal_propagation_analysis(model):
    """
    Mean field approximation:
    - Model each layer as linear transformation + noise
    - Track variance of signal vs variance of noise
    - Identify layers where SNR degrades most
    """
    for layer in model.layers:
        input_variance = measure_activation_variance(layer.input)
        weight_variance = measure_weight_variance(layer)
        noise_variance = quantization_noise(layer, bits=4)

        snr = input_variance * weight_variance / noise_variance
        # Lower SNR = more critical layer
```

**Prediction:** L0 and L11 have lowest SNR under quantization.

**Epistemic value:** HIGH - explains mechanism, not just correlation.

---

#### Exp-082: Quantization Noise Threshold per Layer

**Question:** Does the √3 threshold from Soudry apply per-layer?

**Soudry finding:** Training fails when gradient_norm < √3 × quantization_noise.

**Our analog:** Disparity spikes when layer_signal < √3 × quantization_noise.

**Method:**
```python
def layer_threshold_analysis(model, texts):
    """
    For each layer:
    1. Measure activation magnitude (signal)
    2. Compute quantization noise at INT4
    3. Check if signal/noise ratio crosses √3 threshold
    4. Correlate with disparity from Exp-080
    """
    for layer_idx in range(num_layers):
        signal = activation_norm(model, layer_idx, texts)
        noise = quantization_noise_norm(model, layer_idx, bits=4)
        ratio = signal / noise

        crosses_threshold = ratio < math.sqrt(3)
        # Compare with known disparity
```

**Prediction:** Layers where ratio < √3 for low-resource texts = critical layers.

**Epistemic value:** HIGH - connects our findings to established theory.

---

### Tier 2: Distribution Analysis

#### Exp-083: Per-Language Activation Distribution

**Question:** Do different languages have different effective kurtosis?

**ACIQ insight:** Optimal clipping α depends on distribution shape.

**Method:**
```python
def per_language_distribution(model, tokenizer, texts):
    """
    For each language and layer:
    1. Run forward pass
    2. Extract activations
    3. Fit distribution (Gaussian, Laplace, or estimate kurtosis)
    4. Compute ACIQ-optimal α per language
    """
    for lang, text in texts.items():
        activations = extract_activations(model, tokenizer, text)
        for layer_idx in range(num_layers):
            layer_acts = activations[layer_idx]
            kurtosis = compute_kurtosis(layer_acts)
            optimal_alpha = aciq_clip(kurtosis, bits=4)
            # Compare α across languages
```

**Prediction:** Low-resource languages have higher effective kurtosis (more outliers in their activation path).

**Epistemic value:** MEDIUM-HIGH - extends ACIQ framework to multilingual.

---

#### Exp-084: Language-Specific Optimal Clipping

**Question:** Does language-aware α reduce disparity?

**Method:**
```python
def language_aware_quantization(model, texts):
    """
    Instead of global α, use per-language optimal α:
    1. Compute language-specific activation statistics
    2. Derive per-language α using ACIQ formula
    3. Simulate quantization with language-weighted α
    4. Measure disparity improvement
    """
    # This tests LA-ACIQ without GPU
```

**Prediction:** Language-aware α reduces disparity by 30-50%.

**Epistemic value:** HIGH - if successful, validates core contribution.

---

### Tier 3: Cross-Track Connections

#### Exp-085: Outlier Weights × Language-Specific Heads

**Question:** Are outlier weights concentrated in language-specific heads?

**Track B finding:** 16.7% of heads are language-specific.
**Track A finding:** Outliers concentrate in attention projections.

**Method:**
```python
def outlier_head_correlation(model):
    """
    1. Identify language-specific heads (from B-001)
    2. Measure outlier ratio in each head
    3. Test correlation: language_specific ~ outlier_ratio
    """
    for layer in range(num_layers):
        for head in range(num_heads):
            is_lang_specific = language_specificity_score(model, layer, head)
            outlier_ratio = compute_outlier_ratio(model, layer, head)
            # Correlate
```

**Prediction:** Language-specific heads have HIGHER outlier ratios (they encode more specialized information).

**Epistemic value:** MEDIUM - connects two tracks mechanistically.

---

#### Exp-086: Morphological Complexity × Layer Criticality

**Question:** Do morphologically rich languages (MRLs) depend more on specific layers?

**Track D hypothesis:** MRLs need more computational depth for morphological disambiguation.

**Method:**
```python
def morphological_layer_analysis(model, tokenizer):
    """
    Languages: Hebrew, Arabic (MRL) vs English, French (non-MRL)
    1. Run each language through model
    2. Measure per-layer contribution to output (gradient magnitude)
    3. Compare layer importance profiles across language types
    """
    mrl_langs = ['he', 'ar']
    non_mrl_langs = ['en', 'fr']

    for lang in mrl_langs + non_mrl_langs:
        layer_importance = compute_layer_gradients(model, texts[lang])
        # MRLs may rely more on middle layers for disambiguation
```

**Prediction:** MRLs have broader layer importance distribution (need more layers to process).

**Epistemic value:** MEDIUM - bridges Track A and Track D.

---

### Tier 4: Theoretical Derivation

#### Exp-087: Closed-Form Disparity Predictor

**Question:** Can we derive a formula predicting disparity from layer statistics?

**Soudry approach:** Closed-form solutions enable rapid deployment.

**Method:**
```python
def derive_disparity_formula():
    """
    From our 80 experiments, we have:
    - Layer variance (Exp-057-058)
    - Outlier ratio (Exp-066)
    - Sparsity (Exp-048)
    - Position (layer index)

    Fit: disparity = f(variance, outliers, sparsity, position)

    Goal: Closed-form that predicts disparity from weight statistics alone
    """
    # Multiple regression on existing data
    features = ['variance', 'outlier_ratio', 'sparsity', 'position']
    target = 'disparity'

    # Fit model, extract coefficients
    # This gives ACIQ-style formula for multilingual quantization
```

**Prediction:** Formula achieves r > 0.9 prediction of disparity.

**Epistemic value:** VERY HIGH - publishable theoretical contribution.

---

### Tier 5: Phase Transition Analysis

#### Exp-088: Disparity Phase Transition

**Question:** Is there a critical protection threshold where disparity suddenly drops?

**Soudry finding:** L_max shows phase transition at specific weight variance.

**Our analog:** Disparity may have phase transition at specific protection %.

**Method:**
```python
def protection_phase_transition(model, tokenizer, texts):
    """
    Sweep protection from 0% to 50% in fine increments (0.5%)
    Plot: protection % vs disparity
    Look for: sudden drop (phase transition)
    """
    for protection_pct in range(0, 50, 0.5):
        apply_protection(model, protection_pct)
        disparity = measure_disparity(model, tokenizer, texts)
        # Plot and find inflection point
```

**Prediction:** Phase transition at ~11% (where L0+L11 kicks in).

**Epistemic value:** HIGH - if phase transition exists, it's theoretically interesting.

---

## Experiment Priority Matrix

| Exp | Name | Epistemic Value | Time | Connects To |
|-----|------|-----------------|------|-------------|
| 087 | Closed-form formula | VERY HIGH | 2h | ACIQ, all findings |
| 082 | √3 threshold | HIGH | 3h | Soudry FP4 paper |
| 081 | Mean field analysis | HIGH | 4h | Theoretical grounding |
| 084 | LA-ACIQ validation | HIGH | 4h | Core contribution |
| 088 | Phase transition | HIGH | 3h | Soudry L_max paper |
| 083 | Per-language distribution | MEDIUM-HIGH | 2h | ACIQ extension |
| 085 | Outlier × heads | MEDIUM | 2h | Track B connection |
| 086 | Morphological layers | MEDIUM | 3h | Track D connection |

---

## Recommended Sequence

1. **Exp-087** first - Uses existing data, high ROI
2. **Exp-082** - Tests Soudry's √3 threshold, novel connection
3. **Exp-088** - Quick sweep, may reveal new insight
4. **Exp-083 + 084** - LA-ACIQ validation sequence
5. **Exp-085 + 086** - Cross-track connections

Total: ~23 hours of experiments, all CPU-feasible.

---

## What This Gets Us

If successful:
1. **Theoretical paper** with closed-form disparity predictor (Exp-087)
2. **Connection to Soudry framework** via √3 threshold (Exp-082)
3. **LA-ACIQ validation** on small scale (Exp-083-084)
4. **Unified cross-track story** (Exp-085-086)

This prepares us for GPU stage by having theoretical predictions to validate.

---

*Analysis date: 2026-01-09*
