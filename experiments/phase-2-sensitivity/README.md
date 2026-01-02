# Phase 2: Layer Sensitivity

**Status:** Future
**Blocked by:** Phase 1 validation
**Compute:** GPU cluster

## Purpose

Build (layer x language) sensitivity matrix to identify which layers are most sensitive for which languages.

## Method

```python
for layer_idx in range(model.num_layers):
    for lang in languages:
        # Quantize only this layer
        model_quant = quantize_layer(model, layer_idx, bits=4)

        # Measure degradation
        ppl_base = perplexity(model, lang_data[lang])
        ppl_quant = perplexity(model_quant, lang_data[lang])

        sensitivity[layer_idx, lang] = (ppl_quant - ppl_base) / ppl_base
```

## Expected Findings

1. FFN layers more sensitive for morphologically complex languages
2. Attention layers more sensitive for word-order flexible languages
3. Later layers more sensitive overall

## Application

Mixed-precision quantization:
- More bits for high-sensitivity (layer, language) pairs
- Standard bits for low-sensitivity pairs

## Files

| File | Purpose |
|------|---------|
| `layer_quant.py` | Single-layer quantization (TODO) |
| `sensitivity.py` | Matrix construction (TODO) |
| `matrix.json` | Sensitivity matrix (TODO) |
