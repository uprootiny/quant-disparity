"""
Quantization simulation and disparity measurement.
"""

import torch
from typing import Dict, List, Set


def simulate_int4(model, protect_layers: Set[int] = None, protect_biases: bool = True):
    """
    Simulate INT4 quantization with optional layer protection.

    Args:
        model: HuggingFace model
        protect_layers: Set of layer indices to keep in FP16
        protect_biases: Whether to keep biases in FP16 (recommended)

    Returns:
        None (modifies model in-place)
    """
    if protect_layers is None:
        protect_layers = set()

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip biases if protected
            if protect_biases and 'bias' in name:
                continue

            # Skip final layernorm
            if 'ln_f' in name:
                continue

            # Check if this layer should be protected
            is_protected = False
            for layer_idx in protect_layers:
                if f'h.{layer_idx}.' in name or f'layers.{layer_idx}.' in name:
                    is_protected = True
                    break

            if is_protected:
                continue

            # Only quantize weights
            if 'weight' not in name:
                continue

            # INT4 simulation: scale to [-8, 7], round, scale back
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7
                quantized = torch.round(flat / scale).clamp(-8, 7) * scale
                param.data.copy_(quantized.view(param.shape))


def perplexity(model, tokenizer, text: str) -> float:
    """Calculate perplexity for a text sample."""
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()


def measure_disparity(
    model,
    tokenizer,
    texts: Dict[str, str],
    baseline_ppl: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Measure quantization disparity across languages.

    Args:
        model: Quantized model
        tokenizer: Model tokenizer
        texts: Dict mapping language codes to text samples
        baseline_ppl: Pre-computed baseline perplexities (optional)

    Returns:
        Dict with disparity ratios (1.0 = same as English degradation)
    """
    # Get current perplexities
    current_ppl = {lang: perplexity(model, tokenizer, text)
                   for lang, text in texts.items()}

    if baseline_ppl is None:
        return current_ppl

    # Calculate degradation percentages
    degradation = {}
    for lang in texts:
        if baseline_ppl[lang] > 0:
            degradation[lang] = (current_ppl[lang] - baseline_ppl[lang]) / baseline_ppl[lang] * 100
        else:
            degradation[lang] = float('inf')

    # Calculate disparity relative to English
    en_deg = degradation.get('en', 1)
    if en_deg <= 0:
        en_deg = 1

    disparity = {lang: deg / en_deg for lang, deg in degradation.items()}

    return disparity
