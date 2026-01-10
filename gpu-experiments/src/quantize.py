"""
Quantization utilities for GPU experiments.
"""
import torch
from typing import Set, Optional


def get_num_layers(model) -> int:
    """Detect number of transformer layers."""
    if hasattr(model, 'config'):
        for attr in ['n_layer', 'num_hidden_layers', 'num_layers']:
            if hasattr(model.config, attr):
                return getattr(model.config, attr)

    # Fallback: count layers in state dict
    layer_indices = set()
    for name in model.state_dict().keys():
        for pattern in ['h.', 'layers.', 'model.layers.']:
            if pattern in name:
                try:
                    idx = int(name.split(pattern)[1].split('.')[0])
                    layer_indices.add(idx)
                except (IndexError, ValueError):
                    continue

    return max(layer_indices) + 1 if layer_indices else 12


def get_layer_pattern(model) -> str:
    """Detect layer naming pattern (h.X or model.layers.X)."""
    for name in model.state_dict().keys():
        if 'model.layers.' in name:
            return 'model.layers.'
        if 'h.' in name:
            return 'h.'
    return 'layers.'


def simulate_int4(
    model,
    exclude: Optional[Set[int]] = None,
    bits: int = 4,
    protect_biases: bool = True,
    protect_ln: bool = True,
):
    """
    Simulate INT4 quantization with optional layer protection.

    Args:
        model: HuggingFace model
        exclude: Set of layer indices to keep in FP16
        bits: Bit width (4 or 8)
        protect_biases: Keep biases in FP16
        protect_ln: Keep LayerNorm in FP16
    """
    if exclude is None:
        exclude = set()

    max_val = (2 ** (bits - 1)) - 1  # 7 for INT4, 127 for INT8
    layer_pattern = get_layer_pattern(model)

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip biases
            if protect_biases and 'bias' in name:
                continue

            # Skip LayerNorm
            if protect_ln and ('ln' in name.lower() or 'layernorm' in name.lower() or 'norm' in name.lower()):
                continue

            # Check if in protected layer
            is_protected = False
            for layer_idx in exclude:
                if f'{layer_pattern}{layer_idx}.' in name:
                    is_protected = True
                    break

            if is_protected:
                continue

            # Only quantize weights
            if 'weight' not in name:
                continue

            # Quantize
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / max_val
                quantized = torch.round(flat / scale).clamp(-max_val - 1, max_val) * scale
                param.data.copy_(quantized.view(param.shape))


def protect_layers(model, layers: Set[int]):
    """
    Mark layers for protection (no-op, just documentation).
    Actual protection happens in simulate_int4 via exclude parameter.
    """
    print(f"Layers {sorted(layers)} marked for FP16 protection")
    return layers


def restore_model(model, state_dict: dict):
    """Restore model from saved state dict."""
    model.load_state_dict(state_dict)


def save_state(model) -> dict:
    """Save model state for restoration."""
    return {k: v.clone() for k, v in model.state_dict().items()}
