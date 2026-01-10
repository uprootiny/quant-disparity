"""
Layer sweep methods to identify critical layers.
"""

import torch
from typing import Dict, List, Tuple
from .quantize import simulate_int4, perplexity, measure_disparity


# Default test texts for sweep
DEFAULT_TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog near the river bank.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד גדת הנהר.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من ضفة النهر.',
    'zh': '敏捷的棕色狐狸跳过河边的懒狗。',
}


def get_num_layers(model) -> int:
    """Detect number of transformer layers in model."""
    # Try common attribute names
    if hasattr(model, 'config'):
        for attr in ['n_layer', 'num_hidden_layers', 'num_layers']:
            if hasattr(model.config, attr):
                return getattr(model.config, attr)

    # Fallback: count h.* or layers.* parameters
    layer_indices = set()
    for name in model.state_dict().keys():
        if 'h.' in name:
            idx = int(name.split('h.')[1].split('.')[0])
            layer_indices.add(idx)
        elif 'layers.' in name:
            idx = int(name.split('layers.')[1].split('.')[0])
            layer_indices.add(idx)

    return max(layer_indices) + 1 if layer_indices else 12


def layer_sweep(
    model,
    tokenizer,
    texts: Dict[str, str] = None,
    verbose: bool = True
) -> List[Tuple[int, float]]:
    """
    Sweep all layers to find critical ones for multilingual fairness.

    Protects each layer individually and measures disparity.
    Lower disparity = more critical layer.

    Args:
        model: HuggingFace model
        tokenizer: Model tokenizer
        texts: Dict of language code -> text (uses defaults if None)
        verbose: Print progress

    Returns:
        List of (layer_idx, avg_disparity) sorted by criticality (lowest first)
    """
    if texts is None:
        texts = DEFAULT_TEXTS

    # Store original state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Get baseline perplexities
    baseline = {lang: perplexity(model, tokenizer, text)
                for lang, text in texts.items()}

    if verbose:
        print(f"Baseline PPL: en={baseline['en']:.1f}")

    num_layers = get_num_layers(model)
    results = []

    for layer_idx in range(num_layers):
        # Restore and quantize with this layer protected
        model.load_state_dict(original_state)
        simulate_int4(model, protect_layers={layer_idx})

        # Measure disparity
        disparity = measure_disparity(model, tokenizer, texts, baseline)

        # Average non-English disparity
        non_en = [v for k, v in disparity.items() if k != 'en' and v != float('inf')]
        avg_disp = sum(non_en) / len(non_en) if non_en else float('inf')

        results.append((layer_idx, avg_disp))

        if verbose:
            print(f"  L{layer_idx}: {avg_disp:.2f}x")

    # Restore original
    model.load_state_dict(original_state)

    # Sort by disparity (lower = more critical)
    results.sort(key=lambda x: x[1])

    return results


def quick_sweep(
    model,
    tokenizer,
    texts: Dict[str, str] = None,
    top_n: int = 3
) -> List[int]:
    """
    Quick layer sweep returning recommended layers to protect.

    Args:
        model: HuggingFace model
        tokenizer: Model tokenizer
        texts: Dict of language code -> text
        top_n: Number of layers to recommend

    Returns:
        List of layer indices to protect
    """
    results = layer_sweep(model, tokenizer, texts, verbose=False)
    return [layer_idx for layer_idx, _ in results[:top_n]]
