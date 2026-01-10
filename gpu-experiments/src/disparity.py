"""
Disparity measurement utilities.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .quantize import simulate_int4, save_state, restore_model, get_num_layers


# Default test texts
DEFAULT_TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog near the river bank.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund am Flussufer.',
    'fr': 'Le renard brun rapide saute par-dessus le chien paresseux près de la rivière.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד גדת הנהר.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من ضفة النهر.',
    'zh': '敏捷的棕色狐狸跳过河边的懒狗。',
    'ja': '素早い茶色の狐が川辺で怠惰な犬を飛び越えた。',
    'ko': '빠른 갈색 여우가 강가에서 게으른 개를 뛰어넘었다.',
    'ru': 'Быстрая коричневая лиса прыгает через ленивую собаку у реки.',
    'es': 'El rápido zorro marrón salta sobre el perro perezoso cerca del río.',
}


def perplexity(model, tokenizer, text: str) -> float:
    """Calculate perplexity for a text sample."""
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()


def measure_disparity(
    model,
    tokenizer,
    texts: Dict[str, str] = None,
    baseline_ppl: Dict[str, float] = None,
    reference_lang: str = 'en',
) -> Dict:
    """
    Measure quantization disparity across languages.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        texts: Dict of lang -> text
        baseline_ppl: Pre-computed FP16 perplexities
        reference_lang: Reference language for disparity ratio

    Returns:
        Dict with perplexities, degradation, and disparity ratios
    """
    if texts is None:
        texts = DEFAULT_TEXTS

    # Current perplexities
    current_ppl = {lang: perplexity(model, tokenizer, text)
                   for lang, text in texts.items()}

    if baseline_ppl is None:
        return {'ppl': current_ppl}

    # Degradation (% increase)
    degradation = {}
    for lang in texts:
        if baseline_ppl[lang] > 0:
            degradation[lang] = (current_ppl[lang] - baseline_ppl[lang]) / baseline_ppl[lang] * 100
        else:
            degradation[lang] = float('inf')

    # Disparity ratio (relative to reference)
    ref_deg = degradation.get(reference_lang, 1)
    if ref_deg <= 0:
        ref_deg = 1

    disparity = {lang: deg / ref_deg for lang, deg in degradation.items()}

    # Summary stats
    non_ref = [v for k, v in disparity.items() if k != reference_lang and v != float('inf')]
    avg_disparity = np.mean(non_ref) if non_ref else float('inf')
    max_disparity = max(non_ref) if non_ref else float('inf')

    return {
        'ppl': current_ppl,
        'degradation': degradation,
        'disparity': disparity,
        'avg_disparity': avg_disparity,
        'max_disparity': max_disparity,
    }


def quick_sweep(
    model,
    tokenizer,
    texts: Dict[str, str] = None,
    top_n: int = 3,
    verbose: bool = True,
) -> List[int]:
    """
    Quick layer sweep to find critical layers.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        texts: Test texts (uses defaults if None)
        top_n: Number of layers to recommend
        verbose: Print progress

    Returns:
        List of layer indices to protect
    """
    if texts is None:
        texts = {k: v for k, v in DEFAULT_TEXTS.items() if k in ['en', 'he', 'ar', 'zh']}

    state = save_state(model)
    num_layers = get_num_layers(model)

    # Baseline
    baseline = {lang: perplexity(model, tokenizer, text) for lang, text in texts.items()}
    if verbose:
        print(f"Baseline PPL: en={baseline.get('en', 0):.1f}")

    results = []
    iterator = tqdm(range(num_layers), desc="Layer sweep") if verbose else range(num_layers)

    for layer_idx in iterator:
        restore_model(model, state)
        simulate_int4(model, exclude={layer_idx})

        metrics = measure_disparity(model, tokenizer, texts, baseline)
        results.append((layer_idx, metrics['avg_disparity']))

        if verbose and not isinstance(iterator, tqdm):
            print(f"  L{layer_idx}: {metrics['avg_disparity']:.2f}x")

    restore_model(model, state)

    # Sort by disparity (lower = more critical)
    results.sort(key=lambda x: x[1])

    if verbose:
        print("\nTop critical layers:")
        for layer, disp in results[:top_n]:
            print(f"  L{layer}: {disp:.2f}x disparity when protected")

    return [layer for layer, _ in results[:top_n]]


def full_sweep(
    model,
    tokenizer,
    texts: Dict[str, str] = None,
    verbose: bool = True,
) -> List[Tuple[int, float, Dict]]:
    """
    Full layer sweep with detailed metrics.

    Returns:
        List of (layer_idx, avg_disparity, full_metrics)
    """
    if texts is None:
        texts = DEFAULT_TEXTS

    state = save_state(model)
    num_layers = get_num_layers(model)

    baseline = {lang: perplexity(model, tokenizer, text) for lang, text in texts.items()}

    results = []
    for layer_idx in tqdm(range(num_layers), desc="Full sweep", disable=not verbose):
        restore_model(model, state)
        simulate_int4(model, exclude={layer_idx})

        metrics = measure_disparity(model, tokenizer, texts, baseline)
        results.append((layer_idx, metrics['avg_disparity'], metrics))

    restore_model(model, state)
    results.sort(key=lambda x: x[1])

    return results
