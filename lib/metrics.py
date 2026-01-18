"""
Evaluation metrics for quantization disparity research.

Canonical implementations of perplexity, fertility, and layer statistics.
"""
from typing import Dict, Optional, List
import numpy as np

# Lazy imports to avoid torch dependency when not needed
_torch = None
_scipy_stats = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_scipy_stats():
    global _scipy_stats
    if _scipy_stats is None:
        from scipy import stats
        _scipy_stats = stats
    return _scipy_stats


# =============================================================================
# PERPLEXITY
# =============================================================================

def perplexity(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
    stride: int = 256
) -> float:
    """
    Compute perplexity for a causal language model.

    PPL = exp(mean(NLL))

    where NLL is the negative log-likelihood per token.

    Args:
        model: HuggingFace causal LM (e.g., GPT2, OPT, BLOOM)
        tokenizer: Corresponding tokenizer
        text: Input text
        max_length: Maximum sequence length
        stride: Stride for sliding window (for texts > max_length)

    Returns:
        Perplexity (float). Lower is better.

    Note: For masked LMs (BERT, XLM-R), use pseudo_perplexity() instead.
    """
    torch = _get_torch()

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss).item())


def perplexity_sliding_window(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
    stride: int = 256
) -> float:
    """
    Compute perplexity with sliding window for long texts.

    More accurate than truncation for texts longer than max_length.
    """
    torch = _get_torch()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    nlls = []
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        chunk = input_ids[:, begin:end].to(model.device)

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * (end - begin))

        if end == input_ids.size(1):
            break

    return float(np.exp(sum(nlls) / input_ids.size(1)))


def pseudo_perplexity_mlm(
    model,
    tokenizer,
    text: str,
    mask_ratio: float = 0.15,
    n_samples: int = 10
) -> float:
    """
    Pseudo-perplexity for masked language models (BERT, XLM-R).

    Approximates perplexity by masking tokens and measuring prediction loss.

    Args:
        model: HuggingFace masked LM
        tokenizer: Corresponding tokenizer
        text: Input text
        mask_ratio: Fraction of tokens to mask
        n_samples: Number of random masking samples to average

    Returns:
        Pseudo-perplexity (float). Lower is better.
    """
    torch = _get_torch()

    encodings = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = encodings.input_ids.to(model.device)
    n_tokens = input_ids.size(1)
    n_mask = max(1, int(n_tokens * mask_ratio))

    losses = []
    for _ in range(n_samples):
        masked_ids = input_ids.clone()
        mask_positions = torch.randperm(n_tokens)[:n_mask]
        masked_ids[0, mask_positions] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_ids, labels=input_ids)
            losses.append(outputs.loss.item())

    return float(np.exp(np.mean(losses)))


# =============================================================================
# TOKEN FERTILITY
# =============================================================================

def fertility(tokenizer, texts: List[str]) -> float:
    """
    Compute token fertility: tokens / words.

    Higher fertility indicates more subword fragmentation,
    which correlates with worse quantization disparity.

    Args:
        tokenizer: HuggingFace tokenizer
        texts: List of text samples

    Returns:
        Average fertility across texts.
    """
    total_tokens = 0
    total_words = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        total_tokens += len(tokens)
        total_words += len(words)

    if total_words == 0:
        return 0.0

    return total_tokens / total_words


def fertility_per_language(
    tokenizer,
    texts_by_lang: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Compute fertility for each language.

    Args:
        tokenizer: HuggingFace tokenizer
        texts_by_lang: Dict mapping language code to list of texts

    Returns:
        Dict mapping language code to fertility.
    """
    return {lang: fertility(tokenizer, texts) for lang, texts in texts_by_lang.items()}


# =============================================================================
# LAYER STATISTICS
# =============================================================================

def layer_kurtosis(model, component: str = "mlp") -> Dict[int, float]:
    """
    Extract excess kurtosis per layer.

    Args:
        model: HuggingFace transformer model
        component: "mlp", "attn", or "all"

    Returns:
        Dict mapping layer index to kurtosis.
    """
    torch = _get_torch()
    stats = _get_scipy_stats()

    kurtosis_dict = {}

    for name, param in model.named_parameters():
        # Match layer pattern
        if "layers." in name or "h." in name or "block." in name:
            # Extract layer number
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p in ("layers", "h", "block") and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            else:
                continue

            # Filter by component
            if component == "mlp" and "mlp" not in name.lower() and "fc" not in name.lower():
                continue
            if component == "attn" and "attn" not in name.lower() and "attention" not in name.lower():
                continue

            # Compute kurtosis
            weights = param.detach().cpu().numpy().flatten()
            k = float(stats.kurtosis(weights, fisher=True))

            if layer_idx in kurtosis_dict:
                # Average if multiple weight matrices per layer
                kurtosis_dict[layer_idx] = (kurtosis_dict[layer_idx] + k) / 2
            else:
                kurtosis_dict[layer_idx] = k

    return dict(sorted(kurtosis_dict.items()))


def layer_outlier_fraction(
    model,
    threshold_sigma: float = 6.0
) -> Dict[int, float]:
    """
    Compute fraction of outlier weights per layer.

    An outlier is defined as |w| > threshold_sigma * std(W).

    Args:
        model: HuggingFace transformer model
        threshold_sigma: Number of standard deviations for outlier threshold

    Returns:
        Dict mapping layer index to outlier fraction.
    """
    torch = _get_torch()

    outlier_fracs = {}

    for name, param in model.named_parameters():
        if "layers." in name or "h." in name:
            # Extract layer number (same as above)
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p in ("layers", "h") and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            else:
                continue

            weights = param.detach().cpu()
            std = weights.std().item()
            threshold = threshold_sigma * std
            outlier_mask = weights.abs() > threshold
            frac = outlier_mask.float().mean().item()

            if layer_idx in outlier_fracs:
                outlier_fracs[layer_idx] = (outlier_fracs[layer_idx] + frac) / 2
            else:
                outlier_fracs[layer_idx] = frac

    return dict(sorted(outlier_fracs.items()))


# =============================================================================
# DEGRADATION METRICS
# =============================================================================

def degradation_ratio(baseline_ppl: float, quantized_ppl: float) -> float:
    """
    Compute relative degradation from quantization.

    D = (PPL_quant - PPL_base) / PPL_base

    Returns: Degradation ratio (0 = no degradation, 1 = 100% increase)
    """
    if baseline_ppl <= 0:
        return float('inf')
    return (quantized_ppl - baseline_ppl) / baseline_ppl


def disparity(degradations: Dict[str, float]) -> float:
    """
    Compute disparity across languages.

    Disparity = max(D) - min(D)

    Args:
        degradations: Dict mapping language to degradation ratio

    Returns:
        Disparity value.
    """
    if not degradations:
        return 0.0
    values = list(degradations.values())
    return max(values) - min(values)
