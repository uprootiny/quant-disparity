#!/usr/bin/env python3
"""
Compare BLOOM vs XGLM configurations.

Goal: Understand why BLOOM develops outlier weights but XGLM doesn't.
"""

import json
from pathlib import Path


def fetch_config(model_id):
    """Fetch model config from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_id, "config.json")
        return json.loads(Path(path).read_text())
    except Exception as e:
        print(f"Error fetching {model_id}: {e}")
        return None


def compare_configs():
    """Compare BLOOM and XGLM configurations."""

    print("="*70)
    print("BLOOM vs XGLM Configuration Comparison")
    print("="*70)

    bloom = fetch_config("bigscience/bloom-560m")
    xglm = fetch_config("facebook/xglm-564M")

    if not bloom or not xglm:
        return

    # Key architectural parameters
    params = [
        ("Model Type", "model_type", "model_type"),
        ("Hidden Size", "hidden_size", "d_model"),
        ("Num Layers", "n_layer", "num_layers"),
        ("Num Heads", "n_head", "attention_heads"),
        ("FFN Dim", "n_inner", "ffn_dim"),
        ("Vocab Size", "vocab_size", "vocab_size"),
        ("Max Position", "max_position_embeddings", "max_position_embeddings"),
        ("Activation", "activation_function", "activation_function"),
        ("LayerNorm Eps", "layer_norm_epsilon", "layernorm_epsilon"),
        ("Tie Embeddings", "tie_word_embeddings", "tie_word_embeddings"),
    ]

    print(f"\n{'Parameter':<25} {'BLOOM-560M':<25} {'XGLM-564M':<25}")
    print("-"*75)

    for name, bloom_key, xglm_key in params:
        bloom_val = bloom.get(bloom_key, "N/A")
        xglm_val = xglm.get(xglm_key, "N/A")

        # Check if different
        marker = " <<<" if bloom_val != xglm_val else ""
        print(f"{name:<25} {str(bloom_val):<25} {str(xglm_val):<25}{marker}")

    # Additional BLOOM-specific
    print("\n" + "-"*70)
    print("BLOOM-Specific Parameters:")
    print("-"*70)

    bloom_specific = [
        "apply_residual_connection_post_layernorm",
        "attention_softmax_in_fp32",
        "bias_dropout_fusion",
        "masked_softmax_fusion",
        "pretraining_tp",
        "slow_but_exact",
    ]

    for key in bloom_specific:
        val = bloom.get(key, "N/A")
        print(f"  {key}: {val}")

    # Additional XGLM-specific
    print("\n" + "-"*70)
    print("XGLM-Specific Parameters:")
    print("-"*70)

    xglm_specific = [
        "attention_dropout",
        "dropout",
        "activation_dropout",
        "decoder_start_token_id",
        "scale_embedding",
        "normalize_before",
    ]

    for key in xglm_specific:
        val = xglm.get(key, "N/A")
        print(f"  {key}: {val}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS: Potential Causes of Outlier Difference")
    print("="*70)

    differences = []

    # Check activation function
    bloom_act = bloom.get("activation_function", "gelu")
    xglm_act = xglm.get("activation_function", "gelu")
    if bloom_act != xglm_act:
        differences.append(f"Activation: BLOOM={bloom_act}, XGLM={xglm_act}")

    # Check LayerNorm epsilon
    bloom_ln = bloom.get("layer_norm_epsilon", 1e-5)
    xglm_ln = xglm.get("layernorm_epsilon", 1e-5)
    if bloom_ln != xglm_ln:
        differences.append(f"LayerNorm eps: BLOOM={bloom_ln}, XGLM={xglm_ln}")

    # Check normalize_before (pre-LN vs post-LN)
    xglm_prenorm = xglm.get("normalize_before", False)
    bloom_prenorm = bloom.get("apply_residual_connection_post_layernorm", False)
    if xglm_prenorm != bloom_prenorm:
        differences.append(f"Pre-norm: XGLM={xglm_prenorm}, BLOOM post-residual={bloom_prenorm}")

    # Check attention in FP32
    bloom_attn_fp32 = bloom.get("attention_softmax_in_fp32", False)
    differences.append(f"BLOOM attention_softmax_in_fp32: {bloom_attn_fp32}")

    # Check dropouts
    xglm_dropout = xglm.get("dropout", 0)
    differences.append(f"XGLM dropout: {xglm_dropout}")

    print("\nKey Differences Found:")
    for d in differences:
        print(f"  • {d}")

    # Hypotheses
    print("\n" + "-"*70)
    print("HYPOTHESES:")
    print("-"*70)

    print("""
1. Pre-LN vs Post-LN:
   - BLOOM uses post-LayerNorm (apply_residual_connection_post_layernorm)
   - XGLM uses pre-LayerNorm (normalize_before)
   - Pre-LN is known to be more stable during training
   - Post-LN can lead to larger gradient variance → outlier weights?

2. Attention Precision:
   - BLOOM computes attention softmax in FP32
   - This suggests BLOOM training was unstable in lower precision
   - May indicate propensity for outlier formation

3. Training Regime:
   - Not visible in config, but BLOOM was trained with tensor parallelism
   - Different gradient accumulation patterns may cause outliers

4. Data Distribution:
   - BLOOM: 46 languages, but heavily English-weighted
   - XGLM: 30 languages, more balanced?
   - Imbalanced training → specialized layers → outliers?
""")

    # Save
    output = {
        "bloom": bloom,
        "xglm": xglm,
        "differences": differences,
    }
    Path("config_comparison.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to config_comparison.json")


if __name__ == "__main__":
    compare_configs()
