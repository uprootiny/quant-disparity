#!/usr/bin/env python3
"""
EXP-023: Training Config Comparison

Tests H2: Do precision/regularization settings explain kurtosis differences?

Compare: OPT vs BLOOM vs GPT-2 vs Pythia training configs
"""

import json
from pathlib import Path


def fetch_config(model_name: str, repo_id: str) -> dict:
    """Fetch model config from HuggingFace."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, "config.json")
        config = json.loads(Path(path).read_text())
        return config
    except Exception as e:
        print(f"Error fetching {model_name}: {e}")
        return {}


def extract_training_hints(config: dict, model_name: str) -> dict:
    """Extract training-relevant config values."""
    hints = {
        "model": model_name,
        "hidden_size": config.get("hidden_size") or config.get("n_embd"),
        "num_layers": config.get("num_hidden_layers") or config.get("n_layer"),
        "num_heads": config.get("num_attention_heads") or config.get("n_head"),
    }

    # Precision-related
    hints["attention_softmax_in_fp32"] = config.get("attention_softmax_in_fp32", "not_specified")

    # Regularization
    hints["hidden_dropout_prob"] = config.get("hidden_dropout_prob",
                                    config.get("dropout", "not_specified"))
    hints["attention_dropout"] = config.get("attention_dropout",
                                  config.get("attn_pdrop", "not_specified"))
    hints["resid_dropout"] = config.get("resid_pdrop", "not_specified")

    # Initialization
    hints["initializer_range"] = config.get("initializer_range", "not_specified")

    # Scaling
    hints["scale_embedding"] = config.get("scale_embedding", "not_specified")
    hints["tie_word_embeddings"] = config.get("tie_word_embeddings", "not_specified")

    # Architecture specifics
    hints["activation_function"] = config.get("activation_function",
                                    config.get("hidden_act", "not_specified"))
    hints["layer_norm_epsilon"] = config.get("layer_norm_epsilon",
                                   config.get("layer_norm_eps", "not_specified"))

    # OPT-specific
    hints["do_layer_norm_before"] = config.get("do_layer_norm_before", "not_specified")
    hints["word_embed_proj_dim"] = config.get("word_embed_proj_dim", "not_specified")

    return hints


def main():
    print("="*60)
    print("EXP-023: Training Config Comparison")
    print("="*60)

    models = [
        ("OPT-125M", "facebook/opt-125m", 562),
        ("BLOOM-560M", "bigscience/bloom-560m", 504),
        ("GPT-2-small", "openai-community/gpt2", 201),
        ("Pythia-410M", "EleutherAI/pythia-410m", 14),
        ("XGLM-564M", "facebook/xglm-564M", 2),
    ]

    all_hints = []

    for name, repo, max_kurt in models:
        print(f"\nFetching {name}...")
        config = fetch_config(name, repo)
        if config:
            hints = extract_training_hints(config, name)
            hints["max_kurtosis"] = max_kurt
            all_hints.append(hints)
            print(f"  Found {len(config)} config keys")

    # Analysis
    print("\n" + "="*60)
    print("CONFIG COMPARISON")
    print("="*60)

    # Focus on precision and regularization
    print("\n### Precision Settings ###\n")
    print(f"{'Model':<15} {'κ':<8} {'attn_fp32':<12} {'do_ln_before'}")
    print("-"*50)
    for h in sorted(all_hints, key=lambda x: -x["max_kurtosis"]):
        fp32 = str(h.get("attention_softmax_in_fp32", "—"))[:10]
        ln_before = str(h.get("do_layer_norm_before", "—"))[:10]
        print(f"{h['model']:<15} {h['max_kurtosis']:<8} {fp32:<12} {ln_before}")

    print("\n### Regularization Settings ###\n")
    print(f"{'Model':<15} {'κ':<8} {'hidden_drop':<12} {'attn_drop':<12} {'resid_drop'}")
    print("-"*70)
    for h in sorted(all_hints, key=lambda x: -x["max_kurtosis"]):
        hd = str(h.get("hidden_dropout_prob", "—"))[:10]
        ad = str(h.get("attention_dropout", "—"))[:10]
        rd = str(h.get("resid_dropout", "—"))[:10]
        print(f"{h['model']:<15} {h['max_kurtosis']:<8} {hd:<12} {ad:<12} {rd}")

    print("\n### Initialization ###\n")
    print(f"{'Model':<15} {'κ':<8} {'init_range':<12} {'scale_embed'}")
    print("-"*50)
    for h in sorted(all_hints, key=lambda x: -x["max_kurtosis"]):
        init = str(h.get("initializer_range", "—"))[:10]
        scale = str(h.get("scale_embedding", "—"))[:10]
        print(f"{h['model']:<15} {h['max_kurtosis']:<8} {init:<12} {scale}")

    # Pattern analysis
    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)

    heavy = [h for h in all_hints if h["max_kurtosis"] > 100]
    light = [h for h in all_hints if h["max_kurtosis"] < 20]

    print("\n### Heavy-outlier models (κ > 100) ###")
    for h in heavy:
        print(f"  {h['model']}: dropout={h.get('hidden_dropout_prob', '?')}, "
              f"attn_fp32={h.get('attention_softmax_in_fp32', '?')}")

    print("\n### Low-outlier models (κ < 20) ###")
    for h in light:
        print(f"  {h['model']}: dropout={h.get('hidden_dropout_prob', '?')}, "
              f"attn_fp32={h.get('attention_softmax_in_fp32', '?')}")

    # Conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)

    # Check if dropout correlates with low kurtosis
    has_dropout = sum(1 for h in light
                      if h.get("hidden_dropout_prob") not in ["not_specified", 0, 0.0, "0"])
    no_dropout = sum(1 for h in heavy
                     if h.get("hidden_dropout_prob") in ["not_specified", 0, 0.0, "0"])

    print(f"\nDropout hypothesis:")
    print(f"  Low-κ models with dropout: {has_dropout}/{len(light)}")
    print(f"  High-κ models without dropout: {no_dropout}/{len(heavy)}")

    if has_dropout >= len(light) // 2 and no_dropout >= len(heavy) // 2:
        verdict = "H2 SUPPORTED: dropout correlates with lower kurtosis"
    else:
        verdict = "H2 INCONCLUSIVE: no clear dropout pattern"

    print(f"\nVerdict: {verdict}")

    # Save results
    results = {
        "experiment": "EXP-023",
        "configs": all_hints,
        "verdict": verdict,
    }
    Path("exp023_results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nSaved to exp023_results.json")


if __name__ == "__main__":
    main()
