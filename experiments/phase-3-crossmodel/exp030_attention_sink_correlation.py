#!/usr/bin/env python3
"""
EXP-030: Attention Sink and Outlier Correlation

Question: Do outlier weights correlate with attention sink positions?

Background (from literature):
- Attention sinks are tokens receiving disproportionate attention
- They emerge from softmax's sum-to-one constraint
- Super weights (0.01% of weights) are critical and located in attention
- Sinks concentrate in first tokens and later layers

Hypothesis: Layers with highest kurtosis will show strongest attention sink behavior.

Method:
1. Compute attention patterns on sample texts
2. Identify sink positions (tokens with >10% average attention)
3. Compute kurtosis per layer's attention projection
4. Correlate sink strength with kurtosis
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, kurtosis
import torch

# Sample texts for attention analysis
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning was the Word, and the Word was with God.",
    "To be or not to be, that is the question.",
    "All human beings are born free and equal in dignity.",
    "The only thing we have to fear is fear itself.",
]


def compute_attention_sink_strength(model, tokenizer, texts: list) -> dict:
    """
    Compute attention sink strength per layer.
    Sink strength = average attention to first token across all heads and positions.
    """
    sink_strengths = {}

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

        for layer_idx, attn in enumerate(attentions):
            # attn shape: (1, num_heads, seq_len, seq_len)
            attn_matrix = attn[0].numpy()  # (heads, seq, seq)

            # Sink strength = attention to position 0 averaged over all positions
            # Exclude self-attention at position 0
            sink_attn = attn_matrix[:, 1:, 0].mean()  # attention TO first token

            if layer_idx not in sink_strengths:
                sink_strengths[layer_idx] = []
            sink_strengths[layer_idx].append(sink_attn)

    # Average across texts
    return {layer: np.mean(values) for layer, values in sink_strengths.items()}


def compute_layer_kurtosis(model, component: str = "attn") -> dict:
    """
    Compute kurtosis for attention components per layer.
    """
    layer_kurtosis = {}

    for name, param in model.named_parameters():
        if component in name.lower() and 'weight' in name.lower():
            weights = param.detach().cpu().numpy().flatten()
            k = kurtosis(weights, fisher=True)

            # Extract layer number
            import re
            layer_match = re.search(r'layer[._]?(\d+)|layers?[._]?(\d+)|h\.(\d+)|blocks?\.(\d+)', name.lower())
            if layer_match:
                layer_num = int(next(g for g in layer_match.groups() if g is not None))
                if layer_num not in layer_kurtosis:
                    layer_kurtosis[layer_num] = []
                layer_kurtosis[layer_num].append((name, k))

    # Max kurtosis per layer
    return {layer: max(values, key=lambda x: x[1]) for layer, values in layer_kurtosis.items()}


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("EXP-030: Attention Sink and Outlier Correlation")
    print("=" * 60)

    results = {
        "experiment_id": "EXP-030",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Layers with highest kurtosis show strongest attention sinks",
        "model_results": {}
    }

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    models_to_test = [
        ("gpt2", "GPT-2"),
        ("bert-base-uncased", "BERT"),
    ]

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_label}")
        print("=" * 50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Fix for GPT-2 missing pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(model_name, output_attentions=True)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Compute attention sink strength per layer
        print("\n1. Computing attention sink strength per layer...")
        sink_strengths = compute_attention_sink_strength(model, tokenizer, SAMPLE_TEXTS)

        print(f"   Sink strength by layer:")
        for layer, strength in sorted(sink_strengths.items()):
            print(f"   Layer {layer:2d}: {strength:.4f}")

        # Compute kurtosis per layer
        print("\n2. Computing kurtosis per layer...")
        layer_kurtosis = compute_layer_kurtosis(model)

        print(f"   Max kurtosis by layer:")
        for layer in sorted(layer_kurtosis.keys()):
            name, k = layer_kurtosis[layer]
            short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
            print(f"   Layer {layer:2d}: κ={k:8.2f} ({short_name})")

        # Correlate sink strength with kurtosis
        print("\n3. Correlation analysis...")
        common_layers = set(sink_strengths.keys()) & set(layer_kurtosis.keys())

        if len(common_layers) >= 3:
            sinks = [sink_strengths[l] for l in sorted(common_layers)]
            kurtoses = [layer_kurtosis[l][1] for l in sorted(common_layers)]

            r, p = pearsonr(sinks, kurtoses)

            print(f"\n   Correlation (sink strength vs kurtosis):")
            print(f"   r = {r:.4f}, p = {p:.4f}")
            print(f"   n = {len(common_layers)} layers")

            # Interpretation
            if r > 0.3 and p < 0.1:
                interpretation = "POSITIVE correlation: High-kurtosis layers have stronger sinks"
            elif r < -0.3 and p < 0.1:
                interpretation = "NEGATIVE correlation: High-kurtosis layers have weaker sinks"
            else:
                interpretation = "No significant correlation"

            print(f"\n   {interpretation}")

            results["model_results"][model_label] = {
                "n_layers": len(common_layers),
                "correlation": r,
                "p_value": p,
                "sink_strengths": sink_strengths,
                "layer_kurtosis": {l: k for l, (_, k) in layer_kurtosis.items()},
                "interpretation": interpretation
            }

        # Find outlier layers
        print("\n4. Identifying outlier-heavy layers...")
        if layer_kurtosis:
            sorted_layers = sorted(layer_kurtosis.items(), key=lambda x: x[1][1], reverse=True)
            top_3 = sorted_layers[:3]
            print(f"   Top 3 highest-kurtosis layers:")
            for layer, (name, k) in top_3:
                sink = sink_strengths.get(layer, 0)
                print(f"   Layer {layer:2d}: κ={k:.1f}, sink={sink:.4f}")

        del model
        del tokenizer

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"exp030_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
