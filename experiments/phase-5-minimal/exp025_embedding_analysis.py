#!/usr/bin/env python3
"""
Exp-025: Embedding layer analysis
Goal: Understand why protecting embeddings alone INCREASES disparity
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-025: Embedding Analysis")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Analyze embedding layer
    print("Embedding layer analysis:")
    for name, param in model.named_parameters():
        if 'wte' in name or 'wpe' in name:
            print(f"  {name}: {param.shape}, {param.numel():,} params")
            print(f"    mean: {param.mean().item():.6f}")
            print(f"    std:  {param.std().item():.6f}")
            print(f"    max:  {param.abs().max().item():.6f}")

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_except(patterns):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name and 'wte' not in name and 'wpe' not in name:
                    continue
                if any(p in name for p in patterns):
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"\nBaseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test different embedding strategies
    strategies = [
        ("none", []),
        ("wte_only", ["wte"]),  # Token embeddings only
        ("wpe_only", ["wpe"]),  # Position embeddings only
        ("both_embed", ["wte", "wpe"]),  # Both embeddings
        ("layer0", ["h.0."]),  # Layer 0 for comparison
        ("layer0_no_embed", ["h.0.attn", "h.0.mlp", "h.0.ln"]),  # Layer 0 WITHOUT embeddings
    ]

    results = {}
    print(f"\n{'Strategy':<18} {'Protected':>12} {'Disparity':>10}")
    print("-" * 45)

    for strat_name, patterns in strategies:
        restore()
        quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        # Count protected params
        protected = 0
        for name, param in model.named_parameters():
            if any(p in name for p in patterns):
                protected += param.numel()

        results[strat_name] = {
            'protected': protected,
            'disparity': disp,
            'en_deg': deg['en'],
            'he_deg': deg['he'],
        }
        print(f"{strat_name:<18} {protected:>12,} {disp:>10.1f}x")

    # Analysis
    print("\n" + "=" * 60)
    print("Analysis: Why do embeddings hurt?")
    print("=" * 60)

    none_disp = results['none']['disparity']
    embed_disp = results['both_embed']['disparity']
    layer0_disp = results['layer0']['disparity']

    print(f"\nNo protection:       {none_disp:.1f}x")
    print(f"Both embeddings:     {embed_disp:.1f}x")
    print(f"Layer 0:             {layer0_disp:.1f}x")

    if embed_disp > none_disp:
        print("\nHypothesis CONFIRMED: Protecting embeddings makes disparity WORSE")
        print("\nPossible explanations:")
        print("  1. Representation mismatch: FP16 embeddings feed into INT4 layers")
        print("  2. Embeddings are already language-agnostic")
        print("  3. Quantization error in later layers dominates")

    # Check position vs token embeddings
    wte_disp = results['wte_only']['disparity']
    wpe_disp = results['wpe_only']['disparity']
    print(f"\nToken embeddings only: {wte_disp:.1f}x")
    print(f"Position embeddings only: {wpe_disp:.1f}x")

    if wte_disp > wpe_disp:
        print("-> Token embeddings are the problem, not positions")
    else:
        print("-> Position embeddings are the problem, not tokens")

    end = datetime.now()

    result = {
        "id": "Exp-025",
        "name": "Embedding Analysis",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp025_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
