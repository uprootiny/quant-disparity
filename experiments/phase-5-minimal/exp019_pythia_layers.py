#!/usr/bin/env python3
"""
Exp-019: Pythia-160M per-layer validation
Goal: Test if per-layer patterns hold on third model architecture
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-019: Pythia-160M Per-Layer")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Pythia-160M...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {total:,} params")

    # Analyze structure
    components = {'embed': 0, 'attention': 0, 'mlp': 0, 'other': 0}
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        n = param.numel()
        if 'embed' in name:
            components['embed'] += n
        elif 'attention' in name or 'attn' in name:
            components['attention'] += n
        elif 'mlp' in name or 'dense_h_to_4h' in name or 'dense_4h_to_h' in name:
            components['mlp'] += n
        else:
            components['other'] += n

    print("\nComponent breakdown:")
    for c, n in components.items():
        print(f"  {c}: {n:,} ({n/total*100:.1f}%)")

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_except(patterns):
        protected = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                if any(p in name for p in patterns):
                    protected += param.numel()
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))
        return protected

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"\nBaseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Component-level test first
    print("\n--- Component-Level Analysis ---")
    component_configs = [
        ("none", []),
        ("embeddings", ["embed"]),
        ("attention", ["attention", "attn"]),
        ("mlp", ["mlp", "dense_h_to_4h", "dense_4h_to_h"]),
        ("layer0", ["layers.0."]),
    ]

    component_results = {}
    print(f"\n{'Config':<12} {'Disparity':>10}")
    print("-" * 25)

    for name, patterns in component_configs:
        restore()
        quant_except(patterns)
        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
        component_results[name] = disp
        print(f"{name:<12} {disp:>10.1f}x")

    # Per-layer MLP analysis
    print("\n--- Per-Layer MLP Analysis ---")
    n_layers = 12
    layer_results = {}

    disp_none = component_results['none']

    print(f"\n{'Layer':<8} {'Disparity':>12} {'Delta':>10}")
    print("-" * 32)

    for layer_idx in range(n_layers):
        restore()
        # Pythia uses gpt_neox architecture
        patterns = [f'layers.{layer_idx}.mlp']
        quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        delta = disp_none - disp
        layer_results[f'layer_{layer_idx}'] = {
            'disparity': disp,
            'delta': delta,
        }
        print(f"Layer {layer_idx:<3} {disp:>12.1f}x {delta:>+10.1f}")

    # Find critical layers
    sorted_layers = sorted(layer_results.items(), key=lambda x: x[1]['disparity'])
    best = sorted_layers[0]
    worst = sorted_layers[-1]

    print(f"\nMost critical: {best[0]} ({best[1]['disparity']:.1f}x)")
    print(f"Anti-critical: {worst[0]} ({worst[1]['disparity']:.1f}x)")

    # Cross-model comparison
    print("\n" + "=" * 60)
    print("Cross-Model Critical Layer Comparison")
    print("=" * 60)
    print(f"\n{'Model':<12} {'Best Layer':<12} {'Worst Layer':<12} {'Pattern'}")
    print("-" * 50)
    print(f"{'GPT-2':<12} {'Layer 0':<12} {'Layer 1':<12} Early critical")
    print(f"{'OPT-125M':<12} {'Layer 11':<12} {'Layer 7':<12} Late critical")
    print(f"{'Pythia':<12} {best[0]:<12} {worst[0]:<12} ???")

    end = datetime.now()

    result = {
        "id": "Exp-019",
        "name": "Pythia-160M Per-Layer",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "model": "EleutherAI/pythia-160m",
        "component_results": component_results,
        "layer_results": layer_results,
        "best_layer": best[0],
        "worst_layer": worst[0],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp019_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
