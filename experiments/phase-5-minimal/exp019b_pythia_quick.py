#!/usr/bin/env python3
"""
Exp-019b: Pythia-160M quick validation
Goal: Test critical layers only (0, 5, 11) to save memory
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-019b: Pythia Quick")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Pythia-160M...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    model.eval()

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
                if 'weight' not in name:
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
    print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test configs: none, layer0, layer5, layer11, mlp, attention
    configs = [
        ("none", []),
        ("layer_0", ["layers.0."]),
        ("layer_5", ["layers.5."]),
        ("layer_11", ["layers.11."]),
        ("mlp", ["mlp"]),
        ("attention", ["attention"]),
    ]

    results = {}
    print(f"\n{'Config':<12} {'Disparity':>10}")
    print("-" * 25)

    for name, patterns in configs:
        restore()
        quant_except(patterns)
        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
        results[name] = disp
        print(f"{name:<12} {disp:>10.1f}x")
        gc.collect()

    # Determine pattern
    print("\n" + "=" * 50)
    print("Pattern Analysis")
    print("=" * 50)

    mlp_vs_attn = "MLP" if results['mlp'] < results['attention'] else "Attention"
    early_vs_late = "Early" if results['layer_0'] < results['layer_11'] else "Late"

    print(f"\nComponent winner: {mlp_vs_attn}")
    print(f"Layer pattern: {early_vs_late} layers critical")

    print(f"\nComparison:")
    print(f"  GPT-2:   MLP wins, Early critical")
    print(f"  OPT:     Attention wins, Late critical")
    print(f"  Pythia:  {mlp_vs_attn} wins, {early_vs_late} critical")

    end = datetime.now()

    result = {
        "id": "Exp-019b",
        "name": "Pythia Quick Validation",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "model": "EleutherAI/pythia-160m",
        "results": results,
        "component_winner": mlp_vs_attn,
        "layer_pattern": early_vs_late,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp019b_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")

    del model
    gc.collect()
    return result


if __name__ == "__main__":
    main()
