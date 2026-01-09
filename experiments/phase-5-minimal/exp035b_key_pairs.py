#!/usr/bin/env python3
"""
Exp-035b: Key layer pairs (memory-safe)
Goal: Test most promising pairs only
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.', 'ar': 'الثعلب يقفز.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-035b: Key Layer Pairs")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    total = sum(p.numel() for p in model.parameters())
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

    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Key pairs only
    pairs = [
        ("L0", ["h.0."]),
        ("L11", ["h.11."]),
        ("L0+L11", ["h.0.", "h.11."]),
        ("L0+L10", ["h.0.", "h.10."]),
        ("L0+L2", ["h.0.", "h.2."]),
        ("L2+L11", ["h.2.", "h.11."]),
        ("L0+L2+L11", ["h.0.", "h.2.", "h.11."]),
    ]

    results = {}
    print(f"\n{'Pair':<12} {'he':>8} {'ar':>8} {'Avg':>8}")
    print("-" * 40)

    for name, patterns in pairs:
        restore()
        quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
        avg = sum(disp.values()) / len(disp)
        results[name] = avg

        print(f"{name:<12} {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x")

    print(f"\nBest: {min(results, key=results.get)} ({min(results.values()):.1f}x)")

    end = datetime.now()
    result = {
        "id": "Exp-035b",
        "name": "Key Layer Pairs",
        "results": results,
        "best": min(results, key=results.get),
        "status": "SUCCESS"
    }

    with open(Path(__file__).parent / "exp035b_result.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
