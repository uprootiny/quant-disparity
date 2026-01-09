#!/usr/bin/env python3
"""
Exp-030: Anti-critical layer analysis
Goal: Test strategies that avoid anti-critical layers (L1 for GPT-2)
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-030: Anti-Critical Layer Analysis")
    print("=" * 60)

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
    print(f"Baseline: {', '.join([f'{l}={v:.1f}' for l, v in baseline.items()])}")

    # Strategies avoiding anti-critical Layer 1
    strategies = [
        ("none", []),
        ("layer0_only", ["h.0."]),
        ("layer0+2", ["h.0.", "h.2."]),  # Skip L1
        ("layer0+2+3", ["h.0.", "h.2.", "h.3."]),  # More good layers
        ("even_layers", ["h.0.", "h.2.", "h.4.", "h.6.", "h.8.", "h.10."]),
        ("odd_layers", ["h.1.", "h.3.", "h.5.", "h.7.", "h.9.", "h.11."]),  # Includes anti-critical L1
        ("first_half", ["h.0.", "h.1.", "h.2.", "h.3.", "h.4.", "h.5."]),
        ("first_half_no_L1", ["h.0.", "h.2.", "h.3.", "h.4.", "h.5."]),
    ]

    results = {}
    print(f"\n{'Strategy':<20} {'Protected':>10} {'%':>6} {'he':>8} {'ar':>8} {'Avg':>8}")
    print("-" * 65)

    for name, patterns in strategies:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
        avg_disp = sum(disp.values()) / len(disp)

        pct = protected / total * 100
        results[name] = {
            'protected': protected,
            'pct': pct,
            'disparity': disp,
            'avg_disp': avg_disp,
        }

        print(f"{name:<20} {protected:>10,} {pct:>5.1f}% {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg_disp:>7.1f}x")

    # Analysis
    print("\n" + "=" * 60)
    print("Anti-Critical Layer Impact Analysis")
    print("=" * 60)

    # Compare with and without L1
    first_half = results['first_half']['avg_disp']
    first_half_no_L1 = results['first_half_no_L1']['avg_disp']
    L1_impact = first_half - first_half_no_L1

    print(f"\nFirst half (with L1):    {first_half:.1f}x")
    print(f"First half (without L1): {first_half_no_L1:.1f}x")
    print(f"L1 impact: {L1_impact:+.1f}x")

    if L1_impact > 0:
        print("\n-> Removing L1 IMPROVES disparity (confirms anti-critical)")
    else:
        print("\n-> Removing L1 does not help")

    # Best strategy
    best = min(results.items(), key=lambda x: x[1]['avg_disp'])
    print(f"\nBest strategy: {best[0]}")
    print(f"  Avg disparity: {best[1]['avg_disp']:.1f}x")
    print(f"  Overhead: {best[1]['pct']:.1f}%")

    end = datetime.now()

    result = {
        "id": "Exp-030",
        "name": "Anti-Critical Layer Analysis",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "L1_impact": L1_impact,
        "best_strategy": best[0],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp030_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
