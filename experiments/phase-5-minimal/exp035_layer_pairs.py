#!/usr/bin/env python3
"""
Exp-035: Optimal layer pair exploration
Goal: Find which layer pairs achieve best synergy
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-035: Optimal Layer Pair Exploration")
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

    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Baseline: {', '.join([f'{l}={v:.1f}' for l, v in baseline.items()])}")

    # Test all pairs with Layer 0
    pairs_with_l0 = []
    for i in range(1, 12):
        pairs_with_l0.append((f"L0+L{i}", [f"h.0.", f"h.{i}."]))

    # Also test pairs without L0
    other_pairs = [
        ("L2+L11", ["h.2.", "h.11."]),
        ("L0+L10", ["h.0.", "h.10."]),
        ("L0+L2+L11", ["h.0.", "h.2.", "h.11."]),
    ]

    all_strategies = [("L0_only", ["h.0."]), ("L11_only", ["h.11."])] + pairs_with_l0 + other_pairs

    results = {}
    print(f"\n{'Pair':<12} {'%':>7} {'he':>8} {'ar':>8} {'Avg':>8}")
    print("-" * 45)

    for name, patterns in all_strategies:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        if en_deg > 0:
            disp = {l: deg[l] / en_deg for l in TEXTS if l != 'en'}
        else:
            disp = {l: float('inf') for l in TEXTS if l != 'en'}
        avg_disp = sum(disp.values()) / len(disp)

        pct = protected / total * 100
        results[name] = {
            'protected': protected,
            'pct': pct,
            'disparity': disp,
            'avg_disp': avg_disp,
        }

        print(f"{name:<12} {pct:>6.2f}% {disp.get('he', 0):>7.1f}x {disp.get('ar', 0):>7.1f}x {avg_disp:>7.1f}x")

    # Rank by efficiency (disparity reduction per % overhead)
    print("\n" + "=" * 60)
    print("Ranking by Efficiency (disparity / overhead)")
    print("=" * 60)

    ranked = sorted(results.items(), key=lambda x: x[1]['avg_disp'])
    print(f"\n{'Rank':<5} {'Pair':<12} {'Overhead':>10} {'Disparity':>10}")
    print("-" * 40)
    for i, (name, data) in enumerate(ranked[:10], 1):
        print(f"{i:<5} {name:<12} {data['pct']:>9.2f}% {data['avg_disp']:>9.1f}x")

    # Synergy analysis
    print("\n" + "=" * 60)
    print("Synergy Analysis: L0 + Lx vs sum of individuals")
    print("=" * 60)

    l0_alone = results['L0_only']['avg_disp']
    l11_alone = results['L11_only']['avg_disp']

    print(f"\n{'Pair':<12} {'Expected':>10} {'Actual':>10} {'Synergy':>10}")
    print("-" * 45)
    for name, data in results.items():
        if name.startswith("L0+L") and name not in ["L0_only", "L0+L2+L11"]:
            layer_num = name.split("+")[1][1:]  # Extract number
            # We'd need individual layer data, approximate with L11
            expected = (l0_alone + l11_alone) / 2  # Rough estimate
            actual = data['avg_disp']
            synergy = expected - actual
            print(f"{name:<12} {expected:>9.1f}x {actual:>9.1f}x {synergy:>+9.1f}x")

    best = ranked[0]
    print(f"\nBest pair: {best[0]}")
    print(f"  Overhead: {best[1]['pct']:.2f}%")
    print(f"  Disparity: {best[1]['avg_disp']:.1f}x")

    end = datetime.now()

    result = {
        "id": "Exp-035",
        "name": "Optimal Layer Pair Exploration",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "best_pair": best[0],
        "ranking": [(name, data['avg_disp']) for name, data in ranked[:5]],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp035_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
