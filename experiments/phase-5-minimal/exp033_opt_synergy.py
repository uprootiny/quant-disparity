#!/usr/bin/env python3
"""
Exp-033: Validate L0+L11 synergy on OPT-125M
Goal: Check if input-output layer synergy generalizes across architectures
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-033: OPT-125M Input-Output Synergy")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # OPT has 12 layers: model.decoder.layers.0 through 11
    print(f"Total params: {total:,}")

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

    # OPT layer patterns
    strategies = [
        ("none", []),
        ("layer0", ["layers.0."]),
        ("layer11", ["layers.11."]),
        ("layer0+11", ["layers.0.", "layers.11."]),
        ("layer0+10+11", ["layers.0.", "layers.10.", "layers.11."]),
        ("even_layers", ["layers.0.", "layers.2.", "layers.4.", "layers.6.", "layers.8.", "layers.10."]),
        ("odd_layers", ["layers.1.", "layers.3.", "layers.5.", "layers.7.", "layers.9.", "layers.11."]),
    ]

    results = {}
    print(f"\n{'Strategy':<15} {'Protected':>12} {'%':>7} {'he':>8} {'ar':>8} {'Avg':>8}")
    print("-" * 60)

    for name, patterns in strategies:
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

        print(f"{name:<15} {protected:>12,} {pct:>6.2f}% {disp.get('he', 0):>7.1f}x {disp.get('ar', 0):>7.1f}x {avg_disp:>7.1f}x")

    # Analysis
    print("\n" + "=" * 60)
    print("Cross-Architecture Synergy Analysis")
    print("=" * 60)

    l0 = results['layer0']['avg_disp']
    l11 = results['layer11']['avg_disp']
    combined = results['layer0+11']['avg_disp']

    print(f"\nLayer 0 alone:     {l0:.1f}x")
    print(f"Layer 11 alone:    {l11:.1f}x")
    print(f"L0 + L11 combined: {combined:.1f}x")

    expected_additive = (l0 + l11) / 2
    synergy = expected_additive - combined
    print(f"\nExpected (additive): {expected_additive:.1f}x")
    print(f"Actual (combined):   {combined:.1f}x")
    print(f"Synergy bonus:       {synergy:+.1f}x")

    if combined < min(l0, l11):
        print("\n-> SYNERGY CONFIRMED on OPT-125M!")
    else:
        print("\n-> No synergy on OPT-125M")

    # Compare even vs odd
    even = results['even_layers']['avg_disp']
    odd = results['odd_layers']['avg_disp']
    print(f"\nEven layers: {even:.1f}x")
    print(f"Odd layers:  {odd:.1f}x")

    if odd > even * 2:
        print("-> Anti-critical pattern CONFIRMED on OPT")
    else:
        print("-> Pattern differs from GPT-2")

    end = datetime.now()

    result = {
        "id": "Exp-033",
        "name": "OPT-125M Input-Output Synergy",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "synergy_bonus": synergy,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp033_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
