#!/usr/bin/env python3
"""
Exp-032: Output layer analysis
Goal: Test final layers (L10, L11) and lm_head for multilingual disparity
Memory-safe: GPT-2 only
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-032: Output Layer Analysis")
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

    # Test output-related layers
    strategies = [
        ("none", []),
        ("layer0", ["h.0."]),
        ("layer10", ["h.10."]),
        ("layer11", ["h.11."]),
        ("last_2", ["h.10.", "h.11."]),
        ("ln_f", ["ln_f"]),
        ("lm_head", ["lm_head"]),
        ("output_stack", ["h.11.", "ln_f", "lm_head"]),
        ("input_output", ["h.0.", "h.11.", "ln_f"]),
        ("first_last", ["h.0.", "h.11."]),
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
    print("Input vs Output Layer Comparison")
    print("=" * 60)

    l0 = results['layer0']['avg_disp']
    l11 = results['layer11']['avg_disp']
    first_last = results['first_last']['avg_disp']

    print(f"\nLayer 0 alone:    {l0:.1f}x")
    print(f"Layer 11 alone:   {l11:.1f}x")
    print(f"Both L0+L11:      {first_last:.1f}x")

    if first_last < min(l0, l11):
        print("\n-> Combining input + output layers shows synergy")
    else:
        print("\n-> No synergy between input and output layers")

    best = min(results.items(), key=lambda x: x[1]['avg_disp'])
    print(f"\nBest strategy: {best[0]}")
    print(f"  {best[1]['pct']:.2f}% overhead -> {best[1]['avg_disp']:.1f}x disparity")

    end = datetime.now()

    result = {
        "id": "Exp-032",
        "name": "Output Layer Analysis",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "best_strategy": best[0],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp032_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
