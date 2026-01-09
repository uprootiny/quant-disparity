#!/usr/bin/env python3
"""
Exp-031: Fine-grained Layer 0 breakdown
Goal: Identify exact components within Layer 0 most critical for disparity
Memory-safe: GPT-2 only (~500MB base)
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-031: Layer 0 Fine-Grained Breakdown")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Map Layer 0 components
    print("\nLayer 0 component sizes:")
    l0_components = {}
    for name, param in model.named_parameters():
        if name.startswith('transformer.h.0.'):
            short = name.replace('transformer.h.0.', '')
            l0_components[short] = param.numel()
            print(f"  {short:<30} {param.numel():>10,}")

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
    print(f"\nBaseline: {', '.join([f'{l}={v:.1f}' for l, v in baseline.items()])}")

    # Test each Layer 0 sub-component
    strategies = [
        ("none", []),
        ("l0_ln_1", ["h.0.ln_1"]),
        ("l0_ln_2", ["h.0.ln_2"]),
        ("l0_attn_qkv", ["h.0.attn.c_attn"]),
        ("l0_attn_proj", ["h.0.attn.c_proj"]),
        ("l0_attn_all", ["h.0.attn"]),
        ("l0_mlp_fc", ["h.0.mlp.c_fc"]),
        ("l0_mlp_proj", ["h.0.mlp.c_proj"]),
        ("l0_mlp_all", ["h.0.mlp"]),
        ("l0_full", ["h.0."]),
    ]

    results = {}
    print(f"\n{'Component':<15} {'Protected':>12} {'%':>7} {'he':>8} {'ar':>8} {'Avg':>8}")
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
            'degradation': deg,
        }

        print(f"{name:<15} {protected:>12,} {pct:>6.2f}% {disp.get('he', 0):>7.1f}x {disp.get('ar', 0):>7.1f}x {avg_disp:>7.1f}x")

    # Analysis
    print("\n" + "=" * 60)
    print("Efficiency Analysis (disparity reduction per % overhead)")
    print("=" * 60)

    efficiencies = []
    baseline_disp = results['none']['avg_disp']
    for name, data in results.items():
        if name == 'none' or data['pct'] == 0:
            continue
        reduction = baseline_disp - data['avg_disp']
        efficiency = reduction / data['pct']
        efficiencies.append((name, efficiency, data['pct'], data['avg_disp']))

    efficiencies.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Component':<15} {'Efficiency':>10} {'Overhead':>10} {'Disparity':>10}")
    print("-" * 50)
    for name, eff, pct, disp in efficiencies:
        print(f"{name:<15} {eff:>10.1f} {pct:>9.2f}% {disp:>9.1f}x")

    best = efficiencies[0]
    print(f"\nMost efficient: {best[0]}")
    print(f"  {best[2]:.2f}% overhead -> {best[3]:.1f}x disparity")

    end = datetime.now()

    result = {
        "id": "Exp-031",
        "name": "Layer 0 Fine-Grained Breakdown",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "efficiencies": {e[0]: e[1] for e in efficiencies},
        "best_component": best[0],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp031_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
