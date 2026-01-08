#!/usr/bin/env python3
"""
Exp-027: Minimal protection strategy
Goal: Find smallest possible intervention with acceptable disparity
Based on Exp-025 (embeddings) and Exp-026 (layer norms) findings
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
    'zh': '敏捷的棕色狐狸跳过了懒狗。',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-027: Minimal Protection Strategy")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
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
    print("Baseline PPL:", {l: f"{v:.1f}" for l, v in baseline.items()})

    # Minimal strategies ranked by size
    strategies = [
        ("none", []),
        ("layer0_ln_only", ["h.0.ln"]),  # ~3K params
        ("all_ln", ["ln_"]),  # ~38K params
        ("wpe_only", ["wpe"]),  # ~786K params
        ("all_ln+wpe", ["ln_", "wpe"]),  # ~824K params
        ("layer0_attn", ["h.0.attn"]),  # ~2.4M params
        ("layer0_mlp", ["h.0.mlp"]),  # ~4.7M params
        ("layer0", ["h.0."]),  # ~7.1M params
    ]

    results = {}
    print(f"\n{'Strategy':<18} {'Protected':>12} {'%':>6} {'en→he':>8} {'en→ar':>8} {'en→zh':>8}")
    print("-" * 65)

    for strat_name, patterns in strategies:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}

        en_deg = deg['en']
        disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}

        pct = protected / total_params * 100
        results[strat_name] = {
            'protected': protected,
            'pct': pct,
            'disparity': disp,
            'avg_lr_disp': sum(disp.values()) / len(disp),
        }

        he_d = disp.get('he', float('inf'))
        ar_d = disp.get('ar', float('inf'))
        zh_d = disp.get('zh', float('inf'))
        print(f"{strat_name:<18} {protected:>12,} {pct:>5.2f}% {he_d:>7.1f}x {ar_d:>7.1f}x {zh_d:>7.1f}x")

    # Find Pareto frontier
    print("\n" + "=" * 60)
    print("Pareto Optimal Strategies (best disparity for size)")
    print("=" * 60)

    # Sort by size
    sorted_strats = sorted(results.items(), key=lambda x: x[1]['pct'])

    best_disp_so_far = float('inf')
    pareto = []
    for name, data in sorted_strats:
        if data['avg_lr_disp'] < best_disp_so_far:
            best_disp_so_far = data['avg_lr_disp']
            pareto.append((name, data))

    print(f"\n{'Strategy':<18} {'Overhead':>8} {'Avg LR Disp':>12}")
    print("-" * 42)
    for name, data in pareto:
        print(f"{name:<18} {data['pct']:>7.2f}% {data['avg_lr_disp']:>11.1f}x")

    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    for name, data in pareto:
        if data['avg_lr_disp'] < 10:
            print(f"\nMinimal acceptable: {name}")
            print(f"  Overhead: {data['pct']:.2f}%")
            print(f"  Avg LR disparity: {data['avg_lr_disp']:.1f}x")
            break

    end = datetime.now()

    result = {
        "id": "Exp-027",
        "name": "Minimal Protection Strategy",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "pareto_frontier": [(n, d['pct'], d['avg_lr_disp']) for n, d in pareto],
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp027_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
