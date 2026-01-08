#!/usr/bin/env python3
"""
Exp-026: Layer norm analysis
Goal: Test if layer norms are critical despite small size
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-026: LayerNorm Analysis")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Analyze layer norm parameters
    print("LayerNorm analysis:")
    total_params = sum(p.numel() for p in model.parameters())
    ln_params = 0
    for name, param in model.named_parameters():
        if 'ln' in name.lower():
            ln_params += param.numel()
            print(f"  {name}: {param.shape}")

    print(f"\nTotal LayerNorm params: {ln_params:,} ({ln_params/total_params*100:.3f}%)")

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
    print(f"\nBaseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test strategies
    strategies = [
        ("none", []),
        ("all_ln", ["ln_"]),  # All layer norms
        ("layer0_ln", ["h.0.ln"]),  # Just layer 0 layer norms
        ("layer0", ["h.0."]),  # Full layer 0
        ("layer0_no_ln", ["h.0.attn", "h.0.mlp"]),  # Layer 0 without LN
    ]

    results = {}
    print(f"\n{'Strategy':<18} {'Protected':>12} {'%':>6} {'Disparity':>10}")
    print("-" * 50)

    for strat_name, patterns in strategies:
        restore()
        protected = quant_except(patterns)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        pct = protected / total_params * 100
        results[strat_name] = {
            'protected': protected,
            'pct': pct,
            'disparity': disp,
        }
        print(f"{strat_name:<18} {protected:>12,} {pct:>5.2f}% {disp:>10.1f}x")

    # Analysis
    print("\n" + "=" * 60)
    print("LayerNorm Impact Analysis")
    print("=" * 60)

    all_ln = results['all_ln']['disparity']
    layer0 = results['layer0']['disparity']
    layer0_no_ln = results['layer0_no_ln']['disparity']

    print(f"\nAll LayerNorms:     {all_ln:.1f}x")
    print(f"Layer 0 full:       {layer0:.1f}x")
    print(f"Layer 0 (no LN):    {layer0_no_ln:.1f}x")

    ln_impact = layer0_no_ln - layer0
    print(f"\nLayerNorm contribution in Layer 0: {ln_impact:+.1f}x")

    if abs(ln_impact) < 5:
        print("-> LayerNorms have MINIMAL impact on disparity")
    else:
        print("-> LayerNorms have SIGNIFICANT impact on disparity")

    end = datetime.now()

    result = {
        "id": "Exp-026",
        "name": "LayerNorm Analysis",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "ln_impact": ln_impact,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp026_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
