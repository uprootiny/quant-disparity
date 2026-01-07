#!/usr/bin/env python3
"""
Exp-013: Hybrid protection strategy
Goal: Combine Layer 0 + selective MLP for best efficiency
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-013: Hybrid Protection")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}
    total = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)

    # Collect MLP magnitudes separately
    mlp_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name and 'mlp' in name:
            mlp_weights[name] = param.data.abs().view(-1)

    mlp_total = sum(w.numel() for w in mlp_weights.values())
    all_mlp_mags = torch.cat(list(mlp_weights.values()))
    sorted_mlp_mags, _ = all_mlp_mags.sort(descending=True)

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def hybrid_quant(layer0_protect, mlp_top_k_pct):
        """Protect layer 0 + top k% of MLP weights."""
        # Compute MLP threshold
        if mlp_top_k_pct > 0:
            n_keep = int(mlp_total * mlp_top_k_pct / 100)
            mlp_thresh = sorted_mlp_mags[n_keep - 1].item()
        else:
            mlp_thresh = float('inf')

        protected = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue

                # Check protection rules
                is_layer0 = 'h.0.' in name and layer0_protect
                is_top_mlp = 'mlp' in name and param.data.abs().max() >= mlp_thresh

                if is_layer0 or is_top_mlp:
                    protected += param.numel()
                    continue

                # Quantize
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7) * scale
                    param.data.copy_(q.view(param.shape))

        return protected

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Test hybrid configurations
    configs = [
        ("none", False, 0),
        ("layer0_only", True, 0),
        ("layer0+mlp1%", True, 1),
        ("layer0+mlp2%", True, 2),
        ("layer0+mlp5%", True, 5),
        ("layer0+mlp10%", True, 10),
        ("mlp5%_only", False, 5),
        ("mlp10%_only", False, 10),
    ]

    results = {}
    print(f"\n{'Config':<18} {'Protected':>10} {'%':>6} {'Disp':>8}")
    print("-" * 46)

    for name, l0, mlp_k in configs:
        restore()
        protected = hybrid_quant(l0, mlp_k)

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

        pct = protected / total * 100
        results[name] = {
            'layer0': l0,
            'mlp_k': mlp_k,
            'protected': protected,
            'protected_pct': pct,
            'disparity': disp,
        }

        print(f"{name:<18} {protected:>10,} {pct:>6.1f}% {disp:>8.1f}x")
        gc.collect()

    # Find optimal
    print("\n" + "=" * 50)
    print("Optimization Analysis")
    print("=" * 50)

    # Find best under 10% overhead
    under_10 = [(n, r['disparity'], r['protected_pct']) for n, r in results.items()
                if r['protected_pct'] <= 10 and np.isfinite(r['disparity'])]
    if under_10:
        best_10 = min(under_10, key=lambda x: x[1])
        print(f"  Best <10% overhead: {best_10[0]} ({best_10[1]:.1f}x at {best_10[2]:.1f}%)")

    # Find best under 15%
    under_15 = [(n, r['disparity'], r['protected_pct']) for n, r in results.items()
                if r['protected_pct'] <= 15 and np.isfinite(r['disparity'])]
    if under_15:
        best_15 = min(under_15, key=lambda x: x[1])
        print(f"  Best <15% overhead: {best_15[0]} ({best_15[1]:.1f}x at {best_15[2]:.1f}%)")

    end = datetime.now()

    result = {
        "id": "Exp-013",
        "name": "Hybrid Protection",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "results": results,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp013_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
