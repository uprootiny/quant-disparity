#!/usr/bin/env python3
"""
Exp-037: Even layers analysis - is L0 essential?
Goal: Test if even layers without L0 still work
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {'en': 'Fox jumps.', 'he': 'שועל קופץ.', 'ar': 'ثعلب يقفز.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-037: Even Layers Without L0")

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
        gc.collect()

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
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))
        return protected

    baseline = {l: ppl(t) for l, t in TEXTS.items()}

    configs = [
        ("none", []),
        ("L0_only", ["h.0."]),
        ("even_all", ["h.0.", "h.2.", "h.4.", "h.6.", "h.8.", "h.10."]),
        ("even_no_L0", ["h.2.", "h.4.", "h.6.", "h.8.", "h.10."]),
        ("L2+L4+L6", ["h.2.", "h.4.", "h.6."]),
    ]

    print(f"{'Config':<15} {'%':>7} {'he':>8} {'ar':>8} {'Avg':>8}")
    print("-" * 45)

    for name, patterns in configs:
        restore()
        protected = quant_except(patterns)
        pct = protected / total * 100

        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
        avg = sum(disp.values()) / len(disp)

        print(f"{name:<15} {pct:>6.1f}% {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x")

    print(f"\n✓ Done in {(datetime.now()-start).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
