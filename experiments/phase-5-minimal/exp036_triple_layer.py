#!/usr/bin/env python3
"""
Exp-036: Triple layer test (L0+L2+L11)
Goal: Test if adding L2 improves on L0+L11
Minimal memory footprint.
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {'en': 'Fox jumps.', 'he': 'שועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-036: Triple Layer Test")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)
        gc.collect()

    def quant_except(patterns):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name or any(p in name for p in patterns):
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    baseline = {l: ppl(t) for l, t in TEXTS.items()}

    configs = [
        ("L0+L11", ["h.0.", "h.11."]),
        ("L0+L2+L11", ["h.0.", "h.2.", "h.11."]),
        ("L0+L10+L11", ["h.0.", "h.10.", "h.11."]),
    ]

    print(f"{'Config':<15} {'he disp':>10}")
    for name, patterns in configs:
        restore()
        quant_except(patterns)
        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
        print(f"{name:<15} {disp:>9.1f}x")

    print(f"✓ Done in {(datetime.now()-start).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
