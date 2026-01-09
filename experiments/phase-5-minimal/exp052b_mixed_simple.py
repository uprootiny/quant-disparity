#!/usr/bin/env python3
"""
Exp-052b: Mixed precision (simplified)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {'en': 'Fox.', 'he': 'שועל.'}

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

def quant(bits, protect=[]):
    levels = 2 ** (bits - 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name or any(p in name for p in protect):
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / levels
                param.data.copy_((torch.round(flat / scale).clamp(-levels, levels-1) * scale).view(param.shape))

baseline = {l: ppl(t) for l, t in TEXTS.items()}
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

configs = [
    ("INT4_all", 4, []),
    ("INT8_all", 8, []),
    ("INT4_L0L11_FP16", 4, ["h.0.", "h.11."]),
    ("INT8_L0L11_FP16", 8, ["h.0.", "h.11."]),
]

print(f"\n{'Config':<20} {'he disp':>10}")
for name, bits, protect in configs:
    restore()
    quant(bits, protect)
    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    disp = deg['he'] / deg['en'] if deg['en'] > 0 else 0
    print(f"{name:<20} {disp:>9.1f}x")

print("Done.")
