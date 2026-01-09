#!/usr/bin/env python3
"""
Exp-038: Attention component analysis (ultra-minimal)
Goal: Which attention component matters most in L0 vs L11?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {'en': 'Fox.', 'he': 'שועל.'}

print("Loading model...")
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
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

configs = [
    ("L0_attn_qkv", ["h.0.attn.c_attn"]),
    ("L0_attn_proj", ["h.0.attn.c_proj"]),
    ("L11_attn_qkv", ["h.11.attn.c_attn"]),
    ("L11_attn_proj", ["h.11.attn.c_proj"]),
]

print(f"\n{'Config':<15} {'he disp':>10}")
print("-" * 28)

for name, patterns in configs:
    restore()
    quant_except(patterns)
    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
    print(f"{name:<15} {disp:>9.1f}x")

print("\nDone.")
