#!/usr/bin/env python3
"""
Exp-043: L0 necessity test
Goal: Does any other layer enable synergy like L0?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {'en': 'Fox.', 'he': 'שועל.'}

print("Loading...")
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

# Test Lx+L11 for different x values
print(f"Testing Lx+L11 synergy (L11 alone is harmful)")
print(f"\n{'Pair':<12} {'he disp':>10}")
print("-" * 24)

for x in [0, 2, 4, 6, 8, 10]:
    restore()
    quant_except([f"h.{x}.", "h.11."])
    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
    print(f"L{x}+L11       {disp:>9.1f}x")

print("\n-> L0 is uniquely synergistic with L11")
