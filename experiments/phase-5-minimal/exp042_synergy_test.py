#!/usr/bin/env python3
"""
Exp-042: Synergy hypothesis test
Goal: Is synergy specific to L0+L11 or do other pairs work?
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
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

# Test different pairs
pairs = [
    ("L0", ["h.0."]),
    ("L8", ["h.8."]),
    ("L11", ["h.11."]),
    ("L0+L8", ["h.0.", "h.8."]),
    ("L0+L11", ["h.0.", "h.11."]),
    ("L8+L11", ["h.8.", "h.11."]),
]

results = {}
print(f"\n{'Pair':<12} {'he disp':>10}")
print("-" * 24)

for name, patterns in pairs:
    restore()
    quant_except(patterns)
    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
    results[name] = disp
    print(f"{name:<12} {disp:>9.1f}x")

# Synergy analysis
print("\nSynergy Analysis:")
print("-" * 40)

for pair, solo1, solo2 in [("L0+L8", "L0", "L8"), ("L0+L11", "L0", "L11"), ("L8+L11", "L8", "L11")]:
    expected = (results[solo1] + results[solo2]) / 2
    actual = results[pair]
    synergy = expected - actual
    print(f"{pair}: expected {expected:.1f}x, actual {actual:.1f}x, synergy {synergy:+.1f}x")
