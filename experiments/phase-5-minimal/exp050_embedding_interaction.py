#!/usr/bin/env python3
"""
Exp-050: Embedding interaction with L0+L11
Goal: Does protecting embeddings alongside L0+L11 help or hurt?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.', 'ar': 'الثعلب يقفز.'}

print("Loading...")
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
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))
    return protected

baseline = {l: ppl(t) for l, t in TEXTS.items()}
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}, ar={baseline['ar']:.1f}")

configs = [
    ("L0+L11", ["h.0.", "h.11."]),
    ("L0+L11+wte", ["h.0.", "h.11.", "wte"]),
    ("L0+L11+wpe", ["h.0.", "h.11.", "wpe"]),
    ("L0+L11+both_emb", ["h.0.", "h.11.", "wte", "wpe"]),
    ("L0+L11+ln_f", ["h.0.", "h.11.", "ln_f"]),
    ("L0+L11+all_aux", ["h.0.", "h.11.", "wte", "wpe", "ln_f"]),
]

print(f"\n{'Config':<18} {'%':>7} {'he':>8} {'ar':>8} {'Avg':>8}")
print("-" * 50)

for name, patterns in configs:
    restore()
    protected = quant_except(patterns)
    pct = protected / total * 100

    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
    avg = sum(disp.values()) / len(disp)

    print(f"{name:<18} {pct:>6.2f}% {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x")

print("\nDone.")
