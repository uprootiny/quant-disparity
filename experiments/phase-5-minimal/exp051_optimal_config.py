#!/usr/bin/env python3
"""
Exp-051: Find optimal minimal configuration
Goal: Test L0+L11+ln_f+biases as the optimal config
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

def quant_config(protect_patterns, protect_biases=False):
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            is_protected = False

            if protect_biases and 'bias' in name:
                is_protected = True
            if any(p in name for p in protect_patterns):
                is_protected = True

            if is_protected:
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
    ("none", [], False),
    ("L0+L11", ["h.0.", "h.11."], False),
    ("L0+L11+biases", ["h.0.", "h.11."], True),
    ("L0+L11+ln_f", ["h.0.", "h.11.", "ln_f"], False),
    ("L0+L11+ln_f+biases", ["h.0.", "h.11.", "ln_f"], True),
    ("L0+L11+all_ln", ["h.0.", "h.11.", "ln_"], False),
    ("optimal", ["h.0.", "h.11.", "ln_f"], True),  # Same as L0+L11+ln_f+biases
]

print(f"\n{'Config':<20} {'%':>7} {'he':>8} {'ar':>8} {'Avg':>8}")
print("-" * 52)

for name, patterns, biases in configs:
    restore()
    protected = quant_config(patterns, biases)
    pct = protected / total * 100

    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
    avg = sum(disp.values()) / len(disp)

    marker = " <-- OPTIMAL" if name == "optimal" else ""
    print(f"{name:<20} {pct:>6.2f}% {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x{marker}")

print("\nDone.")
