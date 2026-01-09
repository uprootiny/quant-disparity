#!/usr/bin/env python3
"""
Exp-056: Magnitude-based vs L0+L11 selection
Goal: Is protecting high-magnitude weights better than random?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

state = {k: v.clone() for k, v in model.state_dict().items()}
total = sum(p.numel() for p in model.parameters())

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

baseline_en = ppl('Fox.')
baseline_he = ppl('שועל.')
print(f"Baseline: en={baseline_en:.1f}, he={baseline_he:.1f}")

# Magnitude-based: protect top 11.4% by magnitude
restore()
target_pct = 0.114

# Collect all weights and their magnitudes
all_weights = []
for name, param in model.named_parameters():
    if 'weight' in name:
        all_weights.append((name, param.data.abs().mean().item(), param.numel()))

# Sort by magnitude (descending)
all_weights.sort(key=lambda x: x[1], reverse=True)

# Protect top N by magnitude
protected_names = set()
protected_count = 0
for name, mag, numel in all_weights:
    if protected_count / total < target_pct:
        protected_names.add(name)
        protected_count += numel

print(f"Protected by magnitude: {len(protected_names)} tensors ({protected_count/total*100:.1f}%)")
print(f"Top protected: {list(protected_names)[:5]}")

# Quantize everything except protected
with torch.no_grad():
    for name, param in model.named_parameters():
        if name in protected_names:
            continue
        if 'weight' not in name:
            continue
        flat = param.view(-1)
        mx = flat.abs().max()
        if mx > 0:
            scale = mx / 7.0
            param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

en_mag = ppl('Fox.')
he_mag = ppl('שועל.')
deg_mag = ((he_mag - baseline_he) / baseline_he) / ((en_mag - baseline_en) / baseline_en) if (en_mag - baseline_en) > 0 else 0

# Compare with L0+L11
restore()
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        if 'h.0.' in name or 'h.11.' in name:
            continue
        flat = param.view(-1)
        mx = flat.abs().max()
        if mx > 0:
            scale = mx / 7.0
            param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

en_l0l11 = ppl('Fox.')
he_l0l11 = ppl('שועל.')
deg_l0l11 = ((he_l0l11 - baseline_he) / baseline_he) / ((en_l0l11 - baseline_en) / baseline_en) if (en_l0l11 - baseline_en) > 0 else 0

print(f"\nMagnitude-based: {deg_mag:.1f}x")
print(f"L0+L11:          {deg_l0l11:.1f}x")
print(f"Improvement:     {deg_mag/deg_l0l11:.1f}x better" if deg_l0l11 > 0 else "L0+L11 = 0!")
