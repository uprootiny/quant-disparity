#!/usr/bin/env python3
"""
Exp-055: Random vs structured protection
Goal: Is L0+L11 special, or would any 11.4% of weights work?
"""
import torch
import random
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

# Get baseline
baseline_en = ppl('Fox.')
baseline_he = ppl('שועל.')
print(f"Baseline: en={baseline_en:.1f}, he={baseline_he:.1f}")

# L0+L11 protection
restore()
l0l11_params = 0
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        if 'h.0.' in name or 'h.11.' in name:
            l0l11_params += param.numel()
            continue
        flat = param.view(-1)
        mx = flat.abs().max()
        if mx > 0:
            scale = mx / 7.0
            param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

en_l0l11 = ppl('Fox.')
he_l0l11 = ppl('שועל.')
deg_l0l11 = ((he_l0l11 - baseline_he) / baseline_he) / ((en_l0l11 - baseline_en) / baseline_en) if (en_l0l11 - baseline_en) > 0 else 0
print(f"L0+L11 ({l0l11_params/total*100:.1f}%): disparity = {deg_l0l11:.1f}x")

# Random 11.4% protection (3 trials)
target_protect = l0l11_params
print(f"\nRandom protection (~{l0l11_params/total*100:.1f}%):")

for trial in range(3):
    restore()

    # Randomly select weights to protect
    random.seed(42 + trial)
    protected = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue

            flat = param.view(-1)
            if protected < target_protect and random.random() < 0.114:
                protected += param.numel()
                continue

            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    en_rand = ppl('Fox.')
    he_rand = ppl('שועל.')
    deg_rand = ((he_rand - baseline_he) / baseline_he) / ((en_rand - baseline_en) / baseline_en) if (en_rand - baseline_en) > 0 else 0
    print(f"  Trial {trial+1}: disparity = {deg_rand:.1f}x")

print("\nDone.")
