#!/usr/bin/env python3
"""
Exp-053: Single configuration test (memory-safe)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

# Single baseline measurement
en_base = ppl('Fox.')
he_base = ppl('שועל.')
print(f"Baseline: en={en_base:.1f}, he={he_base:.1f}")

# Quantize with L0+L11 protection
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

en_q = ppl('Fox.')
he_q = ppl('שועל.')

en_deg = (en_q - en_base) / en_base * 100
he_deg = (he_q - he_base) / he_base * 100
disp = he_deg / en_deg if en_deg > 0 else 0

print(f"Quantized: en={en_q:.1f}, he={he_q:.1f}")
print(f"Disparity: {disp:.1f}x")
