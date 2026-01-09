#!/usr/bin/env python3
"""
Exp-039: Bias vs Weight analysis (ultra-minimal)
Goal: Do biases contribute to disparity?
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
total = sum(p.numel() for p in model.parameters())

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def quant_weights_only():
    """Quantize weights, leave biases untouched."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

def quant_all():
    """Quantize everything."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

baseline = {l: ppl(t) for l, t in TEXTS.items()}

# Count biases
bias_count = sum(p.numel() for n, p in model.named_parameters() if 'bias' in n)
print(f"Biases: {bias_count:,} ({bias_count/total*100:.3f}%)")
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

# Test weights only
restore()
quant_weights_only()
q1 = {l: ppl(t) for l, t in TEXTS.items()}
deg1 = {l: (q1[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
disp1 = deg1['he'] / deg1['en'] if deg1['en'] > 0 else float('inf')

# Test all
restore()
quant_all()
q2 = {l: ppl(t) for l, t in TEXTS.items()}
deg2 = {l: (q2[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
disp2 = deg2['he'] / deg2['en'] if deg2['en'] > 0 else float('inf')

print(f"\nWeights only: {disp1:.1f}x")
print(f"All (w+bias): {disp2:.1f}x")
print(f"Bias impact:  {disp2 - disp1:+.1f}x")

if disp2 > disp1:
    print("\n-> Quantizing biases HURTS multilingual")
else:
    print("\n-> Biases don't matter much")
