#!/usr/bin/env python3
"""
Exp-057: Variance-based selection
Goal: Do high-variance weights correlate with multilingual importance?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

baseline_en = ppl('Fox.')
baseline_he = ppl('שועל.')

# Analyze variance of L0 vs L11 vs others
print("Weight variance by layer:")
print(f"{'Layer':<10} {'Variance':>12} {'Std':>12}")
print("-" * 36)

variances = {}
for layer_idx in range(12):
    prefix = f"transformer.h.{layer_idx}."
    layer_weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            layer_weights.append(param.data.flatten())

    all_weights = torch.cat(layer_weights)
    var = all_weights.var().item()
    std = all_weights.std().item()
    variances[layer_idx] = var
    print(f"Layer {layer_idx:<4} {var:>12.6f} {std:>12.6f}")

# Sort by variance
sorted_var = sorted(variances.items(), key=lambda x: x[1], reverse=True)
print(f"\nHighest variance: Layer {sorted_var[0][0]}")
print(f"Lowest variance:  Layer {sorted_var[-1][0]}")

# Check if L0 and L11 stand out
l0_rank = next(i for i, (l, v) in enumerate(sorted_var) if l == 0) + 1
l11_rank = next(i for i, (l, v) in enumerate(sorted_var) if l == 11) + 1
print(f"\nL0 variance rank:  {l0_rank}/12")
print(f"L11 variance rank: {l11_rank}/12")

print("\nDone.")
