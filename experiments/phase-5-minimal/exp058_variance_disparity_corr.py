#!/usr/bin/env python3
"""
Exp-058: Variance vs Disparity correlation
Goal: Do high-variance layers reduce disparity more when protected?
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

def protect_layer(layer_idx):
    restore()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            if f'h.{layer_idx}.' in name:
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    en = ppl('Fox.')
    he = ppl('שועל.')
    return he, en

baseline_en = ppl('Fox.')
baseline_he = ppl('שועל.')

# Calculate variance and disparity for each layer
print("Variance vs Disparity by Layer:")
print(f"{'Layer':<8} {'Variance':>10} {'Disparity':>12}")
print("-" * 34)

data = []
for layer_idx in range(12):
    # Get variance
    prefix = f"transformer.h.{layer_idx}."
    layer_weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            layer_weights.append(param.data.flatten())
    all_weights = torch.cat(layer_weights)
    var = all_weights.var().item()

    # Get disparity when protected
    he, en = protect_layer(layer_idx)
    en_deg = (en - baseline_en) / baseline_en
    he_deg = (he - baseline_he) / baseline_he
    disp = he_deg / en_deg if en_deg > 0 else float('inf')

    data.append((layer_idx, var, disp))
    print(f"Layer {layer_idx:<4} {var:>10.6f} {disp:>11.1f}x")

# Calculate correlation
import statistics
vars_list = [d[1] for d in data]
disps_list = [d[2] for d in data if d[2] < 1000]  # Filter inf

if len(disps_list) == len(data):
    mean_var = statistics.mean(vars_list)
    mean_disp = statistics.mean(disps_list)

    num = sum((v - mean_var) * (d - mean_disp) for (_, v, d) in data)
    denom_var = sum((v - mean_var)**2 for v in vars_list) ** 0.5
    denom_disp = sum((d - mean_disp)**2 for d in disps_list) ** 0.5

    corr = num / (denom_var * denom_disp) if denom_var > 0 and denom_disp > 0 else 0
    print(f"\nCorrelation (variance vs disparity): r = {corr:.3f}")

    if corr < -0.5:
        print("-> High variance = lower disparity (good)")
    elif corr > 0.5:
        print("-> High variance = higher disparity (bad)")
    else:
        print("-> Weak correlation")
