#!/usr/bin/env python3
"""
Exp-062: OPT-125M layer criticality sweep
Goal: Find which layers matter most for OPT's multilingual fairness
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
model.eval()

total = sum(p.numel() for p in model.parameters())
state = {k: v.clone() for k, v in model.state_dict().items()}

# Use short texts for speed
en_txt = 'Fox.'
he_txt = 'שועל.'

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def protect_layer(layer_idx):
    """Quantize all except specified layer"""
    restore()
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Always protect biases
            if 'bias' in name:
                protected += param.numel()
                continue
            # Protect specified layer
            if f'layers.{layer_idx}.' in name:
                protected += param.numel()
                continue
            # Quantize the rest
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))
    return protected

print("OPT-125M Layer Criticality Sweep")
print("=" * 50)

# Baseline
baseline_en = ppl(en_txt)
baseline_he = ppl(he_txt)
print(f"Baseline: En={baseline_en:.1f}, He={baseline_he:.1f}")

# Test each layer individually
print(f"\n{'Layer':<8} {'En Deg%':>10} {'He Deg%':>12} {'Disparity':>10}")
print("-" * 44)

results = []
for layer_idx in range(12):
    protected = protect_layer(layer_idx)
    en = ppl(en_txt)
    he = ppl(he_txt)

    en_deg = (en - baseline_en) / baseline_en * 100
    he_deg = (he - baseline_he) / baseline_he * 100
    disp = he_deg / en_deg if en_deg > 0 else float('inf')

    results.append((layer_idx, en_deg, he_deg, disp))
    print(f"Layer {layer_idx:<4} {en_deg:>9.1f}% {he_deg:>11.1f}% {disp:>9.1f}x")

# Find best layers
print("\n" + "=" * 50)
print("Sorted by Disparity (lowest = most critical):")
print("-" * 50)

sorted_results = sorted(results, key=lambda x: x[3])
for layer_idx, en_deg, he_deg, disp in sorted_results[:6]:
    print(f"Layer {layer_idx}: disparity={disp:.1f}x")

best_layers = [r[0] for r in sorted_results[:2]]
print(f"\nRecommended for OPT-125M: layers {best_layers}")
