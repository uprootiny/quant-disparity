#!/usr/bin/env python3
"""
Exp-065: Variance-based automatic layer selection
Goal: Can we use variance to automatically identify critical layers?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

total = sum(p.numel() for p in model.parameters())
state = {k: v.clone() for k, v in model.state_dict().items()}

TEXTS = {
    'en': 'The quick brown fox.',
    'he': 'השועל החום המהיר.',
    'zh': '敏捷的棕色狐狸。',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def get_layer_variance(layer_idx):
    """Get combined variance of all weights in a layer"""
    prefix = f"transformer.h.{layer_idx}."
    weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            weights.append(param.data.flatten())
    if weights:
        return torch.cat(weights).var().item()
    return 0

def protect_layers(layer_indices):
    """Quantize all except specified layers"""
    restore()
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Protect biases
            if 'bias' in name:
                protected += param.numel()
                continue
            # Protect specified layers
            if any(f'h.{l}.' in name for l in layer_indices):
                protected += param.numel()
                continue
            # Protect final LayerNorm
            if 'ln_f' in name:
                protected += param.numel()
                continue
            # Quantize
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))
    return protected

def get_disparity(layers):
    """Get average disparity for a layer configuration"""
    protect_layers(layers)
    baseline = {l: ppl(t) for l, t in TEXTS.items()}

    restore()
    protect_layers(layers)
    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}

    # Use fresh baseline
    restore()
    base = {l: ppl(t) for l, t in TEXTS.items()}

    protect_layers(layers)
    q = {l: ppl(t) for l, t in TEXTS.items()}

    deg = {l: (q[l] - base[l]) / base[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    if en_deg <= 0:
        return float('inf')

    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    return sum(non_en) / len(non_en)

print("=" * 70)
print("Variance-Based Automatic Layer Selection")
print("=" * 70)

# Calculate variance for each layer
print("\n1. Layer Variance Ranking")
print("-" * 40)

variances = [(i, get_layer_variance(i)) for i in range(12)]
sorted_by_var = sorted(variances, key=lambda x: -x[1])  # Highest first

print(f"{'Rank':<6} {'Layer':<8} {'Variance':>12}")
print("-" * 30)
for rank, (layer, var) in enumerate(sorted_by_var, 1):
    print(f"{rank:<6} Layer {layer:<4} {var:>12.6f}")

# Test variance-based selection
print("\n2. Testing Variance-Based Selection")
print("-" * 60)

# Get baseline
restore()
baseline = {l: ppl(t) for l, t in TEXTS.items()}

# Test top-k variance layers
print(f"\n{'Strategy':<30} {'Layers':<15} {'Disparity':>12}")
print("-" * 60)

# Known optimal (L0+L11)
layers_known = [0, 11]
restore()
protected = protect_layers(layers_known)
q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']
if en_deg > 0:
    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    disp_known = sum(non_en) / len(non_en)
else:
    disp_known = float('inf')
print(f"{'Known optimal (L0+L11)':<30} {str(layers_known):<15} {disp_known:>11.2f}x")

# Top-2 by variance
layers_top2 = [l for l, v in sorted_by_var[:2]]
restore()
protect_layers(layers_top2)
q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']
if en_deg > 0:
    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    disp_top2 = sum(non_en) / len(non_en)
else:
    disp_top2 = float('inf')
print(f"{'Top-2 variance':<30} {str(layers_top2):<15} {disp_top2:>11.2f}x")

# Top-3 by variance
layers_top3 = [l for l, v in sorted_by_var[:3]]
restore()
protect_layers(layers_top3)
q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']
if en_deg > 0:
    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    disp_top3 = sum(non_en) / len(non_en)
else:
    disp_top3 = float('inf')
print(f"{'Top-3 variance':<30} {str(layers_top3):<15} {disp_top3:>11.2f}x")

# First + Last (position heuristic)
layers_fl = [0, 11]
restore()
protect_layers(layers_fl)
q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']
if en_deg > 0:
    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    disp_fl = sum(non_en) / len(non_en)
else:
    disp_fl = float('inf')
print(f"{'First + Last':<30} {str(layers_fl):<15} {disp_fl:>11.2f}x")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
top2_match = set(layers_top2) == set([0, 11])
print(f"Top-2 variance matches L0+L11: {top2_match}")
if top2_match:
    print("SUCCESS: Variance can automatically identify critical layers!")
else:
    print(f"Top-2 variance selected: {layers_top2}")
    print("Variance approximates but doesn't perfectly predict criticality")
