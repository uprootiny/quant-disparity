#!/usr/bin/env python3
"""
Exp-076: Why is Layer 9 special?
Goal: Understand what makes L9 uniquely beneficial with L0+L11
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

print("=" * 70)
print("Why is Layer 9 Special? (GPT-2)")
print("=" * 70)

# 1. Weight statistics comparison
print("\n1. LAYER STATISTICS COMPARISON")
print("-" * 60)

def get_layer_stats(layer_idx):
    prefix = f"transformer.h.{layer_idx}."
    weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            weights.append(param.data.flatten())

    all_w = torch.cat(weights)
    std = all_w.std().item()

    return {
        'variance': all_w.var().item(),
        'std': std,
        'mean': all_w.mean().item(),
        'max': all_w.abs().max().item(),
        'outliers': (all_w.abs() > 3 * std).float().mean().item() * 100,
        'sparsity': (all_w.abs() < 0.01).float().mean().item() * 100,
        'kurtosis': ((all_w - all_w.mean()) / std).pow(4).mean().item() - 3,
    }

# Compare L9 with neighbors and extremes
compare_layers = [0, 8, 9, 10, 11]
stats = {i: get_layer_stats(i) for i in compare_layers}

print(f"{'Layer':<8}", end='')
for metric in ['variance', 'outliers', 'sparsity', 'kurtosis']:
    print(f"{metric:>12}", end='')
print()
print("-" * 60)

for layer in compare_layers:
    s = stats[layer]
    print(f"L{layer:<7}", end='')
    print(f"{s['variance']:>12.6f}", end='')
    print(f"{s['outliers']:>11.3f}%", end='')
    print(f"{s['sparsity']:>11.1f}%", end='')
    print(f"{s['kurtosis']:>12.2f}")

# 2. Component-level analysis
print("\n2. COMPONENT-LEVEL ANALYSIS")
print("-" * 60)

def get_component_stats(layer_idx, component):
    for name, param in model.named_parameters():
        if f'h.{layer_idx}.' in name and component in name and 'weight' in name:
            w = param.data.flatten()
            return {
                'var': w.var().item(),
                'max': w.abs().max().item(),
            }
    return None

components = ['c_attn', 'c_proj', 'c_fc', 'mlp.c_proj']
print(f"{'Layer':<8}", end='')
for c in components:
    print(f"{c.split('.')[-1]:>12}", end='')
print()
print("-" * 60)

for layer in compare_layers:
    print(f"L{layer:<7}", end='')
    for c in components:
        s = get_component_stats(layer, c)
        if s:
            print(f"{s['var']:>12.6f}", end='')
        else:
            print(f"{'N/A':>12}", end='')
    print()

# 3. Position analysis
print("\n3. POSITION IN NETWORK")
print("-" * 60)
print("""
L0:  Input gateway (processes raw embeddings)
L9:  Late middle (3rd from output)
L10: 2nd from output
L11: Output gateway (final transformer block)

L9 is at position 9/12 = 75% through the network.
This may be where high-level representations are consolidated
before the final output layers.
""")

# 4. Synergy analysis
print("4. SYNERGY PATTERNS")
print("-" * 60)

state = {k: v.clone() for k, v in model.state_dict().items()}

def restore():
    model.load_state_dict(state)

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def protect_and_measure(layers):
    restore()
    baseline_en = ppl('The fox.')
    baseline_he = ppl('השועל.')

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            if any(f'h.{l}.' in name for l in layers):
                continue
            if 'ln_f' in name:
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    q_en = ppl('The fox.')
    q_he = ppl('השועל.')

    en_deg = (q_en - baseline_en) / baseline_en * 100
    he_deg = (q_he - baseline_he) / baseline_he * 100

    if en_deg <= 0:
        return float('inf')
    return he_deg / en_deg

# Test synergy with different combinations
print("\nAdding middle layer to L0+L11:")
for middle in range(1, 11):
    disp = protect_and_measure([0, middle, 11])
    marker = " <-- L9" if middle == 9 else ""
    print(f"  L0+L{middle}+L11: {disp:.2f}x{marker}")

print("\n" + "=" * 70)
print("HYPOTHESIS")
print("=" * 70)
print("""
L9's special role may be due to:
1. Position at 75% depth - key transition point
2. Statistics intermediate between early/late layers
3. Captures representations that complement L0+L11

The pattern suggests: INPUT (L0) + CONSOLIDATION (L9) + OUTPUT (L11)
creates a more complete coverage of the model's processing stages.
""")
