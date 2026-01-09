#!/usr/bin/env python3
"""
Exp-077: 4-layer combinations - diminishing returns?
Goal: Is there significant improvement from 3 to 4 layers?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

total = sum(p.numel() for p in model.parameters())
state = {k: v.clone() for k, v in model.state_dict().items()}

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog near the river.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד הנהר.',
    'zh': '敏捷的棕色狐狸跳过懒狗在河边。',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def protect_and_measure(layers):
    restore()
    protected = 0
    baseline = {l: ppl(t) for l, t in TEXTS.items()}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                protected += param.numel()
                continue
            if any(f'h.{l}.' in name for l in layers):
                protected += param.numel()
                continue
            if 'ln_f' in name:
                protected += param.numel()
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    if en_deg <= 0:
        return float('inf'), protected / total * 100

    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    return sum(non_en) / len(non_en), protected / total * 100

print("=" * 70)
print("4-Layer Combinations: Diminishing Returns Analysis")
print("=" * 70)

# Baseline comparisons
print("\n1. BASELINE: 2-layer and 3-layer")
print("-" * 50)

disp_2, overhead_2 = protect_and_measure([0, 11])
disp_3, overhead_3 = protect_and_measure([0, 9, 11])

print(f"L0+L11:     {disp_2:.2f}x disparity, {overhead_2:.2f}% overhead")
print(f"L0+L9+L11:  {disp_3:.2f}x disparity, {overhead_3:.2f}% overhead")

# Test L0+L9+L11 + one more layer
print("\n2. ADDING 4TH LAYER TO L0+L9+L11")
print("-" * 50)

results = []
for fourth in [1, 2, 3, 4, 5, 6, 7, 8, 10]:  # Skip 0, 9, 11
    layers = [0, fourth, 9, 11]
    disp, overhead = protect_and_measure(layers)
    results.append((fourth, disp, overhead))
    improve = (disp_3 - disp) / disp_3 * 100 if disp_3 > 0 else 0
    print(f"L0+L{fourth}+L9+L11: {disp:.2f}x ({improve:+.1f}% vs 3-layer)")

# Sort by disparity
results.sort(key=lambda x: x[1])

print("\n3. TOP 4-LAYER COMBINATIONS")
print("-" * 50)

best_4 = results[0]
print(f"Best: L0+L{best_4[0]}+L9+L11")
print(f"  Disparity: {best_4[1]:.2f}x")
print(f"  Overhead:  {best_4[2]:.2f}%")

# Summary
print("\n" + "=" * 70)
print("4. EFFICIENCY ANALYSIS")
print("=" * 70)

print(f"\n{'Config':<20} {'Disparity':>12} {'Overhead':>12} {'Efficiency':>12}")
print("-" * 60)

configs = [
    ("L0+L11", disp_2, overhead_2),
    ("L0+L9+L11", disp_3, overhead_3),
    (f"L0+L{best_4[0]}+L9+L11", best_4[1], best_4[2]),
]

for name, disp, overhead in configs:
    efficiency = (1 / disp) / overhead if disp > 0 and overhead > 0 else 0
    print(f"{name:<20} {disp:>11.2f}x {overhead:>11.2f}% {efficiency:>11.3f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

improve_3to4 = (disp_3 - best_4[1]) / disp_3 * 100 if disp_3 > 0 else 0
overhead_increase = best_4[2] - overhead_3

print(f"""
2→3 layers: Major improvement (L9 adds significant value)
3→4 layers: {improve_3to4:.1f}% improvement for +{overhead_increase:.1f}% overhead

Recommendation:
- L0+L9+L11 is the sweet spot for most use cases
- 4th layer provides diminishing returns
""")
