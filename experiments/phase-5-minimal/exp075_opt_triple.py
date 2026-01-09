#!/usr/bin/env python3
"""
Exp-075: Test triple layer patterns on OPT-125M
Goal: Does L0+L9+L11 pattern help OPT, or need different layers?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
model.eval()

total = sum(p.numel() for p in model.parameters())
state = {k: v.clone() for k, v in model.state_dict().items()}

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'zh': '敏捷的棕色狐狸跳过懒狗。',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def protect_and_measure(layers):
    restore()
    baseline = {l: ppl(t) for l, t in TEXTS.items()}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            if any(f'layers.{l}.' in name for l in layers):
                continue
            if 'final_layer_norm' in name:
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
        return float('inf')

    non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
    return sum(non_en) / len(non_en)

print("=" * 70)
print("OPT-125M: Testing Triple Layer Patterns")
print("=" * 70)

# Test GPT-2's pattern on OPT
print("\n1. GPT-2 Pattern on OPT")
print("-" * 40)

gpt2_patterns = [
    ("L0+L11 (GPT-2 2-layer)", [0, 11]),
    ("L0+L9+L11 (GPT-2 optimal)", [0, 9, 11]),
]

for name, layers in gpt2_patterns:
    disp = protect_and_measure(layers)
    print(f"{name}: {disp:.1f}x")

# Find OPT's optimal triple layer (L4 is known critical)
print("\n2. Finding OPT's Optimal 3-Layer (L4 + two others)")
print("-" * 50)

results = []
for second, third in combinations([i for i in range(12) if i != 4], 2):
    layers = [4, second, third]
    disp = protect_and_measure(layers)
    results.append((layers, disp))

results.sort(key=lambda x: x[1])

print("\nTop 10 combinations:")
for i, (layers, disp) in enumerate(results[:10], 1):
    print(f"#{i}: L4+L{layers[1]}+L{layers[2]}: {disp:.1f}x")

# Compare best OPT triple with GPT-2's pattern
print("\n" + "=" * 70)
print("3. Comparison")
print("=" * 70)

best_opt = results[0]
gpt2_on_opt = protect_and_measure([0, 9, 11])

print(f"GPT-2 pattern (L0+L9+L11) on OPT: {gpt2_on_opt:.1f}x")
print(f"OPT optimal ({best_opt[0]}): {best_opt[1]:.1f}x")
print(f"Improvement: {(gpt2_on_opt - best_opt[1]) / gpt2_on_opt * 100:.0f}%")

# Test if any universal pattern exists
print("\n4. Testing Universal Patterns")
print("-" * 40)

universal = [
    ("First+Middle+Last (0,5,11)", [0, 5, 11]),
    ("First+2nd-last+Last (0,10,11)", [0, 10, 11]),
    ("Evenly spaced (0,4,8,11)", [0, 4, 8, 11]),
]

for name, layers in universal:
    disp = protect_and_measure(layers)
    print(f"{name}: {disp:.1f}x")
