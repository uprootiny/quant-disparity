#!/usr/bin/env python3
"""
Exp-072: Find optimal 3-layer combination
Goal: Is L0+L10+L11 the best, or is there something better?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import combinations

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

state = {k: v.clone() for k, v in model.state_dict().items()}

# Use longer texts for reliability
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
    """Protect specified layers and measure average disparity"""
    restore()
    baseline = {l: ppl(t) for l, t in TEXTS.items()}

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

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    if en_deg <= 0:
        return float('inf')

    non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en']
    return sum(non_en_disps) / len(non_en_disps)

print("=" * 70)
print("Finding Optimal 3-Layer Combination")
print("=" * 70)

# L0 is essential (proven), so test L0 + 2 others
print("\nTesting L0 + two other layers...")
print("-" * 50)

results = []
for second, third in combinations(range(1, 12), 2):
    layers = [0, second, third]
    disp = protect_and_measure(layers)
    results.append((layers, disp))
    if disp < 5:  # Only print good results
        print(f"L0+L{second}+L{third}: {disp:.2f}x")

# Sort and display top 10
results.sort(key=lambda x: x[1])

print("\n" + "=" * 70)
print("TOP 10 Combinations (L0 + two others)")
print("=" * 70)
print(f"{'Rank':<6} {'Layers':<15} {'Disparity':>12}")
print("-" * 35)

for i, (layers, disp) in enumerate(results[:10], 1):
    layers_str = f"L0+L{layers[1]}+L{layers[2]}"
    print(f"#{i:<5} {layers_str:<15} {disp:>11.2f}x")

# Compare best 3-layer with best 2-layer
print("\n" + "=" * 70)
print("Comparison: Best 2-Layer vs Best 3-Layer")
print("=" * 70)

disp_2layer = protect_and_measure([0, 11])
best_3layer = results[0][0]
disp_3layer = results[0][1]

print(f"L0+L11 (2-layer):     {disp_2layer:.2f}x")
print(f"{best_3layer} (3-layer): {disp_3layer:.2f}x")
print(f"Improvement:          {(disp_2layer - disp_3layer) / disp_2layer * 100:.1f}%")

# Check if L0+L10+L11 is indeed optimal
l0l10l11_disp = protect_and_measure([0, 10, 11])
print(f"\nL0+L10+L11:           {l0l10l11_disp:.2f}x")
print(f"vs Top result:        {disp_3layer:.2f}x")
