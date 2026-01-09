#!/usr/bin/env python3
"""
Exp-070: L10 vs L11 - Why is L0+L10 better than L0+L11?
Goal: Verify and understand the L0+L10 finding from quick sweep
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
    'en': 'The quick brown fox jumps over the lazy dog.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
    'zh': '敏捷的棕色狐狸跳过懒狗。',
    'ru': 'Быстрая коричневая лиса прыгает через ленивую собаку.',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def protect_layers(layer_indices, protect_biases=True, protect_ln=True):
    restore()
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if protect_biases and 'bias' in name:
                protected += param.numel()
                continue
            if any(f'h.{l}.' in name for l in layer_indices):
                protected += param.numel()
                continue
            if protect_ln and 'ln_f' in name:
                protected += param.numel()
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))
    return protected

print("=" * 70)
print("L10 vs L11: Why is L0+L10 Better?")
print("=" * 70)

# Get baseline
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("\nBaseline PPL:")
for l, v in baseline.items():
    print(f"  {l}: {v:.1f}")

# Test configurations
configs = [
    ("L0+L11 (original)", [0, 11]),
    ("L0+L10 (sweep)", [0, 10]),
    ("L0+L9", [0, 9]),
    ("L10+L11", [10, 11]),
    ("L0+L10+L11", [0, 10, 11]),
]

print(f"\n{'Config':<25} {'Overhead':>10} {'En Deg%':>10} {'Avg Disp':>10}")
print("-" * 60)

for name, layers in configs:
    protected = protect_layers(layers)
    pct = protected / total * 100

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    if en_deg > 0:
        non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en']
        avg_disp = sum(non_en_disps) / len(non_en_disps)
        disp_str = f"{avg_disp:.2f}x"
    else:
        disp_str = "inf"

    print(f"{name:<25} {pct:>9.2f}% {en_deg:>9.1f}% {disp_str:>10}")

# Detailed comparison for L0+L10 vs L0+L11
print("\n" + "=" * 70)
print("Detailed Comparison: L0+L10 vs L0+L11")
print("=" * 70)

for config_name, layers in [("L0+L10", [0, 10]), ("L0+L11", [0, 11])]:
    protected = protect_layers(layers)

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    print(f"\n{config_name}:")
    print(f"{'Lang':<6} {'PPL':>10} {'Degrad%':>12} {'Disparity':>12}")
    print("-" * 44)

    for l in TEXTS:
        if en_deg > 0:
            disp = deg[l] / en_deg
            print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {disp:>11.2f}x")
        else:
            print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {'N/A':>12}")

# Layer statistics comparison
print("\n" + "=" * 70)
print("Layer Statistics: L10 vs L11")
print("=" * 70)

def get_layer_stats(layer_idx):
    prefix = f"transformer.h.{layer_idx}."
    weights = []
    for name, param in model.named_parameters():
        if name.startswith(prefix) and 'weight' in name:
            weights.append(param.data.flatten())
    all_w = torch.cat(weights)
    return {
        'variance': all_w.var().item(),
        'std': all_w.std().item(),
        'max': all_w.abs().max().item(),
        'sparsity': (all_w.abs() < 0.01).float().mean().item() * 100,
    }

s10 = get_layer_stats(10)
s11 = get_layer_stats(11)

print(f"\n{'Metric':<15} {'Layer 10':>12} {'Layer 11':>12} {'Diff':>12}")
print("-" * 55)
for metric in ['variance', 'std', 'max', 'sparsity']:
    v10, v11 = s10[metric], s11[metric]
    diff = (v10 - v11) / v11 * 100 if v11 != 0 else 0
    if metric == 'sparsity':
        print(f"{metric:<15} {v10:>11.1f}% {v11:>11.1f}% {diff:>11.1f}%")
    else:
        print(f"{metric:<15} {v10:>12.6f} {v11:>12.6f} {diff:>10.1f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
L10 may be better because:
1. It has different variance/statistics than L11
2. L10 + L0 captures both ends of the "variance spectrum"
3. L11 is the final layer before output - may be less critical
   when final LayerNorm (ln_f) is already protected
""")
