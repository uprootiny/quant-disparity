#!/usr/bin/env python3
"""
Exp-063: OPT-125M optimal configuration
Goal: Test L4+L9 and compare to L0+L11
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
model.eval()

total = sum(p.numel() for p in model.parameters())
state = {k: v.clone() for k, v in model.state_dict().items()}

TEXTS = {
    'en': 'The quick brown fox jumps.',
    'de': 'Der schnelle braune Fuchs springt.',
    'he': 'השועל החום המהיר קופץ.',
    'ar': 'الثعلب البني السريع يقفز.',
    'zh': '敏捷的棕色狐狸跳。',
    'ru': 'Быстрая лиса прыгает.',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def apply_config(protect_layers):
    """Quantize all except specified layers and biases"""
    restore()
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                protected += param.numel()
                continue
            if any(f'layers.{l}.' in name for l in protect_layers):
                protected += param.numel()
                continue
            if 'final_layer_norm' in name:
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
print("OPT-125M: Optimal Configuration Search")
print("=" * 70)

# Baseline
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("\nBaseline PPL:")
for l, v in baseline.items():
    print(f"  {l}: {v:.1f}")

# Test configurations
configs = [
    ("L0+L11 (GPT-2 style)", [0, 11]),
    ("L4+L9 (OPT optimal)", [4, 9]),
    ("L4 only", [4]),
    ("L4+L7+L9", [4, 7, 9]),
    ("L0+L4+L9+L11 (both)", [0, 4, 9, 11]),
]

print(f"\n{'Config':<25} {'Overhead':>10} {'En Deg%':>10} {'Avg Disp':>10}")
print("-" * 60)

for name, layers in configs:
    protected = apply_config(layers)
    pct = protected / total * 100

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    # Handle negative English degradation
    if en_deg <= 0:
        avg_disp = float('inf')
        disp_str = "inf"
    else:
        non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en']
        avg_disp = sum(non_en_disps) / len(non_en_disps)
        disp_str = f"{avg_disp:.2f}x"

    print(f"{name:<25} {pct:>9.2f}% {en_deg:>9.1f}% {disp_str:>10}")

# Best config details
print("\n" + "=" * 70)
print("Best Config Details: L4+L9+biases+ln_f")
print("=" * 70)

protected = apply_config([4, 9])
pct = protected / total * 100

q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']

print(f"\n{'Lang':<6} {'PPL':>10} {'Degrad%':>12} {'Disparity':>12}")
print("-" * 44)

for l in TEXTS:
    if en_deg > 0:
        disp = deg[l] / en_deg
        print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {disp:>11.2f}x")
    else:
        print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {'N/A':>12}")

if en_deg > 0:
    non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en']
    avg_disp = sum(non_en_disps) / len(non_en_disps)
    print(f"\nOverhead: {pct:.2f}%")
    print(f"Avg disparity: {avg_disp:.2f}x")
else:
    print(f"\nOverhead: {pct:.2f}%")
    print("English improved by quantization - disparity undefined")
