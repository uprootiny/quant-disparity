#!/usr/bin/env python3
"""
Exp-061: Cross-model consistency validation
Goal: Does the L0+L11+biases pattern hold for OPT-125M?
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

def apply_config(protect_layers, protect_biases=True, protect_ln=True):
    """Apply quantization with specified protection"""
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Protect biases
            if protect_biases and 'bias' in name:
                protected += param.numel()
                continue
            # Protect specified layers (OPT uses decoder.layers.X)
            if any(f'layers.{l}.' in name for l in protect_layers):
                protected += param.numel()
                continue
            # Protect final LayerNorm
            if protect_ln and 'final_layer_norm' in name:
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

print("=" * 70)
print("OPT-125M: Cross-Model Consistency Validation")
print("=" * 70)

# Baseline
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("\nBaseline PPL:")
for l, v in baseline.items():
    print(f"  {l}: {v:.1f}")

# Test configurations
configs = [
    ("No protection", [], False, False),
    ("Biases only", [], True, False),
    ("L0+L11 (GPT-2 analog)", [0, 11], True, True),
    ("L0+last (0,11)", [0, 11], True, True),
]

print(f"\n{'Config':<25} {'Overhead':>10} {'En Deg':>10} {'Avg Disp':>10}")
print("-" * 60)

for name, layers, biases, ln in configs:
    restore()
    protected = apply_config(layers, biases, ln)
    pct = protected / total * 100

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en' and en_deg > 0]
    avg_disp = sum(non_en_disps) / len(non_en_disps) if non_en_disps else 0

    print(f"{name:<25} {pct:>9.2f}% {en_deg:>9.1f}% {avg_disp:>9.2f}x")

# Detailed breakdown for optimal config
print("\n" + "=" * 70)
print("Detailed Results: L0+L11+biases+ln_f")
print("=" * 70)

restore()
protected = apply_config([0, 11], True, True)
pct = protected / total * 100

q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']

print(f"\n{'Lang':<6} {'PPL':>10} {'Degrad%':>12} {'Disparity':>12}")
print("-" * 44)

for l in TEXTS:
    disp = deg[l] / en_deg if en_deg > 0 else 0
    print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {disp:>11.2f}x")

non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en' and en_deg > 0]
avg_disp = sum(non_en_disps) / len(non_en_disps) if non_en_disps else 0

print(f"\nOverhead: {pct:.2f}%")
print(f"Avg disparity: {avg_disp:.2f}x")

if avg_disp < 5:
    print("CONSISTENT: OPT-125M shows similar pattern to GPT-2")
else:
    print("DIVERGENT: OPT-125M requires different configuration")
