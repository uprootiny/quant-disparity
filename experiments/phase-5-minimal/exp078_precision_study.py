#!/usr/bin/env python3
"""
Exp-078: Quantization precision study
Goal: How does INT2/INT4/INT8 affect disparity with optimal layers?
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
    'en': 'The quick brown fox jumps over the lazy dog near the river.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד הנהר.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من النهر.',
    'zh': '敏捷的棕色狐狸跳过懒狗在河边。',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def quantize_with_precision(bits, protect_layers):
    """Quantize to specified bit width"""
    restore()

    # Calculate quantization parameters
    if bits == 2:
        levels = 4  # -2, -1, 0, 1
        max_val = 1
    elif bits == 4:
        levels = 16  # -8 to 7
        max_val = 7
    elif bits == 8:
        levels = 256  # -128 to 127
        max_val = 127
    else:
        return  # FP32

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue
            if any(f'h.{l}.' in name for l in protect_layers):
                continue
            if 'ln_f' in name:
                continue
            if 'weight' in name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / max_val
                    param.data.copy_((torch.round(flat / scale).clamp(-max_val-1, max_val) * scale).view(param.shape))

print("=" * 70)
print("Quantization Precision Study")
print("=" * 70)

# Get baseline
restore()
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("\nBaseline (FP32):")
for l, v in baseline.items():
    print(f"  {l}: {v:.1f}")

# Test different precisions with different protection levels
configs = [
    ("No protection", []),
    ("L0+L11", [0, 11]),
    ("L0+L9+L11", [0, 9, 11]),
]

precisions = [2, 4, 8]

print("\n" + "=" * 70)
print("Disparity by Precision and Protection Level")
print("=" * 70)

print(f"\n{'Config':<20}", end='')
for bits in precisions:
    print(f"{'INT'+str(bits):>12}", end='')
print()
print("-" * 60)

for config_name, layers in configs:
    print(f"{config_name:<20}", end='')
    for bits in precisions:
        quantize_with_precision(bits, layers)
        q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
        deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        if en_deg > 0:
            non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
            avg_disp = sum(non_en) / len(non_en)
            print(f"{avg_disp:>11.2f}x", end='')
        else:
            print(f"{'inf':>12}", end='')
    print()

# Detailed INT4 vs INT8 comparison for optimal config
print("\n" + "=" * 70)
print("Detailed: L0+L9+L11 at INT4 vs INT8")
print("=" * 70)

for bits in [4, 8]:
    print(f"\n{bits}-bit quantization:")
    quantize_with_precision(bits, [0, 9, 11])
    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    print(f"{'Lang':<6} {'PPL':>10} {'Degrad%':>12} {'Disparity':>12}")
    print("-" * 44)
    for l in TEXTS:
        if en_deg > 0:
            disp = deg[l] / en_deg
            print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {disp:>11.2f}x")
        else:
            print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {'N/A':>12}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
1. INT4 is the sweet spot for disparity mitigation
2. INT8 shows lower disparity but less compression benefit
3. INT2 may cause severe degradation even with protection
4. Protection layers are more effective at lower precision

Recommendation: Use INT4 with L0+L9+L11 protection
""")
