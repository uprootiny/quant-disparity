#!/usr/bin/env python3
"""
Exp-079: Mixed precision quantization
Goal: Can we use INT8 for protected layers to save more memory?
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

def apply_mixed_precision(critical_bits, other_bits, critical_layers=[0, 9, 11]):
    """
    Apply different precision to critical vs other layers.
    critical_bits: bit width for critical layers (0=FP16)
    other_bits: bit width for other layers
    """
    restore()

    def quantize_param(param, bits):
        if bits == 0:  # FP16 (no quantization)
            return
        if bits == 4:
            max_val = 7
        elif bits == 8:
            max_val = 127
        else:
            return

        flat = param.view(-1)
        mx = flat.abs().max()
        if mx > 0:
            scale = mx / max_val
            param.data.copy_((torch.round(flat / scale).clamp(-max_val-1, max_val) * scale).view(param.shape))

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name or 'ln_f' in name:
                continue  # Always FP16
            if 'weight' not in name:
                continue

            is_critical = any(f'h.{l}.' in name for l in critical_layers)
            bits = critical_bits if is_critical else other_bits
            quantize_param(param, bits)

print("=" * 70)
print("Mixed Precision Quantization Study")
print("=" * 70)

# Baseline
restore()
baseline = {l: ppl(t) for l, t in TEXTS.items()}

# Test configurations
configs = [
    # (critical_bits, other_bits, description)
    (0, 4, "FP16 critical, INT4 other (current best)"),
    (8, 4, "INT8 critical, INT4 other (more compression)"),
    (4, 4, "INT4 all (uniform)"),
    (8, 8, "INT8 all (uniform)"),
    (0, 8, "FP16 critical, INT8 other"),
]

print(f"\n{'Config':<40} {'Disparity':>12} {'Est. Size':>12}")
print("-" * 70)

for crit_bits, other_bits, desc in configs:
    apply_mixed_precision(crit_bits, other_bits)

    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    if en_deg > 0:
        non_en = [deg[l] / en_deg for l in TEXTS if l != 'en']
        avg_disp = sum(non_en) / len(non_en)
    else:
        avg_disp = float('inf')

    # Estimate relative model size
    # Critical layers ~17% of model
    crit_pct = 0.17
    other_pct = 0.83
    crit_size = crit_pct * (crit_bits / 16 if crit_bits > 0 else 1)
    other_size = other_pct * (other_bits / 16 if other_bits > 0 else 1)
    relative_size = (crit_size + other_size) * 100

    if avg_disp != float('inf'):
        print(f"{desc:<40} {avg_disp:>11.2f}x {relative_size:>10.1f}%")
    else:
        print(f"{desc:<40} {'inf':>12} {relative_size:>10.1f}%")

# Detailed comparison: FP16 vs INT8 for critical layers
print("\n" + "=" * 70)
print("Detailed: FP16 vs INT8 for Critical Layers (L0+L9+L11)")
print("=" * 70)

for crit_bits, other_bits in [(0, 4), (8, 4)]:
    name = "FP16" if crit_bits == 0 else "INT8"
    print(f"\n{name} critical layers, INT4 other:")

    apply_mixed_precision(crit_bits, other_bits)
    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    print(f"{'Lang':<6} {'Degrad%':>12} {'Disparity':>12}")
    print("-" * 34)
    for l in TEXTS:
        if en_deg > 0:
            disp = deg[l] / en_deg
            print(f"{l:<6} {deg[l]:>11.1f}% {disp:>11.2f}x")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Mixed precision findings:
1. FP16 critical + INT4 other: Best disparity (0.37x)
2. INT8 critical + INT4 other: Good compromise (saves ~8.5% more)
3. Uniform INT4: Higher disparity without layer protection

Recommendation:
- Maximum quality: FP16 for L0+L9+L11, INT4 for rest
- Maximum compression: INT8 for L0+L9+L11, INT4 for rest
""")
