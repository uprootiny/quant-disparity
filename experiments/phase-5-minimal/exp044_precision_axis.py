#!/usr/bin/env python3
"""
Exp-044: Quantization precision axis
Goal: How does bit-width affect disparity? INT2 vs INT4 vs INT8
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.', 'ar': 'الثعلب يقفز.'}

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

state = {k: v.clone() for k, v in model.state_dict().items()}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def quantize_to_bits(bits, protect_patterns=None):
    """Quantize to specified bit-width."""
    protect_patterns = protect_patterns or []
    levels = 2 ** (bits - 1)  # e.g., INT4 = 8 levels each side

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name or any(p in name for p in protect_patterns):
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / levels
                q = torch.round(flat / scale).clamp(-levels, levels-1) * scale
                param.data.copy_(q.view(param.shape))

baseline = {l: ppl(t) for l, t in TEXTS.items()}
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}, ar={baseline['ar']:.1f}")

# Test different bit-widths
print(f"\n{'Bits':<8} {'he':>8} {'ar':>8} {'Avg':>8}")
print("-" * 35)

for bits in [8, 4, 3, 2]:
    restore()
    quantize_to_bits(bits)
    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
    avg = sum(disp.values()) / len(disp)

    print(f"INT{bits:<5} {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x")

# Now test L0+L11 protection at each bit-width
print(f"\nWith L0+L11 protection:")
print(f"{'Bits':<8} {'he':>8} {'ar':>8} {'Avg':>8}")
print("-" * 35)

for bits in [8, 4, 3, 2]:
    restore()
    quantize_to_bits(bits, ["h.0.", "h.11."])
    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
    avg = sum(disp.values()) / len(disp)

    print(f"INT{bits:<5} {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x")

print("\nDone.")
