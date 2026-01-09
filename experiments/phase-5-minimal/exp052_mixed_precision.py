#!/usr/bin/env python3
"""
Exp-052: Mixed precision quantization
Goal: Test INT4 for some layers, INT8 for others
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {'en': 'Fox jumps.', 'he': 'שועל קופץ.', 'ar': 'ثعلب يقفز.'}

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

def quant_mixed(int8_patterns, fp16_patterns):
    """INT4 default, INT8 for some, FP16 for others."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue

            # FP16 (no quantization)
            if any(p in name for p in fp16_patterns):
                continue

            flat = param.view(-1)
            mx = flat.abs().max()
            if mx == 0:
                continue

            # INT8 for specified patterns
            if any(p in name for p in int8_patterns):
                levels = 128  # INT8
            else:
                levels = 8    # INT4

            scale = mx / levels
            q = torch.round(flat / scale).clamp(-levels, levels-1) * scale
            param.data.copy_(q.view(param.shape))

baseline = {l: ppl(t) for l, t in TEXTS.items()}
print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

configs = [
    ("all_INT4", [], []),
    ("all_INT8", ["h."], []),
    ("L0L11_FP16_rest_INT4", [], ["h.0.", "h.11."]),
    ("L0L11_FP16_rest_INT8", ["h."], ["h.0.", "h.11."]),
    ("L0L11_INT8_rest_INT4", ["h.0.", "h.11."], []),
    ("critical_INT8_anti_INT4", ["h.0.", "h.11.", "h.2."], ["h.1."]),  # Protect good, expose anti-critical
]

print(f"\n{'Config':<25} {'he':>8} {'ar':>8} {'Avg':>8}")
print("-" * 52)

for name, int8_pat, fp16_pat in configs:
    restore()
    quant_mixed(int8_pat, fp16_pat)

    q = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    disp = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS if l != 'en'}
    avg = sum(disp.values()) / len(disp)

    print(f"{name:<25} {disp['he']:>7.1f}x {disp['ar']:>7.1f}x {avg:>7.1f}x")

print("\nDone.")
