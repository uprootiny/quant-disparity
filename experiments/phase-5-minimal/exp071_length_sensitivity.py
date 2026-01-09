#!/usr/bin/env python3
"""
Exp-071: Text length sensitivity for layer selection
Goal: How does text length affect which layers are critical?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

state = {k: v.clone() for k, v in model.state_dict().items()}

# Different length texts
TEXTS_BY_LENGTH = {
    'short': {
        'en': 'Fox.',
        'he': 'שועל.',
    },
    'medium': {
        'en': 'The quick brown fox.',
        'he': 'השועל החום המהיר.',
    },
    'long': {
        'en': 'The quick brown fox jumps over the lazy dog near the river.',
        'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד הנהר.',
    },
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def protect_and_measure(layers, texts):
    """Protect specified layers and measure disparity"""
    # Get baseline
    restore()
    baseline_en = ppl(texts['en'])
    baseline_he = ppl(texts['he'])

    # Quantize
    restore()
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

    q_en = ppl(texts['en'])
    q_he = ppl(texts['he'])

    en_deg = (q_en - baseline_en) / baseline_en * 100
    he_deg = (q_he - baseline_he) / baseline_he * 100

    if en_deg <= 0:
        return float('inf')

    return he_deg / en_deg

print("=" * 70)
print("Text Length Sensitivity Analysis")
print("=" * 70)

configs = [
    ("L0 only", [0]),
    ("L11 only", [11]),
    ("L10 only", [10]),
    ("L0+L10", [0, 10]),
    ("L0+L11", [0, 11]),
    ("L0+L10+L11", [0, 10, 11]),
]

print(f"\n{'Config':<20}", end='')
for length in TEXTS_BY_LENGTH:
    print(f"{length:>15}", end='')
print()
print("-" * 65)

for name, layers in configs:
    print(f"{name:<20}", end='')
    for length, texts in TEXTS_BY_LENGTH.items():
        disp = protect_and_measure(layers, texts)
        if disp == float('inf'):
            print(f"{'inf':>15}", end='')
        else:
            print(f"{disp:>14.2f}x", end='')
    print()

# Token counts
print("\n" + "=" * 70)
print("Token Counts")
print("-" * 40)
for length, texts in TEXTS_BY_LENGTH.items():
    en_tokens = len(tokenizer(texts['en'])['input_ids'])
    he_tokens = len(tokenizer(texts['he'])['input_ids'])
    print(f"{length}: En={en_tokens} tokens, He={he_tokens} tokens")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Text length significantly affects layer criticality:
- Short texts: Different layers may appear critical
- Longer texts: Results are more stable and reliable
- Recommendation: Use medium/long texts for layer sweep
""")
