#!/usr/bin/env python3
"""
Exp-046: Language family axis
Goal: Test L0+L11 across diverse language families
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Languages by family (using "fox jumps" equivalent)
TEXTS = {
    # Germanic
    'en': 'The fox jumps.',
    'de': 'Der Fuchs springt.',
    # Romance
    'fr': 'Le renard saute.',
    'es': 'El zorro salta.',
    # Slavic
    'ru': 'Лиса прыгает.',
    'pl': 'Lis skacze.',
    # Semitic
    'he': 'השועל קופץ.',
    'ar': 'الثعلب يقفز.',
    # Sinitic
    'zh': '狐狸跳。',
    # Japonic
    'ja': 'キツネが跳ぶ。',
    # Koreanic
    'ko': '여우가 뛴다.',
}

FAMILIES = {
    'Germanic': ['en', 'de'],
    'Romance': ['fr', 'es'],
    'Slavic': ['ru', 'pl'],
    'Semitic': ['he', 'ar'],
    'CJK': ['zh', 'ja', 'ko'],
}

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

def quant_except(patterns):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name or any(p in name for p in patterns):
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

# Get baselines
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("Baseline PPL (token fertility indicator):")
for l, v in sorted(baseline.items(), key=lambda x: x[1]):
    tokens = len(tokenizer(TEXTS[l])['input_ids'])
    print(f"  {l}: {v:.1f} ({tokens} tokens)")

# Quantize without protection
restore()
quant_except([])
q_none = {l: ppl(t) for l, t in TEXTS.items()}

# Quantize with L0+L11
restore()
quant_except(["h.0.", "h.11."])
q_l0l11 = {l: ppl(t) for l, t in TEXTS.items()}

# Calculate disparities
deg_none = {l: (q_none[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
deg_l0l11 = {l: (q_l0l11[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}

en_deg_none = deg_none['en']
en_deg_l0l11 = deg_l0l11['en']

disp_none = {l: deg_none[l] / en_deg_none if en_deg_none > 0 else 0 for l in TEXTS}
disp_l0l11 = {l: deg_l0l11[l] / en_deg_l0l11 if en_deg_l0l11 > 0 else 0 for l in TEXTS}

print(f"\n{'Lang':<6} {'Family':<10} {'Baseline':>10} {'L0+L11':>10} {'Improve':>10}")
print("-" * 50)

for lang in TEXTS:
    family = next((f for f, langs in FAMILIES.items() if lang in langs), "?")
    improve = disp_none[lang] / disp_l0l11[lang] if disp_l0l11[lang] > 0 else float('inf')
    print(f"{lang:<6} {family:<10} {disp_none[lang]:>9.1f}x {disp_l0l11[lang]:>9.1f}x {improve:>9.1f}x")

# Summary by family
print("\nBy Family (average):")
for family, langs in FAMILIES.items():
    avg_none = sum(disp_none[l] for l in langs) / len(langs)
    avg_l0l11 = sum(disp_l0l11[l] for l in langs) / len(langs)
    print(f"  {family:<10}: {avg_none:.1f}x → {avg_l0l11:.1f}x")

print("\nDone.")
