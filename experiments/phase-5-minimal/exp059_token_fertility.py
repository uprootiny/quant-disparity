#!/usr/bin/env python3
"""
Exp-059: Token fertility and disparity
Goal: How does token count affect disparity?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Same semantic content, different languages
TEXTS = {
    'en': 'Hello world',
    'de': 'Hallo Welt',
    'fr': 'Bonjour monde',
    'es': 'Hola mundo',
    'ru': 'Привет мир',
    'ar': 'مرحبا بالعالم',
    'he': 'שלום עולם',
    'zh': '你好世界',
    'ja': 'こんにちは世界',
    'ko': '안녕하세요 세계',
}

print("Token Fertility Analysis:")
print(f"{'Lang':<6} {'Text':<20} {'Tokens':>8} {'Fertility':>10}")
print("-" * 50)

en_tokens = len(tokenizer(TEXTS['en'])['input_ids'])
fertility_data = []

for lang, text in TEXTS.items():
    tokens = len(tokenizer(text)['input_ids'])
    fertility = tokens / en_tokens
    fertility_data.append((lang, text, tokens, fertility))
    print(f"{lang:<6} {text:<20} {tokens:>8} {fertility:>9.2f}x")

# Now test disparity vs fertility correlation
print("\n" + "=" * 60)
print("Disparity vs Token Fertility:")
print(f"{'Lang':<6} {'Fertility':>10} {'Baseline':>10} {'L0+L11':>10}")
print("-" * 40)

# Get baseline PPL
baseline = {l: ppl(t) for l, t in TEXTS.items()}

# Quantize without protection
restore()
quant_except([])
q_none = {l: ppl(t) for l, t in TEXTS.items()}

# Quantize with L0+L11
restore()
quant_except(["h.0.", "h.11."])
q_l0l11 = {l: ppl(t) for l, t in TEXTS.items()}

deg_none = {l: (q_none[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
deg_l0l11 = {l: (q_l0l11[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}

en_deg_none = deg_none['en']
en_deg_l0l11 = deg_l0l11['en']

disp_none = {l: deg_none[l] / en_deg_none if en_deg_none > 0 else 0 for l in TEXTS}
disp_l0l11 = {l: deg_l0l11[l] / en_deg_l0l11 if en_deg_l0l11 > 0 else 0 for l in TEXTS}

for lang, text, tokens, fertility in fertility_data:
    print(f"{lang:<6} {fertility:>9.2f}x {disp_none[lang]:>9.1f}x {disp_l0l11[lang]:>9.1f}x")

# Correlation
fertilities = [f[3] for f in fertility_data if f[0] != 'en']
disps = [disp_none[f[0]] for f in fertility_data if f[0] != 'en']

import statistics
mean_f = statistics.mean(fertilities)
mean_d = statistics.mean(disps)
num = sum((f - mean_f) * (d - mean_d) for f, d in zip(fertilities, disps))
denom_f = sum((f - mean_f)**2 for f in fertilities) ** 0.5
denom_d = sum((d - mean_d)**2 for d in disps) ** 0.5
corr = num / (denom_f * denom_d) if denom_f > 0 and denom_d > 0 else 0

print(f"\nCorrelation (fertility vs baseline disparity): r = {corr:.3f}")
