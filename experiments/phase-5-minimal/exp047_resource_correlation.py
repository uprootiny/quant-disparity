#!/usr/bin/env python3
"""
Exp-047: Resource level correlation
Goal: Quantify correlation between baseline disparity and improvement from L0+L11
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXTS = {
    'en': 'Fox.', 'de': 'Fuchs.', 'fr': 'Renard.',
    'ru': 'Лиса.', 'he': 'שועל.', 'ar': 'ثعلب.',
    'zh': '狐狸。', 'ja': 'キツネ。', 'ko': '여우.',
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

baseline = {l: ppl(t) for l, t in TEXTS.items()}

# Get disparities
restore()
quant_except([])
q_none = {l: ppl(t) for l, t in TEXTS.items()}

restore()
quant_except(["h.0.", "h.11."])
q_l0l11 = {l: ppl(t) for l, t in TEXTS.items()}

deg_none = {l: (q_none[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
deg_l0l11 = {l: (q_l0l11[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}

en_deg_none = deg_none['en']
en_deg_l0l11 = deg_l0l11['en']

disp_none = {l: deg_none[l] / en_deg_none if en_deg_none > 0 else 0 for l in TEXTS}
disp_l0l11 = {l: deg_l0l11[l] / en_deg_l0l11 if en_deg_l0l11 > 0 else 0 for l in TEXTS}

# Calculate improvement ratio
print("Baseline disparity vs Improvement from L0+L11:")
print(f"\n{'Lang':<6} {'Baseline':>10} {'L0+L11':>10} {'Change':>12}")
print("-" * 42)

data_points = []
for lang in TEXTS:
    if lang == 'en':
        continue
    base = disp_none[lang]
    prot = disp_l0l11[lang]
    change = base - prot  # positive = improvement
    data_points.append((base, change))
    direction = "better" if change > 0 else "worse"
    print(f"{lang:<6} {base:>9.1f}x {prot:>9.1f}x {change:>+10.1f}x ({direction})")

# Correlation
import statistics
bases = [d[0] for d in data_points]
changes = [d[1] for d in data_points]

mean_base = statistics.mean(bases)
mean_change = statistics.mean(changes)

numerator = sum((b - mean_base) * (c - mean_change) for b, c in data_points)
denom_base = sum((b - mean_base) ** 2 for b in bases) ** 0.5
denom_change = sum((c - mean_change) ** 2 for c in changes) ** 0.5

if denom_base > 0 and denom_change > 0:
    correlation = numerator / (denom_base * denom_change)
else:
    correlation = 0

print(f"\nCorrelation (baseline vs improvement): r = {correlation:.3f}")

if correlation > 0.7:
    print("-> Strong positive: L0+L11 helps most where it's needed most")
elif correlation > 0.3:
    print("-> Moderate positive: L0+L11 tends to help high-disparity languages")
elif correlation < -0.3:
    print("-> Negative: L0+L11 helps low-disparity languages more")
else:
    print("-> Weak correlation: Effect varies")

print("\nDone.")
