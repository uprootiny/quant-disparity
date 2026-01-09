#!/usr/bin/env python3
"""
Exp-045: Text length axis
Goal: Does L0+L11 effectiveness vary with sequence length?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Texts of varying lengths
TEXTS_SHORT = {'en': 'Fox.', 'he': 'שועל.'}
TEXTS_MED = {'en': 'The quick brown fox jumps over the lazy dog.', 'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.'}
TEXTS_LONG = {
    'en': 'The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully in the warm afternoon sun.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן. הכלב ישן בשלווה בשמש החמימה של אחר הצהריים.'
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

def measure_disparity(texts, protect=[]):
    restore()
    if protect:
        quant_except(protect)
    else:
        quant_except([])  # quantize all

    baseline = {l: ppl(t) for l, t in texts.items()}

    # Get quantized values (model already quantized)
    restore()
    quant_except(protect)
    q = {l: ppl(t) for l, t in texts.items()}

    # Calculate disparity
    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in texts}
    en_deg = deg['en']
    he_disp = deg['he'] / en_deg if en_deg > 0 else float('inf')
    return he_disp

# Actually need to recalculate properly
def test_config(texts, protect=[]):
    # Get baseline (FP16)
    restore()
    baseline = {l: ppl(t) for l, t in texts.items()}

    # Quantize
    restore()
    quant_except(protect)
    q = {l: ppl(t) for l, t in texts.items()}

    deg = {l: (q[l] - baseline[l]) / baseline[l] * 100 for l in texts}
    en_deg = deg['en']
    he_disp = deg['he'] / en_deg if en_deg > 0 else float('inf')
    return he_disp, len(tokenizer(texts['en'])['input_ids'])

print(f"{'Length':<12} {'Tokens':>8} {'No prot':>10} {'L0+L11':>10}")
print("-" * 45)

for name, texts in [("short", TEXTS_SHORT), ("medium", TEXTS_MED), ("long", TEXTS_LONG)]:
    disp_none, tokens = test_config(texts, [])
    disp_l0l11, _ = test_config(texts, ["h.0.", "h.11."])
    print(f"{name:<12} {tokens:>8} {disp_none:>9.1f}x {disp_l0l11:>9.1f}x")

print("\nDone.")
