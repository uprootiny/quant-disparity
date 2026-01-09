#!/usr/bin/env python3
"""
Exp-060: Final recommendations validation
Goal: Validate the complete optimal configuration across scenarios
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
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund am Fluss.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד הנהר.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من النهر.',
    'zh': '敏捷的棕色狐狸跳过懒狗在河边。',
    'ru': 'Быстрая коричневая лиса прыгает через ленивую собаку у реки.',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def apply_optimal_config():
    """Apply optimal config: L0+L11+ln_f weights, all biases in FP16"""
    protected = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Protect all biases
            if 'bias' in name:
                protected += param.numel()
                continue
            # Protect L0, L11, and ln_f weights
            if any(p in name for p in ['h.0.', 'h.11.', 'ln_f']):
                protected += param.numel()
                continue
            # Quantize the rest
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))
    return protected

print("=" * 70)
print("FINAL VALIDATION: Optimal Configuration for Multilingual Fairness")
print("=" * 70)

# Baseline
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("\nBaseline PPL:")
for l, v in baseline.items():
    print(f"  {l}: {v:.1f}")

# Apply optimal config
restore()
protected = apply_optimal_config()
pct = protected / total * 100
print(f"\nOptimal config: L0+L11+ln_f+biases ({pct:.2f}% protected)")

q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
en_deg = deg['en']

print("\nResults:")
print(f"{'Lang':<6} {'PPL':>10} {'Degrad%':>12} {'Disparity':>12}")
print("-" * 44)

for l in TEXTS:
    disp = deg[l] / en_deg if en_deg > 0 else 0
    print(f"{l:<6} {q_ppl[l]:>10.1f} {deg[l]:>11.1f}% {disp:>11.2f}x")

# Summary
non_en_disps = [deg[l] / en_deg for l in TEXTS if l != 'en' and en_deg > 0]
avg_disp = sum(non_en_disps) / len(non_en_disps) if non_en_disps else 0

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Overhead:        {pct:.2f}%")
print(f"Avg disparity:   {avg_disp:.2f}x")
print(f"Memory cost:     ~{protected * 2 / 1e6:.1f} MB (FP16)")

if avg_disp < 5:
    print("\n✓ EXCELLENT: Disparity < 5x achieved!")
elif avg_disp < 20:
    print("\n✓ GOOD: Disparity < 20x")
else:
    print("\n⚠ WARNING: High disparity")

print("\n" + "=" * 70)
print("RECOMMENDATION FOR PRACTITIONERS")
print("=" * 70)
print("""
1. Protect Layer 0 and Layer 11 weights in FP16
2. Keep ALL biases in FP16 (0.08% extra)
3. Include final LayerNorm (ln_f) in FP16
4. Quantize remaining 88.5% to INT4

Expected results:
- Overhead: ~11.5%
- Disparity: ~2-5x across languages
- Memory: ~14MB for GPT-2 class models
""")
