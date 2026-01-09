#!/usr/bin/env python3
"""
Exp-074: Final Summary and Recommendations
Goal: Comprehensive validation of all findings
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
    'ja': '素早い茶色の狐が怠惰な犬を飛び越えた。',
    'ko': '빠른 갈색 여우가 게으른 개를 뛰어넘었다.',
}

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

def restore():
    model.load_state_dict(state)

def apply_config(name):
    """Apply named configuration and return overhead%"""
    restore()
    protected = 0

    if name == "none":
        # Quantize everything
        pass
    elif name == "biases_only":
        pass  # Biases protected by default
    elif name == "L0+L11":
        layers = [0, 11]
    elif name == "L0+L9+L11":
        layers = [0, 9, 11]
    else:
        layers = []

    with torch.no_grad():
        for param_name, param in model.named_parameters():
            # Always protect biases
            if 'bias' in param_name:
                protected += param.numel()
                continue

            # Protect specified layers
            if name not in ["none", "biases_only"]:
                if any(f'h.{l}.' in param_name for l in layers):
                    protected += param.numel()
                    continue

            # Protect final LayerNorm
            if name not in ["none"]:
                if 'ln_f' in param_name:
                    protected += param.numel()
                    continue

            # Quantize the rest
            if 'weight' in param_name:
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    return protected / total * 100

print("=" * 80)
print("FINAL SUMMARY: Multilingual Quantization Disparity Research")
print("73 Experiments | GPT-2 Model")
print("=" * 80)

# Baseline
restore()
baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("\n1. BASELINE PERPLEXITY (FP32)")
print("-" * 50)
for l, v in baseline.items():
    tokens = len(tokenizer(TEXTS[l])['input_ids'])
    print(f"  {l}: {v:>8.1f} PPL ({tokens} tokens)")

# Test configurations
configs = ["none", "biases_only", "L0+L11", "L0+L9+L11"]

print("\n" + "=" * 80)
print("2. CONFIGURATION COMPARISON")
print("=" * 80)

results = {}
for config in configs:
    overhead = apply_config(config)
    q_ppl = {l: ppl(t) for l, t in TEXTS.items()}
    deg = {l: (q_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    en_deg = deg['en']

    disparities = {}
    for l in TEXTS:
        if en_deg > 0:
            disparities[l] = deg[l] / en_deg
        else:
            disparities[l] = float('inf')

    non_en_avg = sum(v for k, v in disparities.items() if k != 'en') / (len(TEXTS) - 1)
    results[config] = {
        'overhead': overhead,
        'en_deg': en_deg,
        'avg_disp': non_en_avg,
        'disparities': disparities,
    }

print(f"\n{'Config':<15} {'Overhead':>10} {'En Degrad':>12} {'Avg Disp':>12}")
print("-" * 55)
for config in configs:
    r = results[config]
    print(f"{config:<15} {r['overhead']:>9.2f}% {r['en_deg']:>10.1f}% {r['avg_disp']:>11.2f}x")

# Per-language breakdown for optimal config
print("\n" + "=" * 80)
print("3. OPTIMAL CONFIG (L0+L9+L11) PER-LANGUAGE BREAKDOWN")
print("=" * 80)

r = results["L0+L9+L11"]
print(f"\n{'Lang':<6} {'Disparity':>12} {'Assessment':>20}")
print("-" * 40)
for l, d in r['disparities'].items():
    if l == 'en':
        assess = "Reference"
    elif d < 1.0:
        assess = "EXCELLENT (< 1.0x)"
    elif d < 2.0:
        assess = "GOOD (< 2.0x)"
    elif d < 5.0:
        assess = "ACCEPTABLE"
    else:
        assess = "NEEDS WORK"
    print(f"{l:<6} {d:>11.2f}x {assess:>20}")

# Key numbers
print("\n" + "=" * 80)
print("4. KEY FINDINGS")
print("=" * 80)
print(f"""
A. DISPARITY WITHOUT MITIGATION:
   - Hebrew:  {results['none']['disparities']['he']:.0f}x worse than English
   - Arabic:  {results['none']['disparities']['ar']:.0f}x worse than English
   - Chinese: {results['none']['disparities']['zh']:.0f}x worse than English

B. OPTIMAL MITIGATION (L0+L9+L11 + biases + ln_f):
   - Overhead: {results['L0+L9+L11']['overhead']:.2f}%
   - Average disparity: {results['L0+L9+L11']['avg_disp']:.2f}x
   - Improvement: {(results['none']['avg_disp'] - results['L0+L9+L11']['avg_disp']) / results['none']['avg_disp'] * 100:.0f}%

C. MINIMUM VIABLE (L0+L11 + biases + ln_f):
   - Overhead: {results['L0+L11']['overhead']:.2f}%
   - Average disparity: {results['L0+L11']['avg_disp']:.2f}x
""")

print("=" * 80)
print("5. RECOMMENDATIONS FOR PRACTITIONERS")
print("=" * 80)
print("""
FOR GPT-2 / GPT-2-LIKE ARCHITECTURES:

1. MINIMUM RECOMMENDED:
   - Protect: Layer 0, Layer 11, all biases, final LayerNorm
   - Overhead: ~11.5%
   - Expected disparity: ~1.0x

2. OPTIMAL:
   - Protect: Layer 0, Layer 9, Layer 11, all biases, final LayerNorm
   - Overhead: ~17%
   - Expected disparity: ~0.7x

3. DO NOT:
   - Use random weight selection (catastrophic: 44-327x)
   - Use magnitude-based selection (catastrophic: 125,480x)
   - Protect token embeddings (harmful)
   - Trust short-text experiments (misleading)

4. FOR OTHER ARCHITECTURES:
   - Run quick layer sweep with medium-length texts
   - Layer criticality is model-specific
   - OPT-125M needs different layers (L4+L7)
""")

print("=" * 80)
print("END OF SUMMARY")
print("=" * 80)
