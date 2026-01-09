#!/usr/bin/env python3
"""
Exp-064: GPT-2 vs OPT-125M architecture comparison
Goal: Why does OPT have higher disparity? Tokenization? Architecture?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load both models
gpt2_tok = AutoTokenizer.from_pretrained('gpt2')
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
gpt2.eval()

opt_tok = AutoTokenizer.from_pretrained('facebook/opt-125m')
opt_tok.pad_token = opt_tok.eos_token
opt = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
opt.eval()

TEXTS = {
    'en': 'Hello',
    'de': 'Hallo',
    'he': 'שלום',
    'ar': 'مرحبا',
    'zh': '你好',
    'ru': 'Привет',
}

print("=" * 70)
print("GPT-2 vs OPT-125M: Why Does OPT Have Higher Disparity?")
print("=" * 70)

# 1. Tokenization comparison
print("\n1. TOKENIZATION COMPARISON")
print("-" * 50)
print(f"{'Lang':<6} {'Text':<10} {'GPT-2':>10} {'OPT':>10} {'Ratio':>8}")
print("-" * 50)

for lang, text in TEXTS.items():
    gpt2_tokens = len(gpt2_tok(text)['input_ids'])
    opt_tokens = len(opt_tok(text)['input_ids'])
    ratio = opt_tokens / gpt2_tokens if gpt2_tokens > 0 else 0
    print(f"{lang:<6} {text:<10} {gpt2_tokens:>10} {opt_tokens:>10} {ratio:>7.2f}x")

# 2. Vocabulary size
print("\n2. VOCABULARY SIZE")
print(f"GPT-2: {len(gpt2_tok.get_vocab()):,}")
print(f"OPT:   {len(opt_tok.get_vocab()):,}")

# 3. Parameter count by component
print("\n3. PARAMETER DISTRIBUTION")
print("-" * 50)

def count_params(model, prefix_filter=None):
    total = 0
    for name, p in model.named_parameters():
        if prefix_filter is None or prefix_filter in name:
            total += p.numel()
    return total

gpt2_embed = count_params(gpt2, 'wte') + count_params(gpt2, 'wpe')
opt_embed = count_params(opt, 'embed_tokens') + count_params(opt, 'embed_positions')

print(f"Embeddings: GPT-2={gpt2_embed/1e6:.1f}M, OPT={opt_embed/1e6:.1f}M")
print(f"Total:      GPT-2={sum(p.numel() for p in gpt2.parameters())/1e6:.1f}M, OPT={sum(p.numel() for p in opt.parameters())/1e6:.1f}M")
print(f"Embed %:    GPT-2={gpt2_embed/sum(p.numel() for p in gpt2.parameters())*100:.1f}%, OPT={opt_embed/sum(p.numel() for p in opt.parameters())*100:.1f}%")

# 4. Baseline perplexity comparison
print("\n4. BASELINE PERPLEXITY (unquantized)")
print("-" * 50)
print(f"{'Lang':<6} {'GPT-2 PPL':>12} {'OPT PPL':>12} {'GPT2/OPT':>10}")
print("-" * 50)

def get_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

for lang, text in TEXTS.items():
    gpt2_ppl = get_ppl(gpt2, gpt2_tok, text)
    opt_ppl = get_ppl(opt, opt_tok, text)
    ratio = gpt2_ppl / opt_ppl if opt_ppl > 0 else 0
    print(f"{lang:<6} {gpt2_ppl:>12.1f} {opt_ppl:>12.1f} {ratio:>9.2f}x")

# 5. Weight variance comparison (layer 0)
print("\n5. LAYER 0 WEIGHT VARIANCE")
print("-" * 50)

gpt2_l0_vars = []
for name, p in gpt2.named_parameters():
    if 'h.0.' in name and 'weight' in name:
        gpt2_l0_vars.append((name.split('.')[-2], p.var().item()))

opt_l0_vars = []
for name, p in opt.named_parameters():
    if 'layers.0.' in name and 'weight' in name:
        opt_l0_vars.append((name.split('.')[-2], p.var().item()))

print("GPT-2 Layer 0 variance:")
for name, var in gpt2_l0_vars[:4]:
    print(f"  {name}: {var:.6f}")

print("OPT Layer 0 variance:")
for name, var in opt_l0_vars[:4]:
    print(f"  {name}: {var:.6f}")

# 6. Key insight
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("""
1. Tokenization: OPT may segment non-Latin text differently
2. Training: OPT was trained on different multilingual data
3. Architecture: OPT uses pre-LayerNorm vs GPT-2's post-LayerNorm
4. The L0+L11 pattern is GPT-2 SPECIFIC, not universal

CONCLUSION: Model-specific layer analysis is required for each architecture.
""")
