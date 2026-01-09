#!/usr/bin/env python3
"""
Exp-054: Attention head granularity analysis
Goal: Are specific attention heads more critical than others?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

gc.collect()

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.eval()

# GPT-2 has 12 heads, head_dim = 64, so c_attn is (768, 2304) = (768, 3*768)
# Each head's Q,K,V is 64 dims

n_heads = 12
head_dim = 64

def ppl(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

# Analyze which heads have highest weight magnitude
print("Analyzing attention head statistics in L0:")

c_attn = dict(model.named_parameters())['transformer.h.0.attn.c_attn.weight']
# Shape: (768, 2304) -> Q, K, V each (768, 768)
q_weights = c_attn[:, :768]
k_weights = c_attn[:, 768:1536]
v_weights = c_attn[:, 1536:]

print(f"\nPer-head weight magnitude (L0 Q projection):")
for head_idx in range(n_heads):
    start = head_idx * head_dim
    end = start + head_dim
    head_weights = q_weights[:, start:end]
    mag = head_weights.abs().mean().item()
    max_val = head_weights.abs().max().item()
    print(f"  Head {head_idx:2d}: mean={mag:.4f}, max={max_val:.4f}")

# Check V weights (often more important for content)
print(f"\nPer-head weight magnitude (L0 V projection):")
for head_idx in range(n_heads):
    start = head_idx * head_dim
    end = start + head_dim
    head_weights = v_weights[:, start:end]
    mag = head_weights.abs().mean().item()
    max_val = head_weights.abs().max().item()
    print(f"  Head {head_idx:2d}: mean={mag:.4f}, max={max_val:.4f}")

# Compare L0 vs L11
print("\n" + "=" * 50)
print("L0 vs L11 attention weight comparison:")

l0_attn = dict(model.named_parameters())['transformer.h.0.attn.c_attn.weight']
l11_attn = dict(model.named_parameters())['transformer.h.11.attn.c_attn.weight']

l0_v = l0_attn[:, 1536:]
l11_v = l11_attn[:, 1536:]

print(f"\nL0 V weights:  mean={l0_v.abs().mean():.4f}, std={l0_v.std():.4f}")
print(f"L11 V weights: mean={l11_v.abs().mean():.4f}, std={l11_v.std():.4f}")

# Quantization sensitivity per head
print("\nDone.")
