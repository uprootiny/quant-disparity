#!/usr/bin/env python3
"""
GPU-001: Test if L0+L11 Pattern Transfers to Llama-2-7B

Hypothesis: Gateway layers exist at input (L0), consolidation (~75%), output (L31)

Requirements:
- GPU with 16GB+ VRAM
- transformers, torch, accelerate

Run time: ~30 minutes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-2-7b-hf"
# For testing without approval: "NousResearch/Llama-2-7b-hf"

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog near the river bank.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund am Flussufer.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד גדת הנהר.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من ضفة النهر.',
    'zh': '敏捷的棕色狐狸跳过河边的懒狗。',
}


def load_model():
    """Load Llama-2-7B with float16."""
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer


def ppl(model, tokenizer, text):
    """Calculate perplexity."""
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()


def simulate_int4_layer(model, layer_idx, protect=False):
    """
    Simulate INT4 quantization for a specific layer.

    Note: This is simulation. Real GPTQ tested in gpu004.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if f'model.layers.{layer_idx}.' not in name:
                continue

            if 'bias' in name or 'layernorm' in name.lower():
                continue

            if protect:
                continue  # Keep FP16

            # INT4 simulation
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))


def test_gpt2_pattern():
    """Test if GPT-2's L0+L31 pattern works for Llama."""
    model, tokenizer = load_model()
    state = {k: v.clone() for k, v in model.state_dict().items()}

    num_layers = model.config.num_hidden_layers  # Should be 32
    print(f"Llama-2-7B has {num_layers} layers")

    # Baseline
    baseline = {l: ppl(model, tokenizer, t) for l, t in TEXTS.items()}
    print(f"\nBaseline PPL (FP16):")
    for l, v in sorted(baseline.items(), key=lambda x: x[1]):
        print(f"  {l}: {v:.1f}")

    # Test configurations (mapping from GPT-2's 12 layers to Llama's 32)
    configs = {
        "no_protection": [],
        "L0+L31": [0, 31],  # Input + output
        "L0+L24+L31": [0, 24, 31],  # Input + 75% + output (GPT-2's L0+L9+L11)
        "L0+L16+L31": [0, 16, 31],  # Input + 50% + output
    }

    print("\n" + "=" * 60)
    print("TESTING GPT-2 PATTERN TRANSFER")
    print("=" * 60)

    for config_name, protect_layers in configs.items():
        model.load_state_dict(state)

        # Quantize all layers except protected
        for layer_idx in range(num_layers):
            protect = layer_idx in protect_layers
            simulate_int4_layer(model, layer_idx, protect=protect)

        # Measure
        quant_ppl = {l: ppl(model, tokenizer, t) for l, t in TEXTS.items()}
        deg = {l: (quant_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        disparities = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS}
        non_en = [v for k, v in disparities.items() if k != 'en' and v != float('inf')]
        avg_disp = sum(non_en) / len(non_en) if non_en else float('inf')

        overhead = len(protect_layers) / num_layers * 100

        print(f"\n{config_name} ({overhead:.1f}% overhead):")
        print(f"  Avg disparity: {avg_disp:.2f}x")
        for l in ['en', 'he', 'ar', 'zh']:
            print(f"  {l}: {disparities[l]:.2f}x")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
If L0+L24+L31 achieves <1.0x disparity:
  → Gateway pattern TRANSFERS to larger models
  → Paper claim strengthened

If L0+L31 achieves <1.0x but L0+L24+L31 doesn't help:
  → Only input/output gateways matter
  → Consolidation layer may be GPT-2 specific

If no config achieves <1.0x:
  → Architecture-specific analysis needed
  → Run gpu002 for full layer sweep
""")


if __name__ == '__main__':
    test_gpt2_pattern()
