#!/usr/bin/env python3
"""
GPU-002: Full Layer Sweep on Llama-2-7B

Find critical layers for Llama architecture (32 layers).

Requirements:
- GPU with 16GB+ VRAM
- transformers, torch, accelerate

Run time: ~2 hours (test 32 layers)
"""

import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-2-7b-hf"

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog near the river bank.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ליד גדת הנהר.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول بالقرب من ضفة النهر.',
    'zh': '敏捷的棕色狐狸跳过河边的懒狗。',
}


def load_model():
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
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()


def quantize_except(model, state, protect_layers):
    """Quantize all layers except those in protect_layers."""
    model.load_state_dict(state)
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip biases and layernorms
            if 'bias' in name or 'layernorm' in name.lower() or 'norm' in name.lower():
                continue

            # Check if this layer is protected
            is_protected = False
            for layer_idx in protect_layers:
                if f'model.layers.{layer_idx}.' in name:
                    is_protected = True
                    break

            if is_protected:
                continue

            # INT4 simulation
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7
                param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))


def main():
    model, tokenizer = load_model()
    state = {k: v.clone() for k, v in model.state_dict().items()}

    num_layers = model.config.num_hidden_layers
    print(f"Llama-2-7B: {num_layers} layers")

    # Baseline
    baseline = {l: ppl(model, tokenizer, t) for l, t in TEXTS.items()}
    print(f"\nBaseline PPL: en={baseline['en']:.1f}")

    # Single layer sweep
    print(f"\nSweeping {num_layers} layers...")
    results = []

    for layer_idx in range(num_layers):
        quantize_except(model, state, protect_layers={layer_idx})

        quant_ppl = {l: ppl(model, tokenizer, t) for l, t in TEXTS.items()}
        deg = {l: (quant_ppl[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
        en_deg = deg['en']

        disparities = {l: deg[l] / en_deg if en_deg > 0 else float('inf') for l in TEXTS}
        non_en = [v for k, v in disparities.items() if k != 'en' and v != float('inf')]
        avg_disp = sum(non_en) / len(non_en) if non_en else float('inf')

        results.append({
            'layer': layer_idx,
            'disparity': avg_disp,
            'en_deg': en_deg,
            'disparities': disparities,
        })

        print(f"  L{layer_idx:2d}: {avg_disp:>8.2f}x")

    # Sort by criticality
    results.sort(key=lambda x: x['disparity'])

    print("\n" + "=" * 60)
    print("RESULTS (sorted by criticality)")
    print("=" * 60)

    for i, r in enumerate(results[:10]):
        print(f"#{i+1:2d}: L{r['layer']:2d} = {r['disparity']:.2f}x")

    # Save results
    output = {
        'model': MODEL,
        'timestamp': datetime.now().isoformat(),
        'num_layers': num_layers,
        'baseline': baseline,
        'results': results,
        'top_3': [r['layer'] for r in results[:3]],
    }

    with open('gpu002_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to gpu002_results.json")
    print(f"\nRECOMMENDATION: Protect layers {output['top_3']}")

    # Compare to GPT-2 pattern
    gpt2_equiv = [0, int(num_layers * 0.75), num_layers - 1]
    print(f"\nGPT-2 equivalent (L0 + 75% + last): {gpt2_equiv}")


if __name__ == '__main__':
    main()
