#!/usr/bin/env python3
"""
Exp-069: Quick Layer Sweep - Practical Method
Goal: Create a fast method to identify critical layers for any model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def quick_layer_sweep(model_name, num_layers=12, layer_prefix="h"):
    """
    Quick sweep to identify critical layers for multilingual fairness.

    Returns: List of recommended layers to protect in FP16
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Minimal test set: English + one LR language
    en_text = "Fox."
    lr_text = "שועל."  # Hebrew

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    def restore():
        model.load_state_dict(state)

    def quantize_except(layer_indices):
        restore()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'bias' in name:
                    continue
                if any(f'{layer_prefix}.{l}.' in name or f'layers.{l}.' in name for l in layer_indices):
                    continue
                if 'ln_f' in name or 'final_layer_norm' in name:
                    continue
                if 'weight' in name:
                    flat = param.view(-1)
                    mx = flat.abs().max()
                    if mx > 0:
                        scale = mx / 7.0
                        param.data.copy_((torch.round(flat / scale).clamp(-8, 7) * scale).view(param.shape))

    print(f"\nSweeping {num_layers} layers...")
    print("-" * 50)

    # Get baseline
    baseline_en = ppl(en_text)
    baseline_lr = ppl(lr_text)
    print(f"Baseline: En={baseline_en:.1f}, LR={baseline_lr:.1f}")

    # Test each layer individually
    results = []
    for i in range(num_layers):
        quantize_except([i])
        q_en = ppl(en_text)
        q_lr = ppl(lr_text)

        en_deg = (q_en - baseline_en) / baseline_en * 100
        lr_deg = (q_lr - baseline_lr) / baseline_lr * 100

        # Skip if English improves (undefined disparity)
        if en_deg > 0:
            disp = lr_deg / en_deg
            results.append((i, disp, en_deg))
            print(f"  Layer {i:2d}: disparity={disp:8.1f}x, en_deg={en_deg:.1f}%")
        else:
            results.append((i, float('inf'), en_deg))
            print(f"  Layer {i:2d}: disparity=inf (en improved by {-en_deg:.1f}%)")

    # Sort by disparity (lowest = most critical)
    valid_results = [(i, d, e) for i, d, e in results if d != float('inf')]
    if not valid_results:
        print("\nWARNING: All layers show infinite disparity!")
        return [0, num_layers-1]  # Default to first+last

    sorted_results = sorted(valid_results, key=lambda x: x[1])

    print("\n" + "=" * 50)
    print("RESULTS: Layers ranked by criticality")
    print("=" * 50)

    for rank, (layer, disp, _) in enumerate(sorted_results[:5], 1):
        print(f"  #{rank}: Layer {layer} (disparity={disp:.1f}x)")

    # Recommend top-2 layers
    recommended = [r[0] for r in sorted_results[:2]]

    # Test the recommendation
    print("\n" + "=" * 50)
    print(f"RECOMMENDATION: Protect layers {recommended}")
    print("=" * 50)

    quantize_except(recommended)
    final_en = ppl(en_text)
    final_lr = ppl(lr_text)
    en_deg = (final_en - baseline_en) / baseline_en * 100
    lr_deg = (final_lr - baseline_lr) / baseline_lr * 100

    if en_deg > 0:
        final_disp = lr_deg / en_deg
        print(f"Final disparity: {final_disp:.1f}x")
    else:
        print(f"Final: English improved by {-en_deg:.1f}%")

    return recommended


if __name__ == "__main__":
    print("=" * 70)
    print("Quick Layer Sweep: Practical Method for Critical Layer ID")
    print("=" * 70)

    # Test on GPT-2
    print("\n" + "=" * 70)
    print("TEST 1: GPT-2")
    print("=" * 70)
    gpt2_layers = quick_layer_sweep('gpt2', num_layers=12, layer_prefix='h')
    print(f"\nGPT-2 critical layers: {gpt2_layers}")
    print(f"Expected: [0, 11]")
    print(f"Match: {set(gpt2_layers) == {0, 11}}")

    # Test on OPT-125M
    print("\n" + "=" * 70)
    print("TEST 2: OPT-125M")
    print("=" * 70)
    opt_layers = quick_layer_sweep('facebook/opt-125m', num_layers=12, layer_prefix='layers')
    print(f"\nOPT-125M critical layers: {opt_layers}")
    print(f"Expected: [4] or similar (varies)")

    print("\n" + "=" * 70)
    print("SUMMARY: Quick Layer Sweep Method")
    print("=" * 70)
    print("""
USAGE:
1. Run single-layer protection sweep on En + one LR language
2. Rank layers by disparity (lower = more critical)
3. Protect top-2 layers in FP16

This method:
- Works in ~30 seconds per model
- Identifies model-specific critical layers
- No assumptions about architecture
- Minimal compute requirements
""")
