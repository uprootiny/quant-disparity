#!/usr/bin/env python3
"""
Simple, fast experiment to validate disparity pattern.
Uses GPT-2 with minimal text and single preservation test.
"""

import json
from pathlib import Path
from datetime import datetime
import torch

# Very short texts for speed
TEXTS = {
    'en': 'The cat sits on the mat.',
    'he': 'החתול יושב על המזרן.',
}


def main():
    print("Simple Disparity Test")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Save state
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(out.loss).item()

    def restore():
        model.load_state_dict(state)

    def quant_int4():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'weight' not in n:
                    continue
                flat = p.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7)
                    p.data.copy_((q * scale).view(p.shape))

    # Baseline
    print("\nBaseline:")
    base_en = ppl(TEXTS['en'])
    base_he = ppl(TEXTS['he'])
    print(f"  en: {base_en:.2f}")
    print(f"  he: {base_he:.2f}")

    # Quantized
    print("\nINT4 Quantized:")
    quant_int4()
    q_en = ppl(TEXTS['en'])
    q_he = ppl(TEXTS['he'])
    print(f"  en: {q_en:.2f}")
    print(f"  he: {q_he:.2f}")

    # Degradation
    deg_en = (q_en - base_en) / base_en * 100
    deg_he = (q_he - base_he) / base_he * 100
    print(f"\nDegradation:")
    print(f"  en: {deg_en:+.0f}%")
    print(f"  he: {deg_he:+.0f}%")

    # Disparity
    disparity = deg_he / deg_en if deg_en > 0 else float('inf')
    print(f"\nDisparity ratio: {disparity:.2f}x")

    # Save
    result = {
        "id": "simple-disparity",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2",
        "baseline": {"en": base_en, "he": base_he},
        "quantized": {"en": q_en, "he": q_he},
        "degradation_pct": {"en": deg_en, "he": deg_he},
        "disparity_ratio": disparity,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "results" / f"simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {out_file}")
    print("\n✓ Experiment completed successfully")
    return result


if __name__ == "__main__":
    main()
