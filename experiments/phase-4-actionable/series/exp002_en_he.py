#!/usr/bin/env python3
"""
Exp-002: English vs Hebrew comparison
Goal: Measure disparity between high and low resource language
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-002: English vs Hebrew")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # Save state for restoration
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(out.loss).item()

    def restore():
        model.load_state_dict(state)

    def quantize():
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' not in name:
                    continue
                flat = param.view(-1)
                mx = flat.abs().max()
                if mx > 0:
                    scale = mx / 7.0
                    q = torch.round(flat / scale).clamp(-8, 7)
                    param.data.copy_((q * scale).view(param.shape))

    # Baseline
    print("\nBaseline PPL:")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        print(f"  {lang}: {baseline[lang]:.2f}")

    # Quantized
    print("\nQuantized PPL:")
    quantize()
    quant = {}
    for lang, text in TEXTS.items():
        quant[lang] = ppl(text)
        print(f"  {lang}: {quant[lang]:.2f}")

    # Degradation
    deg = {lang: (quant[lang] - baseline[lang]) / baseline[lang] * 100
           for lang in TEXTS}
    print("\nDegradation:")
    for lang, d in deg.items():
        print(f"  {lang}: {d:+.0f}%")

    # Disparity
    disparity = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')
    print(f"\nDisparity ratio (he/en): {disparity:.2f}x")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-002",
        "name": "English vs Hebrew",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "baseline_ppl": baseline,
        "quantized_ppl": quant,
        "degradation_pct": deg,
        "disparity_ratio": disparity,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp002_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    print(f"Saved: {out_file}")
    return result


if __name__ == "__main__":
    main()
