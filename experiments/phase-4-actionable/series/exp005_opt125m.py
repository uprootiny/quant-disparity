#!/usr/bin/env python3
"""
Exp-005: OPT-125M scaling test
Goal: Verify disparity pattern holds for different model architecture
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
    print(f"[{start.strftime('%H:%M:%S')}] Exp-005: OPT-125M")
    print("=" * 40)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading OPT-125M...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    model.eval()

    load_time = (datetime.now() - start).total_seconds()
    print(f"Model loaded in {load_time:.1f}s")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        return torch.exp(out.loss).item()

    # Baseline
    print("\nBaseline PPL:")
    baseline = {}
    for lang, text in TEXTS.items():
        baseline[lang] = ppl(text)
        print(f"  {lang}: {baseline[lang]:.2f}")

    # Quantize
    print("\nApplying INT4 quantization...")
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

    # Quantized
    print("Quantized PPL:")
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
    print(f"\nDisparity ratio: {disparity:.2f}x")

    # Compare to GPT-2
    gpt2_disparity = 213.82  # From Exp-004
    print(f"\nComparison:")
    print(f"  GPT-2 (124M):    213.82x")
    print(f"  OPT-125M (125M): {disparity:.2f}x")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-005",
        "name": "OPT-125M Scaling",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "model": "facebook/opt-125m",
        "parameters": params,
        "baseline_ppl": baseline,
        "quantized_ppl": quant,
        "degradation_pct": deg,
        "disparity_ratio": disparity,
        "gpt2_disparity": gpt2_disparity,
        "pattern_consistent": abs(disparity - gpt2_disparity) / gpt2_disparity < 0.5,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp005_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    return result


if __name__ == "__main__":
    main()
