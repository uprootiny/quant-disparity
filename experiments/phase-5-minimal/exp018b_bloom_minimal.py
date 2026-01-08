#!/usr/bin/env python3
"""
Exp-018b: BLOOM-560M minimal validation
Goal: Quick disparity check on truly multilingual model
Uses minimal memory footprint
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import gc

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-018b: BLOOM Minimal")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading BLOOM-560M...")
    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
    model = AutoModelForCausalLM.from_pretrained(
        'bigscience/bloom-560m',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {total:,} params")

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

    # Baseline
    baseline = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Baseline: en={baseline['en']:.1f}, he={baseline['he']:.1f}")

    # Clone state for comparison
    state_sample = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            state_sample[name] = param.data.clone()
            break

    # Quantize
    print("Quantizing...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                q = torch.round(flat / scale).clamp(-8, 7) * scale
                param.data.copy_(q.view(param.shape))

    gc.collect()

    # Quantized PPL
    quant = {l: ppl(t) for l, t in TEXTS.items()}
    print(f"Quantized: en={quant['en']:.1f}, he={quant['he']:.1f}")

    # Calculate
    deg = {l: (quant[l] - baseline[l]) / baseline[l] * 100 for l in TEXTS}
    disp = deg['he'] / deg['en'] if deg['en'] > 0 else float('inf')

    print(f"\nDegradation:")
    print(f"  en: {deg['en']:+.0f}%")
    print(f"  he: {deg['he']:+.0f}%")
    print(f"  Disparity: {disp:.1f}x")

    print(f"\nComparison to English-centric models:")
    print(f"  GPT-2:      214x")
    print(f"  OPT-125M:   153x")
    print(f"  BLOOM-560M: {disp:.1f}x")

    if disp < 50:
        conclusion = "SIGNIFICANTLY LOWER - multilingual training helps"
    elif disp < 100:
        conclusion = "LOWER than English-centric models"
    else:
        conclusion = "SIMILAR to English-centric models"

    print(f"\nConclusion: {conclusion}")

    end = datetime.now()

    result = {
        "id": "Exp-018b",
        "name": "BLOOM Minimal Validation",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "model": "bigscience/bloom-560m",
        "baseline": baseline,
        "quantized": quant,
        "degradation": deg,
        "disparity": disp,
        "conclusion": conclusion,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp018b_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")

    del model
    gc.collect()
    return result


if __name__ == "__main__":
    main()
