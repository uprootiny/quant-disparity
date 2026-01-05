#!/usr/bin/env python3
"""
Exp-001: Single language baseline (English only)
Goal: Establish baseline timing and verify pipeline works
"""

import json
from pathlib import Path
from datetime import datetime
import torch

def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-001: English Baseline")
    print("=" * 40)

    # Load model
    print("Loading GPT-2...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    load_time = (datetime.now() - start).total_seconds()
    print(f"Model loaded in {load_time:.1f}s")

    # Single text
    text = "The quick brown fox jumps over the lazy dog."

    # Baseline perplexity
    print("\nComputing baseline PPL...")
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, labels=inputs['input_ids'])
    baseline_ppl = torch.exp(out.loss).item()
    print(f"Baseline PPL: {baseline_ppl:.2f}")

    # INT4 quantization
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

    # Quantized perplexity
    print("Computing quantized PPL...")
    with torch.no_grad():
        out = model(**inputs, labels=inputs['input_ids'])
    quant_ppl = torch.exp(out.loss).item()
    print(f"Quantized PPL: {quant_ppl:.2f}")

    # Result
    degradation = (quant_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"\nDegradation: {degradation:+.0f}%")

    end = datetime.now()
    duration = (end - start).total_seconds()

    result = {
        "id": "Exp-001",
        "name": "English Baseline",
        "timestamp": end.isoformat(),
        "duration_sec": duration,
        "language": "en",
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quant_ppl,
        "degradation_pct": degradation,
        "status": "SUCCESS"
    }

    # Save
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "exp001_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Completed in {duration:.1f}s")
    print(f"Saved: {out_file}")
    return result


if __name__ == "__main__":
    main()
