#!/usr/bin/env python3
"""
Exp-019c: Pythia diagnostic
Goal: Understand why disparity is infinite
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-019c: Pythia Diagnostic")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    model.eval()

    # Check tokenization
    print("\nTokenization analysis:")
    for lang, text in TEXTS.items():
        tokens = tokenizer.encode(text)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  {lang}: {len(tokens)} tokens")
        print(f"      {decoded[:10]}...")

    def ppl(text):
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
            return torch.exp(out.loss).item(), out.loss.item()

    # Baseline PPL and loss
    print("\nBaseline perplexity:")
    baseline = {}
    for lang, text in TEXTS.items():
        ppl_val, loss_val = ppl(text)
        baseline[lang] = {'ppl': ppl_val, 'loss': loss_val}
        print(f"  {lang}: PPL={ppl_val:.1f}, loss={loss_val:.3f}")

    # Save state
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # Quantize
    print("\nQuantizing...")
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

    # Quantized PPL
    print("\nQuantized perplexity:")
    quantized = {}
    for lang, text in TEXTS.items():
        ppl_val, loss_val = ppl(text)
        quantized[lang] = {'ppl': ppl_val, 'loss': loss_val}
        print(f"  {lang}: PPL={ppl_val:.1f}, loss={loss_val:.3f}")

    # Calculate degradation properly
    print("\nDegradation analysis:")
    for lang in TEXTS:
        base_ppl = baseline[lang]['ppl']
        quant_ppl = quantized[lang]['ppl']
        if base_ppl > 0:
            deg = (quant_ppl - base_ppl) / base_ppl * 100
        else:
            deg = float('inf')
        print(f"  {lang}: {base_ppl:.1f} -> {quant_ppl:.1f} ({deg:+.0f}%)")

    en_deg = (quantized['en']['ppl'] - baseline['en']['ppl']) / baseline['en']['ppl'] * 100
    he_deg = (quantized['he']['ppl'] - baseline['he']['ppl']) / baseline['he']['ppl'] * 100

    if en_deg > 0:
        disp = he_deg / en_deg
    else:
        disp = float('inf')

    print(f"\nDisparity: {disp:.1f}x")
    print(f"  en_deg: {en_deg:+.1f}%")
    print(f"  he_deg: {he_deg:+.1f}%")

    # Check if the issue is that English PPL is already very high
    if baseline['en']['ppl'] > 1000:
        print("\nDIAGNOSIS: English baseline PPL is very high (>1000)")
        print("  This suggests the model may already be nearly random for English")
        print("  Quantization impact is masked by poor baseline")

    if baseline['he']['ppl'] < baseline['en']['ppl']:
        print("\nDIAGNOSIS: Hebrew PPL is LOWER than English")
        print("  This is unexpected and may indicate tokenization issues")

    end = datetime.now()

    result = {
        "id": "Exp-019c",
        "name": "Pythia Diagnostic",
        "timestamp": end.isoformat(),
        "duration_sec": (end - start).total_seconds(),
        "baseline": baseline,
        "quantized": quantized,
        "en_degradation": en_deg,
        "he_degradation": he_deg,
        "disparity": disp,
        "status": "SUCCESS"
    }

    out_file = Path(__file__).parent / "exp019c_result.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Completed in {(end-start).total_seconds():.1f}s")
    return result


if __name__ == "__main__":
    main()
