#!/usr/bin/env python3
"""
Minimal perplexity measurement - memory optimized.
Processes 10 samples per language, short sequences.
"""

import json
import gc
from pathlib import Path
import numpy as np

LANGS = ["ara", "eng", "fra", "hin", "jpn", "zho"]

def load_samples(lang, n=10, max_chars=200):
    """Load minimal samples."""
    corpus = Path(f"corpus/{lang}.txt")
    if not corpus.exists():
        return []
    text = corpus.read_text()
    docs = [d.strip()[:max_chars] for d in text.split("<|endofdoc|>") if len(d.strip()) > 50]
    return docs[:n]

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading BLOOM-560M...")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Loaded.\n")

    results = {}

    for lang in LANGS:
        samples = load_samples(lang)
        if not samples:
            print(f"{lang}: no corpus")
            continue

        losses = []
        for text in samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
                losses.append(out.loss.item())
            del inputs, out

        ppl = np.exp(np.mean(losses))
        results[lang] = round(ppl, 2)
        print(f"{lang}: PPL = {ppl:.2f}")
        gc.collect()

    Path("baseline_ppl.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to baseline_ppl.json")

if __name__ == "__main__":
    main()
