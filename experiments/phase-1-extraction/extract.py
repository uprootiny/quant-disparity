#!/usr/bin/env python3
"""
Phase 1: Weight Extraction from Real Model

Stages (run gradually):
  Stage 0: Test model loading
  Stage 1: Single language, single layer
  Stage 2: All languages, single layer
  Stage 3: All languages, all layers

Usage:
    python3 extract.py --stage 0
    python3 extract.py --stage 1 --lang eng
    python3 extract.py --stage 2
    python3 extract.py --stage 3
"""

import argparse
import json
from pathlib import Path

# Check dependencies
def check_deps():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        missing.append("transformers")
    try:
        from scipy import stats
    except ImportError:
        missing.append("scipy")

    if missing:
        print(f"Missing: {missing}")
        print("Install: pip install torch transformers scipy")
        return False
    return True


# Sample sentences per language (native content)
SAMPLES = {
    "eng": "London is the capital and largest city of England and the United Kingdom.",
    "fra": "Paris est la capitale de la France et le chef-lieu de la region Ile-de-France.",
    "deu": "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.",
    "ara": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها.",
    "heb": "ירושלים היא בירת ישראל והעיר הגדולה ביותר בה.",
    "jpn": "東京は日本の首都であり、世界最大の都市圏を形成している。",
    "zho": "北京是中华人民共和国的首都，是全国政治中心。",
    "kor": "서울은 대한민국의 수도이자 최대 도시이다.",
    "rus": "Москва является столицей Российской Федерации и крупнейшим городом страны.",
    "hin": "नई दिल्ली भारत की राजधानी है और दिल्ली राष्ट्रीय राजधानी क्षेत्र का हिस्सा है।",
    "tha": "กรุงเทพมหานครเป็นเมืองหลวงและเมืองที่มีประชากรมากที่สุดของประเทศไทย",
    "vie": "Hanoi la thu do cua Viet Nam va la thanh pho lon thu hai cua ca nuoc.",
    "fin": "Helsinki on Suomen paakaupunki ja samalla maan suurin kaupunki.",
    "tur": "Ankara, Turkiye'nin baskenti ve Istanbul'dan sonra en kalabalik ikinci sehridir.",
}


def stage_0():
    """Stage 0: Just test we can load the model."""
    print("=" * 60)
    print("STAGE 0: Model Loading Test")
    print("=" * 60)

    if not check_deps():
        return False

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    print("Loading BLOOM-560M...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Load model on CPU with low memory
        model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-560m",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

        # Quick test
        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"  Forward pass OK: output shape {outputs.logits.shape}")

        print()
        print("[OK] Stage 0 complete. Model loads successfully.")
        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def stage_1(lang="eng"):
    """Stage 1: Single language, analyze weight statistics."""
    print("=" * 60)
    print(f"STAGE 1: Single Language Analysis ({lang})")
    print("=" * 60)

    if not check_deps():
        return None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy import stats as sp_stats

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    text = SAMPLES.get(lang, SAMPLES["eng"])
    print(f"Text: {text[:50]}...")
    print()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Tokens: {inputs['input_ids'].shape[1]}")

    # Get activations using hooks
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook

    # Register hooks on MLP layers
    hooks = []
    for i, layer in enumerate(model.transformer.h):
        h = layer.mlp.register_forward_hook(hook_fn(f"layer_{i}_mlp"))
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Analyze activations
    print()
    print("Activation Statistics per Layer (MLP output):")
    print("-" * 60)
    print("Layer    Mean       Std       Kurt      Outlier%")

    results = {}
    for name, act in sorted(activations.items()):
        flat = act.flatten().numpy()
        mean = float(flat.mean())
        std = float(flat.std())
        kurt = float(sp_stats.kurtosis(flat))
        outlier = float((abs(flat) > 3 * std).mean() * 100)

        layer_num = name.split("_")[1]
        print(f"{layer_num:>5}   {mean:+.4f}   {std:.4f}   {kurt:+.2f}      {outlier:.2f}%")

        results[name] = {
            "mean": mean,
            "std": std,
            "kurtosis": kurt,
            "outlier_pct": outlier,
        }

    # Aggregate
    all_kurt = [r["kurtosis"] for r in results.values()]
    all_outlier = [r["outlier_pct"] for r in results.values()]

    print()
    print(f"Aggregate: kurtosis={sum(all_kurt)/len(all_kurt):.2f}, outlier={sum(all_outlier)/len(all_outlier):.2f}%")

    return results


def stage_2():
    """Stage 2: All languages, aggregate statistics."""
    print("=" * 60)
    print("STAGE 2: All Languages")
    print("=" * 60)

    results = {}
    for lang in sorted(SAMPLES.keys()):
        print(f"\n--- {lang} ---")
        lang_results = stage_1(lang)
        if lang_results:
            # Aggregate across layers
            all_kurt = [r["kurtosis"] for r in lang_results.values()]
            all_outlier = [r["outlier_pct"] for r in lang_results.values()]
            results[lang] = {
                "mean_kurtosis": sum(all_kurt) / len(all_kurt),
                "mean_outlier_pct": sum(all_outlier) / len(all_outlier),
            }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Lang   Kurtosis   Outlier%")
    for lang in sorted(results.keys()):
        r = results[lang]
        print(f"{lang}    {r['mean_kurtosis']:+.2f}      {r['mean_outlier_pct']:.2f}%")

    # Save
    Path("activations.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to activations.json")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, help="Stage to run (0-3)")
    parser.add_argument("--lang", default="eng", help="Language for stage 1")
    args = parser.parse_args()

    if args.stage == 0:
        stage_0()
    elif args.stage == 1:
        stage_1(args.lang)
    elif args.stage == 2:
        stage_2()
    else:
        print(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
