#!/usr/bin/env python3
"""
C-001: Distillation Disparity Analysis

Question: Does knowledge distillation amplify language gaps?

Method:
1. Load BERT-base-multilingual and DistilmBERT
2. Evaluate on multilingual benchmark (XNLI subset)
3. Compare performance drop across language resource levels

Hypothesis: Low-resource languages suffer more from distillation.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats


def get_model_embeddings(model_name: str, texts: list) -> np.ndarray:
    """Get sentence embeddings from a model."""
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=128)
            outputs = model(**inputs)
            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(emb[0])

    return np.array(embeddings)


def compute_representation_quality(embeddings: np.ndarray) -> float:
    """Compute representation quality via isotropy."""
    # Higher isotropy = better distributed representations
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Isotropy: how uniform are eigenvalues?
    if len(eigenvalues) == 0:
        return 0.0
    normalized = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    max_entropy = np.log(len(normalized))

    return entropy / max_entropy if max_entropy > 0 else 0.0


def main():
    print("="*60)
    print("C-001: Distillation Disparity Analysis")
    print("="*60)

    # Sample texts per language (small test set)
    # In full experiment, use XNLI or similar
    test_texts = {
        # High-resource
        "en": [
            "The weather is nice today.",
            "I went to the store yesterday.",
            "Science helps us understand the world.",
        ],
        "de": [
            "Das Wetter ist heute schön.",
            "Ich ging gestern in den Laden.",
            "Wissenschaft hilft uns die Welt zu verstehen.",
        ],
        "fr": [
            "Le temps est beau aujourd'hui.",
            "Je suis allé au magasin hier.",
            "La science nous aide à comprendre le monde.",
        ],
        # Low-resource (using languages mBERT was trained on but with less data)
        "sw": [
            "Hali ya hewa ni nzuri leo.",
            "Nilikwenda dukani jana.",
            "Sayansi inatusaidia kuelewa dunia.",
        ],
        "vi": [
            "Thời tiết hôm nay đẹp.",
            "Tôi đã đi đến cửa hàng hôm qua.",
            "Khoa học giúp chúng ta hiểu thế giới.",
        ],
        "ar": [
            "الطقس جميل اليوم.",
            "ذهبت إلى المتجر أمس.",
            "العلم يساعدنا على فهم العالم.",
        ],
    }

    resource_level = {
        "en": "high", "de": "high", "fr": "high",
        "sw": "low", "vi": "medium", "ar": "medium",
    }

    models = [
        ("bert-base-multilingual-cased", "BERT-mBERT"),
        ("distilbert-base-multilingual-cased", "DistilmBERT"),
    ]

    results = {}

    for model_id, model_name in models:
        print(f"\n### {model_name} ###")
        results[model_name] = {}

        for lang, texts in test_texts.items():
            try:
                embeddings = get_model_embeddings(model_id, texts)
                quality = compute_representation_quality(embeddings)
                results[model_name][lang] = {
                    "quality": quality,
                    "resource": resource_level[lang],
                }
                print(f"  {lang} ({resource_level[lang]}): quality = {quality:.4f}")
            except Exception as e:
                print(f"  {lang}: ERROR - {e}")
                results[model_name][lang] = {"quality": 0, "resource": resource_level[lang]}

    # Analysis
    print("\n" + "="*60)
    print("DISPARITY ANALYSIS")
    print("="*60)

    bert_results = results.get("BERT-mBERT", {})
    distil_results = results.get("DistilmBERT", {})

    if bert_results and distil_results:
        # Compute drop per language
        drops = {}
        for lang in bert_results:
            bert_q = bert_results[lang]["quality"]
            distil_q = distil_results[lang]["quality"]
            drop = bert_q - distil_q
            drops[lang] = {
                "drop": drop,
                "resource": bert_results[lang]["resource"],
            }

        print(f"\n{'Language':<10} {'Resource':<10} {'BERT':<10} {'Distil':<10} {'Drop'}")
        print("-"*50)

        for lang, data in drops.items():
            bert_q = bert_results[lang]["quality"]
            distil_q = distil_results[lang]["quality"]
            print(f"{lang:<10} {data['resource']:<10} {bert_q:<10.4f} {distil_q:<10.4f} {data['drop']:.4f}")

        # Group by resource level
        high_drops = [d["drop"] for d in drops.values() if d["resource"] == "high"]
        low_drops = [d["drop"] for d in drops.values() if d["resource"] == "low"]
        med_drops = [d["drop"] for d in drops.values() if d["resource"] == "medium"]

        print("\n--- Summary by Resource Level ---")
        if high_drops:
            print(f"High-resource mean drop: {np.mean(high_drops):.4f}")
        if med_drops:
            print(f"Medium-resource mean drop: {np.mean(med_drops):.4f}")
        if low_drops:
            print(f"Low-resource mean drop: {np.mean(low_drops):.4f}")

        # Disparity ratio
        if high_drops and low_drops:
            high_mean = np.mean(high_drops)
            low_mean = np.mean(low_drops)
            if high_mean != 0:
                disparity_ratio = low_mean / high_mean
                print(f"\nDisparity ratio (low/high): {disparity_ratio:.2f}")

                if disparity_ratio > 1.5:
                    verdict = "H-C1 SUPPORTED: Low-resource languages suffer more"
                elif disparity_ratio < 0.67:
                    verdict = "H-C1 REVERSED: High-resource languages suffer more"
                else:
                    verdict = "H-C1 INCONCLUSIVE: Similar impact across resource levels"
            else:
                verdict = "H-C1 INCONCLUSIVE: No measurable drop in high-resource"
                disparity_ratio = float('nan')
        else:
            verdict = "H-C1 INSUFFICIENT DATA"
            disparity_ratio = float('nan')

        print(f"\nVerdict: {verdict}")

        # Save results
        output = {
            "experiment": "C-001",
            "models": list(results.keys()),
            "languages": list(drops.keys()),
            "drops": {k: v["drop"] for k, v in drops.items()},
            "disparity_ratio": disparity_ratio if not np.isnan(disparity_ratio) else None,
            "verdict": verdict,
        }
        Path("c001_results.json").write_text(json.dumps(output, indent=2))
        print("\nSaved to c001_results.json")


if __name__ == "__main__":
    main()
