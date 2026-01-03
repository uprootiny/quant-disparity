#!/usr/bin/env python3
"""
C-001b: Tokenizer Efficiency Gap (Lightweight version)

Question: Do tokenizer inefficiencies vary by language resource level?

Method:
1. Measure fertility (tokens/word) across languages
2. Compare BERT vs DistilBERT tokenizers
3. Correlate with language resource level

This is a CPU-light experiment that doesn't require loading full models.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats


def compute_fertility(tokenizer, texts: list) -> float:
    """Compute average tokens per word."""
    total_tokens = 0
    total_words = 0

    for text in texts:
        words = text.split()
        tokens = tokenizer.tokenize(text)
        total_words += len(words)
        total_tokens += len(tokens)

    return total_tokens / total_words if total_words > 0 else 0


def main():
    print("="*60)
    print("C-001b: Tokenizer Efficiency Gap")
    print("="*60)

    from transformers import AutoTokenizer

    # Sample texts per language
    test_texts = {
        # High-resource European
        "en": [
            "The weather is nice today and I am happy.",
            "Scientists discovered a new species in the rainforest.",
            "The economic situation requires careful consideration.",
        ],
        "de": [
            "Das Wetter ist heute schön und ich bin glücklich.",
            "Wissenschaftler entdeckten eine neue Art im Regenwald.",
            "Die wirtschaftliche Situation erfordert sorgfältige Überlegung.",
        ],
        "fr": [
            "Le temps est beau aujourd'hui et je suis content.",
            "Les scientifiques ont découvert une nouvelle espèce.",
            "La situation économique nécessite une réflexion approfondie.",
        ],
        # Medium-resource
        "vi": [
            "Thời tiết hôm nay đẹp và tôi rất vui.",
            "Các nhà khoa học đã phát hiện một loài mới.",
            "Tình hình kinh tế cần được xem xét cẩn thận.",
        ],
        "ar": [
            "الطقس جميل اليوم وأنا سعيد.",
            "اكتشف العلماء نوعا جديدا في الغابة.",
            "الوضع الاقتصادي يتطلب دراسة متأنية.",
        ],
        # Lower-resource
        "hi": [
            "आज मौसम अच्छा है और मैं खुश हूं।",
            "वैज्ञानिकों ने एक नई प्रजाति की खोज की।",
            "आर्थिक स्थिति पर सावधानीपूर्वक विचार आवश्यक है।",
        ],
        "sw": [
            "Hali ya hewa ni nzuri leo na mimi nina furaha.",
            "Wanasayansi waligundua spishi mpya msituni.",
            "Hali ya uchumi inahitaji kuzingatiwa kwa makini.",
        ],
        "th": [
            "วันนี้อากาศดีและฉันมีความสุข",
            "นักวิทยาศาสตร์ค้นพบสายพันธุ์ใหม่ในป่าฝน",
            "สถานการณ์ทางเศรษฐกิจต้องพิจารณาอย่างรอบคอบ",
        ],
    }

    resource_level = {
        "en": "high", "de": "high", "fr": "high",
        "vi": "medium", "ar": "medium",
        "hi": "low", "sw": "low", "th": "low",
    }

    tokenizers = [
        ("bert-base-multilingual-cased", "mBERT"),
        ("distilbert-base-multilingual-cased", "DistilmBERT"),
    ]

    results = {}

    for tok_id, tok_name in tokenizers:
        print(f"\n### {tok_name} ###")
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
        results[tok_name] = {}

        for lang, texts in test_texts.items():
            fertility = compute_fertility(tokenizer, texts)
            results[tok_name][lang] = {
                "fertility": fertility,
                "resource": resource_level[lang],
            }
            print(f"  {lang} ({resource_level[lang]:>6}): fertility = {fertility:.2f}")

    # Analysis
    print("\n" + "="*60)
    print("EFFICIENCY GAP ANALYSIS")
    print("="*60)

    # Compare fertility by resource level
    for tok_name, lang_data in results.items():
        print(f"\n### {tok_name} by Resource Level ###")

        by_resource = {"high": [], "medium": [], "low": []}
        for lang, data in lang_data.items():
            by_resource[data["resource"]].append(data["fertility"])

        for level, fertilities in by_resource.items():
            if fertilities:
                mean_f = np.mean(fertilities)
                print(f"  {level:>6}: mean fertility = {mean_f:.2f}")

    # Efficiency gap metric
    print("\n### Efficiency Gap (Fertility Ratio) ###")

    mbert_data = results.get("mBERT", {})
    if mbert_data:
        high_fert = [d["fertility"] for d in mbert_data.values() if d["resource"] == "high"]
        low_fert = [d["fertility"] for d in mbert_data.values() if d["resource"] == "low"]

        if high_fert and low_fert:
            high_mean = np.mean(high_fert)
            low_mean = np.mean(low_fert)
            efficiency_gap = low_mean / high_mean

            print(f"High-resource mean fertility: {high_mean:.2f}")
            print(f"Low-resource mean fertility:  {low_mean:.2f}")
            print(f"Efficiency gap (low/high):    {efficiency_gap:.2f}x")

            # Correlation with resource level
            fertilities = [d["fertility"] for d in mbert_data.values()]
            resource_scores = [{"high": 3, "medium": 2, "low": 1}[d["resource"]]
                             for d in mbert_data.values()]

            r, p = sp_stats.pearsonr(fertilities, resource_scores)
            print(f"\nCorrelation (fertility vs resource): r = {r:.3f}, p = {p:.4f}")

            if r < -0.5 and p < 0.1:
                verdict = "EFFICIENCY GAP CONFIRMED: Low-resource = higher fertility"
            elif r > 0.5 and p < 0.1:
                verdict = "EFFICIENCY GAP REVERSED: High-resource = higher fertility"
            else:
                verdict = "EFFICIENCY GAP INCONCLUSIVE"

            print(f"\nVerdict: {verdict}")
        else:
            efficiency_gap = None
            verdict = "INSUFFICIENT DATA"
    else:
        efficiency_gap = None
        verdict = "NO DATA"

    # Save results
    output = {
        "experiment": "C-001b",
        "results": {
            tok: {lang: data["fertility"] for lang, data in langs.items()}
            for tok, langs in results.items()
        },
        "efficiency_gap": efficiency_gap,
        "verdict": verdict,
    }
    Path("c001b_results.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to c001b_results.json")


if __name__ == "__main__":
    main()
