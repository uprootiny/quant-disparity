#!/usr/bin/env python3
"""
C-002: Quantization Fairness Across Languages

Question: Does quantization create or amplify language disparities?

Method:
1. Compute perplexity on parallel corpus at FP16
2. Compute perplexity at simulated INT8/INT4
3. Measure relative degradation per language
4. Correlate with language resource level

Connects to Track A: Tests whether outlier-based quantization disparity
is consistent across different efficiency metrics.

Hypothesis H-C2: Low-resource languages degrade more under quantization.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

# Parallel sentences for perplexity measurement
PARALLEL_CORPUS = {
    "en": [
        "The weather is nice today and I am happy.",
        "She reads books every evening before sleeping.",
        "We need food and water to survive.",
        "The children play in the park after school.",
        "He works at a large company in the city.",
    ],
    "de": [
        "Das Wetter ist heute schön und ich bin glücklich.",
        "Sie liest jeden Abend vor dem Schlafen Bücher.",
        "Wir brauchen Essen und Wasser zum Überleben.",
        "Die Kinder spielen nach der Schule im Park.",
        "Er arbeitet in einer großen Firma in der Stadt.",
    ],
    "fr": [
        "Le temps est beau aujourd'hui et je suis heureux.",
        "Elle lit des livres chaque soir avant de dormir.",
        "Nous avons besoin de nourriture et d'eau pour survivre.",
        "Les enfants jouent dans le parc après l'école.",
        "Il travaille dans une grande entreprise en ville.",
    ],
    "es": [
        "El tiempo está agradable hoy y estoy feliz.",
        "Ella lee libros cada noche antes de dormir.",
        "Necesitamos comida y agua para sobrevivir.",
        "Los niños juegan en el parque después de la escuela.",
        "Él trabaja en una gran empresa en la ciudad.",
    ],
    "zh": [
        "今天天气很好，我很高兴。",
        "她每天晚上睡前都读书。",
        "我们需要食物和水才能生存。",
        "孩子们放学后在公园玩耍。",
        "他在城市的一家大公司工作。",
    ],
    "ar": [
        "الطقس جميل اليوم وأنا سعيد.",
        "تقرأ الكتب كل مساء قبل النوم.",
        "نحتاج إلى الطعام والماء للبقاء على قيد الحياة.",
        "يلعب الأطفال في الحديقة بعد المدرسة.",
        "يعمل في شركة كبيرة في المدينة.",
    ],
    "he": [
        "מזג האוויר יפה היום ואני שמח.",
        "היא קוראת ספרים כל ערב לפני השינה.",
        "אנחנו צריכים אוכל ומים כדי לשרוד.",
        "הילדים משחקים בפארק אחרי בית הספר.",
        "הוא עובד בחברה גדולה בעיר.",
    ],
    "ru": [
        "Сегодня хорошая погода и я счастлив.",
        "Она читает книги каждый вечер перед сном.",
        "Нам нужна еда и вода, чтобы выжить.",
        "Дети играют в парке после школы.",
        "Он работает в крупной компании в городе.",
    ],
    "sw": [
        "Hali ya hewa ni nzuri leo na mimi nina furaha.",
        "Anasoma vitabu kila jioni kabla ya kulala.",
        "Tunahitaji chakula na maji ili kuishi.",
        "Watoto wanacheza kwenye bustani baada ya shule.",
        "Anafanya kazi katika kampuni kubwa mjini.",
    ],
}

# Resource levels (Wikipedia article count proxy, normalized)
RESOURCE_LEVELS = {
    "en": 1.0,
    "de": 0.8,
    "fr": 0.75,
    "es": 0.6,
    "zh": 0.5,
    "ru": 0.45,
    "ar": 0.35,
    "he": 0.2,
    "sw": 0.05,
}


def compute_pseudo_perplexity_mlm(model, tokenizer, text: str) -> float:
    """Compute pseudo-perplexity for masked language models."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]

    total_loss = 0.0
    n_tokens = 0

    for i in range(1, min(input_ids.shape[1] - 1, 20)):  # Limit for speed
        masked_input = input_ids.clone()
        original_token = input_ids[0, i].item()

        if original_token in [tokenizer.cls_token_id, tokenizer.sep_token_id,
                              tokenizer.pad_token_id]:
            continue

        masked_input[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input, attention_mask=inputs["attention_mask"])
            logits = outputs.logits

        loss = F.cross_entropy(
            logits[0, i:i+1],
            torch.tensor([original_token])
        )
        total_loss += loss.item()
        n_tokens += 1

    if n_tokens == 0:
        return float('inf')

    return np.exp(total_loss / n_tokens)


def simulate_quantization_effect(model, bits: int = 8) -> None:
    """
    Simulate quantization by adding noise proportional to weight magnitude.
    This modifies weights in place (call before/after comparison).
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Quantization noise proportional to scale
                abs_max = param.abs().max()
                if abs_max > 0:
                    scale = abs_max / (2 ** (bits - 1) - 1)
                    # Add quantization noise
                    noise = torch.randn_like(param) * scale * 0.5
                    param.add_(noise)


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("C-002: Quantization Fairness Across Languages")
    print("=" * 60)

    results = {
        "experiment_id": "C-002",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-C2: Low-resource languages degrade more under quantization",
        "findings": {}
    }

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import copy
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    models_to_test = [
        ("bert-base-multilingual-cased", "mBERT"),
    ]

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_label}")
        print("=" * 50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_fp16 = AutoModelForMaskedLM.from_pretrained(model_name)
            model_fp16.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Compute FP16 baseline perplexity per language
        print("\n1. Computing FP16 baseline perplexity...")
        lang_ppl_fp16 = {}

        for lang, sentences in PARALLEL_CORPUS.items():
            ppls = []
            for sent in sentences[:3]:  # Limit for speed
                try:
                    ppl = compute_pseudo_perplexity_mlm(model_fp16, tokenizer, sent)
                    if not np.isinf(ppl):
                        ppls.append(ppl)
                except Exception as e:
                    pass

            if ppls:
                lang_ppl_fp16[lang] = np.mean(ppls)
                print(f"   {lang}: PPL = {lang_ppl_fp16[lang]:.2f}")

        # Create quantized copy and compute INT8 perplexity
        print("\n2. Computing INT8 perplexity (simulated)...")
        model_int8 = copy.deepcopy(model_fp16)
        simulate_quantization_effect(model_int8, bits=8)

        lang_ppl_int8 = {}
        for lang, sentences in PARALLEL_CORPUS.items():
            ppls = []
            for sent in sentences[:3]:
                try:
                    ppl = compute_pseudo_perplexity_mlm(model_int8, tokenizer, sent)
                    if not np.isinf(ppl):
                        ppls.append(ppl)
                except Exception:
                    pass

            if ppls:
                lang_ppl_int8[lang] = np.mean(ppls)
                print(f"   {lang}: PPL = {lang_ppl_int8[lang]:.2f}")

        del model_int8

        # Create INT4 simulation
        print("\n3. Computing INT4 perplexity (simulated)...")
        model_int4 = copy.deepcopy(model_fp16)
        simulate_quantization_effect(model_int4, bits=4)

        lang_ppl_int4 = {}
        for lang, sentences in PARALLEL_CORPUS.items():
            ppls = []
            for sent in sentences[:3]:
                try:
                    ppl = compute_pseudo_perplexity_mlm(model_int4, tokenizer, sent)
                    if not np.isinf(ppl):
                        ppls.append(ppl)
                except Exception:
                    pass

            if ppls:
                lang_ppl_int4[lang] = np.mean(ppls)
                print(f"   {lang}: PPL = {lang_ppl_int4[lang]:.2f}")

        del model_int4
        del model_fp16

        # Compute degradation
        print("\n4. Computing degradation ratios...")
        lang_degradation = {}
        common_langs = set(lang_ppl_fp16.keys()) & set(lang_ppl_int8.keys()) & set(lang_ppl_int4.keys())

        for lang in common_langs:
            fp16 = lang_ppl_fp16[lang]
            int8 = lang_ppl_int8[lang]
            int4 = lang_ppl_int4[lang]

            lang_degradation[lang] = {
                "ppl_fp16": fp16,
                "ppl_int8": int8,
                "ppl_int4": int4,
                "deg_int8": (int8 - fp16) / fp16,  # Relative increase
                "deg_int4": (int4 - fp16) / fp16,
                "resource_level": RESOURCE_LEVELS.get(lang, 0.5)
            }

        # Correlation analysis
        print(f"\n{'='*50}")
        print("Correlation Analysis: Resource Level vs Degradation")
        print("=" * 50)

        resource_levels = [lang_degradation[l]["resource_level"] for l in lang_degradation]
        deg_int8 = [lang_degradation[l]["deg_int8"] for l in lang_degradation]
        deg_int4 = [lang_degradation[l]["deg_int4"] for l in lang_degradation]

        if len(resource_levels) >= 4:
            r8, p8 = pearsonr(resource_levels, deg_int8)
            r4, p4 = pearsonr(resource_levels, deg_int4)

            print(f"\nINT8 degradation:")
            print(f"  Pearson r = {r8:.4f}, p = {p8:.4f}")

            print(f"\nINT4 degradation:")
            print(f"  Pearson r = {r4:.4f}, p = {p4:.4f}")

            # Hypothesis test
            h_c2_supported = r4 < -0.3 or r8 < -0.3  # Negative = low resource degrades more

            print(f"\nH-C2 (low resource → more degradation):")
            print(f"  Prediction: r < 0")
            print(f"  Result: INT8 r={r8:.3f}, INT4 r={r4:.3f}")
            print(f"  Status: {'SUPPORTED' if h_c2_supported else 'NOT SUPPORTED'}")

            results["findings"][model_label] = {
                "per_language": lang_degradation,
                "correlation_int8": {"r": r8, "p": p8},
                "correlation_int4": {"r": r4, "p": p4},
                "h_c2_supported": h_c2_supported
            }

        # Summary table
        print(f"\n{'='*50}")
        print("Summary: Quantization Fairness")
        print("=" * 50)
        print(f"{'Lang':<6} {'Resource':<10} {'FP16':<10} {'INT8':<10} {'INT4':<10} {'Deg8%':<10} {'Deg4%':<10}")
        print("-" * 66)

        for lang in sorted(lang_degradation.keys(), key=lambda x: -lang_degradation[x]["resource_level"]):
            d = lang_degradation[lang]
            print(f"{lang:<6} {d['resource_level']:<10.2f} {d['ppl_fp16']:<10.1f} {d['ppl_int8']:<10.1f} {d['ppl_int4']:<10.1f} {d['deg_int8']*100:<10.1f} {d['deg_int4']*100:<10.1f}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"c002_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
