#!/usr/bin/env python3
"""
B-003: Circuit Ablation by Language

Question: Which circuit components are critical for each language?

Method:
1. Identify top-k important heads per language (via activation magnitude)
2. Ablate heads and measure perplexity increase
3. Compare ablation sensitivity across languages

Hypothesis H-B3: Low-resource languages are more sensitive to ablation (less redundancy).

Connects to: Belinkov's causal mediation analysis
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, ttest_ind
import copy

# Test sentences per language
TEST_SENTENCES = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
    ],
    "de": [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Sie verkauft Muscheln am Meeresufer.",
        "Eine Reise von tausend Meilen beginnt mit einem einzigen Schritt.",
        "Sein oder Nichtsein, das ist hier die Frage.",
        "Es ist nicht alles Gold, was glänzt.",
    ],
    "fr": [
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Elle vend des coquillages au bord de la mer.",
        "Un voyage de mille lieues commence par un premier pas.",
        "Être ou ne pas être, telle est la question.",
        "Tout ce qui brille n'est pas or.",
    ],
    "zh": [
        "敏捷的棕色狐狸跳过懒惰的狗。",
        "她在海边卖贝壳。",
        "千里之行始于足下。",
        "生存还是毁灭，这是个问题。",
        "闪光的不一定是金子。",
    ],
    "ar": [
        "الثعلب البني السريع يقفز فوق الكلب الكسول.",
        "تبيع صدفات البحر على شاطئ البحر.",
        "رحلة الألف ميل تبدأ بخطوة واحدة.",
        "أكون أو لا أكون، هذا هو السؤال.",
        "ليس كل ما يلمع ذهباً.",
    ],
    "he": [
        "השועל החום המהיר קופץ מעל הכלב העצלן.",
        "היא מוכרת צדפים על שפת הים.",
        "מסע של אלף מילים מתחיל בצעד אחד.",
        "להיות או לא להיות, זאת השאלה.",
        "לא כל הנוצץ זהב.",
    ],
    "sw": [
        "Mbweha wa kahawia haraka anaruka juu ya mbwa wavivu.",
        "Yeye anauza kombe za bahari pwani.",
        "Safari ya maili elfu inaanza na hatua moja.",
        "Kuwa au kutokuwa, hiyo ndiyo swali.",
        "Si dhahabu yote inayong'aa.",
    ],
}

# Resource level estimates
RESOURCE_LEVELS = {
    "en": 1.0,
    "de": 0.8,
    "fr": 0.75,
    "zh": 0.6,
    "ar": 0.4,
    "he": 0.25,
    "sw": 0.05,
}


def compute_pseudo_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute pseudo-perplexity using masked language modeling.
    For each token, mask it and compute cross-entropy loss.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]

    total_loss = 0.0
    n_tokens = 0

    for i in range(1, input_ids.shape[1] - 1):  # Skip [CLS] and [SEP]
        masked_input = input_ids.clone()
        masked_input[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input, attention_mask=inputs["attention_mask"])
            logits = outputs.logits

        # Get loss for the masked position
        target = input_ids[0, i]
        loss = F.cross_entropy(logits[0, i:i+1], target.unsqueeze(0))
        total_loss += loss.item()
        n_tokens += 1

    if n_tokens == 0:
        return float('inf')

    return np.exp(total_loss / n_tokens)


def get_head_importance(model, tokenizer, text: str, layer: int, head: int) -> float:
    """
    Compute importance of a head based on activation magnitude.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention weights for the specified head
    attention = outputs.attentions[layer][0, head]  # (seq_len, seq_len)

    # Importance = mean attention magnitude
    return attention.abs().mean().item()


def ablate_head(model, layer: int, head: int):
    """
    Zero out a specific attention head.
    Returns a function to restore the head.
    """
    # Store original parameters
    layer_module = model.encoder.layer[layer].attention.self

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # Store original values
    start_idx = head * head_dim
    end_idx = (head + 1) * head_dim

    original_query = layer_module.query.weight.data[:, start_idx:end_idx].clone()
    original_key = layer_module.key.weight.data[:, start_idx:end_idx].clone()
    original_value = layer_module.value.weight.data[:, start_idx:end_idx].clone()

    # Zero out
    layer_module.query.weight.data[:, start_idx:end_idx] = 0
    layer_module.key.weight.data[:, start_idx:end_idx] = 0
    layer_module.value.weight.data[:, start_idx:end_idx] = 0

    def restore():
        layer_module.query.weight.data[:, start_idx:end_idx] = original_query
        layer_module.key.weight.data[:, start_idx:end_idx] = original_key
        layer_module.value.weight.data[:, start_idx:end_idx] = original_value

    return restore


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("B-003: Circuit Ablation by Language")
    print("=" * 60)

    results = {
        "experiment_id": "B-003",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-B3: Low-resource languages are more sensitive to ablation",
        "model_results": {}
    }

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    models_to_test = [
        ("bert-base-multilingual-cased", "mBERT"),
    ]

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_label} ({model_name})")
        print("=" * 50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        print(f"Architecture: {num_layers} layers, {num_heads} heads per layer")

        # Step 1: Compute baseline perplexity per language
        print("\n1. Computing baseline perplexity...")
        baseline_ppl = {}

        for lang, sentences in TEST_SENTENCES.items():
            ppls = []
            for sent in sentences:
                try:
                    ppl = compute_pseudo_perplexity(model, tokenizer, sent)
                    ppls.append(ppl)
                except Exception as e:
                    print(f"  Warning: {lang} failed: {e}")
            if ppls:
                baseline_ppl[lang] = np.mean(ppls)
                print(f"  {lang}: PPL = {baseline_ppl[lang]:.2f}")

        # Step 2: Identify important heads per language
        print("\n2. Computing head importance per language...")
        lang_head_importance = {lang: {} for lang in TEST_SENTENCES}

        for lang, sentences in TEST_SENTENCES.items():
            print(f"  Processing {lang}...")
            for layer in range(num_layers):
                for head in range(num_heads):
                    importances = []
                    for sent in sentences[:2]:  # Use subset for speed
                        try:
                            imp = get_head_importance(model, tokenizer, sent, layer, head)
                            importances.append(imp)
                        except Exception:
                            pass
                    if importances:
                        lang_head_importance[lang][(layer, head)] = np.mean(importances)

        # Step 3: Ablate top-k heads and measure sensitivity
        print("\n3. Ablating top heads and measuring sensitivity...")
        top_k = 5  # Ablate top 5 most important heads per language

        lang_sensitivity = {}

        for lang in TEST_SENTENCES:
            if lang not in lang_head_importance or not lang_head_importance[lang]:
                continue

            # Sort heads by importance for this language
            sorted_heads = sorted(
                lang_head_importance[lang].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            print(f"\n  {lang}: Top {top_k} heads: {[h[0] for h in sorted_heads]}")

            ppl_increases = []

            for (layer, head), importance in sorted_heads:
                try:
                    # Ablate head
                    restore_fn = ablate_head(model.bert, layer, head)

                    # Compute perplexity with ablated head
                    ablated_ppls = []
                    for sent in TEST_SENTENCES[lang][:2]:
                        ppl = compute_pseudo_perplexity(model, tokenizer, sent)
                        ablated_ppls.append(ppl)

                    # Restore head
                    restore_fn()

                    avg_ablated_ppl = np.mean(ablated_ppls)
                    ppl_increase = (avg_ablated_ppl - baseline_ppl[lang]) / baseline_ppl[lang]
                    ppl_increases.append(ppl_increase)

                    print(f"    Layer {layer}, Head {head}: +{ppl_increase*100:.1f}% PPL")

                except Exception as e:
                    print(f"    Layer {layer}, Head {head}: FAILED ({e})")

            if ppl_increases:
                lang_sensitivity[lang] = {
                    "mean_ppl_increase": np.mean(ppl_increases),
                    "max_ppl_increase": np.max(ppl_increases),
                    "resource_level": RESOURCE_LEVELS.get(lang, 0.5),
                    "top_heads_ablated": top_k,
                    "individual_increases": ppl_increases
                }

        # Step 4: Analyze correlation with resource level
        print(f"\n{'='*50}")
        print("Correlation Analysis: Resource Level vs Ablation Sensitivity")
        print("=" * 50)

        resource_levels = []
        sensitivities = []

        for lang, data in lang_sensitivity.items():
            resource_levels.append(data["resource_level"])
            sensitivities.append(data["mean_ppl_increase"])

        if len(resource_levels) >= 3:
            r, p = pearsonr(resource_levels, sensitivities)
            print(f"\nCorrelation: r = {r:.3f}, p = {p:.4f}")
            print(f"Prediction (H-B3): r < 0 (low resource → higher sensitivity)")

            h_b3_supported = r < 0 and p < 0.1
            print(f"H-B3: {'SUPPORTED' if h_b3_supported else 'NOT SUPPORTED'}")

            results["model_results"][model_label] = {
                "correlation": r,
                "p_value": p,
                "h_b3_supported": h_b3_supported,
                "per_language": lang_sensitivity
            }

        # Summary table
        print(f"\n{'='*50}")
        print("Summary: Ablation Sensitivity by Language")
        print("=" * 50)
        print(f"{'Lang':<6} {'Resource':<10} {'Mean ↑PPL':<12} {'Max ↑PPL':<12}")
        print("-" * 40)

        for lang in sorted(lang_sensitivity.keys(), key=lambda x: -lang_sensitivity[x]["resource_level"]):
            data = lang_sensitivity[lang]
            print(f"{lang:<6} {data['resource_level']:<10.2f} {data['mean_ppl_increase']*100:<12.1f}% {data['max_ppl_increase']*100:<12.1f}%")

        # High vs low resource comparison
        high_resource = [d["mean_ppl_increase"] for l, d in lang_sensitivity.items() if d["resource_level"] > 0.5]
        low_resource = [d["mean_ppl_increase"] for l, d in lang_sensitivity.items() if d["resource_level"] <= 0.5]

        if high_resource and low_resource:
            t_stat, t_p = ttest_ind(high_resource, low_resource)
            print(f"\nHigh vs Low resource t-test:")
            print(f"  High resource mean: {np.mean(high_resource)*100:.1f}%")
            print(f"  Low resource mean:  {np.mean(low_resource)*100:.1f}%")
            print(f"  t = {t_stat:.3f}, p = {t_p:.4f}")

        del model
        del tokenizer

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"b003_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
