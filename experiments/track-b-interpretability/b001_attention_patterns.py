#!/usr/bin/env python3
"""
B-001: Cross-Lingual Attention Pattern Analysis

Question: Do attention heads specialize by language?

Method:
1. Run mBERT/BLOOM on parallel sentences (same meaning, different languages)
2. Extract attention patterns per head
3. Compute cross-lingual similarity using Jensen-Shannon divergence
4. Identify language-specific vs universal heads

Connects to: Belinkov Lab's probing methodology
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import torch

# Parallel sentences (same meaning, different languages)
# From Tatoeba / common phrases
PARALLEL_CORPUS = [
    {
        "en": "The cat sits on the mat.",
        "de": "Die Katze sitzt auf der Matte.",
        "fr": "Le chat est assis sur le tapis.",
        "es": "El gato está sentado en la alfombra.",
        "zh": "猫坐在垫子上。",
        "ar": "القطة تجلس على الحصيرة.",
        "he": "החתול יושב על המחצלת.",
    },
    {
        "en": "I am learning a new language.",
        "de": "Ich lerne eine neue Sprache.",
        "fr": "J'apprends une nouvelle langue.",
        "es": "Estoy aprendiendo un nuevo idioma.",
        "zh": "我正在学习一门新语言。",
        "ar": "أنا أتعلم لغة جديدة.",
        "he": "אני לומד שפה חדשה.",
    },
    {
        "en": "The weather is nice today.",
        "de": "Das Wetter ist heute schön.",
        "fr": "Le temps est beau aujourd'hui.",
        "es": "El tiempo está agradable hoy.",
        "zh": "今天天气很好。",
        "ar": "الطقس جميل اليوم.",
        "he": "מזג האוויר יפה היום.",
    },
    {
        "en": "She reads books every day.",
        "de": "Sie liest jeden Tag Bücher.",
        "fr": "Elle lit des livres chaque jour.",
        "es": "Ella lee libros todos los días.",
        "zh": "她每天都读书。",
        "ar": "هي تقرأ الكتب كل يوم.",
        "he": "היא קוראת ספרים כל יום.",
    },
    {
        "en": "We need water to live.",
        "de": "Wir brauchen Wasser zum Leben.",
        "fr": "Nous avons besoin d'eau pour vivre.",
        "es": "Necesitamos agua para vivir.",
        "zh": "我们需要水来生活。",
        "ar": "نحتاج الماء للعيش.",
        "he": "אנחנו צריכים מים כדי לחיות.",
    },
]


def extract_attention_patterns(model, tokenizer, text: str) -> np.ndarray:
    """
    Extract attention patterns from all heads.
    Returns: (num_layers, num_heads, seq_len, seq_len)
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attention from all layers: (num_layers, batch, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions)
    # Remove batch dimension and convert to numpy
    return attentions[:, 0, :, :, :].cpu().numpy()


def attention_to_distribution(attn: np.ndarray) -> np.ndarray:
    """
    Convert attention matrix to a flat probability distribution.
    Average over sequence positions to get head-level distribution.
    """
    # attn shape: (seq_len, seq_len)
    # Average attention pattern
    avg_attn = attn.mean(axis=0)  # Average over query positions
    # Normalize to sum to 1
    avg_attn = avg_attn / (avg_attn.sum() + 1e-10)
    return avg_attn


def compute_js_divergence(attn1: np.ndarray, attn2: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two attention distributions.
    Lower = more similar. Range: [0, 1] for log base 2.
    """
    # Pad to same length
    max_len = max(len(attn1), len(attn2))
    p1 = np.zeros(max_len)
    p2 = np.zeros(max_len)
    p1[:len(attn1)] = attn1
    p2[:len(attn2)] = attn2

    # Normalize
    p1 = p1 / (p1.sum() + 1e-10)
    p2 = p2 / (p2.sum() + 1e-10)

    # Add small epsilon to avoid log(0)
    p1 = p1 + 1e-10
    p2 = p2 + 1e-10
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()

    return jensenshannon(p1, p2, base=2)


def classify_head(js_values: dict, threshold_universal: float = 0.3,
                  threshold_specific: float = 0.5) -> str:
    """
    Classify a head as universal, language-specific, or mixed.

    Universal: Low JS divergence across all language pairs
    Language-specific: High JS divergence for certain languages
    Mixed: Intermediate behavior
    """
    avg_js = np.mean(list(js_values.values()))
    max_js = np.max(list(js_values.values()))

    if avg_js < threshold_universal:
        return "universal"
    elif max_js > threshold_specific:
        return "language_specific"
    else:
        return "mixed"


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("B-001: Cross-Lingual Attention Pattern Analysis")
    print("=" * 60)

    results = {
        "experiment_id": "B-001",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-B1: Different languages activate different circuit subsets",
        "model_results": {}
    }

    # Test with mBERT (multilingual BERT)
    models_to_test = [
        ("bert-base-multilingual-cased", "mBERT"),
    ]

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    languages = ["en", "de", "fr", "es", "zh", "ar", "he"]

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_label} ({model_name})")
        print("=" * 50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, output_attentions=True)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Get model config
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        print(f"Architecture: {num_layers} layers, {num_heads} heads per layer")

        # Store attention patterns per language per sentence
        lang_attentions = {lang: [] for lang in languages}

        print("\nExtracting attention patterns...")
        for sent_idx, parallel_sent in enumerate(PARALLEL_CORPUS):
            for lang in languages:
                if lang in parallel_sent:
                    text = parallel_sent[lang]
                    attn = extract_attention_patterns(model, tokenizer, text)
                    lang_attentions[lang].append(attn)

        # Compute per-head JS divergence between language pairs
        print("\nComputing cross-lingual JS divergence...")
        head_js_scores = {}  # (layer, head) -> {(lang1, lang2): js_value}

        for layer in range(num_layers):
            for head in range(num_heads):
                head_key = (layer, head)
                head_js_scores[head_key] = {}

                for i, lang1 in enumerate(languages):
                    for lang2 in languages[i+1:]:
                        js_values = []

                        for sent_idx in range(len(PARALLEL_CORPUS)):
                            if sent_idx < len(lang_attentions[lang1]) and sent_idx < len(lang_attentions[lang2]):
                                attn1 = lang_attentions[lang1][sent_idx][layer, head]
                                attn2 = lang_attentions[lang2][sent_idx][layer, head]

                                dist1 = attention_to_distribution(attn1)
                                dist2 = attention_to_distribution(attn2)

                                js = compute_js_divergence(dist1, dist2)
                                js_values.append(js)

                        if js_values:
                            head_js_scores[head_key][(lang1, lang2)] = np.mean(js_values)

        # Classify heads
        print("\nClassifying heads...")
        universal_heads = []
        specific_heads = []
        mixed_heads = []

        layer_universality = {l: {"universal": 0, "specific": 0, "mixed": 0}
                            for l in range(num_layers)}

        for (layer, head), js_values in head_js_scores.items():
            if js_values:
                classification = classify_head(js_values)

                if classification == "universal":
                    universal_heads.append((layer, head))
                    layer_universality[layer]["universal"] += 1
                elif classification == "language_specific":
                    specific_heads.append((layer, head))
                    layer_universality[layer]["specific"] += 1
                else:
                    mixed_heads.append((layer, head))
                    layer_universality[layer]["mixed"] += 1

        total_heads = num_layers * num_heads
        pct_universal = len(universal_heads) / total_heads * 100
        pct_specific = len(specific_heads) / total_heads * 100
        pct_mixed = len(mixed_heads) / total_heads * 100

        print(f"\nHead Classification Summary:")
        print(f"  Universal heads:          {len(universal_heads):3d} ({pct_universal:.1f}%)")
        print(f"  Language-specific heads:  {len(specific_heads):3d} ({pct_specific:.1f}%)")
        print(f"  Mixed heads:              {len(mixed_heads):3d} ({pct_mixed:.1f}%)")

        # Per-layer analysis
        print(f"\nPer-layer distribution:")
        print(f"{'Layer':<8} {'Universal':<12} {'Specific':<12} {'Mixed':<12}")
        print("-" * 44)
        for layer in range(num_layers):
            u = layer_universality[layer]["universal"]
            s = layer_universality[layer]["specific"]
            m = layer_universality[layer]["mixed"]
            print(f"{layer:<8} {u:<12} {s:<12} {m:<12}")

        # Find most language-specific heads
        print(f"\nMost language-specific heads (top 10):")
        head_max_js = []
        for (layer, head), js_values in head_js_scores.items():
            if js_values:
                max_js = max(js_values.values())
                max_pair = max(js_values.items(), key=lambda x: x[1])
                head_max_js.append((layer, head, max_js, max_pair[0]))

        head_max_js.sort(key=lambda x: x[2], reverse=True)
        for layer, head, max_js, pair in head_max_js[:10]:
            print(f"  Layer {layer:2d}, Head {head:2d}: JS={max_js:.4f} ({pair[0]}-{pair[1]})")

        # Language pair similarity matrix
        print(f"\nLanguage pair average JS divergence (lower = more similar):")
        lang_pair_avg = {}
        for (layer, head), js_values in head_js_scores.items():
            for pair, js in js_values.items():
                if pair not in lang_pair_avg:
                    lang_pair_avg[pair] = []
                lang_pair_avg[pair].append(js)

        for pair, values in sorted(lang_pair_avg.items(), key=lambda x: np.mean(x[1])):
            print(f"  {pair[0]}-{pair[1]}: {np.mean(values):.4f}")

        # Test hypothesis H-B1
        print(f"\n{'='*50}")
        print("HYPOTHESIS TEST: H-B1")
        print("='*50")
        print(f"Prediction: >10% of heads are language-specific")
        print(f"Result: {pct_specific:.1f}% language-specific")

        h_b1_supported = pct_specific > 10.0
        print(f"H-B1: {'SUPPORTED' if h_b1_supported else 'NOT SUPPORTED'}")

        # Store results
        results["model_results"][model_label] = {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "total_heads": total_heads,
            "universal_count": len(universal_heads),
            "specific_count": len(specific_heads),
            "mixed_count": len(mixed_heads),
            "pct_universal": pct_universal,
            "pct_specific": pct_specific,
            "pct_mixed": pct_mixed,
            "layer_distribution": layer_universality,
            "top_specific_heads": [(l, h, js) for l, h, js, _ in head_max_js[:10]],
            "language_pair_js": {f"{p[0]}-{p[1]}": np.mean(v) for p, v in lang_pair_avg.items()},
            "h_b1_supported": h_b1_supported
        }

        # Clean up
        del model
        del tokenizer

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"b001_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
