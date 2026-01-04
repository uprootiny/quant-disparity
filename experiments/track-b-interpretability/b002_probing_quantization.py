#!/usr/bin/env python3
"""
B-002: Probing Quantization Effects

Question: Does quantization affect linguistic probe accuracy?

Method:
1. Train probing classifiers on BLOOM/mBERT FP16 representations
2. Extract representations from quantized models (INT8, simulated INT4)
3. Measure probe accuracy drop per language
4. Correlate with resource level

Hypothesis: Low-resource languages lose more probe accuracy under quantization.

Connects to: Belinkov's probing methodology
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr

# POS tag mappings (Universal Dependencies tagset, simplified)
POS_TAGS = {
    "NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, "PRON": 4,
    "DET": 5, "ADP": 6, "CONJ": 7, "PUNCT": 8, "OTHER": 9
}

# Synthetic labeled data for probing (word, POS tag)
# In production, use UD Treebanks
PROBE_DATA = {
    "en": [
        ("cat", "NOUN"), ("dog", "NOUN"), ("house", "NOUN"), ("book", "NOUN"),
        ("run", "VERB"), ("eat", "VERB"), ("sleep", "VERB"), ("walk", "VERB"),
        ("big", "ADJ"), ("small", "ADJ"), ("fast", "ADJ"), ("slow", "ADJ"),
        ("quickly", "ADV"), ("slowly", "ADV"), ("very", "ADV"), ("well", "ADV"),
        ("he", "PRON"), ("she", "PRON"), ("it", "PRON"), ("they", "PRON"),
        ("the", "DET"), ("a", "DET"), ("this", "DET"), ("that", "DET"),
    ],
    "de": [
        ("Katze", "NOUN"), ("Hund", "NOUN"), ("Haus", "NOUN"), ("Buch", "NOUN"),
        ("laufen", "VERB"), ("essen", "VERB"), ("schlafen", "VERB"), ("gehen", "VERB"),
        ("groß", "ADJ"), ("klein", "ADJ"), ("schnell", "ADJ"), ("langsam", "ADJ"),
        ("schnell", "ADV"), ("langsam", "ADV"), ("sehr", "ADV"), ("gut", "ADV"),
        ("er", "PRON"), ("sie", "PRON"), ("es", "PRON"), ("wir", "PRON"),
        ("der", "DET"), ("die", "DET"), ("das", "DET"), ("ein", "DET"),
    ],
    "fr": [
        ("chat", "NOUN"), ("chien", "NOUN"), ("maison", "NOUN"), ("livre", "NOUN"),
        ("courir", "VERB"), ("manger", "VERB"), ("dormir", "VERB"), ("marcher", "VERB"),
        ("grand", "ADJ"), ("petit", "ADJ"), ("rapide", "ADJ"), ("lent", "ADJ"),
        ("vite", "ADV"), ("lentement", "ADV"), ("très", "ADV"), ("bien", "ADV"),
        ("il", "PRON"), ("elle", "PRON"), ("nous", "PRON"), ("vous", "PRON"),
        ("le", "DET"), ("la", "DET"), ("un", "DET"), ("une", "DET"),
    ],
    "zh": [
        ("猫", "NOUN"), ("狗", "NOUN"), ("房子", "NOUN"), ("书", "NOUN"),
        ("跑", "VERB"), ("吃", "VERB"), ("睡", "VERB"), ("走", "VERB"),
        ("大", "ADJ"), ("小", "ADJ"), ("快", "ADJ"), ("慢", "ADJ"),
        ("快地", "ADV"), ("慢慢", "ADV"), ("很", "ADV"), ("好", "ADV"),
        ("他", "PRON"), ("她", "PRON"), ("它", "PRON"), ("我们", "PRON"),
        ("这", "DET"), ("那", "DET"), ("一个", "DET"), ("每", "DET"),
    ],
    "ar": [
        ("قطة", "NOUN"), ("كلب", "NOUN"), ("بيت", "NOUN"), ("كتاب", "NOUN"),
        ("يجري", "VERB"), ("يأكل", "VERB"), ("ينام", "VERB"), ("يمشي", "VERB"),
        ("كبير", "ADJ"), ("صغير", "ADJ"), ("سريع", "ADJ"), ("بطيء", "ADJ"),
        ("بسرعة", "ADV"), ("ببطء", "ADV"), ("جدا", "ADV"), ("حسنا", "ADV"),
        ("هو", "PRON"), ("هي", "PRON"), ("هم", "PRON"), ("نحن", "PRON"),
        ("ال", "DET"), ("هذا", "DET"), ("ذلك", "DET"), ("كل", "DET"),
    ],
    "he": [
        ("חתול", "NOUN"), ("כלב", "NOUN"), ("בית", "NOUN"), ("ספר", "NOUN"),
        ("רץ", "VERB"), ("אוכל", "VERB"), ("ישן", "VERB"), ("הולך", "VERB"),
        ("גדול", "ADJ"), ("קטן", "ADJ"), ("מהיר", "ADJ"), ("איטי", "ADJ"),
        ("מהר", "ADV"), ("לאט", "ADV"), ("מאוד", "ADV"), ("טוב", "ADV"),
        ("הוא", "PRON"), ("היא", "PRON"), ("הם", "PRON"), ("אנחנו", "PRON"),
        ("ה", "DET"), ("זה", "DET"), ("זאת", "DET"), ("כל", "DET"),
    ],
    "sw": [  # Swahili - low resource
        ("paka", "NOUN"), ("mbwa", "NOUN"), ("nyumba", "NOUN"), ("kitabu", "NOUN"),
        ("kukimbia", "VERB"), ("kula", "VERB"), ("kulala", "VERB"), ("kutembea", "VERB"),
        ("kubwa", "ADJ"), ("ndogo", "ADJ"), ("haraka", "ADJ"), ("polepole", "ADJ"),
        ("haraka", "ADV"), ("polepole", "ADV"), ("sana", "ADV"), ("vizuri", "ADV"),
        ("yeye", "PRON"), ("sisi", "PRON"), ("wao", "PRON"), ("mimi", "PRON"),
        ("hii", "DET"), ("hilo", "DET"), ("kila", "DET"), ("baadhi", "DET"),
    ],
}

# Resource level estimates (Wikipedia article count proxy)
RESOURCE_LEVELS = {
    "en": 1.0,    # High
    "de": 0.8,    # High
    "fr": 0.75,   # High
    "zh": 0.6,    # Medium-High
    "ar": 0.4,    # Medium
    "he": 0.25,   # Medium-Low
    "sw": 0.05,   # Low
}


def get_word_embedding(model, tokenizer, word: str, layer: int = -1) -> np.ndarray:
    """Extract contextual embedding for a word from specified layer."""
    inputs = tokenizer(word, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states from specified layer
    hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)
    # Average over tokens (handles subword tokenization)
    embedding = hidden_states[0].mean(dim=0).cpu().numpy()

    return embedding


def simulate_quantization(embedding: np.ndarray, bits: int = 8) -> np.ndarray:
    """
    Simulate quantization effect on embedding.
    Maps values to discrete levels based on bit width.
    """
    # Compute scale and zero point (symmetric quantization)
    abs_max = np.abs(embedding).max()
    scale = abs_max / (2 ** (bits - 1) - 1)

    if scale == 0:
        return embedding

    # Quantize
    quantized = np.round(embedding / scale)
    quantized = np.clip(quantized, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

    # Dequantize
    dequantized = quantized * scale

    return dequantized


def train_probe(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, float]:
    """Train a probing classifier and return it with cross-val accuracy."""
    probe = LogisticRegression(max_iter=1000, multi_class='multinomial')
    scores = cross_val_score(probe, X, y, cv=min(5, len(y) // 2), scoring='accuracy')
    probe.fit(X, y)
    return probe, scores.mean()


def run_experiment():
    """Main experiment execution."""
    print("=" * 60)
    print("B-002: Probing Quantization Effects")
    print("=" * 60)

    results = {
        "experiment_id": "B-002",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "H-B2 variant: Low-resource languages lose more probe accuracy under quantization",
        "model_results": {}
    }

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return None

    models_to_test = [
        ("bert-base-multilingual-cased", "mBERT"),
    ]

    bit_widths = [16, 8, 4]  # FP16 baseline, INT8, INT4

    for model_name, model_label in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_label} ({model_name})")
        print("=" * 50)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        hidden_dim = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")

        # Extract embeddings for each language
        print("\nExtracting embeddings...")
        lang_embeddings = {}
        lang_labels = {}

        for lang, data in PROBE_DATA.items():
            embeddings = []
            labels = []

            for word, pos in data:
                try:
                    emb = get_word_embedding(model, tokenizer, word, layer=-1)
                    embeddings.append(emb)
                    labels.append(POS_TAGS.get(pos, POS_TAGS["OTHER"]))
                except Exception as e:
                    print(f"  Warning: Failed to embed '{word}' ({lang}): {e}")

            if embeddings:
                lang_embeddings[lang] = np.array(embeddings)
                lang_labels[lang] = np.array(labels)
                print(f"  {lang}: {len(embeddings)} samples")

        # Train probes and evaluate under different quantization levels
        print("\nTraining probes and evaluating quantization effects...")
        lang_results = {}

        for lang in lang_embeddings:
            X = lang_embeddings[lang]
            y = lang_labels[lang]

            if len(np.unique(y)) < 2:
                print(f"  {lang}: Skipping (insufficient label diversity)")
                continue

            lang_results[lang] = {
                "resource_level": RESOURCE_LEVELS.get(lang, 0.5),
                "n_samples": len(y),
                "accuracies": {},
                "accuracy_drops": {}
            }

            baseline_acc = None

            for bits in bit_widths:
                if bits == 16:
                    X_q = X  # No quantization for FP16 baseline
                else:
                    X_q = np.array([simulate_quantization(emb, bits) for emb in X])

                try:
                    _, acc = train_probe(X_q, y)
                    lang_results[lang]["accuracies"][bits] = acc

                    if bits == 16:
                        baseline_acc = acc
                    elif baseline_acc is not None:
                        drop = baseline_acc - acc
                        lang_results[lang]["accuracy_drops"][bits] = drop

                    print(f"  {lang} @ {bits}bit: {acc:.3f}")
                except Exception as e:
                    print(f"  {lang} @ {bits}bit: FAILED ({e})")

        # Analyze correlation between resource level and accuracy drop
        print(f"\n{'='*50}")
        print("Correlation Analysis")
        print("=" * 50)

        for bits in [8, 4]:
            resource_levels = []
            accuracy_drops = []

            for lang, res in lang_results.items():
                if bits in res["accuracy_drops"]:
                    resource_levels.append(res["resource_level"])
                    accuracy_drops.append(res["accuracy_drops"][bits])

            if len(resource_levels) >= 3:
                r, p = pearsonr(resource_levels, accuracy_drops)
                print(f"\n{bits}-bit quantization:")
                print(f"  Correlation (resource vs drop): r = {r:.3f}, p = {p:.4f}")
                print(f"  Prediction: r < 0 (low resource → more drop)")
                print(f"  Result: r = {r:.3f}")

                results["model_results"][f"{model_label}_{bits}bit"] = {
                    "correlation": r,
                    "p_value": p,
                    "n_languages": len(resource_levels),
                    "supported": r < 0 and p < 0.1
                }

        # Summary table
        print(f"\n{'='*50}")
        print("Summary: Accuracy by Language and Quantization")
        print("=" * 50)
        print(f"{'Lang':<6} {'Resource':<10} {'FP16':<8} {'INT8':<8} {'INT4':<8} {'Drop8':<8} {'Drop4':<8}")
        print("-" * 62)

        for lang in sorted(lang_results.keys(), key=lambda x: -lang_results[x]["resource_level"]):
            res = lang_results[lang]
            fp16 = res["accuracies"].get(16, 0)
            int8 = res["accuracies"].get(8, 0)
            int4 = res["accuracies"].get(4, 0)
            drop8 = res["accuracy_drops"].get(8, 0)
            drop4 = res["accuracy_drops"].get(4, 0)
            rlvl = res["resource_level"]
            print(f"{lang:<6} {rlvl:<10.2f} {fp16:<8.3f} {int8:<8.3f} {int4:<8.3f} {drop8:<8.3f} {drop4:<8.3f}")

        results["model_results"][model_label] = {
            "per_language": lang_results
        }

        del model
        del tokenizer

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"b002_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_experiment()
