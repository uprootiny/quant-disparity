#!/usr/bin/env python3
"""
Parse results from Colab GPU experiments.

Usage:
  1. Run the Colab notebook
  2. Copy the JSON output between COMPLETE_RESULTS_JSON_START and COMPLETE_RESULTS_JSON_END
  3. Paste into a file called 'colab_results.json'
  4. Run: python3 parse_colab_results.py colab_results.json

Or pipe directly:
  python3 parse_colab_results.py < colab_results.json
"""
import json
import sys
import numpy as np
from scipy import stats

def parse_and_analyze(data):
    """Parse Colab results and perform analysis."""

    print("=" * 70)
    print("COLAB RESULTS ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    print(f"GPU: {data.get('gpu', 'Unknown')}")
    print()

    # G1: Tokenization Analysis
    if 'G1_results' in data and data['G1_results'] != 'Not run':
        print("-" * 70)
        print("G1: TOKENIZATION ANALYSIS")
        print("-" * 70)

        g1 = data['G1_results']

        # Sort by alignment
        langs_sorted = sorted(g1.keys(), key=lambda l: g1[l].get('alignment_proxy', 0), reverse=True)

        print(f"\n{'Lang':<8} {'Tokens':<10} {'Words':<8} {'Ratio':<10} {'Alignment':<10}")
        print("-" * 50)

        alignments = []
        ratios = []

        for lang in langs_sorted:
            r = g1[lang]
            print(f"{lang:<8} {r['avg_tokens']:<10.1f} {r['avg_words']:<8.1f} {r['token_word_ratio']:<10.2f} {r['alignment_proxy']:<10.3f}")
            alignments.append(r['alignment_proxy'])
            ratios.append(r['token_word_ratio'])

        # Compare to our simulated values
        SIMULATED_ALIGNMENT = {
            'en': 0.72, 'de': 0.58, 'fr': 0.62, 'he': 0.24,
            'ar': 0.28, 'zh': 0.55, 'ja': 0.38, 'ko': 0.32
        }

        real_aligns = [g1[l]['alignment_proxy'] for l in g1 if l in SIMULATED_ALIGNMENT]
        sim_aligns = [SIMULATED_ALIGNMENT[l] for l in g1 if l in SIMULATED_ALIGNMENT]

        if len(real_aligns) > 2:
            r_val, p_val = stats.pearsonr(real_aligns, sim_aligns)
            print(f"\nCorrelation with simulated alignment: r = {r_val:.3f}, p = {p_val:.4f}")
            print(f"Validation: {'CONFIRMED' if r_val > 0.7 else 'PARTIAL' if r_val > 0.4 else 'FAILED'}")

    # G2: Quantization Effects
    if 'G2_results' in data and data['G2_results'] != 'Not run':
        print("\n" + "-" * 70)
        print("G2: QUANTIZATION EFFECTS")
        print("-" * 70)

        g2 = data['G2_results']

        # Separate HR and LR
        HR_LANGS = ['en', 'de', 'fr']
        LR_LANGS = ['he', 'ar', 'ko', 'ja', 'zh']

        hr_deg = [g2[l]['degradation_pct'] for l in HR_LANGS if l in g2]
        lr_deg = [g2[l]['degradation_pct'] for l in LR_LANGS if l in g2]

        print(f"\n{'Lang':<8} {'FP32 PPL':<12} {'INT8 PPL':<12} {'Degradation':<12}")
        print("-" * 50)

        for lang in g2:
            r = g2[lang]
            print(f"{lang:<8} {r['fp32_ppl']:<12.2f} {r['int8_ppl']:<12.2f} {r['degradation_pct']:>+10.1f}%")

        if hr_deg and lr_deg:
            print(f"\nHR mean degradation: {np.mean(hr_deg):+.1f}%")
            print(f"LR mean degradation: {np.mean(lr_deg):+.1f}%")
            disparity = np.mean(lr_deg) / np.mean(hr_deg) if np.mean(hr_deg) != 0 else float('inf')
            print(f"Disparity (LR/HR): {disparity:.2f}x")

            # Statistical test
            if len(hr_deg) > 1 and len(lr_deg) > 1:
                t_stat, p_val = stats.ttest_ind(lr_deg, hr_deg)
                print(f"t-test: t={t_stat:.2f}, p={p_val:.4f}")
                print(f"Validation: {'CONFIRMED' if p_val < 0.05 else 'NOT SIGNIFICANT'}")

    # G3: Layer Importance
    if 'G3_results' in data and data['G3_results'] != 'Not run':
        print("\n" + "-" * 70)
        print("G3: LAYER IMPORTANCE")
        print("-" * 70)

        g3 = data['G3_results']

        for lang in g3:
            print(f"\n{lang}:")
            layers = g3[lang]

            # Get layer indices
            layer_nums = sorted([int(l[1:]) for l in layers.keys()])
            n_layers = max(layer_nums) + 1

            gateway_layers = [0, n_layers - 1]
            middle_layers = [i for i in range(2, n_layers - 2)]

            gateway_deg = np.mean([layers[f'L{i}']['degradation_pct'] for i in gateway_layers if f'L{i}' in layers])
            middle_deg = np.mean([layers[f'L{i}']['degradation_pct'] for i in middle_layers if f'L{i}' in layers])

            print(f"  Gateway (L0, L{n_layers-1}): {gateway_deg:.1f}% avg")
            print(f"  Middle layers: {middle_deg:.1f}% avg")
            print(f"  Gateway/Middle ratio: {gateway_deg/middle_deg:.2f}x")

        # Compare HR vs LR gateway importance
        hr_gateway = []
        lr_gateway = []

        for lang in g3:
            layers = g3[lang]
            layer_nums = sorted([int(l[1:]) for l in layers.keys()])
            n_layers = max(layer_nums) + 1
            gateway_deg = np.mean([layers[f'L{i}']['degradation_pct'] for i in [0, n_layers-1] if f'L{i}' in layers])

            if lang in ['en', 'de', 'fr']:
                hr_gateway.append(gateway_deg)
            else:
                lr_gateway.append(gateway_deg)

        if hr_gateway and lr_gateway:
            print(f"\nHR gateway importance: {np.mean(hr_gateway):.1f}%")
            print(f"LR gateway importance: {np.mean(lr_gateway):.1f}%")
            print(f"LR/HR ratio: {np.mean(lr_gateway)/np.mean(hr_gateway):.2f}x")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
    else:
        # Read from stdin
        data = json.load(sys.stdin)

    parse_and_analyze(data)


if __name__ == '__main__':
    main()
