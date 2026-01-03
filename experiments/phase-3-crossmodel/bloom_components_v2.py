#!/usr/bin/env python3
"""
EXP-031: BLOOM Component Analysis (Memory-Efficient)

Analyze which components have outliers without loading full model.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_bloom_components():
    """Analyze BLOOM weight components via state dict."""
    print("="*60)
    print("EXP-031: BLOOM Component Analysis")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download
        import torch

        print("\nDownloading BLOOM-560M weights...")
        path = hf_hub_download("bigscience/bloom-560m", "pytorch_model.bin")
        state_dict = torch.load(path, map_location="cpu")

        print(f"Loaded {len(state_dict)} tensors\n")

        # Classify components
        component_stats = {}

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            # Parse layer number
            layer_id = None
            if '.h.' in name:
                parts = name.split('.')
                for i, p in enumerate(parts):
                    if p == 'h' and i + 1 < len(parts):
                        try:
                            layer_id = int(parts[i + 1])
                        except:
                            pass
                        break

            # Classify component type
            name_lower = name.lower()
            if 'query_key_value' in name_lower:
                comp = 'qkv'
            elif 'dense' in name_lower and 'attention' in name_lower:
                comp = 'attn_out'
            elif 'dense_h_to_4h' in name_lower:
                comp = 'mlp_up'
            elif 'dense_4h_to_h' in name_lower:
                comp = 'mlp_down'
            elif 'embed' in name_lower:
                comp = 'embed'
            elif 'ln' in name_lower or 'norm' in name_lower:
                continue  # skip layernorm
            else:
                comp = 'other'

            # Compute stats
            w = tensor.numpy().flatten()
            if len(w) > 50000:
                w = np.random.choice(w, 50000, replace=False)

            kurt = float(sp_stats.kurtosis(w))
            max_abs = float(np.max(np.abs(w)))

            key = (layer_id, comp)
            if key not in component_stats:
                component_stats[key] = {'kurtosis': [], 'max_abs': []}
            component_stats[key]['kurtosis'].append(kurt)
            component_stats[key]['max_abs'].append(max_abs)

        del state_dict
        gc.collect()

        # Aggregate
        agg_stats = {}
        for (layer_id, comp), data in component_stats.items():
            if layer_id is None:
                continue
            key = f"L{layer_id}_{comp}"
            agg_stats[key] = {
                'layer': layer_id,
                'component': comp,
                'kurtosis': float(np.mean(data['kurtosis'])),
                'max_abs': float(np.max(data['max_abs'])),
            }

        # Print by layer for outlier layers
        print("OUTLIER LAYERS (5, 21, 22) - Component Breakdown:")
        print("-" * 60)
        print(f"{'Layer':<8} {'Component':<12} {'Kurtosis':<12} {'Max|W|':<12}")
        print("-" * 60)

        for layer_id in [5, 21, 22]:
            for comp in ['qkv', 'attn_out', 'mlp_up', 'mlp_down']:
                key = f"L{layer_id}_{comp}"
                if key in agg_stats:
                    s = agg_stats[key]
                    marker = "<<<" if s['kurtosis'] > 50 else ""
                    print(f"{layer_id:<8} {comp:<12} {s['kurtosis']:<12.1f} {s['max_abs']:<12.3f} {marker}")
            print()

        # Print normal layers for comparison
        print("\nNORMAL LAYERS (0, 10) - Component Breakdown:")
        print("-" * 60)

        for layer_id in [0, 10]:
            for comp in ['qkv', 'attn_out', 'mlp_up', 'mlp_down']:
                key = f"L{layer_id}_{comp}"
                if key in agg_stats:
                    s = agg_stats[key]
                    print(f"{layer_id:<8} {comp:<12} {s['kurtosis']:<12.1f} {s['max_abs']:<12.3f}")
            print()

        # Summary by component type
        print("\n" + "="*60)
        print("SUMMARY BY COMPONENT TYPE")
        print("="*60)

        comp_summary = {}
        for key, s in agg_stats.items():
            comp = s['component']
            if comp not in comp_summary:
                comp_summary[comp] = {'kurtosis': [], 'max_abs': []}
            comp_summary[comp]['kurtosis'].append(s['kurtosis'])
            comp_summary[comp]['max_abs'].append(s['max_abs'])

        print(f"\n{'Component':<12} {'Mean κ':<12} {'Max κ':<12} {'Max|W|':<12}")
        print("-" * 50)

        for comp in ['qkv', 'attn_out', 'mlp_up', 'mlp_down']:
            if comp in comp_summary:
                data = comp_summary[comp]
                print(f"{comp:<12} {np.mean(data['kurtosis']):<12.1f} "
                      f"{np.max(data['kurtosis']):<12.1f} {np.max(data['max_abs']):<12.3f}")

        # Identify which component has outliers
        print("\n" + "="*60)
        print("CONCLUSION: WHERE ARE THE OUTLIERS?")
        print("="*60)

        outlier_components = []
        for comp, data in comp_summary.items():
            if np.max(data['kurtosis']) > 50:
                outlier_components.append((comp, np.max(data['kurtosis'])))

        outlier_components.sort(key=lambda x: -x[1])

        print("\nComponents with κ > 50:")
        for comp, kurt in outlier_components:
            print(f"  {comp}: max κ = {kurt:.1f}")

        if not outlier_components:
            print("  None")

        # Save
        Path("bloom_component_stats.json").write_text(json.dumps(agg_stats, indent=2))
        print("\nSaved to bloom_component_stats.json")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_bloom_components()
