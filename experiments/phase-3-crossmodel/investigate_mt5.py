#!/usr/bin/env python3
"""
Investigate mT5-small's extreme outliers.

mT5 shows kurtosis=568, much higher than BLOOM's 164.
Which layers/components have the outliers?
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def main():
    print("="*60)
    print("mT5-small Outlier Investigation")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download
        import torch

        path = hf_hub_download("google/mt5-small", "pytorch_model.bin")
        state_dict = torch.load(path, map_location="cpu")

        print(f"Loaded {len(state_dict)} tensors\n")

        # Detailed per-tensor analysis
        tensor_stats = []

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            w = tensor.numpy().flatten()
            kurt = sp_stats.kurtosis(w)
            max_abs = np.max(np.abs(w))

            tensor_stats.append({
                "name": name,
                "shape": list(tensor.shape),
                "kurtosis": kurt,
                "max_abs": max_abs,
                "std": np.std(w),
            })

        # Sort by kurtosis
        tensor_stats.sort(key=lambda x: x["kurtosis"], reverse=True)

        # Top 10 outliers
        print("TOP 10 OUTLIER TENSORS:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Kurtosis':<12} {'Max|W|':<10} {'Name':<50}")
        print("-" * 80)

        for i, t in enumerate(tensor_stats[:10]):
            print(f"{i+1:<5} {t['kurtosis']:<12.1f} {t['max_abs']:<10.3f} {t['name'][:50]:<50}")

        # Analyze patterns
        print("\n" + "="*60)
        print("PATTERN ANALYSIS")
        print("="*60)

        # By component type
        component_kurt = {}
        for t in tensor_stats:
            name = t["name"].lower()
            if "embed" in name:
                comp = "embedding"
            elif "encoder" in name:
                comp = "encoder"
            elif "decoder" in name:
                comp = "decoder"
            elif "shared" in name:
                comp = "shared"
            else:
                comp = "other"

            if comp not in component_kurt:
                component_kurt[comp] = []
            component_kurt[comp].append(t["kurtosis"])

        print("\nBy Component:")
        for comp, kurts in sorted(component_kurt.items(), key=lambda x: -np.max(x[1])):
            print(f"  {comp:<12}: mean={np.mean(kurts):>8.1f}, max={np.max(kurts):>8.1f}")

        # Check if it's the embedding
        embed_stats = [t for t in tensor_stats if "embed" in t["name"].lower()]
        if embed_stats:
            print("\nEmbedding layers:")
            for t in embed_stats[:3]:
                print(f"  {t['name']}: κ={t['kurtosis']:.1f}, max|W|={t['max_abs']:.3f}")

        # Summary
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)

        max_tensor = tensor_stats[0]
        print(f"""
Most extreme outlier:
  {max_tensor['name']}
  Kurtosis: {max_tensor['kurtosis']:.1f}
  Max|W|:   {max_tensor['max_abs']:.3f}
  Shape:    {max_tensor['shape']}
""")

        if "embed" in max_tensor["name"].lower():
            print("The outlier is in EMBEDDING layer.")
            print("This is different from BLOOM (MLP layers).")
        elif "encoder.block.0" in max_tensor["name"]:
            print("The outlier is in FIRST ENCODER BLOCK.")
            print("Similar to BLOOM's early layer outliers (layer 5).")
        elif "decoder" in max_tensor["name"]:
            print("The outlier is in DECODER.")

        # Save
        Path("mt5_outliers.json").write_text(json.dumps({
            "top_outliers": tensor_stats[:20],
            "by_component": {k: {"mean": np.mean(v), "max": np.max(v)}
                             for k, v in component_kurt.items()},
        }, indent=2))
        print("\nSaved to mt5_outliers.json")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
