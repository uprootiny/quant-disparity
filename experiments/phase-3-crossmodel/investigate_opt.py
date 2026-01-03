#!/usr/bin/env python3
"""
Investigate OPT-125M's extreme outliers (κ=372).

OPT has even higher kurtosis than BLOOM (164).
Where are the outliers? Why?
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def main():
    print("="*60)
    print("OPT-125M Outlier Investigation")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download
        import torch

        print("\nDownloading OPT-125M weights...")
        path = hf_hub_download("facebook/opt-125m", "pytorch_model.bin")
        state_dict = torch.load(path, map_location="cpu")

        print(f"Loaded {len(state_dict)} tensors\n")

        # Analyze each tensor
        tensor_stats = []

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue

            w = tensor.numpy().flatten()
            if len(w) > 50000:
                idx = np.random.choice(len(w), 50000, replace=False)
                w = w[idx]

            try:
                kurt = float(sp_stats.kurtosis(w))
                max_abs = float(np.max(np.abs(w)))
                std = float(np.std(w))

                if not np.isnan(kurt) and not np.isinf(kurt):
                    tensor_stats.append({
                        "name": name,
                        "kurtosis": kurt,
                        "max_abs": max_abs,
                        "std": std,
                        "shape": list(tensor.shape),
                    })
            except:
                continue

        del state_dict
        gc.collect()

        # Sort by kurtosis
        tensor_stats.sort(key=lambda x: -x["kurtosis"])

        # Top outliers
        print("TOP 10 OUTLIER TENSORS:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Kurtosis':<12} {'Max|W|':<10} {'Name':<45}")
        print("-" * 70)

        for i, t in enumerate(tensor_stats[:10]):
            print(f"{i+1:<5} {t['kurtosis']:<12.1f} {t['max_abs']:<10.3f} "
                  f"{t['name'][:45]:<45}")

        # Compare with BLOOM pattern
        print("\n" + "="*60)
        print("COMPARISON WITH BLOOM")
        print("="*60)

        bloom_outliers = ["layers 5, 21, 22", "MLP weights", "κ up to 164"]
        print("\nBLOOM outlier pattern:")
        for b in bloom_outliers:
            print(f"  - {b}")

        # Categorize OPT outliers
        print("\nOPT outlier pattern:")

        outlier_tensors = [t for t in tensor_stats if t["kurtosis"] > 50]
        print(f"  - {len(outlier_tensors)} tensors with κ > 50")

        # Check component types
        components = {}
        for t in outlier_tensors:
            name = t["name"].lower()
            if "embed" in name:
                comp = "embedding"
            elif "fc1" in name or "fc2" in name:
                comp = "MLP"
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                comp = "attention_qkv"
            elif "out_proj" in name:
                comp = "attention_out"
            else:
                comp = "other"

            if comp not in components:
                components[comp] = []
            components[comp].append(t["kurtosis"])

        print("\n  By component type:")
        for comp, kurts in sorted(components.items(), key=lambda x: -max(x[1])):
            print(f"    {comp}: {len(kurts)} tensors, max κ = {max(kurts):.1f}")

        # Check layer distribution
        layer_kurts = {}
        for t in tensor_stats:
            name = t["name"]
            # Extract layer number
            layer_id = None
            if "layers." in name:
                try:
                    layer_id = int(name.split("layers.")[1].split(".")[0])
                except:
                    pass

            if layer_id is not None:
                if layer_id not in layer_kurts:
                    layer_kurts[layer_id] = []
                layer_kurts[layer_id].append(t["kurtosis"])

        print("\n  By layer (max κ per layer):")
        for layer_id in sorted(layer_kurts.keys()):
            max_k = max(layer_kurts[layer_id])
            marker = "<<<" if max_k > 50 else ""
            print(f"    Layer {layer_id:2d}: max κ = {max_k:6.1f} {marker}")

        # Conclusion
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)

        max_tensor = tensor_stats[0]
        print(f"""
Most extreme outlier: {max_tensor['name']}
  Kurtosis: {max_tensor['kurtosis']:.1f}
  Max|W|:   {max_tensor['max_abs']:.3f}
  Shape:    {max_tensor['shape']}
""")

        if "embed" in max_tensor["name"].lower():
            print("Outlier is in EMBEDDING layer.")
        elif "fc" in max_tensor["name"].lower():
            print("Outlier is in MLP layer (like BLOOM).")
        elif "proj" in max_tensor["name"].lower():
            print("Outlier is in attention projection.")

        # Save summary
        summary = {
            "max_kurtosis": tensor_stats[0]["kurtosis"],
            "top_outliers": tensor_stats[:10],
            "outlier_components": {k: len(v) for k, v in components.items()},
            "outlier_layers": [l for l, k in layer_kurts.items() if max(k) > 50],
        }
        Path("opt_outliers.json").write_text(json.dumps(summary, indent=2))
        print("\nSaved to opt_outliers.json")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
