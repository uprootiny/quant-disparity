#!/usr/bin/env python3
"""
EXP-027: Checkpoint Evolution Analysis

Pythia provides training checkpoints. When do outliers form?

Hypothesis: Outliers emerge late in training as weights specialize.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
import gc


def analyze_checkpoint(model_name: str, repo_id: str, revision: str) -> dict:
    """Analyze a single checkpoint."""
    from huggingface_hub import hf_hub_download
    import torch

    try:
        path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        return None

    max_kurt = 0
    max_tensor = ""
    all_kurts = []

    for name, tensor in state_dict.items():
        if tensor.dim() < 2:
            continue

        w = tensor.numpy().flatten()
        if len(w) > 30000:
            idx = np.random.choice(len(w), 30000, replace=False)
            w = w[idx]

        try:
            kurt = float(sp_stats.kurtosis(w))
            if np.isnan(kurt) or np.isinf(kurt):
                continue
            all_kurts.append(kurt)
            if kurt > max_kurt:
                max_kurt = kurt
                max_tensor = name
        except:
            continue

    del state_dict
    gc.collect()

    return {
        "revision": revision,
        "max_kurtosis": max_kurt,
        "mean_kurtosis": np.mean(all_kurts) if all_kurts else 0,
        "max_tensor": max_tensor,
    }


def main():
    print("="*60)
    print("EXP-027: Checkpoint Evolution Analysis")
    print("="*60)

    # Pythia checkpoints (subset for speed)
    checkpoints = [
        ("step1000", "step1000"),
        ("step8000", "step8000"),
        ("step16000", "step16000"),
        ("step32000", "step32000"),
        ("step64000", "step64000"),
        ("step143000", "step143000"),  # Final
    ]

    model_repo = "EleutherAI/pythia-70m"
    results = []

    print(f"\nAnalyzing {model_repo} across training...")
    print("-"*50)

    for name, revision in checkpoints:
        print(f"  Checkpoint: {name}...", end=" ", flush=True)
        r = analyze_checkpoint("Pythia-70M", model_repo, revision)
        if r:
            r["checkpoint"] = name
            results.append(r)
            print(f"max κ = {r['max_kurtosis']:.1f}")
        else:
            print("failed")

    if not results:
        print("No checkpoints loaded successfully")
        return

    # Analysis
    print("\n" + "="*60)
    print("EVOLUTION ANALYSIS")
    print("="*60)

    print(f"\n{'Checkpoint':<15} {'Step':<10} {'Max κ':<10} {'Mean κ'}")
    print("-"*45)

    steps = []
    max_kurts = []

    for r in results:
        step = int(r["revision"].replace("step", ""))
        steps.append(step)
        max_kurts.append(r["max_kurtosis"])
        print(f"{r['checkpoint']:<15} {step:<10} {r['max_kurtosis']:<10.1f} {r['mean_kurtosis']:.2f}")

    # Trend analysis
    if len(steps) >= 3:
        r_val, p_val = sp_stats.pearsonr(steps, max_kurts)
        print(f"\nCorrelation (step vs max κ): r = {r_val:.3f}, p = {p_val:.4f}")

        if r_val > 0.5 and p_val < 0.1:
            verdict = "OUTLIERS GROW: kurtosis increases during training"
        elif r_val < -0.5 and p_val < 0.1:
            verdict = "OUTLIERS SHRINK: kurtosis decreases during training"
        else:
            verdict = "NO CLEAR TREND: kurtosis varies non-monotonically"
    else:
        verdict = "INSUFFICIENT DATA"

    print(f"\nVerdict: {verdict}")

    # When did outliers first appear?
    threshold = 20
    first_outlier = next((r for r in results if r["max_kurtosis"] > threshold), None)
    if first_outlier:
        print(f"\nFirst κ > {threshold}: {first_outlier['checkpoint']}")
    else:
        print(f"\nNo checkpoint exceeded κ = {threshold}")

    # Save
    output = {
        "experiment": "EXP-027",
        "model": model_repo,
        "checkpoints": results,
        "verdict": verdict,
    }
    Path("exp027_results.json").write_text(json.dumps(output, indent=2, default=float))
    print("\nSaved to exp027_results.json")


if __name__ == "__main__":
    main()
