#!/usr/bin/env python3
"""
EXP-008c: Bootstrap validation of EXP-007 correlation

Since memory constraints prevent large-corpus validation,
we validate using bootstrap resampling and permutation tests
on the existing EXP-007 data.

This tells us whether r=-0.834 is robust to:
1. Sampling variability (bootstrap CI)
2. Random chance (permutation test)
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

# Load EXP-007 data
DATA = json.loads(Path("outlier_activation.json").read_text())

DEGRADATION = {
    "eng": 0.005, "fra": 0.007, "deu": 0.008, "vie": 0.009,
    "rus": 0.012, "zho": 0.013, "tur": 0.015, "fin": 0.016,
    "kor": 0.018, "heb": 0.020, "tha": 0.020, "hin": 0.021,
    "jpn": 0.022, "ara": 0.025,
}


def main():
    print("=" * 60)
    print("EXP-008c: Bootstrap Validation of EXP-007")
    print("=" * 60)

    langs = sorted(set(DATA.keys()) & set(DEGRADATION.keys()))
    n = len(langs)

    outlier = np.array([DATA[l]["outlier_frac"] for l in langs])
    combined = np.array([DATA[l]["combined_frac"] for l in langs])
    degrad = np.array([DEGRADATION[l] for l in langs])

    # Original correlation
    r_orig, p_orig = sp_stats.pearsonr(outlier, degrad)
    print(f"\nOriginal correlation (n={n}):")
    print(f"  r = {r_orig:+.4f}")
    print(f"  p = {p_orig:.6f}")

    # Bootstrap CI (case resampling)
    print("\n" + "-" * 40)
    print("Bootstrap 95% CI (10,000 resamples):")
    n_boot = 10000
    boot_r = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = sp_stats.pearsonr(outlier[idx], degrad[idx])
        boot_r.append(r)

    boot_r = np.array(boot_r)
    ci_low = np.percentile(boot_r, 2.5)
    ci_high = np.percentile(boot_r, 97.5)
    print(f"  95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"  Median: {np.median(boot_r):+.3f}")
    print(f"  Std: {np.std(boot_r):.3f}")

    # Permutation test (is this better than chance?)
    print("\n" + "-" * 40)
    print("Permutation test (10,000 permutations):")
    n_perm = 10000
    perm_r = []
    for _ in range(n_perm):
        perm_degrad = np.random.permutation(degrad)
        r, _ = sp_stats.pearsonr(outlier, perm_degrad)
        perm_r.append(r)

    perm_r = np.array(perm_r)
    p_perm = np.mean(perm_r <= r_orig)  # One-sided: r <= observed
    print(f"  p-value (one-sided, r <= {r_orig:.3f}): {p_perm:.4f}")
    print(f"  Null distribution: mean={np.mean(perm_r):.3f}, std={np.std(perm_r):.3f}")

    # Leave-one-out sensitivity
    print("\n" + "-" * 40)
    print("Leave-one-out sensitivity:")
    for i, l in enumerate(langs):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r_loo, _ = sp_stats.pearsonr(outlier[mask], degrad[mask])
        diff = r_loo - r_orig
        print(f"  Without {l}: r = {r_loo:+.3f} (Δ = {diff:+.3f})")

    # Combined metric
    print("\n" + "-" * 40)
    print("Combined outlier layers (4-7, 20-23):")
    r_comb, p_comb = sp_stats.pearsonr(combined, degrad)
    print(f"  r = {r_comb:+.4f}")
    print(f"  p = {p_comb:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    ci_excludes_zero = ci_low < 0 and ci_high < 0
    perm_sig = p_perm < 0.05

    print(f"\n  Original r: {r_orig:+.3f}")
    print(f"  Bootstrap 95% CI excludes zero: {ci_excludes_zero}")
    print(f"  Permutation p < 0.05: {perm_sig} (p = {p_perm:.4f})")

    if ci_excludes_zero and perm_sig:
        print("\n  [*] ROBUST: Correlation is statistically robust")
    elif ci_excludes_zero or perm_sig:
        print("\n  [~] MODERATE: Some evidence for robustness")
    else:
        print("\n  [!] WEAK: Correlation may be spurious")

    # Save results
    results = {
        "original": {"r": float(r_orig), "p": float(p_orig)},
        "bootstrap": {
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "median": float(np.median(boot_r)),
            "std": float(np.std(boot_r)),
            "n_resamples": n_boot,
        },
        "permutation": {
            "p_value": float(p_perm),
            "null_mean": float(np.mean(perm_r)),
            "null_std": float(np.std(perm_r)),
            "n_permutations": n_perm,
        },
        "combined_layers": {"r": float(r_comb), "p": float(p_comb)},
        "robust": bool(ci_excludes_zero and perm_sig),
    }
    Path("bootstrap_validation.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to bootstrap_validation.json")


if __name__ == "__main__":
    main()
