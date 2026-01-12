# Theory Tracks: Mise-en-Place

*Preparation for rigorous theoretical work*

---

## Two Parallel Tracks

### Track S: Soudry-Quality (Optimization Theory)
```
Goal: Derive closed-form LA-ACIQ with proof of optimality
Style: Mathematical derivation, theorem-proof structure
Output: Theory paper suitable for ICML/NeurIPS
```

### Track G: Goldberg-Quality (Causal Analysis)
```
Goal: Run causal intervention experiments (do-calculus)
Style: Controlled experiments, causal inference
Output: Empirical paper suitable for ACL/EMNLP
```

---

## Directory Structure

```
theory-tracks/
├── soudry-optimal/           # Track S
│   ├── derivations/          # Mathematical work
│   │   ├── 01_problem_setup.md
│   │   ├── 02_mse_decomposition.md
│   │   ├── 03_effective_kurtosis.md
│   │   ├── 04_optimal_alpha.md
│   │   └── 05_disparity_bound.md
│   ├── proofs/               # Formal proofs
│   │   ├── theorem_1_existence.md
│   │   ├── theorem_2_uniqueness.md
│   │   └── theorem_3_bound.md
│   └── experiments/          # Numerical validation
│       └── validate_derivations.py
│
├── goldberg-causal/          # Track G
│   ├── protocols/            # Experiment designs
│   │   ├── 01_scm_specification.md
│   │   ├── 02_do_tokenization.md
│   │   ├── 03_do_alignment.md
│   │   └── 04_mediation_analysis.md
│   ├── experiments/          # Implementations
│   │   ├── intervention_framework.py
│   │   └── causal_tests.py
│   └── derivations/          # Causal calculus
│       └── identifiability.md
│
└── README.md                 # This file
```

---

## Prerequisites

### For Track S (Soudry)
- Convex optimization (Boyd & Vandenberghe)
- Probability theory (moments, distributions)
- Rate-distortion theory (Cover & Thomas Ch. 10)
- ACIQ paper (Banner 2019) — thorough understanding

### For Track G (Goldberg)
- Causal inference (Pearl, Causality)
- Structural causal models
- do-calculus and identifiability
- Mediation analysis (VanderWeele)

---

## Could Be Split Into Separate Repo

If these tracks grow substantially:

```
quant-disparity/          # Empirical work (current)
quant-theory/             # Track S (mathematical)
quant-causal/             # Track G (causal)
```

For now, keep together to share data and infrastructure.
