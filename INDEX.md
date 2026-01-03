# Document Index

## Research Status

| Doc | Purpose | Updated |
|-----|---------|---------|
| [README.md](README.md) | Project overview | 2026-01-03 |
| [STATUS.md](STATUS.md) | Current findings | 2026-01-03 |
| [VISION.md](VISION.md) | Research tiers & goals | 2026-01-03 |

## Experiments

| Doc | Purpose |
|-----|---------|
| [experiments/PROTOCOL.md](experiments/PROTOCOL.md) | Pre-registered hypotheses & results |
| [experiments/EXPERIMENTAL_PLAN.md](experiments/EXPERIMENTAL_PLAN.md) | Full experimental design |
| [experiments/CPU_ROADMAP.md](experiments/CPU_ROADMAP.md) | CPU-only directions |
| [experiments/COMPUTE.md](experiments/COMPUTE.md) | Resource requirements |
| [experiments/HYPOTHESIS.md](experiments/HYPOTHESIS.md) | Hypothesis evolution |

## Theory

| Doc | Purpose |
|-----|---------|
| [theory/la_aciq_formalization.md](theory/la_aciq_formalization.md) | LA-ACIQ v1 framework |
| [theory/la_aciq_v2.md](theory/la_aciq_v2.md) | Refined theory with predictions |
| [theory/la_aciq_math.md](theory/la_aciq_math.md) | Compact mathematical summary |

## Cross-Model Analysis

| Doc | Purpose |
|-----|---------|
| [experiments/phase-3-crossmodel/MODEL_TAXONOMY.md](experiments/phase-3-crossmodel/MODEL_TAXONOMY.md) | Full model census (12 models) |
| [experiments/phase-3-crossmodel/OPT_HYPOTHESES.md](experiments/phase-3-crossmodel/OPT_HYPOTHESES.md) | OPT kurtosis investigation |
| [experiments/phase-3-crossmodel/config_analysis.md](experiments/phase-3-crossmodel/config_analysis.md) | BLOOM vs XGLM training |
| [experiments/phase-3-crossmodel/FINDINGS.md](experiments/phase-3-crossmodel/FINDINGS.md) | Cross-model patterns |

## PhD Planning / Lab Analysis

| Doc | Purpose |
|-----|---------|
| [docs/soudry_lab_narrative.md](docs/soudry_lab_narrative.md) | Soudry Lab deep dive |
| [docs/soudry_lab_analysis.md](docs/soudry_lab_analysis.md) | Paper-by-paper analysis |
| [docs/soudry_lab_literature_review.md](docs/soudry_lab_literature_review.md) | 12+ papers reviewed |
| [docs/israeli_labs_narratives.md](docs/israeli_labs_narratives.md) | Schwartz, Goldberg, Belinkov |
| [docs/internal_synthesis.md](docs/internal_synthesis.md) | Strategic positioning |
| [docs/research_fit_analysis.md](docs/research_fit_analysis.md) | Where our work fits |

## Key Data Files

| File | Contents |
|------|----------|
| `experiments/phase-1-extraction/outlier_activation.json` | Per-language outlier activation |
| `experiments/phase-1-extraction/bloom_architecture.json` | BLOOM layer kurtosis |
| `experiments/phase-3-crossmodel/exp022_results.json` | Architecture comparison (H1, H4) |
| `experiments/phase-3-crossmodel/exp024_results.json` | Layer position analysis (H3) |
| `experiments/phase-3-crossmodel/opt_outliers.json` | OPT outlier details |
| `experiments/phase-3-crossmodel/gpt2_analysis.json` | GPT-2 outlier analysis |
| `experiments/phase-3-crossmodel/pythia_survey.json` | Pythia family analysis |
| `theory/theory_validation.json` | Theory validation results |

## Key Experiment Scripts

| Script | Purpose |
|--------|---------|
| `phase-3-crossmodel/exp022_architecture.py` | Tests H1, H4 (dimension, component) |
| `phase-3-crossmodel/exp024_layer_position.py` | Tests H3 (layer position) |
| `phase-3-crossmodel/investigate_opt.py` | OPT outlier deep dive |
| `phase-3-crossmodel/investigate_gpt2.py` | GPT-2 outlier deep dive |

---

## Quick Navigation

**"What did we find?"** → [STATUS.md](STATUS.md)

**"What's the theory?"** → [theory/la_aciq_v2.md](theory/la_aciq_v2.md)

**"Which models have outliers?"** → [MODEL_TAXONOMY.md](experiments/phase-3-crossmodel/MODEL_TAXONOMY.md)

**"What experiments are planned?"** → [EXPERIMENTAL_PLAN.md](experiments/EXPERIMENTAL_PLAN.md)

**"Who should I contact?"** → [israeli_labs_narratives.md](docs/israeli_labs_narratives.md)

---
*Index created: 2026-01-03*
