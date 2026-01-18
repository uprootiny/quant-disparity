# Architecture Analysis: Partial Solutions & Proper Layering

## Current State: Duplication Map

### 1. Kurtosis Computation (8 implementations)

| File | Function | Notes |
|------|----------|-------|
| `theory/validate_theory.py:34` | `compute_effective_kurtosis()` | Hardcoded layer lists |
| `theory-tracks/formal-proofs/laaciq_bridge.py:98` | `effective_kurtosis()` | Full mixture formula |
| `experiments/phase-0-validation/distrib_analysis.py:136` | uses scipy.stats.kurtosis | Inline |
| `experiments/phase-1-extraction/xglm_validation.py:102` | `get_layer_kurtosis()` | Per-model |
| `experiments/phase-2-corpus/theoretical_quant_error.py:85` | `compute_effective_kurtosis()` | Copy of theory/ |
| `experiments/phase-3-crossmodel/exp030_attention_sink_correlation.py:70` | `compute_layer_kurtosis()` | Yet another |
| `experiments/phase-4-actionable/exp036_layer_contribution.py:98` | `get_layer_kurtosis()` | Copy |
| `theory-tracks/experiments/theory_validation_suite.py:93` | `get_layer_kurtosis()` | Simulated |

### 2. Perplexity Computation (10+ implementations)

| File | Function | Notes |
|------|----------|-------|
| `quant-fairness/src/quant_fairness/quantize.py:57` | `perplexity()` | Production-ready |
| `gpu-experiments/src/disparity.py:27` | `perplexity()` | Near-identical |
| `experiments/phase-2-corpus/bitwidth_sweep.py:59` | `compute_perplexity()` | Different signature |
| `experiments/phase-4-actionable/exp039_*.py` | `compute_ppl()` | Multiple copies |
| `experiments/phase-4-actionable/run_*.py` | `compute_ppl()` | More copies |
| `experiments/track-b-interpretability/b003_circuit_ablation.py:92` | `compute_pseudo_perplexity()` | MLM variant |

### 3. MSE / Quantization Error (9 implementations)

| File | Function | Notes |
|------|----------|-------|
| `theory/validate_theory.py:69` | `compute_quant_mse()` | Gaussian approx |
| `theory/redundancy_formalization.py:200` | `mse()` | Symbolic |
| `theory-tracks/formal-proofs/laaciq_bridge.py:167` | `mse()` | With decomposition |
| `theory-tracks/experiments/validated_experiment.py:53` | `mse()` | Bridge to Lean |
| `experiments/phase-0-validation/distrib_analysis.py:101` | `quant_error()` | Original |
| `experiments/phase-2-corpus/theoretical_quant_error.py:59` | `compute_quant_error()` | Copy |
| `experiments/phase-5-minimal/exp090_*.py` | `compute_quantization_error()` | Another copy |
| `experiments/phase-7-hypothesis/exp110_*.py:92` | `compute_mse()` | Yet another |

### 4. Optimal Alpha / Banner Formula (7 implementations)

| File | Function | Notes |
|------|----------|-------|
| `theory/validate_theory.py:62` | `optimal_alpha()` | Simple |
| `theory/redundancy_formalization.py:182` | `optimal_alpha_gaussian()` | Gaussian only |
| `theory-tracks/formal-proofs/laaciq_bridge.py:54` | `banner_approximation()` | **Canonical** |
| `theory-tracks/formal-proofs/laaciq_bridge.py:121` | `laaciq_optimal_alpha()` | Full LA-ACIQ |
| `experiments/phase-2-corpus/theoretical_quant_error.py:44` | `compute_optimal_alpha()` | Copy |
| `experiments/phase-5-minimal/exp092_*.py:175` | `compute_optimal_alpha()` | Another copy |
| `experiments/phase-7-hypothesis/exp110_*.py:98` | `find_optimal_alpha()` | Numerical search |

### 5. Token Fertility (2 implementations)

| File | Function |
|------|----------|
| `experiments/phase-3-crossmodel/exp033_token_fertility_prediction.py:112` | `compute_fertility()` |
| `experiments/track-c-efficiency/c001b_tokenizer_efficiency.py:21` | `compute_fertility()` |

---

## Proper Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  Notebooks, Reports, Visualizations                          │
│  - experiments/gpu-colab/*.ipynb                             │
│  - site/*.md                                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER                          │
│  Property Tests, Experiments, Proof Verification             │
│  - theory-tracks/experiments/  (KEEP)                        │
│  - theory-tracks/formal-proofs/ (KEEP)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    COMPUTATION LAYER                         │
│  Pure Functions: kurtosis, MSE, perplexity, quantize         │
│  - lib/laaciq.py  (TO CREATE - canonical implementations)    │
│  - lib/metrics.py (TO CREATE - perplexity, fertility)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  Corpus Acquisition, Model Loading, Caching                  │
│  - data/hebrew-corpus/ (KEEP)                                │
│  - data/cache/ (model weights, activations)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Proposed Consolidation

### 1. Create `lib/laaciq.py` - Canonical Theory Implementation

```python
# lib/laaciq.py - THE source of truth for LA-ACIQ computations
"""
LA-ACIQ: Language-Aware Analytical Clipping for Integer Quantization

All formulas match spec.json and Lean proofs.
"""
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class MixtureComponent:
    weight: float
    mean: float
    variance: float
    kurtosis: float  # excess kurtosis

def clip(x: np.ndarray, alpha: float) -> np.ndarray:
    """Symmetric clipping. Matches Lean: clip_in_range theorem."""
    return np.clip(x, -alpha, alpha)

def step_size(alpha: float, bits: int) -> float:
    """Quantization step. Δ = 2α/(2^B - 1)"""
    return 2 * alpha / (2**bits - 1)

def banner_approximation(sigma: float, kappa: float) -> float:
    """
    Optimal clipping threshold approximation.
    α* ≈ σ · (2.5 + 0.3·ln(1 + max(0, κ)))

    From Banner et al. (2019), validated in spec.json.
    """
    return sigma * (2.5 + 0.3 * np.log(1 + max(0, kappa)))

def effective_kurtosis(components: list[MixtureComponent]) -> float:
    """
    Kurtosis of a Gaussian mixture.
    κ_eff = (Σ wᵢ(κᵢ+3)σᵢ⁴ + 6Σ wᵢσᵢ²δᵢ² + Σ wᵢδᵢ⁴) / σ_eff⁴ - 3

    Matches Lean: Probability/Mixture.lean
    """
    # ... full implementation from laaciq_bridge.py:98

def mse_decomposition(x: np.ndarray, alpha: float, bits: int) -> tuple[float, float]:
    """
    MSE = E_clip + E_quant

    Matches Lean: mse_decomposition theorem (scaffolded)
    """
    # ... implementation

def quantization_mse(x: np.ndarray, alpha: float, bits: int) -> float:
    """Total MSE for symmetric uniform quantization."""
    e_clip, e_quant = mse_decomposition(x, alpha, bits)
    return e_clip + e_quant
```

### 2. Create `lib/metrics.py` - Evaluation Metrics

```python
# lib/metrics.py - Model evaluation utilities
"""
Metrics for quantization disparity evaluation.
"""
import torch

def perplexity(model, tokenizer, text: str, max_length: int = 512) -> float:
    """
    Compute perplexity for causal LM.
    Single canonical implementation.
    """
    # ... from quant-fairness/src/quant_fairness/quantize.py

def fertility(tokenizer, texts: list[str]) -> float:
    """
    Token fertility = tokens / words.
    Higher fertility → more subword fragmentation.
    """
    # ... from exp033_token_fertility_prediction.py

def layer_kurtosis(model, component: str = "mlp") -> dict[int, float]:
    """
    Extract kurtosis per layer.
    """
    # ... consolidated from multiple implementations
```

### 3. Migrate Experiments

Each experiment should:
1. Import from `lib/laaciq` and `lib/metrics`
2. Focus ONLY on experimental logic
3. NOT redefine core formulas

```python
# experiments/phase-X/expNNN.py
from lib.laaciq import banner_approximation, effective_kurtosis, mse_decomposition
from lib.metrics import perplexity, layer_kurtosis

# Experiment-specific code only
```

---

## Files to Consolidate/Remove

### Delete (redundant):
- `theory/validate_theory.py` → merged into `lib/laaciq.py`
- `theory/redundancy_formalization.py` → symbolic parts to docs, numeric to lib
- `experiments/phase-2-corpus/theoretical_quant_error.py` → use lib

### Keep as-is:
- `theory-tracks/formal-proofs/` - Lean proofs (source of truth for theorems)
- `theory-tracks/formal-proofs/spec.json` - Shared specification
- `theory-tracks/formal-proofs/laaciq_bridge.py` - Bridge (becomes lib/laaciq.py)

### Consolidate markdown:
- 135 .md files → reduce to ~20 essential docs
- Keep: README, THEORY.md, HONEST_ASSESSMENT.md, ARCHITECTURE.md
- Archive or delete: redundant theory/*.md, site/*.md duplicates

---

## Migration Path

1. **Create lib/** with canonical implementations
2. **Update imports** in actively-used experiments
3. **Archive** old theory/ directory
4. **Run validation** to ensure behavior unchanged
5. **Delete** redundant implementations

## Constraints Encoded as Tests

Each function in `lib/` should have:
- Docstring linking to spec.json theorem
- Property test in `tests/test_laaciq.py`
- Reference to Lean theorem (proved or sorry)

```python
def test_clip_in_range():
    """Matches Lean: clip_in_range (proved)"""
    for _ in range(100):
        x = np.random.randn()
        alpha = abs(np.random.randn()) + 0.1
        result = clip(x, alpha)
        assert -alpha <= result <= alpha, "Violates clip_in_range"
```
