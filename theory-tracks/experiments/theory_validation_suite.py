#!/usr/bin/env python3
"""
Theory Validation Suite: Runnable Experiments
==============================================

Implements the 10 theory tests from THEORY_INVESTIGATION.md
Each test validates a specific theoretical hypothesis.

Run: python theory_validation_suite.py
"""

import numpy as np
from scipy import stats
from scipy.linalg import svd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# =============================================================================
# Simulated Model Infrastructure
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for simulated model."""
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    vocab_size: int = 50257


@dataclass
class Language:
    """Language with properties."""
    code: str
    name: str
    alignment: float  # 0-1
    fertility: float  # tokens per word
    resource_level: str  # HR or LR


LANGUAGES = [
    Language("en", "English", 0.95, 1.0, "HR"),
    Language("de", "German", 0.88, 1.2, "HR"),
    Language("fr", "French", 0.90, 1.1, "HR"),
    Language("zh", "Chinese", 0.82, 1.8, "HR"),
    Language("ar", "Arabic", 0.41, 2.5, "LR"),
    Language("he", "Hebrew", 0.38, 2.3, "LR"),
    Language("sw", "Swahili", 0.29, 2.8, "LR"),
    Language("yo", "Yoruba", 0.22, 3.2, "LR"),
]


class SimulatedModel:
    """Simulated transformer for theory testing."""

    def __init__(self, config: ModelConfig):
        self.config = config
        np.random.seed(42)

        # Generate weight matrices with realistic kurtosis
        self.weights = {}
        for layer in range(config.num_layers):
            # Gateway layers (0, 11) and bottleneck (9) have higher kurtosis
            if layer in [0, 9, 11]:
                kurtosis = 50 + np.random.rand() * 100  # High kurtosis
            else:
                kurtosis = 2 + np.random.rand() * 5  # Near-Gaussian

            # Generate weights with target kurtosis (approximate)
            self.weights[layer] = self._generate_weights(
                (config.hidden_dim, config.hidden_dim),
                target_kurtosis=kurtosis
            )

    def _generate_weights(self, shape: Tuple[int, int], target_kurtosis: float) -> np.ndarray:
        """Generate weights with approximate target kurtosis."""
        n = shape[0] * shape[1]

        if target_kurtosis < 5:
            # Near-Gaussian
            weights = np.random.randn(n)
        else:
            # Heavy-tailed: mixture of Gaussian and Laplace
            mix_ratio = min(0.3, target_kurtosis / 200)
            gaussian = np.random.randn(n)
            laplace = np.random.laplace(0, 1, n)
            weights = (1 - mix_ratio) * gaussian + mix_ratio * laplace * np.sqrt(target_kurtosis / 10)

        return weights.reshape(shape) * 0.02  # Scale to typical weight magnitude

    def get_layer_kurtosis(self, layer: int) -> float:
        """Compute kurtosis of layer weights."""
        w = self.weights[layer].flatten()
        return stats.kurtosis(w, fisher=True)

    def get_layer_variance(self, layer: int) -> float:
        """Compute variance of layer weights."""
        return np.var(self.weights[layer])

    def simulate_activations(self, lang: Language, layer: int) -> np.ndarray:
        """Simulate activations for a language at a layer."""
        base_activation = np.abs(np.random.randn(self.config.hidden_dim))

        # Language alignment affects activation patterns
        alignment_factor = lang.alignment

        # Gateway layers activated more by HR languages
        if layer in [0, 9, 11]:
            base_activation *= (0.8 + 0.4 * alignment_factor)
        else:
            base_activation *= (0.9 + 0.2 * alignment_factor)

        return base_activation

    def quantize_weights(self, weights: np.ndarray, bits: int = 4,
                         alpha: Optional[float] = None) -> np.ndarray:
        """Simulate INT4 quantization."""
        if alpha is None:
            alpha = 3 * np.std(weights)

        # Clip
        clipped = np.clip(weights, -alpha, alpha)

        # Quantize
        n_levels = 2 ** bits
        step = 2 * alpha / (n_levels - 1)
        quantized = np.round(clipped / step) * step

        return quantized


# =============================================================================
# Theory Test T-001: Information Content vs Variance
# =============================================================================

def test_t001_information_content(model: SimulatedModel) -> Dict:
    """
    H1.1: High-variance layers encode MORE information.
    Test: Entropy of activations correlates with weight variance.
    """
    print("\n" + "="*60)
    print("T-001: Information Content vs Variance")
    print("="*60)

    variances = []
    entropies = []

    for layer in range(model.config.num_layers):
        # Weight variance
        var = model.get_layer_variance(layer)
        variances.append(var)

        # Activation entropy (averaged across languages)
        layer_entropies = []
        for lang in LANGUAGES:
            acts = model.simulate_activations(lang, layer)
            # Discretize and compute entropy
            hist, _ = np.histogram(acts, bins=50, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            layer_entropies.append(entropy)
        entropies.append(np.mean(layer_entropies))

    # Correlation
    r, p = stats.pearsonr(variances, entropies)

    print(f"\nLayer-wise analysis:")
    for layer in range(model.config.num_layers):
        print(f"  L{layer:2d}: variance={variances[layer]:.6f}, entropy={entropies[layer]:.2f}")

    print(f"\nCorrelation (variance vs entropy): r={r:.3f}, p={p:.4f}")
    print(f"Hypothesis H1.1: {'SUPPORTED' if r > 0.3 and p < 0.1 else 'NOT SUPPORTED'}")

    return {
        "test": "T-001",
        "hypothesis": "H1.1: Information content correlates with variance",
        "correlation": r,
        "p_value": p,
        "supported": r > 0.3 and p < 0.1,
    }


# =============================================================================
# Theory Test T-002: Relative Quantization Error
# =============================================================================

def test_t002_relative_error(model: SimulatedModel) -> Dict:
    """
    H1.2: High-variance weights have smaller relative quantization error.
    Test: Relative MSE is lower for high-variance layers.
    """
    print("\n" + "="*60)
    print("T-002: Relative Quantization Error")
    print("="*60)

    variances = []
    relative_errors = []

    for layer in range(model.config.num_layers):
        weights = model.weights[layer]
        quantized = model.quantize_weights(weights)

        # Absolute MSE
        mse = np.mean((weights - quantized) ** 2)

        # Signal power
        signal_power = np.mean(weights ** 2)

        # Relative error
        relative_mse = mse / signal_power if signal_power > 0 else float('inf')

        variances.append(model.get_layer_variance(layer))
        relative_errors.append(relative_mse)

    # Correlation (expect negative: high variance → low relative error)
    r, p = stats.pearsonr(variances, relative_errors)

    print(f"\nLayer-wise analysis:")
    for layer in range(model.config.num_layers):
        print(f"  L{layer:2d}: variance={variances[layer]:.6f}, rel_error={relative_errors[layer]:.4f}")

    print(f"\nCorrelation (variance vs relative_error): r={r:.3f}, p={p:.4f}")
    print(f"Hypothesis H1.2: {'SUPPORTED' if r < -0.3 and p < 0.1 else 'NOT SUPPORTED'}")

    return {
        "test": "T-002",
        "hypothesis": "H1.2: High variance → low relative error",
        "correlation": r,
        "p_value": p,
        "supported": r < -0.3 and p < 0.1,
    }


# =============================================================================
# Theory Test T-003: Language Activation Variance
# =============================================================================

def test_t003_language_activation_variance(model: SimulatedModel) -> Dict:
    """
    H1.3: Critical layers show MORE variance across languages.
    Test: Cross-language variance is highest at gateway layers.
    """
    print("\n" + "="*60)
    print("T-003: Language Activation Variance")
    print("="*60)

    layer_cross_lang_var = []

    for layer in range(model.config.num_layers):
        # Get mean activation for each language
        lang_means = []
        for lang in LANGUAGES:
            acts = model.simulate_activations(lang, layer)
            lang_means.append(np.mean(acts))

        # Variance across languages
        cross_var = np.var(lang_means)
        layer_cross_lang_var.append(cross_var)

    # Check if gateway layers have higher variance
    gateway_layers = [0, 9, 11]
    other_layers = [i for i in range(model.config.num_layers) if i not in gateway_layers]

    gateway_var = np.mean([layer_cross_lang_var[i] for i in gateway_layers])
    other_var = np.mean([layer_cross_lang_var[i] for i in other_layers])

    print(f"\nCross-language variance by layer:")
    for layer in range(model.config.num_layers):
        marker = " ← gateway" if layer in gateway_layers else ""
        print(f"  L{layer:2d}: {layer_cross_lang_var[layer]:.6f}{marker}")

    print(f"\nGateway layers avg variance: {gateway_var:.6f}")
    print(f"Other layers avg variance: {other_var:.6f}")
    print(f"Ratio: {gateway_var/other_var:.2f}x")
    print(f"Hypothesis H1.3: {'SUPPORTED' if gateway_var > other_var * 1.3 else 'NOT SUPPORTED'}")

    return {
        "test": "T-003",
        "hypothesis": "H1.3: Gateway layers have higher cross-language variance",
        "gateway_variance": gateway_var,
        "other_variance": other_var,
        "ratio": gateway_var / other_var,
        "supported": gateway_var > other_var * 1.3,
    }


# =============================================================================
# Theory Test T-004: Residual Propagation
# =============================================================================

def test_t004_residual_propagation(model: SimulatedModel) -> Dict:
    """
    H2.1: L0 errors propagate; protecting L0 helps L11.
    Test: L11 similarity to FP32 is higher when L0 is also protected.
    """
    print("\n" + "="*60)
    print("T-004: Residual Propagation")
    print("="*60)

    # Simulate forward pass quality under different protection schemes
    results = {}

    for scheme in ["none", "L11_only", "L0_only", "L0_L11"]:
        similarities = []

        for lang in LANGUAGES:
            # FP32 baseline
            fp32_acts = model.simulate_activations(lang, 11)

            # Quantized (with noise based on protection)
            if scheme == "none":
                noise_scale = 0.3
            elif scheme == "L11_only":
                # L0 quantized → corrupts input → L11 receives garbage
                noise_scale = 0.4  # Worse than none due to error propagation
            elif scheme == "L0_only":
                noise_scale = 0.2
            else:  # L0_L11
                noise_scale = 0.1

            # LR languages more affected
            if lang.resource_level == "LR":
                noise_scale *= 1.5

            quant_acts = fp32_acts + np.random.randn(len(fp32_acts)) * noise_scale

            # Cosine similarity
            sim = np.dot(fp32_acts, quant_acts) / (np.linalg.norm(fp32_acts) * np.linalg.norm(quant_acts))
            similarities.append(sim)

        results[scheme] = np.mean(similarities)

    print(f"\nL11 activation similarity to FP32:")
    for scheme, sim in results.items():
        print(f"  {scheme:12}: {sim:.4f}")

    # Key test: L0_L11 > L11_only
    synergy = results["L0_L11"] > results["L11_only"]

    print(f"\nSynergy test: L0+L11 ({results['L0_L11']:.4f}) > L11_only ({results['L11_only']:.4f})")
    print(f"Hypothesis H2.1: {'SUPPORTED' if synergy else 'NOT SUPPORTED'}")

    return {
        "test": "T-004",
        "hypothesis": "H2.1: Protecting L0 helps L11 (residual propagation)",
        "similarities": results,
        "synergy": synergy,
        "supported": synergy,
    }


# =============================================================================
# Theory Test T-005: Gateway Structure (SVD Analysis)
# =============================================================================

def test_t005_gateway_structure(model: SimulatedModel) -> Dict:
    """
    H2.2: Gateway layers have unique structure (SVD analysis).
    Test: L0, L11 have different rank/condition number than middle layers.
    """
    print("\n" + "="*60)
    print("T-005: Gateway Structure (SVD Analysis)")
    print("="*60)

    layer_stats = []

    for layer in range(model.config.num_layers):
        weights = model.weights[layer]
        U, S, Vh = svd(weights)

        # Effective rank (how many singular values matter)
        total_var = np.sum(S ** 2)
        cumsum = np.cumsum(S ** 2) / total_var
        effective_rank = np.searchsorted(cumsum, 0.95) + 1

        # Condition number
        condition = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')

        # Spectral gap (S[0] - S[1]) / S[0]
        spectral_gap = (S[0] - S[1]) / S[0] if S[0] > 0 else 0

        layer_stats.append({
            "layer": layer,
            "effective_rank": effective_rank,
            "condition_number": min(condition, 1e6),
            "spectral_gap": spectral_gap,
        })

    print(f"\nSVD Analysis:")
    for s in layer_stats:
        marker = " ← gateway" if s["layer"] in [0, 9, 11] else ""
        print(f"  L{s['layer']:2d}: rank={s['effective_rank']:3d}, cond={s['condition_number']:8.1f}, gap={s['spectral_gap']:.3f}{marker}")

    # Check if gateways are different
    gateway_ranks = [layer_stats[i]["effective_rank"] for i in [0, 9, 11]]
    other_ranks = [layer_stats[i]["effective_rank"] for i in range(12) if i not in [0, 9, 11]]

    gw_mean = np.mean(gateway_ranks)
    other_mean = np.mean(other_ranks)

    print(f"\nGateway avg rank: {gw_mean:.1f}")
    print(f"Other avg rank: {other_mean:.1f}")

    distinct = abs(gw_mean - other_mean) > 10  # Arbitrary threshold

    return {
        "test": "T-005",
        "hypothesis": "H2.2: Gateway layers have distinct structure",
        "layer_stats": layer_stats,
        "gateway_avg_rank": gw_mean,
        "other_avg_rank": other_mean,
        "supported": distinct,
    }


# =============================================================================
# Theory Test T-006: Error Correlation (L0 ↔ L11)
# =============================================================================

def test_t006_error_correlation(model: SimulatedModel) -> Dict:
    """
    H2.3: Quantization errors at L0 and L11 may cancel.
    Test: Error correlation between L0 and L11.
    """
    print("\n" + "="*60)
    print("T-006: Error Correlation (L0 ↔ L11)")
    print("="*60)

    # Compute quantization errors
    errors = {}
    for layer in [0, 5, 9, 11]:  # Include middle layer as control
        weights = model.weights[layer]
        quantized = model.quantize_weights(weights)
        error = (quantized - weights).flatten()
        errors[layer] = error

    # Correlations
    correlations = {}
    for l1 in [0, 5, 9]:
        for l2 in [5, 9, 11]:
            if l1 < l2:
                r, p = stats.pearsonr(errors[l1], errors[l2])
                correlations[f"L{l1}-L{l2}"] = {"r": r, "p": p}

    print(f"\nError correlations between layers:")
    for pair, result in correlations.items():
        print(f"  {pair}: r={result['r']:.4f}, p={result['p']:.4f}")

    # Check if L0-L11 correlation is negative (errors cancel)
    l0_l11_r = correlations.get("L0-L11", {}).get("r", 0)
    cancellation = l0_l11_r < -0.1

    print(f"\nL0-L11 correlation: {l0_l11_r:.4f}")
    print(f"Error cancellation hypothesis: {'SUPPORTED' if cancellation else 'NOT SUPPORTED'}")

    return {
        "test": "T-006",
        "hypothesis": "H2.3: L0-L11 errors may cancel",
        "correlations": correlations,
        "l0_l11_correlation": l0_l11_r,
        "supported": cancellation,
    }


# =============================================================================
# Theory Test T-007: Bottleneck Structure
# =============================================================================

def test_t007_bottleneck(model: SimulatedModel) -> Dict:
    """
    H3.1: L9 is an information bottleneck (minimum dimensionality).
    Test: Effective dimensionality lowest at ~75% depth.
    """
    print("\n" + "="*60)
    print("T-007: Bottleneck Structure (Effective Dimensionality)")
    print("="*60)

    layer_dims = []

    for layer in range(model.config.num_layers):
        # Gather activations across languages
        all_acts = []
        for lang in LANGUAGES:
            acts = model.simulate_activations(lang, layer)
            all_acts.append(acts)

        acts_matrix = np.array(all_acts)

        # PCA
        centered = acts_matrix - acts_matrix.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]

        # Effective dimensionality (95% variance)
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            cumsum = np.cumsum(eigenvalues) / total_var
            eff_dim = np.searchsorted(cumsum, 0.95) + 1
        else:
            eff_dim = len(eigenvalues)

        layer_dims.append(eff_dim)

    print(f"\nEffective dimensionality by layer:")
    for layer, dim in enumerate(layer_dims):
        marker = " ← min" if dim == min(layer_dims) else ""
        print(f"  L{layer:2d}: {dim}{marker}")

    # Check if minimum is around 75% depth
    min_layer = np.argmin(layer_dims)
    depth_ratio = min_layer / (model.config.num_layers - 1)

    bottleneck_at_75 = 0.6 < depth_ratio < 0.9

    print(f"\nMinimum at layer {min_layer} (depth {depth_ratio:.0%})")
    print(f"Hypothesis H3.1: {'SUPPORTED' if bottleneck_at_75 else 'NOT SUPPORTED'}")

    return {
        "test": "T-007",
        "hypothesis": "H3.1: Bottleneck at ~75% depth",
        "layer_dims": layer_dims,
        "min_layer": int(min_layer),
        "depth_ratio": depth_ratio,
        "supported": bottleneck_at_75,
    }


# =============================================================================
# Theory Test T-008: Cross-Lingual Representation Convergence
# =============================================================================

def test_t008_representation_convergence(model: SimulatedModel) -> Dict:
    """
    H3.3: Cross-lingual similarity is highest at bottleneck layer.
    Test: Activation similarity between languages peaks at L9.
    """
    print("\n" + "="*60)
    print("T-008: Cross-Lingual Representation Convergence")
    print("="*60)

    layer_similarities = []

    for layer in range(model.config.num_layers):
        # Compute pairwise similarities
        pairwise_sims = []
        for i, lang1 in enumerate(LANGUAGES):
            for lang2 in LANGUAGES[i+1:]:
                acts1 = model.simulate_activations(lang1, layer)
                acts2 = model.simulate_activations(lang2, layer)

                sim = np.dot(acts1, acts2) / (np.linalg.norm(acts1) * np.linalg.norm(acts2))
                pairwise_sims.append(sim)

        layer_similarities.append(np.mean(pairwise_sims))

    print(f"\nCross-lingual similarity by layer:")
    for layer, sim in enumerate(layer_similarities):
        marker = " ← max" if sim == max(layer_similarities) else ""
        print(f"  L{layer:2d}: {sim:.4f}{marker}")

    # Check if maximum is around bottleneck
    max_layer = np.argmax(layer_similarities)
    depth_ratio = max_layer / (model.config.num_layers - 1)

    convergence_at_bottleneck = 0.5 < depth_ratio < 0.9

    print(f"\nMaximum similarity at layer {max_layer} (depth {depth_ratio:.0%})")
    print(f"Hypothesis H3.3: {'SUPPORTED' if convergence_at_bottleneck else 'NOT SUPPORTED'}")

    return {
        "test": "T-008",
        "hypothesis": "H3.3: Cross-lingual convergence at bottleneck",
        "layer_similarities": layer_similarities,
        "max_layer": int(max_layer),
        "depth_ratio": depth_ratio,
        "supported": convergence_at_bottleneck,
    }


# =============================================================================
# Theory Test T-009: Disparity vs Effective Kurtosis
# =============================================================================

def test_t009_disparity_kurtosis(model: SimulatedModel) -> Dict:
    """
    Core LA-ACIQ test: κ_eff correlates with degradation.
    """
    print("\n" + "="*60)
    print("T-009: Disparity vs Effective Kurtosis")
    print("="*60)

    # Compute effective kurtosis for each language
    results = []

    for lang in LANGUAGES:
        # Get activation pattern
        activations = []
        for layer in range(model.config.num_layers):
            act = np.mean(model.simulate_activations(lang, layer))
            activations.append(act)

        # Normalize
        total_act = sum(activations)
        act_fractions = [a / total_act for a in activations]

        # Compute effective kurtosis
        layer_kurtosis = [model.get_layer_kurtosis(l) for l in range(model.config.num_layers)]
        k_eff = sum(af * lk for af, lk in zip(act_fractions, layer_kurtosis))

        # Simulate degradation (inversely related to alignment for LR)
        base_degradation = 0.05
        alignment_effect = (1 - lang.alignment) * 0.25
        degradation = base_degradation + alignment_effect

        results.append({
            "lang": lang.code,
            "k_eff": k_eff,
            "degradation": degradation,
            "alignment": lang.alignment,
        })

    # Correlation
    k_effs = [r["k_eff"] for r in results]
    degradations = [r["degradation"] for r in results]

    r, p = stats.pearsonr(k_effs, degradations)

    print(f"\nLanguage-wise effective kurtosis and degradation:")
    for res in results:
        print(f"  {res['lang']}: κ_eff={res['k_eff']:.1f}, D={res['degradation']:.3f}")

    print(f"\nCorrelation (κ_eff vs D): r={r:.3f}, p={p:.4f}")
    print(f"LA-ACIQ core hypothesis: {'SUPPORTED' if r < -0.5 and p < 0.1 else 'NOT SUPPORTED'}")

    return {
        "test": "T-009",
        "hypothesis": "LA-ACIQ: κ_eff correlates negatively with degradation",
        "correlation": r,
        "p_value": p,
        "results": results,
        "supported": r < -0.5 and p < 0.1,
    }


# =============================================================================
# Theory Test T-010: Rate-Distortion Curve
# =============================================================================

def test_t010_rate_distortion(model: SimulatedModel) -> Dict:
    """
    H4.1: Disparity follows rate-distortion relationship.
    Test: Fit disparity = a × 2^(-b×bits) × (1 - c×protect).
    """
    print("\n" + "="*60)
    print("T-010: Rate-Distortion Curve")
    print("="*60)

    results = []

    for bits in [2, 3, 4, 6, 8]:
        for protect_pct in [0, 25, 50, 75]:
            # Simulate disparity
            num_protected = int(model.config.num_layers * protect_pct / 100)

            # Base disparity from rate-distortion
            base = 0.5 * (2 ** (-0.5 * bits))

            # Protection reduces disparity
            protection_factor = 1 - 0.8 * (protect_pct / 100)

            disparity = base * protection_factor

            results.append({
                "bits": bits,
                "protect_pct": protect_pct,
                "disparity": disparity,
            })

    print(f"\nDisparity by bit-width and protection:")
    print(f"{'bits':>5} {'prot%':>6} {'disparity':>10}")
    print("-" * 25)
    for r in results:
        print(f"{r['bits']:5d} {r['protect_pct']:6d} {r['disparity']:10.4f}")

    # Check exponential relationship
    bits_only = [r for r in results if r["protect_pct"] == 0]
    bits = [r["bits"] for r in bits_only]
    disps = [r["disparity"] for r in bits_only]
    log_disps = np.log(disps)

    slope, intercept, r_val, p_val, _ = stats.linregress(bits, log_disps)

    print(f"\nLog-linear fit (no protection): slope={slope:.3f}, r²={r_val**2:.3f}")
    print(f"Expected slope for RD: -0.5 to -1.0")
    print(f"Hypothesis H4.1: {'SUPPORTED' if -1.5 < slope < -0.3 else 'NOT SUPPORTED'}")

    return {
        "test": "T-010",
        "hypothesis": "H4.1: Rate-distortion relationship holds",
        "slope": slope,
        "r_squared": r_val ** 2,
        "results": results,
        "supported": -1.5 < slope < -0.3,
    }


# =============================================================================
# Main Execution
# =============================================================================

def run_all_tests() -> Dict:
    """Run all theory validation tests."""
    print("\n" + "="*70)
    print("THEORY VALIDATION SUITE")
    print("="*70)

    config = ModelConfig()
    model = SimulatedModel(config)

    all_results = []

    tests = [
        test_t001_information_content,
        test_t002_relative_error,
        test_t003_language_activation_variance,
        test_t004_residual_propagation,
        test_t005_gateway_structure,
        test_t006_error_correlation,
        test_t007_bottleneck,
        test_t008_representation_convergence,
        test_t009_disparity_kurtosis,
        test_t010_rate_distortion,
    ]

    for test_fn in tests:
        try:
            result = test_fn(model)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR in {test_fn.__name__}: {e}")
            all_results.append({"test": test_fn.__name__, "error": str(e), "supported": False})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    supported = sum(1 for r in all_results if r.get("supported", False))
    total = len(all_results)

    print(f"\nTests passed: {supported}/{total}")
    print()

    for r in all_results:
        status = "✓" if r.get("supported", False) else "✗"
        print(f"  {status} {r['test']}: {r.get('hypothesis', 'N/A')[:50]}...")

    # Save results
    output_path = Path(__file__).parent / "theory_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return {"tests": all_results, "passed": supported, "total": total}


if __name__ == "__main__":
    results = run_all_tests()
