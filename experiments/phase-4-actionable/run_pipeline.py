#!/usr/bin/env python3
"""
Systematic Experiment Pipeline Runner

Runs experiments sequentially with:
- Long timeouts (configurable per experiment)
- Result evaluation and scoring
- Automatic synthesis of findings
- Error handling and recovery

Usage:
    python3 run_pipeline.py [--start-from PHASE]
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import gc
import sys
import argparse

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PIPELINE_STATE_FILE = RESULTS_DIR / "pipeline_state.json"

# Test texts - same across all experiments for consistency
TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog and runs through the forest looking for food.',
    'de': 'Der schnelle braune Fuchs springt über den faulen Hund und rennt durch den Wald.',
    'fr': 'Le renard brun rapide saute par-dessus le chien paresseux et court dans la forêt.',
    'zh': '敏捷的棕色狐狸跳过懒狗，穿过森林寻找食物。',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן ורץ ביער בחיפוש אחר אוכל.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول ويجري عبر الغابة.',
}

RESOURCE_LEVELS = {
    'en': 1.0,
    'de': 0.85,
    'fr': 0.80,
    'zh': 0.50,
    'he': 0.15,
    'ar': 0.25,
}


def get_hr_lr_langs():
    """Split languages into high and low resource."""
    hr = [l for l, r in RESOURCE_LEVELS.items() if r > 0.5]
    lr = [l for l, r in RESOURCE_LEVELS.items() if r <= 0.5]
    return hr, lr


def log(msg: str, level: str = "INFO"):
    """Timestamped logging."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def save_state(state: dict):
    """Save pipeline state for recovery."""
    with open(PIPELINE_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_state() -> dict:
    """Load pipeline state if exists."""
    if PIPELINE_STATE_FILE.exists():
        with open(PIPELINE_STATE_FILE) as f:
            return json.load(f)
    return {"completed": [], "current": None, "results": {}}


class ExperimentRunner:
    """Runs a series of experiments with state tracking."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.original_state = None
        self.state = load_state()

    def setup_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return

        log(f"Loading model: {self.model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

        # Save original state
        log("Saving original model state")
        self.original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Count weights
        self.total_weights = sum(
            p.numel() for n, p in self.model.named_parameters() if 'weight' in n
        )
        log(f"Total weight parameters: {self.total_weights:,}")

    def restore_model(self):
        """Restore model to original state."""
        if self.original_state is None:
            return
        self.model.load_state_dict(self.original_state)
        self.model.eval()

    def compute_ppl(self, text: str) -> float:
        """Compute perplexity for a single text."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()

    def compute_all_ppl(self) -> Dict[str, float]:
        """Compute perplexity for all languages."""
        return {lang: self.compute_ppl(text) for lang, text in TEXTS.items()}

    def quantize_int4(self, preserve_pct: float = 0.0) -> int:
        """
        Apply INT4 quantization with optional preservation.
        Returns number of preserved weights.
        """
        preserved_count = 0

        # Compute global threshold if preserving
        if preserve_pct > 0:
            all_mags = []
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    all_mags.append(param.data.abs().view(-1))
            all_mags = torch.cat(all_mags)
            n_preserve = max(1, int(len(all_mags) * preserve_pct / 100))
            sorted_mags, _ = all_mags.sort(descending=True)
            threshold = sorted_mags[n_preserve - 1].item()
            del all_mags, sorted_mags
        else:
            threshold = float('inf')

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' not in name:
                    continue

                flat = param.data.view(-1)
                abs_vals = flat.abs()

                # INT4 quantization
                abs_max = abs_vals.max()
                if abs_max > 0:
                    scale = abs_max / 7.0
                    quantized = torch.round(flat / scale)
                    quantized = torch.clamp(quantized, -8, 7)
                    dequantized = quantized * scale

                    # Preservation mask
                    if preserve_pct > 0:
                        mask = abs_vals >= threshold
                        new_weights = torch.where(mask, flat, dequantized)
                        preserved_count += mask.sum().item()
                    else:
                        new_weights = dequantized

                    param.data.copy_(new_weights.view(param.shape))

        gc.collect()
        return preserved_count

    # === PHASE A: BASELINE ===

    def run_A001_baseline(self) -> dict:
        """A-001: Compute baseline perplexity per language."""
        log("Running A-001: Baseline Perplexity")
        self.setup_model()
        self.restore_model()

        ppls = self.compute_all_ppl()

        result = {
            "id": "A-001",
            "name": "Baseline Perplexity",
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "perplexity": ppls,
            "status": "SUCCESS",
        }

        log(f"A-001 Complete: {ppls}")
        return result

    def run_A002_int4_degradation(self) -> dict:
        """A-002: Compute INT4 degradation per language."""
        log("Running A-002: INT4 Degradation")
        self.setup_model()

        # Need baseline first
        if "A-001" not in self.state["results"]:
            self.state["results"]["A-001"] = self.run_A001_baseline()
            save_state(self.state)

        baseline = self.state["results"]["A-001"]["perplexity"]

        # Apply INT4 quantization (no preservation)
        self.restore_model()
        self.quantize_int4(preserve_pct=0)

        quant_ppl = self.compute_all_ppl()

        # Compute degradation
        degradation = {}
        for lang in baseline:
            if lang in quant_ppl:
                deg_pct = (quant_ppl[lang] - baseline[lang]) / baseline[lang] * 100
                degradation[lang] = deg_pct

        self.restore_model()

        result = {
            "id": "A-002",
            "name": "INT4 Degradation",
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "baseline_ppl": baseline,
            "quantized_ppl": quant_ppl,
            "degradation_pct": degradation,
            "status": "SUCCESS",
        }

        log(f"A-002 Complete: Degradation = {degradation}")
        return result

    def run_A003_disparity_ratio(self) -> dict:
        """A-003: Compute disparity ratio (LR/HR)."""
        log("Running A-003: Disparity Ratio")

        # Need A-002 first
        if "A-002" not in self.state["results"]:
            self.state["results"]["A-002"] = self.run_A002_int4_degradation()
            save_state(self.state)

        degradation = self.state["results"]["A-002"]["degradation_pct"]

        hr, lr = get_hr_lr_langs()

        hr_degs = [degradation[l] for l in hr if l in degradation]
        lr_degs = [degradation[l] for l in lr if l in degradation]

        hr_avg = np.mean(hr_degs) if hr_degs else 0
        lr_avg = np.mean(lr_degs) if lr_degs else 0

        if hr_avg > 0:
            disparity_ratio = lr_avg / hr_avg
        else:
            disparity_ratio = float('inf')

        result = {
            "id": "A-003",
            "name": "Disparity Ratio",
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "high_resource_langs": hr,
            "low_resource_langs": lr,
            "hr_avg_degradation_pct": hr_avg,
            "lr_avg_degradation_pct": lr_avg,
            "disparity_ratio": disparity_ratio,
            "status": "SUCCESS",
            "interpretation": self._interpret_disparity(disparity_ratio),
        }

        log(f"A-003 Complete: Disparity = {disparity_ratio:.2f}x")
        return result

    def _interpret_disparity(self, ratio: float) -> str:
        if ratio > 50:
            return "MASSIVE disparity - low-resource languages severely impacted"
        elif ratio > 10:
            return "SIGNIFICANT disparity - fairness concern for deployment"
        elif ratio > 2:
            return "MODERATE disparity - measurable but manageable"
        elif ratio > 1:
            return "MILD disparity - low-resource slightly worse"
        else:
            return "NO disparity - languages affected equally"

    # === PHASE B: PRESERVATION ===

    def run_B_preservation(self, pct: float) -> dict:
        """B-series: Test specific preservation percentage."""
        exp_id = f"B-{int(pct):03d}"
        log(f"Running {exp_id}: {pct}% Preservation")
        self.setup_model()

        # Need baseline
        if "A-001" not in self.state["results"]:
            self.state["results"]["A-001"] = self.run_A001_baseline()
            save_state(self.state)

        baseline = self.state["results"]["A-001"]["perplexity"]

        # Apply quantization with preservation
        self.restore_model()
        preserved = self.quantize_int4(preserve_pct=pct)

        quant_ppl = self.compute_all_ppl()

        # Compute degradation
        degradation = {}
        for lang in baseline:
            if lang in quant_ppl:
                deg_pct = (quant_ppl[lang] - baseline[lang]) / baseline[lang] * 100
                degradation[lang] = deg_pct

        hr, lr = get_hr_lr_langs()
        hr_degs = [degradation[l] for l in hr if l in degradation]
        lr_degs = [degradation[l] for l in lr if l in degradation]

        hr_avg = np.mean(hr_degs) if hr_degs else 0
        lr_avg = np.mean(lr_degs) if lr_degs else 0
        disparity = lr_avg / hr_avg if hr_avg > 0 else float('inf')

        self.restore_model()

        result = {
            "id": exp_id,
            "name": f"{pct}% Preservation",
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "preserve_pct": pct,
            "weights_preserved": preserved,
            "weights_preserved_pct_actual": preserved / self.total_weights * 100,
            "degradation_pct": degradation,
            "hr_avg_degradation": hr_avg,
            "lr_avg_degradation": lr_avg,
            "disparity_ratio": disparity,
            "status": "SUCCESS",
        }

        log(f"{exp_id} Complete: Disparity = {disparity:.2f}x (HR={hr_avg:.0f}%, LR={lr_avg:.0f}%)")
        return result

    # === PHASE B SYNTHESIS ===

    def run_B004_optimal(self) -> dict:
        """B-004: Determine optimal preservation percentage."""
        log("Running B-004: Optimal Preservation Analysis")

        # Run all preservation experiments
        for pct in [0, 5, 10, 20]:
            exp_id = f"B-{pct:03d}"
            if exp_id not in self.state["results"]:
                self.state["results"][exp_id] = self.run_B_preservation(pct)
                save_state(self.state)

        # Analyze
        pcts = [0, 5, 10, 20]
        disparities = []
        hr_degs = []
        lr_degs = []

        for pct in pcts:
            exp_id = f"B-{pct:03d}"
            r = self.state["results"][exp_id]
            disparities.append(r["disparity_ratio"])
            hr_degs.append(r["hr_avg_degradation"])
            lr_degs.append(r["lr_avg_degradation"])

        # Find minimum disparity
        finite_disps = [(p, d) for p, d in zip(pcts, disparities) if np.isfinite(d)]

        if finite_disps:
            optimal_pct, min_disparity = min(finite_disps, key=lambda x: x[1])
        else:
            optimal_pct, min_disparity = None, float('inf')

        # Correlation analysis
        from scipy.stats import pearsonr
        finite_pairs = [(p, d) for p, d in zip(pcts, disparities) if np.isfinite(d)]
        if len(finite_pairs) >= 3:
            ps, ds = zip(*finite_pairs)
            r, p = pearsonr(ps, ds)
        else:
            r, p = float('nan'), float('nan')

        result = {
            "id": "B-004",
            "name": "Optimal Preservation",
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "tested_percentages": pcts,
            "disparities": disparities,
            "hr_degradations": hr_degs,
            "lr_degradations": lr_degs,
            "optimal_pct": optimal_pct,
            "min_disparity": min_disparity,
            "correlation_r": r,
            "correlation_p": p,
            "hypothesis_supported": r < -0.3 if np.isfinite(r) else False,
            "status": "SUCCESS",
            "conclusion": self._interpret_preservation(r, optimal_pct, min_disparity),
        }

        log(f"B-004 Complete: Optimal = {optimal_pct}%, Min Disparity = {min_disparity:.2f}x")
        return result

    def _interpret_preservation(self, r: float, opt_pct: Optional[float], min_disp: float) -> str:
        if r < -0.5:
            return f"STRONG effect: Higher preservation strongly reduces disparity. Optimal: {opt_pct}%"
        elif r < -0.3:
            return f"MODERATE effect: Preservation helps reduce disparity. Optimal: {opt_pct}%"
        elif r > 0.3:
            return "OPPOSITE effect: Higher preservation increases disparity (unexpected)"
        else:
            return "NO CLEAR effect: Preservation does not consistently affect disparity"

    # === MAIN RUNNER ===

    def run_full_pipeline(self, start_from: str = "A"):
        """Run the complete experiment pipeline."""
        log("=" * 60)
        log("STARTING EXPERIMENT PIPELINE")
        log("=" * 60)

        experiments = []

        # Phase A
        if start_from <= "A":
            experiments.extend([
                ("A-001", self.run_A001_baseline),
                ("A-002", self.run_A002_int4_degradation),
                ("A-003", self.run_A003_disparity_ratio),
            ])

        # Phase B
        if start_from <= "B":
            experiments.extend([
                ("B-000", lambda: self.run_B_preservation(0)),
                ("B-005", lambda: self.run_B_preservation(5)),
                ("B-010", lambda: self.run_B_preservation(10)),
                ("B-020", lambda: self.run_B_preservation(20)),
                ("B-004", self.run_B004_optimal),
            ])

        # Run each experiment
        for exp_id, exp_func in experiments:
            if exp_id in self.state["completed"]:
                log(f"Skipping {exp_id} (already completed)")
                continue

            log(f"Running {exp_id}...")
            self.state["current"] = exp_id

            try:
                result = exp_func()
                self.state["results"][exp_id] = result
                self.state["completed"].append(exp_id)
                save_state(self.state)

                # Save individual result file
                result_file = RESULTS_DIR / f"{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                log(f"Saved: {result_file}")

            except Exception as e:
                log(f"FAILED: {exp_id} - {e}", "ERROR")
                self.state["current"] = None
                save_state(self.state)
                raise

        log("=" * 60)
        log("PIPELINE COMPLETE")
        log("=" * 60)

        # Final synthesis
        self.synthesize_findings()

    def synthesize_findings(self):
        """Create summary of all findings."""
        log("Synthesizing findings...")

        synthesis = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "experiments_completed": self.state["completed"],
        }

        # Phase A findings
        if "A-003" in self.state["results"]:
            a3 = self.state["results"]["A-003"]
            synthesis["disparity_confirmed"] = a3["disparity_ratio"] > 2
            synthesis["baseline_disparity"] = a3["disparity_ratio"]
            synthesis["disparity_interpretation"] = a3["interpretation"]

        # Phase B findings
        if "B-004" in self.state["results"]:
            b4 = self.state["results"]["B-004"]
            synthesis["preservation_helps"] = b4["hypothesis_supported"]
            synthesis["optimal_preservation"] = b4["optimal_pct"]
            synthesis["optimal_disparity"] = b4["min_disparity"]
            synthesis["preservation_conclusion"] = b4["conclusion"]

            # Improvement calculation
            if "A-003" in self.state["results"]:
                baseline_disp = self.state["results"]["A-003"]["disparity_ratio"]
                if np.isfinite(baseline_disp) and np.isfinite(b4["min_disparity"]):
                    improvement = (baseline_disp - b4["min_disparity"]) / baseline_disp * 100
                    synthesis["disparity_improvement_pct"] = improvement

        # Research conclusions
        conclusions = []

        if synthesis.get("disparity_confirmed"):
            conclusions.append(
                f"INT4 quantization creates {synthesis['baseline_disparity']:.1f}x disparity "
                f"between low and high resource languages"
            )

        if synthesis.get("preservation_helps"):
            conclusions.append(
                f"Preserving top {synthesis['optimal_preservation']}% of weights reduces "
                f"disparity to {synthesis['optimal_disparity']:.1f}x"
            )
            if "disparity_improvement_pct" in synthesis:
                conclusions.append(
                    f"This represents a {synthesis['disparity_improvement_pct']:.1f}% improvement"
                )
        elif "preservation_helps" in synthesis:
            conclusions.append(
                "Weight preservation does NOT consistently reduce disparity in this model"
            )

        synthesis["conclusions"] = conclusions

        # Save synthesis
        synthesis_file = RESULTS_DIR / f"SYNTHESIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(synthesis_file, 'w') as f:
            json.dump(synthesis, f, indent=2, default=str)

        log(f"Synthesis saved: {synthesis_file}")

        # Print conclusions
        log("=" * 60)
        log("CONCLUSIONS")
        log("=" * 60)
        for c in conclusions:
            log(f"  - {c}")

        return synthesis


def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline")
    parser.add_argument("--start-from", default="A", choices=["A", "B", "C"],
                        help="Phase to start from (default: A)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset pipeline state and start fresh")
    args = parser.parse_args()

    if args.reset and PIPELINE_STATE_FILE.exists():
        PIPELINE_STATE_FILE.unlink()
        log("Pipeline state reset")

    runner = ExperimentRunner(model_name="gpt2")
    runner.run_full_pipeline(start_from=args.start_from)


if __name__ == "__main__":
    main()
