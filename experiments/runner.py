#!/usr/bin/env python3
"""
Experiment Runner & Orchestrator

Dispatches experiments, tracks status, and collects results.
"""

import subprocess
import json
import time
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys


# Experiment registry
EXPERIMENTS = {
    "exp022": {
        "name": "Architecture Comparison",
        "script": "phase-3-crossmodel/exp022_architecture.py",
        "hypotheses": ["H1", "H4"],
        "timeout": 300,
        "status": "complete",
    },
    "exp023": {
        "name": "Training Config",
        "script": "phase-3-crossmodel/exp023_training_config.py",
        "hypotheses": ["H2"],
        "timeout": 120,
        "status": "complete",
    },
    "exp024": {
        "name": "Layer Position",
        "script": "phase-3-crossmodel/exp024_layer_position.py",
        "hypotheses": ["H3"],
        "timeout": 300,
        "status": "complete",
    },
    "exp025": {
        "name": "Size Scaling",
        "script": "phase-3-crossmodel/exp025_size_scaling.py",
        "hypotheses": ["H5"],
        "timeout": 300,
        "status": "complete",
    },
    "exp026": {
        "name": "Attention Components",
        "script": "phase-3-crossmodel/exp026_attention_components.py",
        "hypotheses": ["H4b"],
        "timeout": 300,
        "status": "pending",
    },
    "exp027": {
        "name": "Checkpoint Evolution",
        "script": "phase-3-crossmodel/exp027_checkpoint_evolution.py",
        "hypotheses": ["H7"],
        "timeout": 600,
        "status": "pending",
    },
}


def run_experiment(exp_id: str, exp_info: dict, base_dir: Path) -> dict:
    """Run a single experiment and capture results."""
    start = time.time()
    script_path = base_dir / exp_info["script"]

    result = {
        "id": exp_id,
        "name": exp_info["name"],
        "hypotheses": exp_info["hypotheses"],
        "started": datetime.now().isoformat(),
        "status": "running",
    }

    try:
        proc = subprocess.run(
            ["python3", str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            timeout=exp_info["timeout"],
        )

        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["returncode"] = proc.returncode
        result["status"] = "completed" if proc.returncode == 0 else "failed"

        # Try to extract verdict from output
        for line in proc.stdout.split("\n"):
            if "verdict" in line.lower() or "SUPPORTED" in line or "REJECTED" in line:
                result.setdefault("verdicts", []).append(line.strip())

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Exceeded {exp_info['timeout']}s"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    result["duration"] = time.time() - start
    result["finished"] = datetime.now().isoformat()

    return result


def run_parallel(exp_ids: list, max_workers: int = 2) -> list:
    """Run multiple experiments in parallel."""
    base_dir = Path(__file__).parent

    results = []
    print(f"\n{'='*60}")
    print(f"DISPATCHING {len(exp_ids)} EXPERIMENTS")
    print(f"{'='*60}\n")

    for exp_id in exp_ids:
        if exp_id in EXPERIMENTS:
            print(f"  [{exp_id}] {EXPERIMENTS[exp_id]['name']}")
        else:
            print(f"  [{exp_id}] UNKNOWN - skipping")

    print(f"\nMax parallel workers: {max_workers}")
    print("-"*60)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for exp_id in exp_ids:
            if exp_id not in EXPERIMENTS:
                continue
            future = executor.submit(
                run_experiment,
                exp_id,
                EXPERIMENTS[exp_id],
                base_dir
            )
            futures[future] = exp_id
            print(f"  Dispatched: {exp_id}")

        for future in as_completed(futures):
            exp_id = futures[future]
            try:
                result = future.result()
                results.append(result)

                status_icon = "✓" if result["status"] == "completed" else "✗"
                print(f"  {status_icon} Finished: {exp_id} ({result['duration']:.1f}s) - {result['status']}")

                if result.get("verdicts"):
                    for v in result["verdicts"][:2]:
                        print(f"      → {v[:70]}")

            except Exception as e:
                print(f"  ✗ Error in {exp_id}: {e}")
                results.append({"id": exp_id, "status": "error", "error": str(e)})

    return results


def print_summary(results: list):
    """Print summary of all experiment results."""
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}\n")

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] in ["failed", "error", "timeout"])

    print(f"Total: {len(results)} | Completed: {completed} | Failed: {failed}\n")

    print(f"{'ID':<10} {'Status':<12} {'Duration':<10} {'Hypotheses'}")
    print("-"*50)

    for r in results:
        hyp = ", ".join(EXPERIMENTS.get(r["id"], {}).get("hypotheses", []))
        dur = f"{r.get('duration', 0):.1f}s"
        print(f"{r['id']:<10} {r['status']:<12} {dur:<10} {hyp}")

    # Verdicts
    print(f"\n{'='*60}")
    print("VERDICTS")
    print(f"{'='*60}\n")

    for r in results:
        if r.get("verdicts"):
            print(f"[{r['id']}]")
            for v in r["verdicts"]:
                print(f"  {v}")
            print()


def save_run(results: list, run_id: str = None):
    """Save run results to JSON."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(__file__).parent / "runs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"run_{run_id}.json"

    # Don't save full stdout/stderr in summary
    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if k not in ["stdout", "stderr"]}
        summary.append(s)

    output_file.write_text(json.dumps({
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": summary,
    }, indent=2))

    print(f"\nRun saved to: {output_file}")
    return output_file


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("experiments", nargs="*", default=["exp023", "exp025"],
                        help="Experiment IDs to run")
    parser.add_argument("-p", "--parallel", type=int, default=2,
                        help="Max parallel workers")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List available experiments")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for exp_id, info in EXPERIMENTS.items():
            print(f"  {exp_id}: {info['name']} (tests {', '.join(info['hypotheses'])})")
        return

    results = run_parallel(args.experiments, args.parallel)
    print_summary(results)
    save_run(results)


if __name__ == "__main__":
    main()
