"""
Utility functions.
"""
import json
import torch
from pathlib import Path
from typing import Dict, Any


def load_texts(path: str = None) -> Dict[str, str]:
    """Load test texts from JSON file."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "test_texts.json"

    with open(path) as f:
        return json.load(f)


def save_results(results: Dict[str, Any], path: str):
    """Save results to JSON."""
    # Convert numpy/torch types to Python types
    def convert(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def print_results(results: Dict, title: str = "Results"):
    """Pretty print results."""
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)

    if 'disparity' in results:
        print(f"\n{'Lang':<6} {'Disparity':>12} {'Assessment':>15}")
        print('-' * 35)
        for lang, disp in sorted(results['disparity'].items(), key=lambda x: x[1]):
            if lang == 'en':
                cat = "Reference"
            elif disp < 0.5:
                cat = "EXCELLENT"
            elif disp < 1.0:
                cat = "VERY GOOD"
            elif disp < 2.0:
                cat = "GOOD"
            else:
                cat = "NEEDS WORK"
            print(f"{lang:<6} {disp:>11.2f}x {cat:>15}")

    if 'avg_disparity' in results:
        print(f"\nAverage disparity: {results['avg_disparity']:.2f}x")

    print('=' * 60)


def get_model_info(model) -> Dict:
    """Get model information."""
    info = {
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_layers': None,
        'hidden_size': None,
        'vocab_size': None,
    }

    if hasattr(model, 'config'):
        config = model.config
        for attr in ['n_layer', 'num_hidden_layers', 'num_layers']:
            if hasattr(config, attr):
                info['num_layers'] = getattr(config, attr)
                break

        if hasattr(config, 'hidden_size'):
            info['hidden_size'] = config.hidden_size
        if hasattr(config, 'vocab_size'):
            info['vocab_size'] = config.vocab_size

    return info
