"""
quant-fairness: Find critical layers for fair multilingual quantization.

Based on 80 experiments showing that protecting specific "gateway" layers
eliminates most quantization disparity between languages.
"""

__version__ = "0.1.0"

from .sweep import layer_sweep, quick_sweep
from .quantize import simulate_int4, measure_disparity

__all__ = ["layer_sweep", "quick_sweep", "simulate_int4", "measure_disparity"]
