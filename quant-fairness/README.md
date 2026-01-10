# quant-fairness

Find critical layers for fair multilingual quantization.

## The Problem

INT4 quantization degrades low-resource languages 160x more than English.
This tool identifies which layers to protect to eliminate that disparity.

## Quick Start

```bash
pip install quant-fairness

# Find critical layers for any model
quant-fairness sweep --model gpt2

# Get quick recommendation (outputs layer indices)
quant-fairness recommend --model gpt2 --top 3
```

## Known Good Configurations

| Model | Protect | Overhead | Avg Disparity |
|-------|---------|----------|---------------|
| GPT-2 | L0, L9, L11 | 17% | 0.59x |
| OPT-125M | L4, L6, L11 | 17% | 12.7x |

## Usage

### Full Layer Sweep

```bash
# Test all layers, find critical ones
quant-fairness sweep --model gpt2 --langs en,he,ar,zh

# Output as JSON
quant-fairness sweep --model gpt2 --json
```

### Quick Recommendation

```bash
# Get recommended layers (pipe-friendly)
LAYERS=$(quant-fairness recommend --model gpt2)
echo "Protect layers: $LAYERS"
```

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant_fairness import quick_sweep, simulate_int4

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Find critical layers
layers = quick_sweep(model, tokenizer, top_n=3)
print(f"Protect: {layers}")  # [0, 9, 11]

# Apply protection during quantization
simulate_int4(model, protect_layers=set(layers))
```

## How It Works

1. Protect each layer individually
2. Measure disparity (LR degradation / English degradation)
3. Rank by disparity (lower = more critical)
4. Recommend top-N layers

The sweep takes ~30 seconds for GPT-2 scale models.

## Background

Based on 80 experiments showing:
- Layer 0 (input gateway) and Layer 11 (output gateway) form a critical pair
- Layer 9 adds consolidation benefit at 75% depth
- Architecture-specific: different models need different layers
- Structure matters: random or magnitude-based selection fails

See: [Multilingual Quantization Disparity Research](https://github.com/uprootiny/quant-disparity)

## Citation

```bibtex
@misc{quantdisparity2026,
  title={Gateway Layers: Multilingual Quantization Disparity},
  year={2026},
  note={80 experiments on GPT-2 and OPT-125M},
  url={https://github.com/uprootiny/quant-disparity}
}
```
