# Multilingual Quantization Disparity - GPU Experiments

Validate gateway layer findings on larger models.

## Quick Start (Colab)

```python
!git clone https://github.com/uprootiny/quant-disparity-gpu.git
%cd quant-disparity-gpu
!pip install -q -r requirements.txt

# Run validation
%run notebooks/01_quick_validation.ipynb
```

## Quick Start (Kaggle)

1. Create new notebook
2. Add this repo as data source, or:
```python
!git clone https://github.com/uprootiny/quant-disparity-gpu.git
%cd quant-disparity-gpu
!pip install -q -r requirements.txt
```

## Experiments

| Notebook | Time | GPU | Goal |
|----------|------|-----|------|
| `01_quick_validation.ipynb` | 30m | T4 | Does L0+L31 work on Llama? |
| `02_layer_sweep.ipynb` | 2h | T4 | Find Llama's critical layers |
| `03_real_gptq.ipynb` | 45m | T4 | Test actual GPTQ quantization |
| `04_cross_model.ipynb` | 3h | T4 | Mistral, Qwen comparison |
| `05_mmlu_eval.ipynb` | 2h | T4 | Task performance vs perplexity |

## Core Findings to Validate

From 80 CPU experiments on GPT-2:

| Config | Overhead | Disparity |
|--------|----------|-----------|
| No protection | 0% | 160x |
| L0 + L11 | 11.5% | 0.92x |
| **L0 + L9 + L11** | **17%** | **0.59x** |

**Key question:** Does this transfer to 7B+ models?

## Usage

```python
from src.disparity import measure_disparity, quick_sweep
from src.quantize import simulate_int4, protect_layers

# Load any model
model = AutoModelForCausalLM.from_pretrained("model_id", torch_dtype=torch.float16)

# Quick sweep to find critical layers
critical = quick_sweep(model, tokenizer, top_n=3)
print(f"Protect layers: {critical}")

# Apply protection and quantize
protect_layers(model, critical)
simulate_int4(model, exclude=critical)

# Measure disparity
results = measure_disparity(model, tokenizer)
```

## Structure

```
gpu-experiments/
├── README.md
├── requirements.txt
├── setup.py
├── notebooks/
│   ├── 01_quick_validation.ipynb
│   ├── 02_layer_sweep.ipynb
│   ├── 03_real_gptq.ipynb
│   ├── 04_cross_model.ipynb
│   └── 05_mmlu_eval.ipynb
├── src/
│   ├── __init__.py
│   ├── disparity.py
│   ├── quantize.py
│   └── utils.py
└── data/
    └── test_texts.json
```

## Expected Results

**If pattern transfers:**
- Llama L0+L31 achieves < 2x disparity
- Critical layers at boundaries (input/output)
- Paper findings validated at scale

**If pattern doesn't transfer:**
- Layer sweep reveals Llama-specific layers
- Different architecture = different criticality
- Still valuable: method generalizes, layers don't

## Citation

```bibtex
@misc{quantdisparity2026,
  title={Gateway Layers: Multilingual Quantization Disparity},
  year={2026},
  url={https://github.com/uprootiny/quant-disparity}
}
```
