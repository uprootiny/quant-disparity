# Gemini Colab Runner Guide

Instructions for Gemini AI to execute the LA-ACIQ quantization disparity experiments within Google Colab.

## Overview

This experimental suite tests 6 hypotheses about multilingual quantization disparity in LLMs. Each notebook is self-contained and designed for T4 GPU (16GB VRAM).

```
experiments/colab-suite/
├── HYPOTHESIS_REGISTRY.json    # Hypothesis definitions and experiment mappings
├── METHODOLOGY.md              # Scientific methodology reference
├── exp001_disparity_validation.ipynb  # H1, H3: Baseline disparity
├── exp002_bloom1b7.ipynb              # H1, H3: Scale validation
├── exp003_kurtosis_analysis.ipynb     # H2: Kurtosis correlation
├── exp004_rate_distortion.ipynb       # H4: Shannon rate-distortion
├── exp005_layer_ablation.ipynb        # H5: Layer criticality
└── exp006_laaciq_intervention.ipynb   # H6: LA-ACIQ effectiveness
```

## Prerequisites

Before running any experiment:

1. **Verify GPU availability:**
   ```python
   import torch
   assert torch.cuda.is_available(), "GPU required"
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   ```

2. **Check available memory:**
   - T4: 16GB VRAM (sufficient for all experiments)
   - A100: 40GB VRAM (can run larger batch sizes)
   - If VRAM < 12GB, only run exp001, exp003, exp005

3. **Runtime settings:**
   - Go to Runtime > Change runtime type > GPU (T4 recommended)
   - Enable "High-RAM" if available for exp002

## Execution Order

Run experiments in this order for logical progression:

```
1. exp001_disparity_validation.ipynb  [~15 min]  - Establishes baseline
2. exp003_kurtosis_analysis.ipynb     [~20 min]  - Tests kurtosis theory
3. exp002_bloom1b7.ipynb              [~30 min]  - Validates at scale
4. exp004_rate_distortion.ipynb       [~45 min]  - Tests rate-distortion
5. exp005_layer_ablation.ipynb        [~60 min]  - Identifies critical layers
6. exp006_laaciq_intervention.ipynb   [~30 min]  - Tests mitigation
```

**Dependency note:** exp002 compares against exp001 results. Run exp001 first.

## Running Each Experiment

### General Procedure

For each notebook:

1. **Open the notebook** in Colab
2. **Run all cells sequentially** (Runtime > Run all)
3. **Monitor for errors** in the output
4. **Verify outputs generated:**
   - `expXXX_results.json` - structured results
   - `expXXX_results.png` - visualization

### Experiment-Specific Instructions

#### EXP-001: Disparity Validation (BLOOM-560M)
```
Purpose: Establish that quantization disparity exists
Runtime: ~15 minutes
Memory: ~4GB peak

Key outputs:
- disparity_ratio: D_LR / D_HR (expect > 1.5 if H1 supported)
- r_fertility: correlation coefficient (expect > 0.7 if H3 supported)

Success criteria:
- H1: disparity_ratio > 1.5
- H3: r_fertility > 0.7 AND p_value < 0.05
```

#### EXP-002: Scale Validation (BLOOM-1B7)
```
Purpose: Verify disparity persists at larger scale
Runtime: ~30 minutes
Memory: ~8GB peak (loads models sequentially)

Special handling:
- Models are loaded/unloaded sequentially to fit memory
- If OOM occurs, restart runtime and run again

Key outputs:
- Same metrics as EXP-001
- Cross-experiment comparison (requires exp001_results.json)

Success criteria:
- Similar disparity_ratio to EXP-001
- Confirms phenomenon is not model-size artifact
```

#### EXP-003: Kurtosis Analysis
```
Purpose: Test if weight distribution kurtosis predicts degradation
Runtime: ~20 minutes
Memory: ~4GB peak

Key outputs:
- per_layer_kurtosis: kurtosis values by layer
- kurtosis_degradation_correlation: r and p-value

Success criteria:
- H2: |r| > 0.7 AND p_value < 0.05
```

#### EXP-004: Rate-Distortion Curve
```
Purpose: Validate Shannon rate-distortion theory applies
Runtime: ~45 minutes
Memory: ~6GB peak (tests multiple precisions)

Special handling:
- Tests FP16, INT8, INT4 (NF4), INT4 (FP4)
- Each precision requires fresh model load

Key outputs:
- empirical_slope: slope of log(D) vs bits
- theoretical_slope: -ln(2)/2 ≈ -0.347

Success criteria:
- H4: empirical_slope within 50% of theoretical_slope
- Negative slope (degradation decreases with more bits)
```

#### EXP-005: Layer Ablation Study
```
Purpose: Identify which layers matter most for LR languages
Runtime: ~60 minutes
Memory: ~2GB peak (uses OPT-125M)

Special handling:
- Tests 8 different layer protection configurations
- Applies manual symmetric quantization (not NF4)

Key outputs:
- disparity_by_config: disparity for each protection strategy
- gateway_reduction: % disparity reduction from protecting L0+L_final

Success criteria:
- H5: gateway_reduction > 30%
```

#### EXP-006: LA-ACIQ Intervention
```
Purpose: Test per-language optimal clipping
Runtime: ~30 minutes
Memory: ~6GB peak

Special handling:
- Requires calibration pass for each language
- Applies language-specific clipping thresholds

Key outputs:
- language_alphas: per-language optimal α values
- disparity_reduction: % improvement over global ACIQ

Success criteria:
- H6: disparity_reduction > 20%
```

## Handling Errors

### Common Issues

**1. CUDA Out of Memory**
```python
# Add after imports:
import gc
torch.cuda.empty_cache()
gc.collect()

# If persists, restart runtime (Runtime > Restart runtime)
```

**2. Model download fails**
```python
# Set cache directory explicitly:
import os
os.environ['TRANSFORMERS_CACHE'] = '/content/cache'
os.environ['HF_HOME'] = '/content/cache'
```

**3. bitsandbytes installation issues**
```python
# Reinstall with specific version:
!pip uninstall -y bitsandbytes
!pip install bitsandbytes==0.41.0
```

**4. Tokenizer warnings**
```python
# Safe to ignore, but to suppress:
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

### Recovery Procedure

If a notebook fails mid-execution:

1. Note which cell failed
2. Restart runtime (Runtime > Restart runtime)
3. Run cells up to the failed one
4. Check error message and apply fix from above
5. Continue from failed cell

## Aggregating Results

After all experiments complete, aggregate results:

```python
import json
import os

results_summary = {}
for exp_num in ['001', '002', '003', '004', '005', '006']:
    results_file = f'exp{exp_num}_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results_summary[f'EXP-{exp_num}'] = json.load(f)

# Summary table
print("=" * 60)
print("HYPOTHESIS RESULTS SUMMARY")
print("=" * 60)

hypothesis_results = {
    'H1': results_summary.get('EXP-001', {}).get('hypotheses', {}).get('H1_disparity_exists', {}).get('result', 'NOT_RUN'),
    'H2': results_summary.get('EXP-003', {}).get('hypothesis', {}).get('H2_kurtosis_correlation', {}).get('result', 'NOT_RUN'),
    'H3': results_summary.get('EXP-001', {}).get('hypotheses', {}).get('H3_fertility_predicts', {}).get('result', 'NOT_RUN'),
    'H4': results_summary.get('EXP-004', {}).get('hypothesis', {}).get('H4_rate_distortion', {}).get('result', 'NOT_RUN'),
    'H5': results_summary.get('EXP-005', {}).get('hypothesis', {}).get('H5_gateway_criticality', {}).get('result', 'NOT_RUN'),
    'H6': results_summary.get('EXP-006', {}).get('hypothesis', {}).get('H6_laaciq_effectiveness', {}).get('result', 'NOT_RUN'),
}

for h, result in hypothesis_results.items():
    status = "✓" if result == "SUPPORTED" else "✗" if "NOT" in result else "?"
    print(f"{h}: {status} {result}")

# Save aggregated results
with open('all_results_summary.json', 'w') as f:
    json.dump({
        'hypothesis_results': hypothesis_results,
        'full_results': results_summary
    }, f, indent=2)

print("\n✓ Aggregated results saved to all_results_summary.json")
```

## Output Artifacts

Each experiment produces:

| File | Description |
|------|-------------|
| `expXXX_results.json` | Structured results with all metrics |
| `expXXX_results.png` | Visualization (3-4 subplots) |

Download artifacts before runtime expires:
```python
from google.colab import files
files.download('exp001_results.json')
files.download('exp001_results.png')
# ... repeat for each experiment
```

Or save to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
for f in ['exp001_results.json', 'exp001_results.png']:
    shutil.copy(f, '/content/drive/MyDrive/laaciq_results/')
```

## Interpretation Guide

### Reading Results

Each `expXXX_results.json` contains:

```json
{
  "experiment": "EXP-XXX: Name",
  "model": "model_name",
  "hypothesis": {
    "HX_name": {
      "prediction": "what we expected",
      "observed_value": 0.123,
      "result": "SUPPORTED | NOT_SUPPORTED | PARTIALLY_SUPPORTED"
    }
  },
  "statistics": { ... },
  "per_language": [ ... ]
}
```

### Result Categories

- **SUPPORTED**: Hypothesis prediction met with statistical significance
- **NOT_SUPPORTED**: Prediction not met
- **PARTIALLY_SUPPORTED**: Trend in expected direction but below threshold
- **INSUFFICIENT_DATA**: Not enough data points for conclusion

### Key Metrics to Report

| Metric | Source | Meaning |
|--------|--------|---------|
| `disparity_ratio` | EXP-001/002 | D_LR / D_HR (>1.5 = significant disparity) |
| `cohens_d` | EXP-001/002 | Effect size (>0.8 = large effect) |
| `r_fertility` | EXP-001/002 | Fertility-degradation correlation |
| `empirical_slope` | EXP-004 | Rate-distortion slope (compare to -0.347) |
| `gateway_reduction` | EXP-005 | % disparity reduction from layer protection |
| `laaciq_reduction` | EXP-006 | % disparity reduction from LA-ACIQ |

## Minimal Reproduction

If time/resources limited, run only:

1. **EXP-001** (15 min) - Core disparity validation
2. **EXP-006** (30 min) - Mitigation effectiveness

This tests the central claims:
- Disparity exists (H1)
- LA-ACIQ reduces it (H6)

## Reporting Template

After running experiments, summarize:

```markdown
## LA-ACIQ Experiment Results

### Environment
- Platform: Google Colab
- GPU: [T4/A100]
- Date: [YYYY-MM-DD]

### Hypothesis Results

| Hypothesis | Prediction | Observed | Result |
|------------|------------|----------|--------|
| H1 | D_LR/D_HR > 1.5 | [value] | [SUPPORTED/NOT] |
| H2 | |r(κ,D)| > 0.7 | [value] | [SUPPORTED/NOT] |
| H3 | r(fertility,D) > 0.7 | [value] | [SUPPORTED/NOT] |
| H4 | slope ≈ -0.347 | [value] | [SUPPORTED/NOT] |
| H5 | gateway reduces >30% | [value]% | [SUPPORTED/NOT] |
| H6 | LA-ACIQ reduces >20% | [value]% | [SUPPORTED/NOT] |

### Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Artifacts
- Results: [link to JSON files]
- Figures: [link to PNG files]
```

## Questions / Issues

If experiments fail or produce unexpected results:

1. Check GPU memory usage
2. Verify model downloads completed
3. Ensure bitsandbytes version compatibility
4. Review cell outputs for warnings

The experiments are designed to be robust to minor variations. If results differ significantly from predictions, this is scientifically interesting data.
