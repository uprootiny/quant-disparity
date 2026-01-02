# Phase 1: Weight Extraction

**Status:** Next
**Blocked by:** Phase 0 validation (complete)
**Compute:** Small GPU or CPU (BLOOM-560M)

## Purpose

Extract real weight statistics from a multilingual model to validate Phase 0 mock assumptions.

## Method

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# For each language:
#   1. Run inference on native text samples
#   2. Collect activation magnitudes per neuron
#   3. Identify top-k activated neurons
#   4. Extract weight statistics for those neurons
```

## Expected Output

```json
{
  "eng": {"mean": 0.001, "std": 0.042, "kurtosis": 0.3, ...},
  "ara": {"mean": 0.003, "std": 0.055, "kurtosis": 2.1, ...},
  ...
}
```

## Validation Criteria

- If real kurtosis correlates with mock: **GO** to Phase 2
- If real kurtosis diverges: **PIVOT** hypothesis

## Files

| File | Purpose |
|------|---------|
| `extract.py` | Weight extraction script (TODO) |
| `weights.json` | Extracted statistics (TODO) |
