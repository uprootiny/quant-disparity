# GPU Compute Estimate

## Minimum Viable Validation

| Experiment | GPU Hours | Cost (Lambda A10) |
|------------|-----------|-------------------|
| gpu001 | 0.5h | $0.60 |
| gpu002 | 2.0h | $2.40 |
| **TOTAL** | **2.5h** | **~$3** |

This validates whether findings transfer to 7B models.

## Full Validation Suite

| Experiment | GPU Hours | Cost |
|------------|-----------|------|
| gpu001-003 | 3.5h | $4.20 |
| gpu004-006 | 4.0h | $4.80 |
| gpu007-009 | 7.0h | $8.40 |
| **TOTAL** | **14.5h** | **~$17** |

## Compute Options

### 1. Colab Pro ($10/month)
- T4 GPU (16GB VRAM) - works for 7B models with FP16
- Limited to ~4 hours continuous
- Good for: gpu001, gpu003

### 2. Lambda Labs (~$1.20/hour for A10)
- A10 GPU (24GB VRAM)
- No time limits
- Good for: Full validation suite

### 3. Runpod (~$0.50/hour for RTX 3090)
- 24GB VRAM
- Spot instances available
- Good for: Cost-sensitive validation

### 4. Academic Credits
- Apply for Google Cloud Research Credits
- Apply for AWS Research Credits
- Apply for Azure Research Credits

## Memory Requirements

| Model | FP16 | INT4 | Minimum VRAM |
|-------|------|------|--------------|
| Llama-2-7B | 14GB | 4GB | 16GB |
| Llama-2-13B | 26GB | 8GB | 32GB |
| Mistral-7B | 14GB | 4GB | 16GB |

## Recommended Approach

1. **Start with Colab Pro** (~$10)
   - Run gpu001 to quick-validate pattern transfer
   - If positive: proceed to full validation

2. **Lambda Labs burst** (~$5)
   - Run gpu002 (full sweep) and gpu004 (real GPTQ)
   - Save results to JSON

3. **Paper submission**
   - Include both positive and negative results
   - Honest limitations section already written
