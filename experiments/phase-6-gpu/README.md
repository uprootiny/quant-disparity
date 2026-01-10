# Phase 6: GPU Validation

*Validating findings on larger models with real quantization*

## Prerequisites

- GPU with 16GB+ VRAM (T4, A10, or better)
- CUDA 11.8+
- ~50GB disk for model weights

```bash
pip install torch transformers auto-gptq bitsandbytes accelerate
```

## Validation Priorities

### P0: Critical (Must validate before publishing)

| Experiment | Model | Goal | Est. Time |
|------------|-------|------|-----------|
| gpu001 | Llama-2-7B | Does L0+L11 pattern hold? | 30 min |
| gpu002 | Llama-2-7B | Find actual critical layers | 2 hours |
| gpu003 | Mistral-7B | Cross-architecture check | 1 hour |

### P1: Important (Strengthens paper)

| Experiment | Model | Goal | Est. Time |
|------------|-------|------|-----------|
| gpu004 | Llama-2-7B | Real GPTQ quantization | 1 hour |
| gpu005 | Llama-2-7B | Real AWQ quantization | 1 hour |
| gpu006 | Llama-2-7B | Downstream task eval (MMLU) | 2 hours |

### P2: Nice to have

| Experiment | Model | Goal | Est. Time |
|------------|-------|------|-----------|
| gpu007 | Llama-2-13B | Scale to 13B | 3 hours |
| gpu008 | Qwen-7B | Non-Western architecture | 2 hours |
| gpu009 | Gemma-7B | Google architecture | 2 hours |

## Hypotheses to Test

### H6.1: Gateway Layer Pattern Transfers

**Prediction**: Llama-2-7B has critical layers at ~0%, ~75%, and ~100% depth.

```python
# Expected: L0, L24 (75% of 32), L31 are critical
# Or equivalent input-consolidation-output pattern
```

### H6.2: GPTQ Already Handles Disparity

**Prediction**: GPTQ's calibration reduces but doesn't eliminate disparity.

If GPTQ already solves this → our contribution is understanding why.
If GPTQ doesn't solve it → our protection strategy adds value.

### H6.3: Perplexity Predicts Task Performance

**Prediction**: Low perplexity disparity → fair task performance.

Test on MMLU per-language subsets.

## Quick Start Scripts

See individual experiment files:
- `gpu001_llama_pattern.py` - Test if L0+L31 works
- `gpu002_llama_sweep.py` - Full layer sweep
- `gpu003_mistral_check.py` - Cross-architecture validation
- `gpu004_real_gptq.py` - GPTQ with layer protection
