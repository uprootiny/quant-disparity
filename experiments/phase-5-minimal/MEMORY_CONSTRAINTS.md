# Memory Constraints for Quantization Disparity Experiments

## System Profile
- **Total RAM**: 12GB
- **Available**: ~3.5GB (Clojure server + Claude sessions use ~8.5GB)
- **No GPU available**

## Model Memory Requirements

| Model | Params | Base Memory | With Quant Ops | Status |
|-------|--------|-------------|----------------|--------|
| GPT-2 | 124M | ~500MB | ~1.5GB | OK |
| OPT-125M | 125M | ~500MB | ~1.5GB | OK |
| Pythia-160M | 160M | ~640MB | ~2GB | Borderline |
| BLOOM-560M | 560M | ~2.2GB | ~4GB | OOM |

## Experiment Types

### Safe (< 2GB peak)
- Per-layer analysis (single model)
- Component-specific protection
- Disparity measurements
- Baseline perplexity

### Risky (2-3GB peak)
- Full model gradient computation
- Multiple model comparison
- Long text processing

### OOM (> 3GB peak)
- Gradient-based weight selection (stores gradients = 2x model)
- BLOOM or larger models
- Simultaneous multi-model loading

## Memory-Safe Experiment Template

```python
# Always use single model loading
model = AutoModelForCausalLM.from_pretrained('gpt2')  # ~500MB

# Store state once, restore to avoid accumulation
state = {k: v.clone() for k, v in model.state_dict().items()}

# Process one strategy at a time
for strategy in strategies:
    restore()  # Reset to FP16
    # ... quantize and measure

# Avoid gradient computation unless essential
model.eval()  # Disable gradients by default
with torch.no_grad():  # Explicit no-grad context
```

## Gradient Experiments

If gradient computation is needed:
1. Use minimal text samples
2. Process one language at a time
3. Clear gradients immediately after use:
   ```python
   model.zero_grad()
   # compute loss
   loss.backward()
   # extract and store gradient info
   gradient_info = {name: p.grad.abs().mean().item() for name, p in model.named_parameters() if p.grad is not None}
   model.zero_grad()  # Free gradient memory
   ```

## Validated Experiments (within constraints)

- Exp-001 to Exp-010: Baseline measurements
- Exp-011: Threshold sweep
- Exp-012 to Exp-017: Layer-specific analysis
- Exp-020 to Exp-027: Component analysis
- Exp-028b: Simplified gradient analysis
- Exp-030 to Exp-032: Anti-critical and fine-grained analysis

## Failed/Blocked Experiments

- Exp-018: BLOOM-560M (OOM)
- Exp-028: Full gradient quantization (OOM)
- Exp-029: Gradient-based selection (OOM)

## Future Work Requiring GPU

1. BLOOM validation (7GB+ VRAM)
2. Llama 2 7B analysis (16GB+ VRAM)
3. Full gradient-based weight selection
4. GPTQ/AWQ real quantization
