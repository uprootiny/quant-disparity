# Research Session Continuation Guide

## Quick Start

```bash
cd /home/uprootiny/ops/quant-disparity/experiments/phase-5-minimal
python3 expNNN_name.py  # Run any experiment
```

---

## Experiment Methodology: Slow Scaling Loop

### Philosophy
Each experiment should complete successfully before scaling up. This prevents wasted compute and ensures reliable incremental progress.

### Loop Structure

```
1. FORMULATE hypothesis (document in HYPOTHESES.md)
2. CREATE minimal experiment script
3. RUN with short timeout (60-120s first)
4. VERIFY output makes sense
5. ITERATE if needed (fix bugs, adjust scope)
6. DOCUMENT results in TECHNICAL_WRITEUP.md
7. COMMIT batch (every 2-3 experiments)
8. SCALE UP if pattern holds (more layers, languages, models)
```

### Timeout Guidelines

| Experiment Type | Initial Timeout | Extended |
|-----------------|-----------------|----------|
| Single model, 2 langs | 60-120s | 180s |
| Per-layer sweep (12 layers) | 180-300s | 400s |
| Multi-model comparison | 300-400s | 600s |
| Memory-intensive (BLOOM) | Skip or use GPU | - |

### Memory Constraints

Current system: ~32GB RAM, CPU-only

| Model | Memory Required | Status |
|-------|-----------------|--------|
| GPT-2 (124M) | ~1GB | OK |
| OPT-125M | ~1GB | OK |
| Pythia-160M | ~1.5GB | OK (but tokenization issues) |
| BLOOM-560M | ~4GB + quantization buffer | FAILS |

---

## Research Roadmap

### Completed Experiments

| Exp | Name | Key Finding |
|-----|------|-------------|
| 001-010 | Baseline series | Disparity exists (78-214x) |
| 011 | Threshold sweep | 5% is optimal |
| 012 | Layer-specific | MLP > Attention (GPT-2) |
| 013 | Hybrid strategy | Layer 0 + MLP = 1.4x |
| 014 | OPT validation | Pattern is model-dependent |
| 015 | Text length | Medium/long texts reliable |
| 016 | Robustness | 0% CV (deterministic) |
| 017 | Per-layer MLP | L0 best (GPT-2), L11 best (OPT) |
| 018 | BLOOM | Memory exceeded |
| 019 | Pythia | Tokenization issues with Hebrew |
| 020 | Per-layer attention | L0 best (both models) |

### Next Experiments Queue

| Exp | Target | Description |
|-----|--------|-------------|
| 021 | Optimal layer combo | Best MLP + best attention layers |
| 022 | Anti-critical analysis | Why do L1/L7 hurt? |
| 023 | 6-language validation | Test all HR/LR languages |
| 024 | Arabic/Chinese focus | Non-Hebrew low-resource |
| 025 | Gradient-based selection | Alternative to magnitude |

### Hypothesis Backlog

| ID | Status | Description |
|----|--------|-------------|
| H5.3b | UNTESTED | Gradient-based selection |
| H5.3c | UNTESTED | Language-activation correlation |
| H5.5c | HYPOTHESIS | Anti-critical = English-specific |

---

## Key Findings Summary

### 1. Disparity is Real and Massive
- 78-214x between high/low resource languages
- Hebrew worst affected (971,648% degradation)
- Statistically significant: r=-0.85, p=0.03

### 2. Component Criticality is MODEL-DEPENDENT

| Model | Best Component | Best Layer | Pattern |
|-------|----------------|------------|---------|
| GPT-2 | MLP | Layer 0 | Early layers critical |
| OPT-125M | Attention | Layer 11 (MLP) / Layer 0 (Attn) | Mixed |

### 3. Anti-Critical Layers Exist
Protecting certain layers INCREASES disparity:
- GPT-2 Layer 1: 381x (vs 214x baseline)
- OPT Layer 7: 245x (vs 153x baseline)

### 4. Optimal Strategies

| Strategy | Overhead | Disparity | Use Case |
|----------|----------|-----------|----------|
| None | 0% | 78-214x | English-only |
| 5% magnitude | 5% | 45x | Quick improvement |
| Layer 0 only | 5.7% | 55x | Balanced |
| Layer 0 + MLP | ~50% | 1.4x | Maximum fairness |

---

## File Structure

```
quant-disparity/
├── docs/
│   ├── TECHNICAL_WRITEUP.md    # Publication-ready draft
│   ├── RESEARCH_GAPS.md        # What's left to validate
│   └── RESEARCH_SESSION_GUIDE.md  # This file
├── experiments/
│   └── phase-5-minimal/
│       ├── HYPOTHESES.md       # Hypothesis tracking
│       ├── exp0XX_*.py         # Experiment scripts
│       └── exp0XX_result.json  # Results
```

---

## Continuing After Session Loss

### 1. Check Current State
```bash
cd /home/uprootiny/ops/quant-disparity
git status
git log --oneline -5
```

### 2. Review Last Results
```bash
ls experiments/phase-5-minimal/exp*_result.json | tail -5
cat experiments/phase-5-minimal/exp020_result.json
```

### 3. Check Hypotheses
```bash
cat experiments/phase-5-minimal/HYPOTHESES.md | head -60
```

### 4. Run Next Experiment
```bash
# Copy pattern from previous experiment
cp experiments/phase-5-minimal/exp020_*.py experiments/phase-5-minimal/exp021_new.py
# Edit and run
timeout 180 python3 experiments/phase-5-minimal/exp021_new.py
```

### 5. Commit Periodically
```bash
git add experiments/phase-5-minimal/
git commit -m "Add exp021: description"
git push origin master
```

---

## Experiment Template

```python
#!/usr/bin/env python3
"""
Exp-XXX: Short description
Goal: What we're testing
"""

import json
from pathlib import Path
from datetime import datetime
import torch

TEXTS = {'en': 'The fox jumps.', 'he': 'השועל קופץ.'}

def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Exp-XXX: Name")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()

    # YOUR EXPERIMENT HERE

    result = {
        "id": "Exp-XXX",
        "timestamp": datetime.now().isoformat(),
        "results": {},
        "status": "SUCCESS"
    }

    with open(Path(__file__).parent / "expXXX_result.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Completed in {(datetime.now()-start).total_seconds():.1f}s")
    return result

if __name__ == "__main__":
    main()
```

---

## Common Patterns

### Quantization Function
```python
def quantize_model(model, exclude_patterns=[]):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            if any(p in name for p in exclude_patterns):
                continue
            flat = param.view(-1)
            mx = flat.abs().max()
            if mx > 0:
                scale = mx / 7.0
                q = torch.round(flat / scale).clamp(-8, 7) * scale
                param.data.copy_(q.view(param.shape))
```

### Disparity Calculation
```python
def calc_disparity(baseline, quantized, texts):
    deg = {l: (quantized[l] - baseline[l]) / baseline[l] * 100
           for l in texts}
    hr_deg = deg['en']
    lr_deg = deg['he']
    return lr_deg / hr_deg if hr_deg > 0 else float('inf')
```

### Layer Patterns by Model
```python
LAYER_PATTERNS = {
    'gpt2': {
        'attention': 'h.{layer}.attn',
        'mlp': 'h.{layer}.mlp',
        'layer': 'h.{layer}.',
    },
    'opt': {
        'attention': 'layers.{layer}.self_attn',
        'mlp': ['layers.{layer}.fc1', 'layers.{layer}.fc2'],
        'layer': 'layers.{layer}.',
    },
}
```

---

*Last updated: 2026-01-08*
*Repository: github.com/uprootiny/quant-disparity*
