# Colab & Kaggle GPU Experiments

*What we can do with free/cheap GPU access*

---

## Platform Comparison

| Platform | GPU | VRAM | Time Limit | Best For |
|----------|-----|------|------------|----------|
| **Colab Free** | T4 | 16GB | ~4h/day | Quick validation |
| **Colab Pro** | T4/A100 | 16-40GB | ~12h/session | Larger models |
| **Kaggle** | P100/T4 | 16GB | 30h/week | Batch experiments |

---

## Tier 1: High-Value Validation (Colab Free)

### COLAB-001: Validate L0+L11 on Llama-2-7B

**Goal:** Test if GPT-2's gateway layer pattern transfers.

**Notebook:**
```python
# Colab cell 1: Setup
!pip install -q transformers accelerate bitsandbytes

# Cell 2: Load Llama
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "NousResearch/Llama-2-7b-hf"  # No approval needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Cell 3: Quick disparity test
TEXTS = {
    'en': 'The quick brown fox jumps over the lazy dog.',
    'he': 'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'ar': 'الثعلب البني السريع يقفز فوق الكلب الكسول.',
}

# Test: L0+L31 (equivalent to GPT-2's L0+L11)
# Llama has 32 layers, so L31 is the output layer
```

**Time:** ~30 min
**Expected outcome:** Pattern transfers (disparity < 2x) or doesn't (needs layer sweep)

---

### COLAB-002: Real GPTQ Quantization

**Goal:** Test if findings transfer to actual GPTQ (not simulated INT4).

**Notebook:**
```python
!pip install -q auto-gptq optimum

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# Load pre-quantized model
model_id = "TheBloke/Llama-2-7B-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    device="cuda:0",
    use_triton=False,
)

# Compare disparity with our simulated results
```

**Time:** ~45 min
**Expected outcome:** GPTQ shows similar or different disparity pattern

---

### COLAB-003: Attention Pattern Analysis on Llama

**Goal:** Find language-specific attention heads at scale.

**Notebook:**
```python
# Extract attention patterns per language
# Compare with GPT-2's 16.7% language-specific heads
# Test if critical layers have more language-specific heads
```

**Time:** ~1 hour
**Expected outcome:** Cross-track validation of B-001 findings

---

## Tier 2: Extended Validation (Colab Pro or Kaggle)

### KAGGLE-001: Full 32-Layer Sweep on Llama

**Goal:** Find Llama's critical layers.

**Approach:**
- Run in Kaggle notebook with 30h/week limit
- Checkpoint every 4 layers
- ~6h total runtime

**Prediction:** Critical layers at L0, ~L24 (75%), L31

---

### KAGGLE-002: Cross-Model Comparison

**Goal:** Test on Mistral-7B and Qwen-7B.

**Value:** Confirm architecture-specific layer patterns.

---

### KAGGLE-003: MMLU Evaluation

**Goal:** Test if perplexity disparity predicts task disparity.

**Approach:**
- Evaluate quantized models on MMLU
- Compare per-language accuracy vs perplexity
- Correlation analysis

---

## Tier 3: Publication-Ready Experiments

### PRO-001: AWQ with Layer Protection

**Goal:** Implement layer-aware AWQ.

**Approach:**
```python
from awq import AutoAWQForCausalLM

# Custom: Set different group sizes per layer
# Critical layers: group_size=128 (more precision)
# Other layers: group_size=64 (more compression)
```

**Value:** Novel contribution if it works.

---

### PRO-002: Calibration Data Ablation

**Goal:** Test if multilingual calibration reduces disparity.

**Approach:**
- Calibrate GPTQ with English-only data
- Calibrate with multilingual data
- Compare disparity

**Prediction:** Multilingual calibration helps but doesn't eliminate disparity.

---

### PRO-003: Layer Importance via Hessian

**Goal:** Connect our findings to GPTQ/OWQ theory.

**Approach:**
```python
# Compute Fisher information / Hessian for each layer
# Compare with our variance-based criticality
# Theoretical connection to optimal quantization
```

**Value:** Strong theoretical contribution.

---

## Notebook Templates

### Template 1: Quick Disparity Check

```python
# disparity_check.ipynb

# === SETUP ===
!pip install -q transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
TEXTS = {
    'en': 'The quick brown fox jumps.',
    'he': 'השועל החום המהיר קופץ.',
    'ar': 'الثعلب البني السريع يقفز.',
    'zh': '敏捷的狐狸跳跃。',
}

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto"
)

# === BASELINE PPL ===
def ppl(text):
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        return torch.exp(model(**inputs, labels=inputs['input_ids']).loss).item()

baseline = {l: ppl(t) for l, t in TEXTS.items()}
print("Baseline:", baseline)

# === QUANTIZE ===
def quantize_except(protect_layers):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model_copy = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    # Apply INT4 simulation...
    return model_copy

# === MEASURE DISPARITY ===
# ...
```

### Template 2: Layer Sweep

```python
# layer_sweep.ipynb
# Systematically test each layer's protection value
# Save results to CSV for analysis
```

---

## Resource Estimates

| Experiment | GPU Hours | Platform | Cost |
|------------|-----------|----------|------|
| COLAB-001 | 0.5h | Free | $0 |
| COLAB-002 | 0.75h | Free | $0 |
| COLAB-003 | 1h | Free | $0 |
| KAGGLE-001 | 6h | Free | $0 |
| KAGGLE-002 | 4h | Free | $0 |
| KAGGLE-003 | 3h | Free | $0 |
| PRO-001 | 2h | Pro | ~$2 |
| PRO-002 | 3h | Pro | ~$3 |
| PRO-003 | 4h | Pro | ~$4 |

**Total free experiments:** ~15h (within Kaggle's 30h/week)
**Total Pro experiments:** ~9h (~$10)

---

## Workflow

### Week 1: Validation
1. COLAB-001: Does L0+L31 work? (30 min)
2. If yes → COLAB-002: Real GPTQ test (45 min)
3. If no → KAGGLE-001: Full layer sweep (6h)

### Week 2: Extension
4. KAGGLE-002: Cross-model comparison
5. KAGGLE-003: MMLU evaluation
6. COLAB-003: Attention patterns

### Week 3: Publication
7. PRO-001: AWQ implementation
8. PRO-002: Calibration ablation
9. PRO-003: Hessian connection

---

## Key Questions to Answer

| Question | Experiment | Platform |
|----------|------------|----------|
| Does pattern transfer to 7B? | COLAB-001 | Free |
| Does real GPTQ show disparity? | COLAB-002 | Free |
| What are Llama's critical layers? | KAGGLE-001 | Free |
| Does multilingual calibration help? | PRO-002 | Pro |
| Can we improve AWQ/GPTQ? | PRO-001 | Pro |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pattern doesn't transfer | HIGH | Layer sweep provides new data |
| GPTQ already handles disparity | MEDIUM | Focus on understanding mechanism |
| Colab session timeout | LOW | Checkpoint frequently |
| Kaggle quota exhausted | LOW | Spread across weeks |

---

*Created: 2026-01-09*
