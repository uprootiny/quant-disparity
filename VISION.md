# Research Vision: From Correlation to Impact

## What We've Established

```
Observation:  Multilingual LLMs degrade non-uniformly under quantization
Finding:      r = -0.834 between outlier layer activation and degradation
Mechanism:    Effective kurtosis (Banner et al.) predicts error
Scope:        BLOOM-specific (XGLM shows no pattern)
```

This is a **diagnostic finding**. We can predict which languages will suffer.
But prediction alone isn't notable research.

---

## What Would Be Notable

### Tier 1: Publishable (Workshop/Short Paper)

**"Why Low-Resource Languages Suffer Under Quantization"**

- Formalize the outlier-activation mechanism
- Show it's architecture-dependent (BLOOM vs XGLM)
- Provide practitioners with a diagnostic tool

*Gap from current work:* Need more models, rigorous methodology write-up.

### Tier 2: Strong Contribution (Main Conference)

**"Language-Aware Quantization for Equitable Multilingual LLMs"**

- Not just diagnose, but **fix** the problem
- Propose: per-language calibration, layer-specific bit-widths, or mixed-precision
- Demonstrate reduced disparity across languages

*Gap from current work:* Need intervention experiments + larger models.

### Tier 3: High Impact (Top Venue + Tech Transfer)

**"Quantization-Robust Multilingual Training"**

- Change how models are trained to prevent outlier formation
- Or: architectural modifications that make quantization equitable
- Deployable solution for production systems

*Gap from current work:* Substantial engineering + industry collaboration.

---

## Soudry Lab Fit

**Daniel Soudry's research themes:**

| Theme | Our Alignment |
|-------|---------------|
| Optimal quantization theory (Banner et al.) | We USE this framework |
| Neural network compression | Direct application |
| FP8 training (Chmiel et al. 2025) | We cite their outlier work |
| Hardware-aware optimization | Bit-width recommendations |

**What Soudry's group could contribute:**

1. **Theoretical depth**: Formalize why BLOOM develops outliers but XGLM doesn't
2. **Optimization expertise**: Design better clipping/quantization algorithms
3. **Scale**: Access to compute for 7B-70B model experiments
4. **Industry connections**: Tech transfer to Meta, Google, startups

**What we bring:**

1. A concrete, reproducible finding (r=-0.834)
2. Multilingual focus (less explored in quantization literature)
3. Hypothesis with theoretical backing
4. Preliminary corpus and experimental infrastructure

---

## Technology Transfer Paths

### Path A: Diagnostic Tool (Shortest)

**Product:** Script that predicts quantization degradation per language

**Users:** MLOps teams deploying multilingual models

**Value:** Avoid shipping models that fail for specific languages

**Timeline:** 1-2 months to polish, open-source

### Path B: Calibration Method (Medium)

**Product:** Language-aware quantization calibration

**Method:** Use per-language activation statistics to set layer-wise bit-widths

**Users:** Cloud providers (AWS, GCP), model hubs (HuggingFace)

**Value:** Better INT4 models for low-resource languages

**Timeline:** 3-6 months research, 3 months engineering

### Path C: Training Intervention (Longest, Highest Impact)

**Product:** Training recipe that prevents outlier formation

**Method:** Regularization, architectural changes, or data balancing

**Users:** Foundation model labs (Anthropic, Meta, Mistral)

**Value:** Fundamentally more equitable models

**Timeline:** 1-2 years, requires significant compute

---

## The Honest Assessment

### What we have:
- Strong correlation (r=-0.834, p=0.0002)
- Theoretical grounding (Banner et al.)
- Bootstrap-validated robustness
- Clear mechanism (outlier layers)

### What we lack:
- Causal proof (need intervention)
- Solution (not just diagnosis)
- Scale (only BLOOM-560M)
- Reproducibility across model families

### The key question:

> "Does fixing the outlier activation pattern actually reduce degradation?"

If YES → Path B or C is viable
If NO → We misidentified the mechanism

This requires EXP-009 (bit-width sweep) + intervention experiments.

---

## Proposed Next Phase

### For Soudry Lab Proposal:

**Title:** "Equitable Quantization for Multilingual LLMs"

**Pitch:**
We've identified why quantization hurts low-resource languages (r=-0.834).
With Technion's expertise in optimal quantization, we can:
1. Formalize the mechanism theoretically
2. Develop language-aware calibration methods
3. Test interventions at scale (7B+ models)

**Deliverables:**
1. Theory paper: Why outliers cause disparity
2. Method paper: Language-aware quantization algorithm
3. Open-source tool: Diagnostic + calibration toolkit

**Resources needed:**
- GPU access for inference experiments
- Collaboration on optimization theory
- Connection to industry partners for deployment

---

## The Bottom Line

We're at a fork:

```
Current: Correlation (diagnostic, predictive)
         │
         ├── Publish finding as-is (modest impact)
         │
         └── Push for intervention → solution (high impact)
                                      │
                                      └── Requires: GPU, theory, scale
```

The Technion pitch should be: "We found the problem. Help us solve it."
