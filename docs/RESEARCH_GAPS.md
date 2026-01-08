# Research Gaps and Validation Needs

## Current Evidence Status

### Strong Evidence (Multiple Experiments)
- ✓ Disparity exists and is massive (78-214x)
- ✓ Hebrew is most affected
- ✓ 5% magnitude preservation helps
- ✓ MLP > Attention > Embeddings for criticality

### Preliminary Evidence (Single Experiment)
- △ Layer 0 efficiency (Exp-012 only)
- △ Hybrid Layer 0 + MLP achieves 1.4x (Exp-013, but had threshold bug)
- △ Non-monotonic preservation curve

### Gaps Requiring Validation
- ✗ Cross-model validation of layer findings
- ✗ Longer text evaluation
- ✗ Statistical robustness (3+ runs)
- ✗ Real quantization (GPTQ, AWQ)
- ✗ Downstream task metrics

---

## Priority Experiments

### P1: Cross-Model Layer Validation
**Goal**: Confirm MLP > Attention finding on OPT-125M

```
Exp-014: OPT-125M layer-specific protection
- Test: none, embeddings, attention, MLP, layer0
- Expected: Same pattern as GPT-2
```

### P2: Text Length Sensitivity
**Goal**: Rule out short-text artifacts

```
Exp-015: Variable text length
- Test: 5, 20, 50, 100 tokens
- Expected: Consistent disparity ratios
```

### P3: Statistical Robustness
**Goal**: Confidence intervals on key findings

```
Exp-016: 5-run validation of key results
- Test: 0%, 5% preservation on GPT-2
- Report: mean ± std for disparity
```

### P4: Component Isolation
**Goal**: Understand WHY MLP matters more

```
Exp-017: Per-layer MLP contribution
- Test: Protect MLP in layers 0, 1, 2, ... separately
- Expected: Identify most critical MLP layers
```

---

## Hypotheses to Test

### H-NEW-1: MLP criticality is universal
MLP > Attention pattern holds across all decoder-only models

### H-NEW-2: First MLP layer is most critical
Layer 0 MLP alone provides most of the benefit

### H-NEW-3: Text length affects disparity measurement
Short texts may overestimate disparity due to tokenization artifacts

### H-NEW-4: Language-specific MLP neurons exist
Specific MLP neurons activate differently for different languages

---

## Validation Matrix

| Finding | GPT-2 | OPT-125M | Pythia | BLOOM |
|---------|-------|----------|--------|-------|
| Disparity exists | ✓ | ✓ | ✓ | - (RAM) |
| MLP > Attention | ✓ | ✗ (Attn wins) | - | - |
| Layer 0 efficient | ✓ | ✓ | - | - |
| 5% optimal | ✓ | ✓ | - | - |
| Hybrid works | ✓ | - | - | - |
| Per-layer MLP | ✓ (L0 best) | ✓ (L11 best) | - | - |

**Key Finding**: Pattern is MODEL-DEPENDENT. GPT-2 needs MLP protection; OPT needs attention.

**BLOOM Note**: 560M model exceeds 32GB RAM during quantization. Requires GPU or more memory.

---

## Experimental Queue

1. **Exp-014**: OPT layer-specific (HIGH PRIORITY)
2. **Exp-015**: Text length sensitivity
3. **Exp-016**: Statistical robustness
4. **Exp-017**: Per-layer MLP analysis

---

*Gaps identified: 2026-01-05*
