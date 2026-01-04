# Systematic Experiment Pipeline

## Design Principles

1. **Sequential Execution**: Each experiment must complete before the next starts
2. **Long Timeouts**: Allow experiments to run to completion (up to 1 hour each)
3. **Evaluation**: Each result is scored and related to research questions
4. **Replication**: Experiments designed for reproducibility
5. **Scalability**: Findings should transfer to larger models

---

## Experiment Sequence

### Phase A: Baseline Establishment

| ID | Question | Expected Runtime | Depends On |
|----|----------|------------------|------------|
| A-001 | What is baseline perplexity per language? | 5 min | None |
| A-002 | What is INT4 degradation per language? | 10 min | A-001 |
| A-003 | What is disparity ratio (LR/HR)? | 5 min | A-002 |

### Phase B: Preservation Study

| ID | Question | Expected Runtime | Depends On |
|----|----------|------------------|------------|
| B-001 | Does 5% preservation reduce degradation? | 15 min | A-002 |
| B-002 | Does 10% preservation reduce degradation? | 15 min | A-002 |
| B-003 | Does 20% preservation reduce degradation? | 15 min | A-002 |
| B-004 | What is optimal preservation %? | 5 min | B-001-003 |

### Phase C: Layer Analysis

| ID | Question | Expected Runtime | Depends On |
|----|----------|------------------|------------|
| C-001 | Which layers are critical per language? | 20 min | A-001 |
| C-002 | Does layer-specific preservation help? | 30 min | C-001 |

### Phase D: Validation

| ID | Question | Expected Runtime | Depends On |
|----|----------|------------------|------------|
| D-001 | Does pattern hold for OPT-125M? | 30 min | A-003 |
| D-002 | Does pattern hold for Pythia-160M? | 30 min | A-003 |

---

## Evaluation Criteria

### Disparity Metrics
- **Primary**: LR/HR degradation ratio
- **Secondary**: Absolute degradation per language
- **Tertiary**: Perplexity variance across languages

### Success Criteria
- Experiment completes without error
- Results are numerically valid (no NaN/Inf in key metrics)
- Pattern is consistent across multiple runs

### Research Questions Addressed

| Question | Experiments | Current Status |
|----------|-------------|----------------|
| Is disparity real? | A-001 to A-003 | CONFIRMED (52x ratio) |
| Does preservation help? | B-001 to B-004 | PENDING |
| Which layers matter? | C-001 to C-002 | PARTIAL (EXP-036) |
| Does pattern scale? | D-001 to D-002 | PENDING |

---

## Implementation Status

- [x] A-001: Baseline perplexity (from EXP-039 v3)
- [x] A-002: INT4 degradation (from EXP-039 v3)
- [x] A-003: Disparity ratio (52.47x confirmed)
- [ ] B-001 to B-004: Preservation study
- [x] C-001: Critical layers (from EXP-036)
- [ ] C-002: Layer-specific preservation
- [ ] D-001, D-002: Model validation

---

*Pipeline designed: 2026-01-04*
