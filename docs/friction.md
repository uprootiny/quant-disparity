# Friction Log

## Active Blockers

### F001: Mock Data Circularity [HIGH]

**Problem:** Mock weight statistics were designed with expected patterns.
**Risk:** Results could be circular.
**Mitigation:** Phase 1 extracts real weights to validate.
**Status:** Active mitigation in progress.

### F002: No GPU Access [MEDIUM]

**Problem:** Cannot run large models.
**Workaround:** Use BLOOM-560M on CPU. Slow but feasible.
**Status:** Acceptable for validation.

## Resolved

### F003: Vocabulary Coverage Confound [RESOLVED]

**Problem:** Tokenizer fertility confounded by vocabulary coverage.
**Resolution:** Pivoted from fertility to weight distribution hypothesis.

### F004: Marchisio Data Not Public [RESOLVED]

**Problem:** Cannot replicate exact degradation numbers.
**Workaround:** Extracted approximate values from paper figures.

## Future Concerns

### F005: Layer Sensitivity Compute [FUTURE]

**Problem:** Full sensitivity matrix requires many quantization runs.
**Mitigation:** Cloud GPU burst when Phase 1 validates.
