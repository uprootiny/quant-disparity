# Decision Log

## D001: Pivot from Fertility to Kurtosis

**Date:** 2026-01-02
**Decision:** Abandon tokenization fertility hypothesis, adopt weight distribution hypothesis.
**Evidence:** Real tokenizer fertility r=0.34 (not significant), mock kurtosis r=0.92 (significant).
**Rationale:** Vocabulary coverage confounds fertility measurement across tokenizers.

## D002: Use BLOOM-560M for Validation

**Date:** 2026-01-02
**Decision:** Use BLOOM-560M for Phase 1 weight extraction.
**Alternatives:** Aya-8B (too large), XGLM-564M (less multilingual coverage).
**Rationale:** Small enough for CPU, good multilingual coverage.

## D003: Target Soudry Lab

**Date:** 2026-01-01
**Decision:** Frame proposal for Soudry Lab (Technion).
**Rationale:** Methodology alignment with Banner and Chmiel papers.
**Alternative:** Belinkov Lab (probing focus, different angle).

## D004: Go/No-Go Thresholds

**Date:** 2026-01-02
**Decision:** Use r > 0.7 as GO threshold, r < 0.3 as PIVOT threshold.
**Rationale:** Conservative given small sample size (14 languages).

## D005: Phased Compute Approach

**Date:** 2026-01-02
**Decision:** Validate on CPU before investing in GPU compute.
**Rationale:** Minimize cost until hypothesis validated.
