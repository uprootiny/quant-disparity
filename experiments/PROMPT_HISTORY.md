# Conversation Prompt History

*Deduplicated user prompts from experimental sessions*

---

## Session Timeline (Reconstructed)

### Phase 1: Initial Experimental Work
**Context:** Continuing from previous session with 102+ experiments across 4 tracks

1. `please proceed`
   - *Action:* Updated track status reports, ran experiments

2. `please proceed`
   - *Action:* Created C-004 carbon cost experiment

### Phase 2: Research Structure
3. `each track should have a very clear research problem, a set of hypotheses, some contextual knowledge -- mostly from surveys and digests of the research group's lead and participants -- and a pre-prepared sequence of experiments picking at these hypotheses trying to gather evidence for or against.`
   - *Action:* Restructured RESEARCH_PLAN.md for all tracks with proper hypothesis format

### Phase 3: Repository Organization
4. `please proceed`
   - *Action:* Checked repo status, organized files

5. `how's our repos? are they up to date? is quant-disparity repo clean enough to show off to potential collaborators, with all the auxiliary files and cogitation and scheming kept safely in *another* repo, perhaps the reader one?`
   - *Action:* Cleaned README, updated .gitignore, committed and pushed

### Phase 4: Hypothesis-Driven Experiments
6. `fabulous. let's do more experiments, and make sure they are actually well designed and do help us move through hypotheses space.`
   - *Action:* Created HYPOTHESIS_AUDIT.md, ran E1-E7 experiments

7. `yes please proceed. and keep scaling to meaningfully large enough models while flying or squeezing under out memory constraints.`
   - *Action:* Continued experiments, discovered Scaling Paradox

### Phase 5: Continued Experimentation
8. `proceed with experiments`
   - *Action:* Created and ran E8-E11 (tokenizer intervention, dynamic protection, LA-ACIQ, middle layer redundancy)

9. `proceed`
   - *Action:* Created and ran E12-E15 (cross-lingual transfer, attention degradation, confounder analysis, confound-resistant tests)

### Phase 6: Critical Self-Examination
10. `reconcile our hypotheses with results and consider that we might be missing lots of *obvious* confounders that render most of our speculation moot. How would we check for that?`
    - *Action:* Created E14 confounder analysis, E15 confound-resistant tests
    - *Key finding:* Vocab coverage and benchmark quality are critical confounders (partial_r collapses)
    - *Key finding:* 3/4 confound-resistant tests pass, suggesting some independent effect

11. `perhaps perhaps some of our initial assumptions are wrong or confused. let's try and make them explicit. and write them to a file. a common one, and four more files, one per research track.`
    - *Action:* Created ASSUMPTIONS_COMMON.md and track-specific ASSUMPTIONS.md files

12. `please write prompt history from that conversation into a file. dedupe. add timestamps.`
    - *Action:* This file

---

## Key Decision Points

### Prompt 6: Hypothesis-Driven Pivot
User pushed for "well designed" experiments that "help move through hypothesis space"
→ Led to formal hypothesis testing framework
→ Created HYPOTHESIS_AUDIT.md

### Prompt 10: Confounder Reckoning
User asked about "obvious confounders that render speculation moot"
→ Discovered critical confounds (vocab coverage r=0.966, benchmark quality r=0.987)
→ Revised claim strength from "causes" to "predicts"
→ But also found 3/4 confound-resistant tests pass

### Prompt 11: Assumption Explication
User asked to make assumptions explicit
→ Created 5 assumption files documenting risks
→ Identified which claims are robust vs contested

---

## Experiment Progression

| Phase | Experiments | Key Findings |
|-------|-------------|--------------|
| Prior | E1-E100+ | Baseline findings |
| E1-E7 | Hypothesis tests | Scaling paradox (r=0.984), family clustering (F=35.71) |
| E8-E11 | Interventions | Tokenizer helps 35%, adaptive 2x efficient, middle layers redundant |
| E12-E15 | Confound analysis | Critical confounds identified, 3/4 robust tests pass |

---

## Evolution of Central Claim

**Initial:** "Alignment causes quantization disparity"

**After E14:** "Alignment predicts disparity but may be confounded with vocab coverage"

**After E15:** "Alignment has independent effect (3/4 tests) but causation not definitive"

**Current:** "Gateway importance and family clustering are robust; alignment-degradation link is real but entangled with training data investment"

---

## Session Statistics

- Total experiments created this session: ~20
- Total experiments in repo: ~120
- Files created: 25+
- Commits: 4
- Key discoveries: Scaling paradox, critical confounders, confound-resistant evidence
