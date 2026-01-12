#!/usr/bin/env bash
# ============================================================================
# SESSION STATE: Quantization Disparity Research
# ============================================================================
# Timestamp: 2026-01-11 03:14:39 CET
# Transcript: 6c9953a1-b872-44d1-a37d-a61530eee047
# Status: 183 experiments across 13 phases/tracks
# ============================================================================
#
# WHAT THIS SCRIPT DOES:
# 1. Verifies the environment and dependencies
# 2. Checks that all research artifacts exist
# 3. Displays current research state
# 4. Resumes the Claude Code session with full context
#
# RUN WITH: bash SESSION_STATE_20260111_031439.sh
# OR: chmod +x SESSION_STATE_20260111_031439.sh && ./SESSION_STATE_20260111_031439.sh
#
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

TRANSCRIPT_ID="6c9953a1-b872-44d1-a37d-a61530eee047"
CLAUDE_PROJECTS_DIR="$HOME/.claude/projects/-home-uprootiny-ops"
TRANSCRIPT_FILE="$CLAUDE_PROJECTS_DIR/$TRANSCRIPT_ID.jsonl"
RESEARCH_ROOT="$HOME/ops/quant-disparity"
EXPERIMENTS_DIR="$RESEARCH_ROOT/experiments"
READER_DIR="$HOME/ops/quant-disparity-reader"

# ----------------------------------------------------------------------------
# RESEARCH CONTEXT
# ----------------------------------------------------------------------------
#
# PROJECT: PhD research on multilingual quantization disparity
# TARGET AUDIENCE: 4 Israeli AI labs
#
# CORE HYPOTHESIS: LLM quantization disproportionately harms low-resource
# languages due to poor BPE-morpheme alignment.
#
# EXPERIMENT PHASES (183 total):
#   phase-0-validation:        1 - Initial validation
#   phase-1-extraction:       10 - Weight/activation extraction
#   phase-2-corpus:            5 - Corpus analysis
#   phase-3-crossmodel:       21 - Cross-model comparison
#   phase-4-actionable:       11 - Actionable findings (series exp001-010)
#   phase-5-minimal:          87 - Minimal protection strategies
#   phase-6-gpu:               3 - GPU experiments
#   phase-7-hypothesis:       15 - Hypothesis testing (exp101-115)
#   track-b-interpretability:  5 - Attention/probing analysis
#   track-c-efficiency:        6 - Efficiency compression methods
#   track-d-syntax:            3 - Syntactic sensitivity
#   track-e-confound-resistant:15 - Confound-resistant tests (latest)
#   gpu-colab:                 1 - Colab notebook
#
# KEY FINDINGS:
#   - Within-language effect: r=-0.998 (confound-free)
#   - Scaling paradox: r=0.984 (larger models = more disparity)
#   - Gateway layers: L0+L9+L11 protection reduces disparity 0.59x
#   - Cross-language: CONFOUNDED (VIF=36, vocab coverage)
#
# CRITICAL GAPS:
#   - No Hebrew corpus (blocks primary narrative)
#   - GPU validation needed
#   - No valid instrumental variable found
#
# ----------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}QUANTIZATION DISPARITY RESEARCH - SESSION RECOVERY${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "Timestamp: ${YELLOW}2026-01-11 03:14:39 CET${NC}"
echo -e "Transcript: ${YELLOW}$TRANSCRIPT_ID${NC}"
echo ""

# ----------------------------------------------------------------------------
# STEP 1: ENVIRONMENT CHECK
# ----------------------------------------------------------------------------

echo -e "${BLUE}[1/5] Checking environment...${NC}"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not found"
        return 1
    fi
}

NEED_NIX=false
check_command python3 || NEED_NIX=true
check_command claude || NEED_NIX=true

if command -v python3 &> /dev/null; then
    python3 -c "import numpy, scipy" 2>/dev/null && \
        echo -e "  ${GREEN}✓${NC} numpy/scipy available" || \
        { echo -e "  ${YELLOW}!${NC} numpy/scipy missing"; NEED_NIX=true; }
fi

if $NEED_NIX; then
    echo ""
    echo -e "${YELLOW}Missing dependencies. Install via:${NC}"
    echo "  nix-shell -p python3 python3Packages.numpy python3Packages.scipy nodejs"
    echo "  # or: pip install numpy scipy"
    echo ""
fi

# ----------------------------------------------------------------------------
# STEP 2: VERIFY RESEARCH ARTIFACTS
# ----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}[2/5] Verifying research artifacts...${NC}"

for dir in "$RESEARCH_ROOT" "$EXPERIMENTS_DIR" "$READER_DIR"; do
    if [[ -d "$dir" ]]; then
        echo -e "  ${GREEN}✓${NC} $(basename "$dir")"
    else
        echo -e "  ${RED}✗${NC} $(basename "$dir") (MISSING)"
    fi
done

# Full experiment count
echo ""
echo "  Experiment inventory:"
total=0
for phase_dir in "$EXPERIMENTS_DIR"/*/; do
    if [[ -d "$phase_dir" ]]; then
        count=$(ls -1 "$phase_dir"*.py 2>/dev/null | wc -l)
        if [[ "$count" -gt 0 ]]; then
            name=$(basename "$phase_dir")
            printf "    %-28s %3d\n" "$name" "$count"
            total=$((total + count))
        fi
    fi
done
echo -e "    ${GREEN}────────────────────────────────${NC}"
printf "    %-28s %3d\n" "TOTAL" "$total"

# ----------------------------------------------------------------------------
# STEP 3: CHECK TRANSCRIPT
# ----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}[3/5] Checking Claude transcript...${NC}"

if [[ -f "$TRANSCRIPT_FILE" ]]; then
    LINES=$(wc -l < "$TRANSCRIPT_FILE")
    SIZE=$(du -h "$TRANSCRIPT_FILE" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Transcript exists"
    echo "    Lines: $LINES, Size: $SIZE"
else
    echo -e "  ${YELLOW}!${NC} Transcript not found, looking for alternatives..."
    LATEST=$(ls -t "$CLAUDE_PROJECTS_DIR"/*.jsonl 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
        echo -e "  ${GREEN}✓${NC} Found: $(basename "$LATEST")"
        TRANSCRIPT_ID=$(basename "$LATEST" .jsonl)
    fi
fi

# ----------------------------------------------------------------------------
# STEP 4: KEY DOCUMENTS
# ----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}[4/5] Key documentation:${NC}"

KEY_DOCS=(
    "$EXPERIMENTS_DIR/track-e-confound-resistant/EXPERIMENT_STATUS.md"
    "$READER_DIR/RESEARCH_ROADMAP.md"
    "$READER_DIR/DATA_AND_PERSPECTIVES.md"
)

for doc in "${KEY_DOCS[@]}"; do
    if [[ -f "$doc" ]]; then
        echo -e "  ${GREEN}✓${NC} $(basename "$doc")"
    else
        echo -e "  ${RED}✗${NC} $(basename "$doc") (MISSING)"
    fi
done

# ----------------------------------------------------------------------------
# STEP 5: DISPLAY STATE & RESUME INSTRUCTIONS
# ----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}[5/5] Current research state:${NC}"
echo ""
echo "  EXPERIMENTS: $total across 13 phases/tracks"
echo "  LATEST: Track E confound-resistant (15 exp) + Track D disambiguating (4 exp)"
echo ""
echo "  KEY RESULTS:"
echo "    - Within-language r=-0.998 (Hebrew), r=-0.996 (Arabic replication)"
echo "    - Gateway protection: L0+L9+L11 in FP16 → 0.59x disparity"
echo "    - Scaling: r=0.984 (larger models = worse disparity)"
echo ""
echo "  BLOCKERS:"
echo -e "    ${RED}!${NC} No Hebrew corpus scraped"
echo -e "    ${YELLOW}!${NC} GPU validation pending"
echo -e "    ${YELLOW}!${NC} Cross-language causation confounded (VIF=36)"
echo ""
echo "  DATA: ~461K lines (French, English, Japanese, Chinese, Arabic, Hindi)"
echo "  NEED: Hebrew (CRITICAL), Korean, Turkish"
echo ""

echo -e "${BLUE}============================================================================${NC}"
echo "To resume this session:"
echo ""
echo -e "  ${GREEN}cd $RESEARCH_ROOT${NC}"
echo -e "  ${GREEN}claude --resume $TRANSCRIPT_ID${NC}"
echo ""
echo "Or paste this context in a new session:"
echo ""
echo -e "${YELLOW}---8<--- COPY FROM HERE ---8<---${NC}"
cat << 'CONTEXT_PROMPT'

Continue PhD research on multilingual quantization disparity.

SESSION: 2026-01-11, 183 experiments across 13 phases
TRANSCRIPT: 6c9953a1-b872-44d1-a37d-a61530eee047

KEY FILES:
1. experiments/track-e-confound-resistant/EXPERIMENT_STATUS.md (latest 15 experiments)
2. ../quant-disparity-reader/RESEARCH_ROADMAP.md (strategic assessment)
3. ../quant-disparity-reader/DATA_AND_PERSPECTIVES.md (data gaps + theory)

PHASES COMPLETED:
- phase-5-minimal: 87 experiments (protection strategies)
- phase-7-hypothesis: 15 experiments (exp101-115)
- track-e-confound-resistant: 15 experiments (latest, confound-free)

CONFIRMED:
- Within-language effect: r=-0.998 (Hebrew), r=-0.996 (Arabic)
- Scaling paradox: r=0.984 (disparity increases with model size)
- Gateway protection: L0+L9+L11 reduces disparity 0.59x

BLOCKED:
- Cross-language causation (VIF=36, vocab coverage confound)
- No valid instrumental variable found

CRITICAL: No Hebrew corpus scraped - blocks primary narrative for Israeli labs

IMMEDIATE TASKS:
1. Scrape Hebrew data (Wikipedia, Sefaria API)
2. Run Colab GPU experiments (notebook in gpu-experiments/)
3. Expand to 8+ language families

CONTEXT_PROMPT
echo -e "${YELLOW}---8<--- END COPY ---8<---${NC}"
echo ""
echo -e "${BLUE}============================================================================${NC}"

# ----------------------------------------------------------------------------
# PHASE DETAILS (for reference)
# ----------------------------------------------------------------------------
#
# PHASE 0: VALIDATION
#   distrib_analysis.py - Distribution analysis
#
# PHASE 1: EXTRACTION (10 experiments)
#   weight_stats.py, layer_activation.py, bloom_architecture.py, etc.
#
# PHASE 2: CORPUS (5 experiments)
#   Corpus-based validation
#
# PHASE 3: CROSS-MODEL (21 experiments)
#   OPT, Pythia, BLOOM comparisons
#
# PHASE 4: ACTIONABLE (11 experiments)
#   series/exp001-010: Baseline → Full preservation
#   exp036-039: Layer contribution, intervention studies
#
# PHASE 5: MINIMAL (87 experiments!)
#   exp011-095: Layer sweeps, protection strategies, language families
#   Key findings: L0+L9+L11 protection, threshold analysis
#
# PHASE 6: GPU (3 experiments)
#   gpu001-004: Llama pattern, sweep, real GPTQ
#
# PHASE 7: HYPOTHESIS (15 experiments)
#   exp101-115: Per-language, bitwidth, families, scale, intervention
#
# TRACK A: ARCHITECTURE
#   Architecture-specific analysis
#
# TRACK B: INTERPRETABILITY (5 experiments)
#   Attention patterns, probing, circuit ablation
#
# TRACK C: EFFICIENCY (6 experiments)
#   Distillation, pruning, carbon cost
#
# TRACK D: SYNTAX (3 experiments)
#   Morphological sensitivity, agreement, alignment
#
# TRACK E: CONFOUND-RESISTANT (15 experiments)
#   exp_e001-e011: Synthetic, ablation, within-language, VIF
#   exp_d001-d004: Power, Arabic replication, IV search, WALS
#
# ----------------------------------------------------------------------------
