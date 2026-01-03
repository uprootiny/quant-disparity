# Compute Requirements Analysis

## Current Experiments

### EXP-009: Bit-Width Sweep

**Task:** Measure perplexity at INT8/4/3/2 for 6 languages

**Model:** BLOOM-560M
- FP32 weights: ~2.2 GB
- FP16 weights: ~1.1 GB
- INT8 weights: ~0.6 GB
- INT4 weights: ~0.3 GB

**Memory Requirements:**
| Configuration | GPU RAM | CPU RAM |
|--------------|---------|---------|
| FP32 inference | 4-6 GB | 8 GB |
| INT8 (bitsandbytes) | 2-3 GB | 6 GB |
| INT4 (bitsandbytes) | 1-2 GB | 4 GB |

**Runtime Estimate:**
- 6 languages × 100 samples × 4 bit-widths
- ~5 min per bit-width per language
- Total: ~2 hours

---

## Spot Instance Options

### AWS

| Instance | GPU | VRAM | Spot $/hr | On-Demand $/hr |
|----------|-----|------|-----------|----------------|
| g4dn.xlarge | T4 | 16 GB | ~$0.16 | $0.52 |
| g5.xlarge | A10G | 24 GB | ~$0.40 | $1.01 |
| p3.2xlarge | V100 | 16 GB | ~$0.92 | $3.06 |

**Recommendation:** g4dn.xlarge ($0.16/hr spot)
- T4 GPU is sufficient for BLOOM-560M
- 16 GB VRAM handles all bit-widths
- Estimated cost: $0.16 × 2 hrs = **$0.32 total**

### GCP

| Instance | GPU | VRAM | Preemptible $/hr |
|----------|-----|------|------------------|
| n1-standard-4 + T4 | T4 | 16 GB | ~$0.11 |
| n1-standard-4 + V100 | V100 | 16 GB | ~$0.74 |

---

## Soudry Lab Context

**Daniel Soudry's research focus:**
- Efficient deep learning
- Quantization and compression
- Neural network optimization
- Hardware-aware training

**Technion resources (typical):**
- NVIDIA DGX systems (A100)
- GPU clusters with V100/A100
- Cloud credits via academic programs

**Their typical experiments:**
- Large-scale quantization studies
- Multi-GPU training runs
- Models up to 7B-70B parameters

**Our experiments (BLOOM-560M) are modest by comparison:**
- Could run on a single consumer GPU (RTX 3080)
- Or cheapest cloud GPU spot instances
- No multi-GPU needed

---

## Execution Plan

### Option A: Local with Colab
1. Use Google Colab Pro ($10/month)
2. Get T4 or A100 depending on availability
3. Upload corpus, run experiments
4. Free tier may suffice for quick tests

### Option B: AWS Spot
1. Launch g4dn.xlarge spot instance (~$0.16/hr)
2. Install dependencies, clone repo
3. Run full bit-width sweep
4. Total cost: <$1

### Option C: Lambda Labs / RunPod
1. On-demand GPU instances
2. A10 GPU: ~$0.50/hr
3. No spot interruption risk
4. Slightly higher cost but simpler

---

## Quick Start (AWS)

```bash
# Launch spot instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot"}' \
  --key-name my-key

# SSH and setup
ssh -i my-key.pem ubuntu@<ip>
pip install torch transformers bitsandbytes scipy datasets
git clone https://github.com/uprootiny/quant-disparity
cd quant-disparity/experiments/phase-2-corpus

# Run experiments
python3 bitwidth_sweep.py --stage 0
python3 bitwidth_sweep.py --stage 1 --bits 8
python3 bitwidth_sweep.py --stage 1 --bits 4
python3 bitwidth_sweep.py --stage 2
```

---

## Summary

| Metric | Value |
|--------|-------|
| Minimum GPU | T4 (16GB) |
| Estimated runtime | 2 hours |
| Estimated cost (spot) | $0.32 |
| Recommended instance | AWS g4dn.xlarge |

**For Soudry Lab proposal:** Our compute needs are minimal. A single
afternoon on their cluster would complete all experiments.
