# Track C Assumptions: Efficiency Techniques

*What we assume about quantization, pruning, distillation, and their fairness*

**Target Lab:** Schwartz Lab (Efficient NLP)

---

## Track-Specific Assumptions

### C-1: All Efficiency Techniques Cause Disparity
**Assumption:** Quantization, pruning, and distillation all disproportionately hurt low-resource languages.

**Could be wrong if:**
- Some techniques are inherently fair
- Disparity is technique-specific, not universal
- We haven't tested all relevant techniques

**Evidence:** E2-E4 confirmed trifecta (4.24x, 3.02x, 3.04x disparity)

**Risk Level:** LOW for tested techniques; HIGH for untested

---

### C-2: INT4 is the Practical Threshold
**Assumption:** INT4 quantization is where disparity becomes severe; INT8 is relatively safe.

**Could be wrong if:**
- Hardware advances change the precision landscape
- Different quantization algorithms have different thresholds
- Threshold varies by application

**Evidence:** E2 showed non-linear acceleration below INT6; INT4 is cliff edge

**Risk Level:** MODERATE - hardware-dependent

---

### C-3: Efficiency-Fairness Trade-Off is Necessary
**Assumption:** There is an inherent trade-off; you cannot have both maximum efficiency and perfect fairness.

**Could be wrong if:**
- Novel techniques achieve both
- The trade-off is an artifact of current methods
- Fairness constraints are less costly than assumed

**Evidence:** All our experiments show trade-off; E9 showed adaptive can improve efficiency

**Risk Level:** MODERATE - may be solvable

---

### C-4: Carbon Cost of Fairness is Measurable
**Assumption:** We can quantify the compute/carbon cost of achieving fair performance.

**Could be wrong if:**
- Costs are context-dependent
- Hardware efficiency varies too much
- Our cost model is unrealistic

**Evidence:** E-C004 estimated 56x compute disparity; E9 showed adaptive reduces cost

**Risk Level:** MODERATE - estimates are rough

---

### C-5: Protection Overhead is Acceptable
**Assumption:** The memory/compute overhead of protecting critical layers (L0+L9+L11) is acceptable for deployment.

**Could be wrong if:**
- Hardware constraints are stricter than assumed
- Real-world deployment has hidden costs
- Users won't accept any overhead

**Evidence:** E9 showed hybrid achieves 68% benefit at 8% overhead

**Risk Level:** LOW for most applications; HIGH for edge deployment

---

## Track C Specific Confounds

1. **Benchmark-specific results** - efficiency gains may not transfer
2. **Hardware assumptions** - different hardware, different results
3. **Cost models are estimates** - real costs may differ significantly

---

## What Would Falsify Track C Claims?

1. Finding an efficiency technique that doesn't cause disparity
2. Showing protection overhead is prohibitive
3. Demonstrating that "fairness cost" is negligible
