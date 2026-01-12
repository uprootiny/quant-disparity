/-
  LA-ACIQ: Language-Aware Analytical Clipping for Integer Quantization

  Formal verification of core theorems for multilingual quantization disparity.

  Main results:
  1. MSE decomposition into clipping + quantization error
  2. Convexity of MSE(α) ensuring unique optimum
  3. Monotonicity of optimal clipping in kurtosis
  4. Effective kurtosis formula for mixture distributions
  5. Disparity bound theorem
-/

import Laaciq.Quantization.Basic
import Laaciq.Quantization.MSE
import Laaciq.Probability.Kurtosis
import Laaciq.Probability.Mixture
import Laaciq.Optimization.Convexity
import Laaciq.Optimization.Optimal
