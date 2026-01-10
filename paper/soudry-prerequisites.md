---
layout: text
title: Prerequisites for Quantization Research - A Soudry Lab Primer
permalink: /soudry-prerequisites/
---

# Prerequisites for Quantization Research

*Everything Soudry's group assumes you already know*

---

## Part I: Mathematical Foundations

### 1. Linear Algebra Essentials

#### Matrices and Vectors

```
Weight matrix W ∈ ℝ^(m×n)
Input vector x ∈ ℝ^n
Output: y = Wx ∈ ℝ^m
```

**Key operations:**
- Matrix multiplication: O(mn²) for naive, O(n^2.37) for Strassen
- Transpose: W^T swaps rows/columns
- Trace: tr(W) = Σᵢ Wᵢᵢ (sum of diagonal)
- Frobenius norm: ||W||_F = √(Σᵢⱼ Wᵢⱼ²)

#### Singular Value Decomposition (SVD)

Any matrix W can be decomposed:

```
W = UΣV^T

Where:
- U ∈ ℝ^(m×m): left singular vectors (orthonormal)
- Σ ∈ ℝ^(m×n): diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
- V ∈ ℝ^(n×n): right singular vectors (orthonormal)
```

**Why it matters for quantization:**
- **Effective rank**: count of σᵢ > ε (how many dimensions matter)
- **Low-rank approximation**: keep top-k singular values
- **Condition number**: κ = σ_max / σ_min (numerical stability)

```python
# Practical: effective rank
def effective_rank(W, threshold=0.01):
    U, S, V = np.linalg.svd(W)
    return (S > S[0] * threshold).sum()
```

#### Eigenvalues and Eigenvectors

For square matrix A:
```
Av = λv

Where:
- λ: eigenvalue (scalar)
- v: eigenvector (direction preserved under transformation)
```

**Connection to variance:**
- Covariance matrix C = E[(x - μ)(x - μ)^T]
- Eigenvalues of C = variances along principal axes
- PCA = eigenvector decomposition of covariance

---

### 2. Probability & Statistics

#### Moments of a Distribution

For random variable X:

| Moment | Formula | Meaning |
|--------|---------|---------|
| Mean (μ) | E[X] | Center |
| Variance (σ²) | E[(X-μ)²] | Spread |
| Skewness | E[(X-μ)³]/σ³ | Asymmetry |
| Kurtosis | E[(X-μ)⁴]/σ⁴ | Tail heaviness |

**Excess kurtosis** = kurtosis - 3 (Gaussian has kurtosis = 3)

```python
from scipy.stats import kurtosis, skew

# Fisher's definition (excess kurtosis)
k = kurtosis(weights, fisher=True)  # Gaussian → 0
# Pearson's definition
k = kurtosis(weights, fisher=False)  # Gaussian → 3
```

**Why kurtosis matters:**
- High kurtosis → heavy tails → more outliers
- Outliers get clipped during quantization
- Clipping causes asymmetric errors

#### Key Distributions

**Gaussian (Normal):**
```
p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

Properties:
- Kurtosis = 3
- ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ
- Sum of Gaussians is Gaussian
```

**Laplace (Double Exponential):**
```
p(x) = (1/2b) exp(-|x-μ|/b)

Properties:
- Kurtosis = 6 (heavier tails than Gaussian)
- Sharper peak, fatter tails
- Common in neural network weights
```

**Lognormal:**
```
If X ~ Normal(μ, σ²), then Y = exp(X) ~ Lognormal

Properties:
- Always positive
- Heavy right tail
- Soudry showed gradients are approximately lognormal
```

**Comparison for quantization:**

| Distribution | Kurtosis | Outlier % (>3σ) | Quantization Impact |
|--------------|----------|-----------------|---------------------|
| Gaussian | 3 | 0.27% | Predictable clipping |
| Laplace | 6 | 1.1% | More clipping needed |
| Lognormal | varies | varies | Asymmetric errors |

#### Maximum Likelihood Estimation (MLE)

Given data {x₁, ..., xₙ}, find parameters θ that maximize:

```
θ_MLE = argmax_θ Π p(xᵢ|θ) = argmax_θ Σ log p(xᵢ|θ)
```

**For Gaussian:**
- μ_MLE = (1/n) Σ xᵢ  (sample mean)
- σ²_MLE = (1/n) Σ (xᵢ - μ)²  (sample variance)

**For quantization:** We use MLE to fit weight distributions and derive optimal clipping.

---

### 3. Information Theory

#### Entropy

Measures uncertainty/information content:

```
H(X) = -Σ p(x) log p(x)    [discrete]
H(X) = -∫ p(x) log p(x) dx [continuous]

Units: bits (log₂) or nats (ln)
```

**Properties:**
- H(X) ≥ 0
- H(X) maximized when X is uniform
- H(Gaussian) = (1/2) log(2πeσ²)

#### Mutual Information

How much knowing X tells you about Y:

```
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

Also:
I(X; Y) = Σ p(x,y) log [p(x,y) / (p(x)p(y))]
```

**For layers:**
- I(input; layer_output) measures information preserved
- Bottleneck layers have lower mutual information

#### KL Divergence

Distance between distributions (not symmetric):

```
D_KL(P || Q) = Σ p(x) log [p(x) / q(x)]

Properties:
- D_KL ≥ 0
- D_KL = 0 iff P = Q
- Asymmetric: D_KL(P||Q) ≠ D_KL(Q||P)
```

**For quantization:**
- Measures how much quantized distribution differs from original
- ACIQ minimizes KL divergence implicitly

#### Rate-Distortion Theory

Fundamental trade-off: bits vs. reconstruction error

```
R(D) = min_{p(x̂|x): E[d(x,x̂)]≤D} I(X; X̂)

Where:
- R: rate (bits needed)
- D: distortion (reconstruction error)
- d(x,x̂): distortion metric (usually MSE)
```

**For quantization:**
- Bits = log₂(quantization_levels)
- Distortion = MSE between original and quantized
- Rate-distortion bound tells us: "You CANNOT do better than this"

---

### 4. Optimization

#### Gradient Descent

```
θ_{t+1} = θ_t - η ∇L(θ_t)

Where:
- θ: parameters
- η: learning rate
- ∇L: gradient of loss
```

**Variants:**
- SGD: Use mini-batch gradient estimate
- Momentum: Add velocity term
- Adam: Adaptive learning rates per parameter

#### Convexity

Function f is convex if:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)  for λ ∈ [0,1]
```

**Why it matters:**
- Convex optimization has global optimum
- ACIQ's MSE minimization is convex → closed-form solution exists

#### Lagrangian and Constrained Optimization

To minimize f(x) subject to g(x) = 0:

```
L(x, λ) = f(x) + λg(x)

Solve: ∇_x L = 0  and  ∇_λ L = 0
```

**For quantization:**
- Minimize quantization error subject to bit-width constraint
- KKT conditions give optimal clipping threshold

---

## Part II: Neural Network Fundamentals

### 5. Backpropagation

#### Chain Rule

For composed function f(g(x)):
```
∂f/∂x = (∂f/∂g)(∂g/∂x)
```

#### Computational Graph

```
Input x → Linear(W₁) → ReLU → Linear(W₂) → Loss L

Forward: compute outputs layer by layer
Backward: compute gradients layer by layer (reversed)
```

#### Gradient Flow

```
∂L/∂W_i = ∂L/∂y_n × ∂y_n/∂y_{n-1} × ... × ∂y_{i+1}/∂y_i × ∂y_i/∂W_i
```

**Problems:**
- **Vanishing gradients**: multiplying small numbers → gradient → 0
- **Exploding gradients**: multiplying large numbers → gradient → ∞

**Solutions:**
- Residual connections (skip connections)
- Layer normalization
- Careful initialization

---

### 6. Transformer Architecture

#### Self-Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q = XW_Q  (queries)
- K = XW_K  (keys)
- V = XW_V  (values)
- d_k = dimension of keys
```

**Complexity:** O(n² × d) where n = sequence length

#### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

Where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

**Why multiple heads:** Different heads can attend to different patterns

#### Transformer Block

```
Input x
  ↓
LayerNorm → MultiHeadAttention → + (residual) → z
  ↓
LayerNorm → MLP → + (residual) → Output

MLP = Linear → GELU/ReLU → Linear
```

#### Residual Stream Perspective

```
x_0 = embed(tokens)
x_1 = x_0 + Attention_0(x_0) + MLP_0(x_0)
x_2 = x_1 + Attention_1(x_1) + MLP_1(x_1)
...
x_n = x_{n-1} + Attention_{n-1}(x_{n-1}) + MLP_{n-1}(x_{n-1})
output = unembed(x_n)
```

**Key insight:** Each layer ADDS to a running residual stream.
- Early errors propagate through residual connections
- This is why L0 errors affect all downstream layers

---

### 7. Language Model Specifics

#### Tokenization

```
Text → Tokens → Token IDs → Embeddings

"Hello" → ["Hel", "lo"] → [1234, 5678] → [0.1, -0.2, ...], [0.3, 0.1, ...]
```

**Byte-Pair Encoding (BPE):**
- Start with characters
- Iteratively merge most frequent pairs
- Results in subword vocabulary

**Token fertility:**
```
fertility(text, lang) = num_tokens(text) / num_words(text)

English: ~1.2 tokens/word
Hebrew: ~2.5 tokens/word (morphologically rich)
```

#### Causal Language Modeling

```
P(x_1, x_2, ..., x_n) = Π P(x_i | x_1, ..., x_{i-1})

Loss = -Σ log P(x_i | x_{<i})
```

**Autoregressive generation:**
- Predict one token at a time
- Feed prediction back as input
- Repeat until done

#### Perplexity

```
PPL = exp(-(1/N) Σ log P(x_i | x_{<i}))

Lower perplexity = better model
PPL = 1 means perfect prediction
```

---

## Part III: Quantization Theory

### 8. Fundamentals of Quantization

#### Uniform Quantization

```
Q(x) = round(x / scale) × scale

Where scale = (max - min) / (2^b - 1)
      b = bit-width
```

**For INT4 (4-bit signed):**
- Range: [-8, 7] (16 levels)
- Scale = max(|W|) / 7

```python
def quantize_int4(W):
    scale = W.abs().max() / 7
    W_q = (W / scale).round().clamp(-8, 7)
    W_dequant = W_q * scale
    return W_dequant, scale
```

#### Quantization Error

```
Error = W - Q(W)

MSE = E[(W - Q(W))²]
```

**For uniform quantization of uniform distribution:**
```
MSE = Δ² / 12

Where Δ = scale (step size)
```

#### Clipping

Values outside [-α, α] are clipped:

```
clip(x, α) = max(-α, min(α, x))
```

**Trade-off:**
- Small α: more clipping error (values truncated)
- Large α: more quantization error (coarser steps)

**ACIQ insight:** Optimal α exists that minimizes total MSE.

---

### 9. ACIQ Framework (Banner et al., 2019)

#### The Problem

Given weight distribution p(w), find optimal clipping threshold α.

#### Total Error Decomposition

```
MSE_total = MSE_quant + MSE_clip

MSE_quant = Δ²/12 × P(|W| ≤ α)
          = (α/(2^{b-1}))² / 12 × (1 - 2×P(|W| > α))

MSE_clip = E[W² | |W| > α] × P(|W| > α)
         = 2∫_α^∞ w² p(w) dw
```

#### Closed-Form Solution

For Gaussian distribution:
```
α* ≈ 2.5σ  for 4-bit
α* ≈ 3.0σ  for 8-bit
```

For Laplace distribution:
```
α* ≈ 3.0σ  for 4-bit
α* ≈ 4.0σ  for 8-bit
```

**General formula:**
```
α* = σ × f(b, kurtosis)

Where f is a function tabulated by ACIQ paper
```

---

### 10. Post-Training Quantization (PTQ)

#### Process

```
1. Train model in FP32
2. Collect calibration data
3. Compute scale factors per layer/channel
4. Quantize weights and activations
5. (Optional) Fine-tune quantized model
```

#### Calibration Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| MinMax | scale = max(|W|) / max_int | Simple | Sensitive to outliers |
| Percentile | scale = percentile(W, 99.9) / max_int | Robust | May clip important values |
| MSE | Minimize ||W - Q(W)||² | Optimal for MSE | Computationally expensive |
| Entropy | Minimize KL(W || Q(W)) | Good for activations | Expensive |

#### Per-Channel vs Per-Tensor

```
Per-tensor: One scale factor for entire weight matrix
Per-channel: One scale factor per output channel

Per-channel is more accurate but requires more storage
```

---

### 11. Key Quantization Metrics

#### Signal-to-Quantization-Noise Ratio (SQNR)

```
SQNR = 10 × log₁₀(signal_power / noise_power)
     = 10 × log₁₀(E[W²] / E[(W - Q(W))²])

Units: dB
```

**Rule of thumb:** SQNR increases ~6 dB per bit

#### Effective Bits

```
Effective bits = SQNR / 6.02

If SQNR = 24 dB, effective bits ≈ 4
```

---

## Part IV: Advanced Topics

### 12. Mean Field Theory for Quantization

Soudry's insight: Use statistical mechanics to analyze deep networks.

#### Key Result

```
L_max ∝ N^1.82

Where:
- L_max = maximum trainable depth
- N = quantization levels (2^b)
```

**Implication:**
- INT4 (N=16): L_max ~ 16^1.82 ≈ 146 layers
- INT2 (N=4): L_max ~ 4^1.82 ≈ 12 layers

This explains why INT2 often fails on deep networks.

### 13. Gradient Statistics

From Soudry's lognormal paper:

```
|∇W| ~ Lognormal(μ, σ²)

Where:
- Log-gradients are approximately Gaussian
- Actual gradients have heavy tails
- Sparsity emerges naturally (many small, few large)
```

**For quantization:**
- Gradient quantization is harder than weight quantization
- Need to account for lognormal distribution
- Optimal threshold differs from Gaussian assumption

### 14. FP8 Training Theory

Key insight from Soudry's FP8 paper:

```
Training fails when: ||∇W|| < √3 × quantization_noise

√3 comes from: variance of uniform quantization noise = Δ²/12
               std = Δ/√12 = Δ/(2√3)
               For signal to dominate: signal > √3 × noise
```

**Practical implication:**
- Monitor gradient magnitude relative to quantization step
- Scale gradients if they fall below threshold
- Use loss scaling in mixed-precision training

---

## Part V: Practical Algorithms

### 15. Quantization-Aware Training (QAT)

```python
class QuantizedLinear(nn.Module):
    def forward(self, x):
        # Forward pass: use quantized weights
        W_q = fake_quantize(self.weight)
        return F.linear(x, W_q, self.bias)

    def backward(self, grad):
        # Backward pass: straight-through estimator
        # Gradient flows through as if no quantization
        return grad  # STE: ∂Q/∂W ≈ 1
```

**Straight-Through Estimator (STE):**
- Quantization is non-differentiable
- STE pretends ∂Q/∂W = 1
- Works surprisingly well in practice

### 16. Mixed-Precision Training

```
FP16 for:
- Forward pass computations
- Most gradients

FP32 for:
- Master weights
- Loss accumulation
- Critical operations
```

**Loss scaling:**
```python
# Scale loss to prevent underflow
scaled_loss = loss * 1024
scaled_loss.backward()
# Unscale gradients
for param in model.parameters():
    param.grad /= 1024
optimizer.step()
```

### 17. Our Contribution: Layer-Selective Protection

```python
def selective_quantize(model, protect_layers):
    """
    Quantize all layers EXCEPT those in protect_layers.

    Based on our findings:
    - L0 + L_last: required for disparity < 1.0
    - Add L_0.75 for additional improvement
    """
    for name, module in model.named_modules():
        layer_idx = extract_layer_index(name)

        if layer_idx in protect_layers:
            # Keep in FP16/FP32
            continue
        else:
            # Quantize to INT4
            module.weight.data = quantize_int4(module.weight.data)
```

---

## Part VI: Key Equations Cheat Sheet

### Distributions
```
Gaussian:   p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
Laplace:    p(x) = (1/2b) exp(-|x-μ|/b)
Lognormal:  p(x) = (1/(xσ√(2π))) exp(-(ln(x)-μ)²/(2σ²))
```

### Quantization
```
Scale:      s = max(|W|) / (2^{b-1} - 1)
Quantize:   Q(W) = round(W/s) × s
MSE:        E[(W - Q(W))²] ≈ s²/12
SQNR:       10 × log₁₀(E[W²] / MSE)
```

### ACIQ Optimal Clipping
```
α* = σ × c(b, κ)

Where c(4, 3) ≈ 2.5 for Gaussian
      c(4, 6) ≈ 3.0 for Laplace
```

### Information Theory
```
Entropy:    H(X) = -Σ p(x) log p(x)
MI:         I(X;Y) = H(X) - H(X|Y)
KL:         D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
```

### Our Results
```
Criticality Score:
  score(L) = 2.5×boundary + 1.5×variance + 0.8×kurtosis
           + 1.2×outliers + 1.0×consolidation - 0.5×distance

Disparity Prediction:
  log(disparity) ≈ 6.0 - 0.5×total_score - 2.0×synergy_bonus

Phase Transition:
  disparity < 1.0 ⟺ (L0 ∈ protected) ∧ (L_last ∈ protected)
```

---

## Part VII: Vocabulary & Jargon

| Term | Definition |
|------|------------|
| **PTQ** | Post-Training Quantization (no retraining) |
| **QAT** | Quantization-Aware Training (finetune with fake quant) |
| **STE** | Straight-Through Estimator (gradient hack for non-differentiable ops) |
| **GPTQ** | GPU-accelerated PTQ using approximate Hessian |
| **AWQ** | Activation-aware Weight Quantization |
| **SQNR** | Signal-to-Quantization-Noise Ratio |
| **Perplexity** | exp(cross-entropy loss), measures LM quality |
| **Calibration** | Collecting statistics to set quantization scales |
| **Fake quantization** | Quantize-dequantize for training (simulates quantization) |
| **Mixed precision** | Different precisions for different operations |
| **Loss scaling** | Multiply loss to prevent gradient underflow |
| **Per-channel** | Separate scale per output dimension |
| **Symmetric** | Quantization centered at zero |
| **Asymmetric** | Quantization with offset (zero-point) |
| **Dynamic range** | Ratio of max to min representable value |
| **Outliers** | Values far from mean (>2.5σ or >3σ) |
| **Clipping** | Truncating values outside range |
| **MRL** | Morphologically Rich Language |
| **Disparity** | Ratio of LR to HR language degradation |

---

## Part VIII: Papers to Know

### Foundational

1. **ACIQ** (Banner et al., 2019): Analytical clipping for PTQ
   - Key: Closed-form optimal α for different distributions

2. **Lognormal Gradients** (Soudry et al., 2020): Gradient statistics
   - Key: Gradients follow lognormal, not Gaussian

3. **Mean Field Quantization** (Soudry et al., 2019): Depth limits
   - Key: L_max ∝ N^1.82

4. **FP8 Training** (Soudry et al., 2025): Low-precision training
   - Key: √3 threshold, SwiGLU sensitivity

### Recent Advances

5. **GPTQ** (Frantar et al., 2022): Efficient GPU quantization
   - Key: Approximate Hessian for layer-wise quantization

6. **AWQ** (Lin et al., 2023): Activation-aware weights
   - Key: Protect salient weights based on activation magnitude

7. **SmoothQuant** (Xiao et al., 2022): Migrate difficulty to weights
   - Key: Transfer quantization difficulty from activations to weights

### Multilingual & Fairness

8. **Tokenizer Fairness** (Ahia et al., 2023): Token fertility disparities
9. **Multilingual BERT Analysis** (Pires et al., 2019): Cross-lingual transfer
10. **Morphological Probing** (Belinkov et al., 2017): What NNs learn about morphology

---

*Reference date: 2026-01-10*
*For: Quantization Disparity Research Project*

