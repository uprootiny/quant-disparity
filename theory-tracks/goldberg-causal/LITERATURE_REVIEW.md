# Track G Literature Review: Causal Inference Foundations

*Prerequisites for do-calculus experiments on quantization disparity*

---

## 1. Causal Inference (Pearl, 2009)

### Key Concepts We Need

#### 1.1 Structural Causal Models (SCM)

**Definition.** An SCM M = (U, V, F) consists of:
- U: Exogenous (background) variables
- V: Endogenous variables {V₁, ..., Vₙ}
- F: Structural equations Vᵢ = fᵢ(PAᵢ, Uᵢ)

where PAᵢ are the parents of Vᵢ.

**Our SCM:**
```
U_T → T (tokenization quality)
       ↓
U_A → A (alignment score)
       ↓
U_R → R (redundancy)
       ↓
U_D → D (disparity)
```

Structural equations:
- T = f_T(U_T)
- A = f_A(T, U_A)
- R = f_R(A, U_R)
- D = f_D(R, U_D)

#### 1.2 The do-Operator

**Definition.** do(X = x) represents an intervention that:
1. Sets X to value x
2. Removes all arrows INTO X
3. Keeps all arrows OUT OF X

**Graphically:**

Before do(T = t):
```
U_T → T → A → R → D
```

After do(T = t):
```
T = t → A → R → D
(U_T disconnected)
```

**Key distinction:**
- P(D | T = t): Observational (conditioning)
- P(D | do(T = t)): Interventional (causal)

These differ when there are confounders!

#### 1.3 Confounding

**Definition.** X and Y are confounded if there's a common cause:
```
    U
   / \
  X   Y
```

**Problem:** P(Y | X) ≠ P(Y | do(X)) when confounded.

**Our situation:**
- T (tokenization) and D (disparity) both affected by training data?
- Resource level U confounds everything?

```
      Training Data (U)
      /      |      \
     T       A       D
```

If this is true, our observed correlations may be confounded.

#### 1.4 The Backdoor Criterion

**Theorem (Pearl).** A set Z satisfies the backdoor criterion relative to (X, Y) if:
1. No node in Z is a descendant of X
2. Z blocks all backdoor paths from X to Y

If Z satisfies backdoor:
$$P(Y | do(X)) = \sum_z P(Y | X, Z = z) P(Z = z)$$

**Our application:**
- To identify P(D | do(T)), need to block backdoor paths
- If training data U is the confounder, need to condition on it

#### 1.5 The Front-Door Criterion

**Theorem (Pearl).** If M mediates X → Y and:
1. M intercepts all directed paths from X to Y
2. No backdoor path from X to M
3. All backdoor paths from M to Y are blocked by X

Then:
$$P(Y | do(X)) = \sum_m P(M = m | X) \sum_{x'} P(Y | M = m, X = x') P(X = x')$$

**Our application:**
- T → A → R → D (A, R are mediators)
- If we can't condition on U, maybe front-door works?

---

## 2. do-Calculus (Pearl, 1995)

### The Three Rules

**Rule 1 (Insertion/deletion of observations):**
$$P(Y | do(X), Z, W) = P(Y | do(X), W)$$
if Z ⊥ Y | X, W in G_{\overline{X}}

**Rule 2 (Action/observation exchange):**
$$P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)$$
if Z ⊥ Y | X, W in G_{\overline{X}\underline{Z}}

**Rule 3 (Insertion/deletion of actions):**
$$P(Y | do(X), do(Z), W) = P(Y | do(X), W)$$
if Z ⊥ Y | X, W in G_{\overline{X}\overline{Z(W)}}

### Identifiability

**Definition.** A causal effect P(Y | do(X)) is identifiable from G if it can be computed from observational distribution P(V) using do-calculus.

**Our question:** Is P(D | do(T)) identifiable from our observations?

**To check:**
1. Draw the causal graph
2. Identify all paths from T to D
3. Apply backdoor/front-door criteria
4. If neither works, use do-calculus rules

---

## 3. Mediation Analysis (VanderWeele, 2015)

### Key Concepts We Need

#### 3.1 Direct and Indirect Effects

**Total effect:**
$$\text{TE} = \mathbb{E}[Y | do(X = 1)] - \mathbb{E}[Y | do(X = 0)]$$

**Natural direct effect (NDE):**
$$\text{NDE} = \mathbb{E}[Y_{1, M_0}] - \mathbb{E}[Y_{0, M_0}]$$

Effect of X on Y holding M at its natural value under X = 0.

**Natural indirect effect (NIE):**
$$\text{NIE} = \mathbb{E}[Y_{1, M_1}] - \mathbb{E}[Y_{1, M_0}]$$

Effect of X on Y through M only.

**Decomposition:**
$$\text{TE} = \text{NDE} + \text{NIE}$$

#### 3.2 Our Mediation Structure

```
T (tokenization)
    ↓
A (alignment)  ← potential mediator
    ↓
R (redundancy) ← potential mediator
    ↓
D (disparity)
```

**Questions:**
1. How much of T → D is mediated by A?
2. How much by R?
3. Is there a direct T → D path?

#### 3.3 Identification Conditions

For NDE and NIE to be identified:
1. No unmeasured confounding of X-M relationship
2. No unmeasured confounding of M-Y relationship
3. No unmeasured confounding of X-Y relationship
4. No M-Y confounder affected by X

**Our situation:**
- Training data volume may confound everything
- Need to control for it or find natural experiments

#### 3.4 Mediation Formula (Baron & Kenny Extended)

**Linear case:**
- Y = c'X + bM + ε₁
- M = aX + ε₂

Then:
- Direct effect = c'
- Indirect effect = a × b
- Total effect = c' + ab

**Our linear model (hypothesized):**
- A = β₁T + ε_A
- R = β₂A + ε_R
- D = γ/R + ε_D (nonlinear!)

The nonlinearity complicates things.

---

## 4. Intervention Designs

### 4.1 Randomized Experiments

**Gold standard:** Randomly assign T, observe D.

**Our problem:** Can't randomly assign tokenization to a trained model.

**Alternatives:**
- Natural experiments (models trained differently)
- Instrumental variables
- Synthetic interventions

### 4.2 Natural Experiments

**Idea:** Find variation in T that is "as good as random."

**Possible sources:**
- Different tokenizers on same data
- Same tokenizer on different data sizes
- Cross-model comparison (BLOOM vs XGLM)

**Our XGLM comparison:**
- BLOOM: high κ, shows disparity
- XGLM: low κ, no disparity
- Is this a natural experiment on "outlier formation"?

### 4.3 Synthetic Interventions

**Idea:** Create artificial scenarios that simulate do(X).

**For do(T):**
- Take well-tokenized language (English)
- Artificially degrade tokenization (random subword splits)
- Measure disparity change

**For do(A):**
- Take embeddings
- Artificially misalign (add noise, rotate)
- Measure disparity change

### 4.4 Within-Language Variation

**Our strongest design:**
- Same language (Hebrew)
- Different word types (alignment varies)
- Correlation r = -0.998

**Why this helps:**
- Controls for language-level confounders
- Variation in T (via word type) is plausibly exogenous
- Approximates do(T) within language

---

## 5. Our Specific Causal Questions

### Q1: Does tokenization CAUSE disparity?

**SCM:**
```
T → A → R → D
```

**To test:** Need P(D | do(T = t)).

**Approach:**
- Use within-language design (controls confounders)
- Use synthetic degradation (approximates intervention)

### Q2: Is alignment a mediator?

**Question:** How much of T → D goes through A?

**To test:** Compute NIE via A.

**Approach:**
- Measure T, A, R, D
- Fit mediation model
- Decompose effects

### Q3: Can we intervene on A directly?

**Question:** If we improve alignment (for fixed T), does D decrease?

**To test:** P(D | do(A = a'))

**Approach:**
- Representation alignment techniques
- Before/after measurement

### Q4: What is the total causal effect?

**Question:** If we improve tokenization by Δ, how much does disparity decrease?

**Estimand:** ∂/∂T E[D | do(T)]

**This is what we ultimately want for policy recommendations.**

---

## 6. Experimental Protocols

### Protocol 1: Within-Language Causal Test

```
1. Select language L with alignment variation (Hebrew)
2. Construct word types: high-align, medium-align, low-align
3. For each word type:
   a. Create test sentences
   b. Measure degradation under INT4
4. Regress D on alignment within L
5. Compare to cross-language correlation
```

**Inference:** If within-language effect matches cross-language, suggests causal.

### Protocol 2: Synthetic Tokenization Intervention

```
1. Take English text (baseline: good tokenization)
2. Create degraded versions:
   a. Random subword splits (simulates bad tokenizer)
   b. Character-level (extreme degradation)
   c. Word-level (minimal subword)
3. For each version:
   a. Run through model
   b. Measure activation patterns
   c. Measure disparity under quantization
4. Test: worse tokenization → higher disparity?
```

**Inference:** Demonstrates causal effect of tokenization quality.

### Protocol 3: Cross-Model Natural Experiment

```
1. Compare BLOOM (has outliers) vs XGLM (no outliers)
2. Same languages, same quantization
3. BLOOM shows disparity, XGLM doesn't
4. Attribute to outlier layer presence

Causal interpretation:
  do(outlier_layers = present) → disparity
  do(outlier_layers = absent) → no disparity
```

**Inference:** Outlier layers are necessary for disparity.

### Protocol 4: Mediation Analysis

```
1. Measure for all languages:
   - T: tokenization quality (fertility, coverage)
   - A: alignment score (cross-lingual similarity)
   - R: redundancy proxy (outlier activation)
   - D: disparity (degradation ratio)

2. Fit structural equations:
   A = β₁T + ε_A
   R = β₂A + ε_R
   D = β₃R + ε_D  (or D = γ/R)

3. Compute:
   - Total effect: T → D
   - Mediated via A: T → A → D
   - Mediated via R: T → A → R → D

4. Test: mediation explains >50% of total effect?
```

---

## 7. Identification Analysis

### Our Causal Graph

```
        U (training data)
       /|\
      / | \
     v  v  v
     T  A  D
     |  |  |
     +->A  |
        |  |
        +->R
           |
           +->D
```

Wait, this is messy. Let me draw it properly:

```
     U_data
    /  |  \
   T   |   (direct to D?)
   |   |
   v   |
   A<--+
   |
   v
   R
   |
   v
   D
```

**Confounding concern:** U_data affects T, A, and possibly D directly.

### Backdoor Analysis

For P(D | do(T)):
- Backdoor path: T ← U_data → D (if direct U → D exists)
- Need to condition on U_data

**Problem:** U_data (training data distribution) is not directly measured.

### Potential Solutions

1. **Proxy for U:** Use language resource level as proxy for training data volume.

2. **Within-language design:** Eliminates U (same language = same training).

3. **Instrumental variable:** Find Z that affects T but not D except through T.

4. **Front-door through A, R:** If T → A → R → D and no backdoor T ← · → A, can use front-door.

---

## 8. Connection to Our Empirical Work

### What We Have

| Finding | Causal Interpretation |
|---------|----------------------|
| r = -0.834 cross-language | Confounded (U_data) |
| r = -0.998 within-language | Less confounded (controls L) |
| XGLM null result | Natural experiment on outliers |
| 42% mediation via tokenization | Suggestive but not causal |

### What We Need

1. **Formal SCM specification** — Write down the model
2. **Identification analysis** — Can we identify P(D | do(T))?
3. **Intervention experiments** — Approximate do() via synthetic data
4. **Sensitivity analysis** — How robust to unmeasured confounding?

---

## References

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge.

2. Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*.

3. VanderWeele, T. J. (2015). *Explanation in Causal Inference*. Oxford.

4. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall.

5. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics*. Cambridge.

---

*Track G Literature Review — 2026-01-11*
