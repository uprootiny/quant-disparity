---
layout: text
title: Prerequisites by Research Track - Belinkov, Schwartz, Goldberg
permalink: /track-prerequisites/
---

# Prerequisites by Research Track

*What each research group assumes you already know*

---

## Track B: Belinkov Lab (Interpretability & Probing)

*Focus: What do neural networks learn? How can we understand their representations?*

### Core Philosophy

Belinkov's approach: Treat neural networks as subjects of scientific inquiry. Use controlled experiments to understand what information is encoded where.

---

### 1. Probing Classifiers

The fundamental tool of interpretability research.

#### Basic Idea

```
Frozen representations → Linear classifier → Linguistic label

If classifier succeeds: Information is "encoded" in representations
If classifier fails: Information is NOT linearly accessible
```

#### Mathematical Setup

```
Given:
- Representations h ∈ ℝ^d from layer L
- Labels y ∈ {1, ..., K} (e.g., POS tags)

Train: Linear probe f(h) = Wh + b
Evaluate: Accuracy on held-out data
```

#### Control Tasks (Hewitt & Liang, 2019)

**Problem:** High probe accuracy might mean:
1. Information is encoded (what we want)
2. Probe is just memorizing (bad)

**Solution:** Compare to random baseline

```
Selectivity = probe_accuracy - control_accuracy

Where control uses same architecture but random labels
High selectivity → Information truly encoded
```

#### Probe Complexity Trade-off

| Probe Type | Capacity | Risk |
|------------|----------|------|
| Linear | Low | Miss non-linear encodings |
| MLP (1 layer) | Medium | Balanced |
| Deep MLP | High | May learn task, not probe |

**Rule of thumb:** Use simplest probe that works. If linear probe succeeds, information is "easily accessible."

---

### 2. Linguistic Annotations

What Belinkov's group probes for:

#### Morphological Features

```python
# Example: Hebrew morphological tags
annotations = {
    'POS': 'VERB',           # Part of speech
    'Gender': 'Masc',        # Grammatical gender
    'Number': 'Plur',        # Singular/Plural
    'Person': '3',           # 1st/2nd/3rd
    'Tense': 'Past',         # Past/Present/Future
    'Binyan': 'Piel',        # Hebrew verb template
}
```

#### Syntactic Features

```python
# Dependency parsing
sentence = "The cat sat on the mat"
dependencies = [
    ('sat', 'nsubj', 'cat'),    # subject
    ('sat', 'prep', 'on'),      # prepositional
    ('on', 'pobj', 'mat'),      # object of prep
    ('cat', 'det', 'The'),      # determiner
]
```

#### Key Datasets

| Dataset | Languages | Annotations |
|---------|-----------|-------------|
| Universal Dependencies | 100+ | POS, morphology, syntax |
| OntoNotes | EN, ZH, AR | NER, coreference, POS |
| SIGMORPHON | 90+ | Morphological inflection |

---

### 3. Attention Analysis

#### Attention Weights as Explanations

```
Attention(Q, K, V) = softmax(QK^T/√d) V
                         ↑
                    These weights tell us "what attends to what"
```

#### Attention Head Patterns (Clark et al., 2019)

Common patterns in BERT-style models:
- **Positional heads:** Attend to adjacent positions
- **Syntactic heads:** Attend to syntactic relations
- **Separator heads:** Attend to [SEP] tokens
- **Rare word heads:** Attend to low-frequency tokens

#### Attention ≠ Explanation

**Caution:** Attention weights are NOT faithful explanations.

```
Attention tells you: What the model "looked at"
Does NOT tell you: What the model "used" for the decision
```

**Better alternative:** Gradient-based attribution
```
Attribution_i = |∂output/∂input_i|
```

---

### 4. Layer-wise Analysis

#### Information Flow Through Layers

```
Layer 0: Token-level features (lexical)
Layer 1-3: Local syntax (POS, local dependencies)
Layer 4-8: Long-range syntax (tree structure)
Layer 9-11: Task-specific / semantic
```

#### The "Onion" Hypothesis

Representations peel back layers of linguistic structure:

```
Surface → Morphology → Syntax → Semantics → Task
 (L0)       (L1-3)      (L4-8)    (L9-11)    (output)
```

#### Layer-wise Probing

```python
def probe_all_layers(model, texts, labels):
    results = {}
    for layer in range(num_layers):
        reps = get_representations(model, texts, layer)
        probe = train_linear_probe(reps, labels)
        results[layer] = evaluate(probe)
    return results  # Accuracy per layer
```

---

### 5. Representation Similarity Analysis (RSA)

#### Compare Two Representation Spaces

```
RSA(R1, R2):
1. Compute pairwise distances in R1: D1[i,j] = ||r1_i - r1_j||
2. Compute pairwise distances in R2: D2[i,j] = ||r2_i - r2_j||
3. Correlate: RSA = corr(flatten(D1), flatten(D2))
```

#### Applications

- Compare model layers to linguistic theories
- Compare different models
- Compare monolingual vs multilingual representations

#### Centered Kernel Alignment (CKA)

More robust alternative to RSA:

```
CKA(X, Y) = ||Y^TX||_F² / (||X^TX||_F × ||Y^TY||_F)

Where X, Y are representation matrices (samples × features)
```

---

### 6. Causal Interventions

Beyond correlation: Does the representation CAUSE the behavior?

#### Interchange Interventions

```python
def interchange_intervention(model, source_input, target_input, layer):
    """
    Replace target's representation at layer with source's.
    If output changes, layer representation is causally important.
    """
    source_rep = get_representation(model, source_input, layer)
    output = forward_with_replacement(model, target_input, layer, source_rep)
    return output
```

#### Causal Tracing (Meng et al., 2022)

```
1. Get clean output
2. Add noise to all representations
3. Restore one layer at a time
4. Which restoration recovers clean output?
   → That layer stores the relevant information
```

---

### 7. Key Papers for Belinkov Track

1. **Probing Classifiers** (Belinkov & Glass, 2019): Survey of probing methods
2. **What do NNs Learn about Morphology?** (Belinkov et al., 2017)
3. **A Primer in BERTology** (Rogers et al., 2020): Comprehensive BERT analysis
4. **Designing Probing Tasks** (Hewitt & Liang, 2019): Control task methodology
5. **Attention is not Explanation** (Jain & Wallace, 2019): Limitations of attention
6. **Locating Facts in GPT** (Meng et al., 2022): Causal interventions

---

## Track C: Schwartz Lab (Efficiency & Green AI)

*Focus: Computational efficiency, environmental cost, practical deployment*

### Core Philosophy

Schwartz's manifesto: "Red AI" (accuracy at any cost) vs "Green AI" (efficiency matters). Report compute costs alongside accuracy.

---

### 1. Computational Complexity

#### FLOPs (Floating Point Operations)

```
Linear layer: FLOPs = 2 × input_dim × output_dim × batch_size
Attention: FLOPs = O(n² × d) where n=sequence, d=dimension
Full transformer: FLOPs ≈ 6 × n_params × n_tokens (rough estimate)
```

#### Memory Complexity

```
Model size: 4 × n_params bytes (FP32)
Activations: O(batch × seq_len × hidden_dim × n_layers)
Attention cache: O(batch × n_heads × seq_len²)
```

#### Inference vs Training

| Operation | Training | Inference |
|-----------|----------|-----------|
| Forward | 1x | 1x |
| Backward | 2x | 0x |
| Memory | High (store activations) | Low (discard) |
| Total | ~3x forward | 1x forward |

---

### 2. Efficiency Metrics

#### Throughput

```
Tokens/second = batch_size × seq_len / time
```

#### Latency

```
Time to first token (TTFT)
Time per output token (TPOT)
End-to-end latency = TTFT + n_tokens × TPOT
```

#### Pareto Efficiency

```
Model A dominates Model B if:
  accuracy(A) ≥ accuracy(B) AND
  compute(A) ≤ compute(B) AND
  at least one inequality is strict
```

**Pareto frontier:** Set of non-dominated models

---

### 3. Model Compression Techniques

#### Pruning

```
Unstructured: Set individual weights to zero
Structured: Remove entire neurons/heads/layers

Magnitude pruning: Remove smallest |w|
Gradient pruning: Remove smallest |w × ∂L/∂w|
```

#### Knowledge Distillation

```
Student loss = α × CE(student, labels) + (1-α) × KL(student, teacher)

Temperature scaling softens distributions:
  soft_logits = logits / temperature
  p = softmax(soft_logits)
```

#### Quantization (covered in Soudry track)

#### Neural Architecture Search (NAS)

Automatically search for efficient architectures:
- Search space: Operations, connections
- Search strategy: Evolutionary, RL, gradient-based
- Evaluation: Accuracy + efficiency

---

### 4. Carbon Footprint

#### Estimation Formula (Strubell et al., 2019)

```
CO2 = Energy × Carbon Intensity

Energy (kWh) = Power (kW) × Time (h) × PUE

Where:
- Power: GPU TDP (e.g., A100 = 400W)
- PUE: Power Usage Effectiveness (~1.1-1.6)
- Carbon Intensity: Depends on grid (50-900 gCO2/kWh)
```

#### Reporting Standards

Always report:
```
- Hardware used (GPU type, count)
- Training time
- Number of hyperparameter trials
- Estimated CO2 (optional but encouraged)
```

---

### 5. Efficient Attention

#### Complexity Reduction

| Method | Complexity | Mechanism |
|--------|------------|-----------|
| Full attention | O(n²) | Baseline |
| Sparse attention | O(n√n) | Attend to subset |
| Linear attention | O(n) | Kernel trick |
| Flash Attention | O(n²) but faster | Memory-efficient |

#### Flash Attention

```
Standard: O(n²) memory for attention matrix
Flash: O(n) memory by tiling and recomputation

Key insight: Recompute attention in backward pass
             instead of storing the full n² matrix
```

#### Sliding Window Attention

```
Each token attends only to local context:
  attention_mask[i, j] = 1 if |i - j| ≤ window_size else 0
```

---

### 6. Inference Optimization

#### KV-Cache

```python
# Without cache: recompute all K, V for each new token
# With cache: store and reuse K, V from previous tokens

class CachedAttention:
    def forward(self, q, k, v, cache):
        if cache is not None:
            k = concat(cache['k'], k)
            v = concat(cache['v'], v)
        cache = {'k': k, 'v': v}
        return attention(q, k, v), cache
```

#### Speculative Decoding

```
1. Small model generates N draft tokens quickly
2. Large model verifies all N in parallel
3. Accept verified tokens, reject others
4. Repeat

Speedup: ~2-3x when draft acceptance rate is high
```

#### Batching Strategies

| Strategy | Latency | Throughput |
|----------|---------|------------|
| No batching | Low | Low |
| Static batching | Medium | Medium |
| Continuous batching | Low | High |

---

### 7. Tokenization Efficiency

#### Token Fertility (Ahia et al., 2023)

```
fertility(text, lang) = n_tokens / n_words

Higher fertility = more compute per word
```

**Cross-lingual disparity:**

| Language | Avg Fertility | Compute Cost (relative) |
|----------|--------------|-------------------------|
| English | 1.2 | 1.0x |
| German | 1.5 | 1.25x |
| Chinese | 1.8 | 1.5x |
| Hebrew | 2.5 | 2.1x |

#### Vocabulary Size Trade-off

```
Larger vocab:
  + Lower fertility
  + Fewer tokens to process
  - Larger embedding matrix
  - Slower softmax

Sweet spot: 32K-100K tokens for multilingual
```

---

### 8. Key Papers for Schwartz Track

1. **Green AI** (Schwartz et al., 2020): The manifesto
2. **Energy and Carbon in NLP** (Strubell et al., 2019): Carbon accounting
3. **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019): Sparse subnetworks
4. **DistilBERT** (Sanh et al., 2019): Knowledge distillation
5. **Language Models are Few-Shot Learners** (Brown et al., 2020): Scale vs efficiency
6. **Flash Attention** (Dao et al., 2022): Memory-efficient attention

---

## Track D: Goldberg Lab (Syntax & Morphology)

*Focus: Linguistic structure, morphological analysis, syntactic parsing*

### Core Philosophy

Goldberg's approach: Rigorous linguistic evaluation. Neural networks should be tested against well-established linguistic phenomena, not just aggregate metrics.

---

### 1. Morphological Analysis

#### Morpheme Types

```
Root: Core meaning (write, walk)
Affix: Added to root
  - Prefix: re-write, un-do
  - Suffix: walk-ed, quick-ly
  - Infix: (rare in English)
  - Circumfix: ge-sung-en (German)
```

#### Morphological Typology

| Type | Example | Description |
|------|---------|-------------|
| Isolating | Chinese | One morpheme per word |
| Agglutinative | Turkish | Morphemes concatenate |
| Fusional | Spanish | Morphemes fuse together |
| Templatic | Arabic/Hebrew | Root + pattern template |

**Hebrew Example (Templatic):**
```
Root: k-t-b (writing)
Template: CaCaC

k-t-b + CaCaC = katav (he wrote)
k-t-b + miCCaC = miktav (letter)
k-t-b + CoCeC = kotev (writer)
```

#### Morphological Inflection

```
Task: Generate inflected form from lemma + features

Input: (walk, VERB, Past)
Output: walked

Input: (גדול [gadol], ADJ, Fem, Plur)
Output: גדולות [gdolot]
```

---

### 2. Syntactic Structure

#### Constituency Parsing

```
Sentence → NP VP
NP → Det N | Det Adj N
VP → V | V NP | V NP PP

Tree:
          S
        /   \
      NP      VP
     / \     / \
   Det  N   V   NP
   The cat sat  ...
```

#### Dependency Parsing

```
Each word depends on exactly one head (except root)

The  cat  sat  on  the  mat
↓    ↓    ↑    ↓   ↓    ↓
det nsubj ROOT prep det pobj
 └────┘    ↑   └──────────┘
           └──────────────┘
```

#### Universal Dependencies (UD)

Standardized annotation scheme across languages:
- 17 universal POS tags
- 37 universal dependency relations
- Morphological features

---

### 3. Agreement & Long-Distance Dependencies

#### Subject-Verb Agreement

```
The keys to the cabinet ARE/*IS on the table.
                         ↑
           Must agree with "keys" (plural), not "cabinet"
```

**Agreement Attraction:** Intervening noun misleads the model

#### Testing Agreement in LMs

```python
def test_agreement(model, sentence_correct, sentence_wrong):
    """
    Model should assign higher probability to grammatical sentence.
    """
    prob_correct = model.score(sentence_correct)
    prob_wrong = model.score(sentence_wrong)
    return prob_correct > prob_wrong
```

#### Benchmark: BLiMP (Warstadt et al., 2020)

```
67 datasets testing:
- Anaphor agreement
- Argument structure
- Binding
- Control/raising
- Determiner-noun agreement
- Ellipsis
- Filler-gap dependencies
- Irregular forms
- Island effects
- NPI licensing
- Quantifiers
- Subject-verb agreement
```

---

### 4. Hebrew/Arabic Specific Linguistics

#### Root-Pattern Morphology

```
Arabic Root: k-t-b (related to writing)

Pattern + Root = Word:
- kitaab (book)
- kaatib (writer)
- maktaba (library)
- kutub (books)
```

#### Diacritics (Niqqud/Tashkeel)

```
Without diacritics: מלך (mlk)
Could be: מֶלֶך (melekh = king)
     or: מָלַך (malakh = he reigned)

Arabic: كتب (ktb)
Could be: كَتَبَ (kataba = he wrote)
     or: كُتُب (kutub = books)
```

#### Cliticization

```
Hebrew: ב + ה + בית = בבית (in the house)
        ש + אני + אמרתי = שאמרתי (that I said)

One token = multiple words
```

---

### 5. Targeted Syntactic Evaluation

#### Minimal Pairs

```
Grammatical: The author that the critics love writes well.
Ungrammatical: *The author that the critics loves write well.

Difference: Just the verb agreement
```

#### Test Suites

| Suite | Focus | Languages |
|-------|-------|-----------|
| BLiMP | English syntax | EN |
| CLiMP | Chinese linguistics | ZH |
| SLING | Hebrew syntax | HE |
| CLAMS | Morphological agreement | EN |

#### Constructing Tests

```python
# Template-based generation
templates = [
    ("The {noun_sing} {verb_sing}.", "correct"),
    ("The {noun_sing} {verb_plur}.", "incorrect"),
]

# Fill slots with controlled vocabulary
nouns_sing = ["cat", "dog", "child"]
verbs_sing = ["runs", "walks", "sleeps"]
verbs_plur = ["run", "walk", "sleep"]
```

---

### 6. Formal Language Theory

#### Chomsky Hierarchy

```
Type 0: Recursively enumerable (Turing machine)
Type 1: Context-sensitive
Type 2: Context-free (pushdown automaton)
Type 3: Regular (finite automaton)
```

#### Why It Matters

Natural language is NOT context-free (requires Type 1 or Type 0).

Examples:
- Center embedding: "The rat the cat the dog chased killed ate the malt"
- Cross-serial dependencies (Dutch/Swiss German)

**Question for neural LMs:** What formal class can they learn?

#### Testing Regular vs Context-Free

```
Regular: a*b* (any number of a's followed by any number of b's)
Context-free: a^n b^n (equal number of a's and b's)
Context-sensitive: a^n b^n c^n (equal number of each)

Can LMs count? Can they match nested structures?
```

---

### 7. Evaluation Metrics for Parsing

#### Unlabeled Attachment Score (UAS)

```
UAS = (correct heads) / (total words)

Only checks if dependency arrow points to correct head
```

#### Labeled Attachment Score (LAS)

```
LAS = (correct heads AND correct labels) / (total words)

More stringent: both head and relation must be correct
```

#### F1 for Constituency

```
Precision = (correct brackets) / (predicted brackets)
Recall = (correct brackets) / (gold brackets)
F1 = 2 × Precision × Recall / (Precision + Recall)
```

---

### 8. Key Papers for Goldberg Track

1. **Assessing BERT's Syntactic Abilities** (Goldberg, 2019)
2. **A Primer on Neural Network Models for NLP** (Goldberg, 2016): Comprehensive tutorial
3. **BLiMP** (Warstadt et al., 2020): Linguistic benchmark
4. **Targeted Syntactic Evaluation** (Marvin & Linzen, 2018)
5. **Character-aware Neural LMs** (Kim et al., 2016): Subword morphology
6. **Morphological Inflection** (Cotterell et al., 2016): SIGMORPHON shared task

---

## Synthesis: How Tracks Intersect with Our Work

### Belinkov + Our Research

```
Probing question: Do quantized models preserve morphological information?

Experiment:
1. Train morphology probe on FP32 model
2. Apply to quantized model
3. Compare accuracy per language
4. Correlate with our disparity metric
```

### Schwartz + Our Research

```
Efficiency question: What's the compute/fairness trade-off?

Experiment:
1. Measure disparity at different quantization levels
2. Measure throughput at each level
3. Plot Pareto frontier: fairness vs efficiency
4. Find optimal operating point
```

### Goldberg + Our Research

```
Linguistic question: Which syntactic phenomena degrade most?

Experiment:
1. Run BLiMP/targeted evaluation on quantized models
2. Break down accuracy by phenomenon
3. Correlate with language morphological complexity
4. Identify which linguistic structures are most vulnerable
```

---

## Common Prerequisites Across All Tracks

### Programming

```python
# PyTorch basics
model = transformers.AutoModel.from_pretrained("...")
outputs = model(input_ids)
hidden_states = outputs.hidden_states  # (layers, batch, seq, hidden)

# HuggingFace ecosystem
tokenizer = AutoTokenizer.from_pretrained("...")
dataset = load_dataset("universal_dependencies", "en_ewt")
```

### Statistics

- Hypothesis testing (t-test, bootstrap)
- Confidence intervals
- Effect sizes
- Multiple comparison correction (Bonferroni)

### Experiment Design

- Control conditions
- Ablation studies
- Statistical significance
- Reproducibility (seeds, hyperparameters)

### Writing

- Clear problem statement
- Related work positioning
- Quantitative claims with support
- Limitations section

---

*Reference date: 2026-01-10*
*For: Quantization Disparity Research Project*

