# Israeli AI Labs: Research Narratives

*Applying the same investigative lens to multiple research groups*

---

## 1. Roy Schwartz Lab (HUJI) — Efficient & Green NLP

### The Central Question

```
How can we make NLP sustainable without sacrificing capability?
```

### Intellectual Trajectory

**Origins: The Carbon Cost Problem (2019-2020)**

The lab emerged from a stark observation: training large language models
produces as much CO2 as five cars over their lifetimes.

```
Strubell et al. (2019): "Energy and Policy Considerations for Deep Learning"
  → Training BERT produces 1,400 lbs CO2
  → GPT-2: even more
  → The trend is unsustainable

Schwartz et al. (2020): "Green AI"
  → Coined the term
  → Called for reporting compute costs
  → Proposed efficiency as a first-class metric
```

**Key Insight:** Accuracy alone is an incomplete metric. We need efficiency.

**Evolution: From Philosophy to Methods (2020-2023)**

```
Phase 1: Awareness
  "Green AI" paper → mainstream recognition
  Media coverage (Bloomberg, podcasts)

Phase 2: Measurement
  "Show Your Work" → require compute reporting
  "Measuring Carbon Intensity" → cloud-specific metrics

Phase 3: Solutions
  "Efficient NLP Survey" → comprehensive methods review
  "Vocab Diet" → vocabulary optimization
  Context length analysis → right-size inputs
```

### Validation Pattern

```
1. QUANTIFY THE PROBLEM
   Measure actual carbon costs
   Make hidden costs visible

2. PROPOSE METRICS
   Efficiency alongside accuracy
   Pareto frontiers

3. DEVELOP SOLUTIONS
   Data efficiency (fewer examples)
   Model efficiency (smaller architectures)
   Inference efficiency (distillation, pruning)

4. EMPIRICAL VALIDATION
   Show equivalent accuracy with less compute
```

### Impact So Far

| Output | Impact |
|--------|--------|
| Green AI paper | Mainstreamed sustainability in ML |
| Efficient NLP survey | Reference for practitioners |
| Vocab optimization | Practical size reductions |
| Media presence | Policy influence |

### Potential Impact (If Everything Pans Out)

```
Near-term (1-3 years):
  - Efficiency metrics become standard in papers
  - Cloud providers offer carbon tracking
  - "Green" becomes a selling point for models

Medium-term (3-5 years):
  - Regulation requires efficiency reporting
  - Models optimized for energy, not just accuracy
  - Edge deployment becomes norm

Long-term (5-10 years):
  - AI carbon footprint stabilizes despite scale
  - Efficiency-first training becomes default
  - Sustainable AI at global scale
```

### Our Work's Fit

```
Connection:  Quantization IS efficiency
             But our work shows efficiency can harm equity

Pitch:       "Efficient NLP must also be equitable NLP.
             We show quantization creates language disparity.
             Green AI needs to be fair AI."

Collaboration:
  - Joint work on efficiency-equity tradeoffs
  - Extend their survey to fairness dimension
  - Quantization methods that are both green AND fair
```

---

## 2. Yoav Goldberg Lab (BIU) — Understanding & Analyzing NLP

### The Central Question

```
How do neural NLP models actually work, and how can we trust them?
```

### Intellectual Trajectory

**Origins: From Parsing to Understanding (2010-2016)**

Goldberg made his name in syntactic parsing—structured prediction with
linguistic constraints.

```
Key Contribution: Transition-based parsing with neural networks
  - Combined symbolic NLP with neural methods
  - Proved you could inject structure into neural learning

The Book: "Neural Network Methods for NLP" (2017)
  - THE reference for a generation of NLP researchers
  - 42,000+ citations
  - Bridged ML and linguistics communities
```

**Evolution: From Building to Analyzing (2017-2024)**

```
Phase 1: Foundations
  Book + tutorials → community education
  Made neural NLP accessible

Phase 2: Probing
  "What do neural NLP models learn?"
  Probing classifiers, attention analysis
  Discovered what BERT knows (and doesn't)

Phase 3: LLM Behavior
  "Shortcut Triggers" → how LLMs cheat
  "Humans Perceive Wrong Narratives" → LLM explanations mislead
  "Diagnosing AI Explanations" → folk concepts of behavior
```

**Key Insight:** Understanding WHY models work (or fail) is as important
as making them work.

### Validation Pattern

```
1. BUILD CAREFULLY
   Deep linguistic knowledge
   Principled architectures

2. TEST RIGOROUSLY
   Probing, adversarial examples
   Find the failure modes

3. EXPLAIN CLEARLY
   Make findings accessible
   Write for practitioners

4. SET HIGH STANDARDS
   Compete with Stanford/MIT/CMU
   No shortcuts in science
```

### Impact So Far

| Output | Impact |
|--------|--------|
| NLP book | Educated thousands of researchers |
| Probing work | Defined subfield of analysis |
| Standards | BIU produces world-class NLP PhDs |
| Alumni | Omer Levy (Meta AI), Gail Weiss (EPFL) |

### Potential Impact (If Everything Pans Out)

```
Near-term:
  - LLM analysis becomes standard practice
  - Behavioral testing before deployment
  - Explanation quality metrics

Medium-term:
  - Interpretable-by-design models
  - Linguistically grounded LLMs
  - Trust certification for NLP systems

Long-term:
  - AI systems we truly understand
  - No more black boxes
  - Reliable language technology
```

### Our Work's Fit

```
Connection:  We're analyzing model BEHAVIOR under quantization
             Differential degradation is a behavioral finding

Pitch:       "Quantization changes model behavior non-uniformly.
             Some languages are affected more than others.
             This is a behavioral analysis problem."

Collaboration:
  - Probing quantized vs full models
  - Linguistic analysis of degradation
  - What capabilities break first?

Caution:     Goldberg has high standards
             Work must be rigorous, compete globally
             On sabbatical 2025-26
```

---

## 3. Yonatan Belinkov Lab (Technion) — Mechanistic Interpretability

### The Central Question

```
Can we reverse-engineer the algorithms that neural networks learn?
```

### Intellectual Trajectory

**Origins: What Does BERT Know? (2017-2020)**

Belinkov pioneered systematic probing of neural representations.

```
Key Question: Do neural models learn linguistic structure?

Method: Train classifiers on hidden states
        → If classifier succeeds, model encodes that info

Findings:
  - Lower layers: surface features
  - Middle layers: syntax
  - Upper layers: semantics
```

**Evolution: From Probing to Circuits (2020-2024)**

```
Phase 1: Probing (descriptive)
  "What information is there?"
  Linear probes, representation analysis

Phase 2: Causal (mechanistic)
  "How does the model use that information?"
  Causal interventions, activation patching

Phase 3: Circuits (algorithmic)
  "What is the algorithm?"
  Sparse feature circuits, component identification
```

**Recent Work (2024-2025):**

```
"Sparse Feature Circuits" (ICLR 2025)
  → Unsupervised discovery of causal graphs
  → Thousands of circuits automatically found

"Have Faith in Faithfulness" (COLM 2024)
  → Improved circuit discovery methods
  → EAP with integrated gradients

"MIB: Mechanistic Interpretability Benchmark"
  → Standard evaluation for interpretability methods
  → Compare circuit localization approaches

"Fine-tuning Enhances Existing Mechanisms"
  → Fine-tuning doesn't create new algorithms
  → It strengthens existing circuits
```

### Validation Pattern

```
1. FORMAL FRAMEWORK
   Grounded in causal inference
   Mathematically precise interventions

2. SCALABLE METHODS
   Work on real LLMs (not toy models)
   Automated discovery pipelines

3. BENCHMARKING
   Standard tasks, metrics
   Compare methods fairly (MIB)

4. BIOLOGICAL ANALOGY
   Like neuroscience for artificial networks
   Circuits, pathways, mechanisms
```

### Impact So Far

| Output | Impact |
|--------|--------|
| Probing methodology | Defined representation analysis |
| Causal interpretability | Moved beyond correlations |
| Circuit analysis | New paradigm for understanding |
| MIB benchmark | Standard for the field |

### Potential Impact (If Everything Pans Out)

```
Near-term:
  - Interpretability becomes standard in deployment
  - "Circuit audits" for safety-critical systems
  - Automated bug detection via circuits

Medium-term:
  - Compile models into understandable programs
  - Targeted editing of model behavior
  - Guaranteed safety properties

Long-term:
  - Fully reverse-engineered AI systems
  - Design-by-circuit model development
  - Human-understandable AI cognition
```

### Our Work's Fit

```
Connection:  We found WHICH LAYERS cause disparity
             That's circuit-level analysis!
             Layers 5, 21, 22 are the "disparity circuit"

Pitch:       "We identified a circuit for language disparity.
             Under quantization, this circuit fails for
             low-resource languages. Can we edit it?"

Collaboration:
  - Use sparse feature circuits on BLOOM
  - Identify language-specific features in outlier layers
  - Edit circuits to reduce disparity

Synergy:     Belinkov has methods
             We have the problem
             Together: first mechanistic analysis of
             quantization disparity

Note:        At Harvard Kempner 2025-26
             Contact spring 2026 for 2026-27 start
```

---

## 4. Soudry Lab (Technion) — Already Detailed

See `docs/soudry_lab_narrative.md` for full analysis.

**Summary:**
- Focus: Quantization, training dynamics, implicit bias
- Method: Theory-first, derive solutions, scale up
- Gap: Language-agnostic, efficiency over equity
- Our Fit: LA-ACIQ extends their framework to fairness

---

## Cross-Group Synthesis

### Complementary Perspectives

| Lab | Lens | Our Work Through That Lens |
|-----|------|---------------------------|
| Soudry | Optimization, efficiency | Optimal α*(λ) per language |
| Schwartz | Sustainability, green AI | Efficient but unfair? |
| Goldberg | Behavior, analysis | Behavioral failure under quantization |
| Belinkov | Circuits, mechanisms | Outlier layers as disparity circuit |

### Potential Multi-Lab Collaboration

```
The Full Story:

Soudry:   Theory — why quantization errors differ by distribution
Belinkov: Mechanism — which circuits are affected
Goldberg: Behavior — what capabilities break
Schwartz: Impact — efficiency-equity tradeoff

Together: Complete framework for equitable efficient NLP
```

### Strategic Positioning

| Target | Pitch | Strength |
|--------|-------|----------|
| Soudry | "Extend ACIQ to fairness" | Direct theory extension |
| Schwartz | "Green AI must be fair AI" | Complements their mission |
| Goldberg | "Behavioral analysis of quantization" | Rigorous analysis tradition |
| Belinkov | "Circuit for language disparity" | Mechanistic novelty |

---

## Recommended Approach

### Primary Target: Soudry

```
Reason: Direct extension of their work
        They have quantization expertise
        Industry connections (Intel)
        Our theory (LA-ACIQ) fits naturally
```

### Secondary Target: Belinkov (for 2026-27)

```
Reason: Mechanistic angle is novel
        Circuits + fairness = new direction
        Interpretability is hot topic
        Could be joint with Soudry
```

### Tertiary Target: Schwartz

```
Reason: Efficiency-equity framing is compelling
        Green AI community would care
        Less competitive (high-resource labs focus elsewhere)
```

---

*Narratives compiled: 2026-01-03*
