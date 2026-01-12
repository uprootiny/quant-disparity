# Literature Foundation: Track B (Interpretability)

*Grounding our work in Belinkov Lab and mechanistic interpretability literature*

**Target: Yonatan Belinkov Lab (Technion)**

---

## Core Publications: Belinkov Lab

### 1. "Analysis Methods in Neural Language Processing" (Belinkov & Glass, 2019)

**Key Claims:**
- Probing classifiers reveal hierarchical encoding in transformers
- Deeper layers encode more abstract features
- Representations are distributed but structured

**Our Extension:**
- Quantization disrupts these encodings non-uniformly
- LR languages show greater probing accuracy loss
- We can MEASURE interpretability damage

**Connection Experiments:**
- B-004: Probing accuracy under quantization
- B-006: Layer-wise feature preservation

---

### 2. "Analyzing Individual Neurons in Pre-trained Models" (Durrani et al., 2020)

**Key Claims:**
- 15-20% of neurons are language-specific
- Neurons specialize for morphology, syntax, semantics
- Cross-lingual transfer relies on language-neutral neurons

**Our Extension:**
- Language-specific neurons may be more quantization-fragile
- LR languages have fewer redundant language-neutral pathways
- Neuron death under quantization is language-biased

**Connection Experiments:**
- B-007: Language-specific neuron survival rate
- B-008: Dead neuron distribution by language

---

### 3. "Backward Lens: Projecting Language Model Gradients" (Belinkov et al., 2024 - Best Paper)

**Key Claims:**
- Gradients reveal token importance
- Backward projection gives interpretable saliency
- Token-level attribution is tractable

**Our Extension:**
- Apply backward lens to quantized vs FP32 comparison
- Identify which tokens lose importance under quantization
- LR languages may lose more semantically critical tokens

**Connection Experiments:**
- B-009: Token saliency shift under quantization
- B-010: Semantic token preservation by language

---

### 4. "The Quest for the Right Mediator" (Belinkov et al., 2024)

**Key Claims:**
- EAP-IG (Edge Attribution Patching) identifies causal paths
- Mediation analysis distinguishes correlation from causation
- Critical circuits can be isolated

**Our Extension:**
- Identify circuits mediating language-specific processing
- Test if quantization damages these mediators
- Provide CAUSAL evidence for disparity mechanism

**Connection Experiments:**
- B-011: Causal mediation analysis for disparity
- B-012: Circuit isolation for LR languages

---

### 5. "Discovering Latent Concepts Learned in BERT" (Dalvi et al., 2022)

**Key Claims:**
- Morphological concepts encoded in early-middle layers
- Syntactic concepts in middle layers
- Semantic concepts in late layers

**Our Extension:**
- MRLs (Hebrew, Arabic) have more morphology-dependent processing
- Quantization may disproportionately harm morphological circuits
- Layer importance varies by language typology

**Connection Experiments:**
- B-013: Morphological concept preservation
- B-014: Syntactic vs semantic layer damage

---

## Attention Mechanisms Literature

### 6. "The Super Weight in Large Language Models" (arXiv:2411.07191)

**Key Claims:**
- 0.01% of weights are "super weights" - orders of magnitude more important
- Located in attention projections (Q, K, V, O)
- Pruning super weight destroys model

**Our Extension:**
- Super weights may serve different languages differently
- LR languages may depend on fewer super weights
- Quantization of super weights is catastrophic

**Connection Experiments:**
- B-015: Super weight language dependency
- B-016: Super weight quantization damage

---

### 7. "When Attention Sink Emerges in Language Models" (ICLR 2025)

**Key Claims:**
- Attention sinks emerge at 1k-2k training steps
- Caused by softmax sum-to-one constraint
- Create massive activations as byproduct

**Our Extension:**
- Do attention sinks serve all languages equally?
- Are LR languages excluded from sink tokens?
- Sink clipping may harm LR disproportionately

**Connection Experiments:**
- B-017: Attention sink distribution by language
- B-018: Sink token language bias

---

### 8. "Massive Activations in Large Language Models" (Sun et al., 2024)

**Key Claims:**
- Massive activations are predictable
- They encode critical information
- Clipping them damages model

**Our Extension:**
- Massive activation preservation is language-biased
- LR languages have fewer redundant pathways
- Activation clipping affects languages differently

**Connection Experiments:**
- B-019: Massive activation distribution by language
- B-020: Activation clipping damage by language

---

## Probing Methodology Literature

### 9. "A Primer in BERTology" (Rogers et al., 2020)

**Key Claims:**
- Probing reveals what models encode
- Different layers encode different linguistic features
- Probing accuracy indicates representation quality

**Our Extension:**
- Probing accuracy DROP under quantization measures damage
- LR languages may show larger accuracy drops
- Feature-specific damage patterns

**Connection Experiments:**
- B-021: Probing accuracy delta by language
- Covered by existing B-004

---

### 10. "What Do NLP Probes Actually Measure?" (Ravichander et al., 2021)

**Key Claims:**
- Probes can memorize rather than measure
- Control tasks needed for valid conclusions
- Probe complexity affects results

**Our Extension:**
- Need control probes for quantization experiments
- Ensure we measure representation damage, not probe artifact
- Use multiple probe architectures

**Connection Experiments:**
- B-022: Control task validation
- Methodological control

---

## Cross-Lingual Representation Literature

### 11. "Cross-Lingual Language Model Pretraining" (Conneau & Lample, 2019)

**Key Claims:**
- Multilingual models create shared representation space
- Cross-lingual transfer depends on representation alignment
- LR languages may be poorly aligned

**Our Extension:**
- Quantization may damage cross-lingual alignment
- LR languages may lose alignment more
- Disparity relates to alignment preservation

**Connection Experiments:**
- B-023: Cross-lingual alignment under quantization
- Links to Track D findings

---

### 12. "MAD-X: An Adapter-Based Framework" (Pfeiffer et al., 2020)

**Key Claims:**
- Language adapters modularize multilingual processing
- LR languages benefit from adapter approach
- Separation of language-specific and shared components

**Our Extension:**
- Can adapter-like protection reduce disparity?
- Language-specific components need different quantization
- Modular approach to fair quantization

**Connection Experiments:**
- B-024: Adapter-style protection for disparity
- Intervention direction

---

## Gaps in Literature We Address

| Gap | Literature Status | Our Contribution |
|-----|-------------------|------------------|
| Probing under quantization | NOT STUDIED | First probing-based damage measurement |
| Circuit analysis for disparity | NOT STUDIED | Identify which circuits fail |
| Attention sink language bias | NOT STUDIED | Test if sinks serve all languages |
| Super weight language dependency | NOT STUDIED | Map super weight usage by language |
| Causal mediation for disparity | NOT STUDIED | EAP-IG applied to quantization |
| Representation damage by language | MINIMAL | Systematic measurement |

---

## 12 New Experiments Derived from Literature

| ID | Name | Literature Connection | Question |
|----|------|----------------------|----------|
| B-004 | Probing accuracy delta | Belinkov 2019 | How much does probing accuracy drop per language? |
| B-005 | Gradient-based circuits | Belinkov 2024 | Which gradients change most under quant? |
| B-006 | Layer-wise feature preservation | Dalvi 2022 | Which features (morph/syn/sem) survive? |
| B-007 | Language-specific neuron survival | Durrani 2020 | Do lang-specific neurons die more? |
| B-008 | Dead neuron distribution | Durrani 2020 | Is neuron death language-biased? |
| B-009 | Token saliency shift | Belinkov 2024 | Which tokens lose importance? |
| B-010 | Semantic token preservation | Belinkov 2024 | Are content words preserved? |
| B-011 | Causal mediation analysis | Belinkov 2024 | What mediates disparity causally? |
| B-012 | Super weight language mapping | Super Weight 2024 | Which languages use which super weights? |
| B-013 | Attention sink language bias | ICLR 2025 | Do sinks serve all languages equally? |
| B-014 | Massive activation by language | Sun 2024 | Activation distribution by language? |
| B-015 | Cross-lingual alignment damage | Conneau 2019 | Does quant break alignment? |

---

## Methodological Principles from Literature

1. **Use probing classifiers** (Belinkov tradition) - not just perplexity
2. **Control for probe complexity** (Ravichander 2021) - validate findings
3. **Apply causal methods** (EAP-IG) - not just correlation
4. **Check multiple layers** (Dalvi 2022) - damage may be layer-specific
5. **Map to linguistic features** - morphology, syntax, semantics separately

---

## Integration with Existing Findings

| Existing B Finding | Literature Support | Strengthens Because |
|--------------------|-------------------|---------------------|
| 16.7% lang-specific heads | Durrani 2020 (15-20%) | Matches prior work |
| 3.3x representation damage | Novel | Extends literature |
| 2.23x ablation sensitivity | Novel | Causal evidence |
| Gateway layers L0/L9/L11 | Dalvi 2022 (layer hierarchy) | Connects to feature levels |

---

*This document grounds Track B in peer-reviewed interpretability literature.*
