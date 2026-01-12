# Research Fit Analysis: Beyond Israeli Academia

*Where would this work find more fertile ground?*

---

## What We're Actually Producing

Before searching for homes, let's be honest about what we've built:

### The Work

| Dimension | Character |
|-----------|-----------|
| **Method** | Empirical, experimental, simulation-based |
| **Style** | Breadth over depth, characterization over proof |
| **Strength** | Volume of evidence, systematic measurement |
| **Weakness** | Lacks GPU validation, theoretical depth thin |
| **Framing** | Fairness, equity, linguistic justice |
| **Impact** | Documentation → advocacy → policy |

### The Evidence

| Finding | Type | Strength |
|---------|------|----------|
| 4.24× disparity | Descriptive | Strong |
| r = −0.998 within-language | Correlational (confound-free) | Very strong |
| Gateway layers L0/L9/L11 | Mechanistic | Moderate |
| 41% reduction from protection | Intervention | Moderate |
| 12 interpretability findings | Mechanistic | Strong |

### The Contribution

1. **Documented a problem** — quantization is not language-neutral
2. **Proposed a metric** — Fair-Efficiency Score
3. **Identified mechanism** — tokenization → alignment → redundancy → disparity
4. **Suggested intervention** — gateway layer protection

---

## Honest Assessment of Israeli Lab Fit

### Soudry Lab (Technion)
| Aspect | Fit |
|--------|-----|
| Theory depth | LOW — we lack proofs |
| Optimization | LOW — no novel optimization |
| Scale | LOW — simulated, not GPU |
| Extension of ACIQ | MODERATE — conceptual, not formal |
| **Overall** | **Marginal** |

### Schwartz Lab (HUJI)
| Aspect | Fit |
|--------|-----|
| Green AI | MODERATE — efficiency-fairness connection |
| Methodology | LOW — no carbon accounting |
| Data efficiency | LOW — not our focus |
| **Overall** | **Moderate** |

### Goldberg Lab (BIU)
| Aspect | Fit |
|--------|-----|
| Rigor | QUESTIONABLE — simulated experiments |
| Linguistic depth | LOW — surface-level typology |
| Standards | HIGH BAR — may not meet it |
| **Overall** | **Low** |

### Belinkov Lab (Technion)
| Aspect | Fit |
|--------|-----|
| Interpretability | HIGH — Track B aligns well |
| Circuit analysis | MODERATE — gateway layers are circuits |
| Probing | MODERATE — we do probing-style analysis |
| **Overall** | **Best Israeli fit** |

---

## Alternative Framings

Our work can be framed multiple ways. Each framing points to different communities:

### Framing 1: AI Ethics & Fairness
```
"Quantization encodes structural inequality into model compression."

Communities:
- FAccT (ACM Conference on Fairness, Accountability, Transparency)
- AIES (AAAI/ACM Conference on AI, Ethics, Society)
- AI ethics centers (Stanford HAI, Berkman Klein, AI Now Institute)

Institutions:
- Stanford HAI (Fei-Fei Li, James Zou)
- Princeton CITP (Arvind Narayanan)
- MIT Media Lab (Joy Buolamwini tradition)
- Algorithmic Justice League
```

### Framing 2: Language Technology for Marginalized Communities
```
"Efficient AI excludes low-resource language speakers."

Communities:
- MASAKHANE (African NLP)
- AmericasNLP
- LT4SG (Language Technology for Social Good)
- LREC-COLING

Institutions:
- Translators Without Borders
- MASAKHANE collective
- Google AI4Social Good (Jeff Dean's team)
- Meta AI LR Language initiative
```

### Framing 3: Digital Divide & Development
```
"Compression techniques widen the digital language divide."

Communities:
- ICT4D (Information & Communication Technologies for Development)
- CHI4Good
- ACM DEV

Institutions:
- Microsoft Research India (Kalika Bali)
- Carnegie Mellon Africa
- NYU Global TIES
- World Bank Development Data Group
```

### Framing 4: Linguistic Diversity Preservation
```
"Model efficiency comes at the cost of linguistic diversity."

Communities:
- Digital humanities
- Endangered languages documentation
- UNESCO linguistic diversity

Institutions:
- ELF (Endangered Languages Fund)
- Wikimedia Language Engineering
- Google Endangered Languages Project
- Unicode Consortium
```

### Framing 5: Practical ML Engineering
```
"Quantization benchmarks need multilingual evaluation."

Communities:
- MLSys
- Quantization practitioners
- Edge deployment engineers

Institutions:
- HuggingFace (democratizing NLP)
- EleutherAI (open-source LLMs)
- BigScience (BLOOM creators)
- Together.AI
```

---

## Better-Fit Institutions (Non-Israeli)

### Tier 1: Excellent Fit

#### 1. Microsoft Research India — Kalika Bali's Team
```
Why:
- Focus on low-resource Indian languages
- Practical impact orientation
- Industry resources + academic freedom
- Already work on tokenization fairness

Pitch:
"We show quantization amplifies tokenization bias.
Your tokenization work + our quantization analysis = full pipeline."

Publications:
- GLUECoS (code-switching)
- Low-resource NLP for Indian languages
```

#### 2. MASAKHANE Collective
```
Why:
- Grassroots African NLP community
- Low-resource languages are their ENTIRE focus
- Community-driven, distributed
- Our work directly serves their mission

Pitch:
"We document how LLM compression harms African languages.
Join us or let us join you to quantify and fix this."

Culture:
- Collaborative, not hierarchical
- Open science, open data
- Researcher-practitioners
```

#### 3. Graham Neubig's Lab (CMU LTI)
```
Why:
- Low-resource NLP focus
- Practical systems + research rigor
- Multilingual MT expertise
- Strong industry connections

Pitch:
"Your masakhane-mt work needs compression-fair deployment.
We show how current methods fail."

Output:
- Actual deployed systems
- Open-source tools (OpenNMT, etc.)
```

#### 4. Google DeepMind — Gemini Fairness Team
```
Why:
- Resources to validate at scale
- Gemini is multilingual by design
- Fairness is a corporate concern
- Could lead to real-world impact

Pitch:
"Your compression pipeline may inadvertently disadvantage
non-English users. We have evidence and mitigation strategies."

Risk:
- Corporate politics
- Less academic freedom
```

### Tier 2: Good Fit

#### 5. Stanford HAI — James Zou's Group
```
Why:
- AI fairness + technical depth
- Intersects with efficiency work
- High-impact publications
- Foundation model expertise

Pitch:
"Compression fairness is an underexplored dimension.
We provide first systematic evidence."
```

#### 6. ETH Zurich — Ryan Cotterell's Lab
```
Why:
- Computational linguistics rigor
- Multilingual NLP focus
- Strong on morphology
- Information-theoretic approach

Pitch:
"Morphologically rich languages suffer most under quantization.
We connect tokenization theory to compression fairness."
```

#### 7. University of Edinburgh — Alexandra Birch / Ivan Titov
```
Why:
- Machine translation focus
- Low-resource translation expertise
- Strong NLP program
- European perspective on language diversity

Pitch:
"MT systems need compression for deployment.
Our work shows which languages get hurt."
```

#### 8. BigScience / HuggingFace
```
Why:
- Created BLOOM (our test model)
- Open science ethos
- Practical impact focus
- Large multilingual community

Pitch:
"BLOOM's efficiency optimizations should be fair.
We provide the analysis and metrics."
```

### Tier 3: Worth Exploring

#### 9. Allen Institute for AI (AI2)
```
Why:
- Open science mission
- OLMo (open LLM) project
- Would care about fair compression

Pitch:
"OLMo's deployment needs fair quantization."
```

#### 10. EleutherAI
```
Why:
- Open-source LLM community
- Produced Pythia (our test model)
- Care about accessibility

Pitch:
"Pythia's compression should serve all languages equally."
```

#### 11. Meta FAIR — Low-Resource Languages
```
Why:
- Massively multilingual (NLLB, etc.)
- Resources for scale
- Deployment focus

Risk:
- Corporate environment
```

---

## Alternative Worldviews

If our work were grounded in different epistemics, where would it fit?

### Critical Data Studies
```
Worldview: Technology encodes power relations.
           Compression is a site of algorithmic injustice.

Institutions:
- NYU AI Now Institute
- Data & Society
- AI Ethics Lab (DAIR)

Scholars:
- Timnit Gebru
- Safiya Noble
- Kate Crawford
```

### Decolonial Computing
```
Worldview: Western AI development marginalizes Global South.
           Efficiency metrics serve dominant languages.

Institutions:
- Decolonise AI collective
- AUB (American University of Beirut)
- UCT (University of Cape Town)

Approach:
- Question efficiency as a universal good
- Center affected communities
- Participatory research
```

### Science & Technology Studies (STS)
```
Worldview: Technical choices are political choices.
           Quantization is not neutral.

Institutions:
- MIT STS Program
- Cornell STS
- Edinburgh Science Studies

Method:
- Ethnography of quantization practice
- Discourse analysis of efficiency rhetoric
- Historical analysis of compression methods
```

### Development Economics / ICT4D
```
Worldview: Technology should serve development.
           Language access is economic access.

Institutions:
- MIT D-Lab
- Berkeley CEGA
- Oxford ODID

Impact:
- Policy briefs
- World Bank engagement
- Development impact assessment
```

---

## Recommended Strategy

### For Academic Career
```
Primary: Graham Neubig (CMU) or ETH Zurich (Cotterell)
         - Rigorous, publishable, well-resourced
         - Clear path to top venues

Backup: Edinburgh or BigScience
        - Strong communities
        - Practical impact
```

### For Maximum Impact
```
Primary: MASAKHANE + industry partner
         - Direct community benefit
         - Real-world deployment

Backup: Google/Microsoft Research
        - Resources to scale
        - Deployment authority
```

### For Intellectual Exploration
```
Primary: Stanford HAI or AI Now Institute
         - Interdisciplinary
         - Ethics + technical depth

Backup: STS program
        - Critical perspective
        - Different questions
```

### For Immediate Collaboration
```
Primary: HuggingFace / BigScience
         - Already built BLOOM
         - Open science ethos
         - Would welcome contribution

Backup: EleutherAI
        - Community-driven
        - Quick to adopt
```

---

## Honest Conclusion

The Israeli labs are **prestigious but not optimal fits** for this work:
- Soudry wants theory; we have experiments
- Goldberg wants rigor; we have breadth
- Schwartz wants efficiency; we complicate their story
- Belinkov is closest but abroad until 2026

Better fits exist in:
1. **Low-resource language communities** (MASAKHANE, CMU, MSR India)
2. **AI fairness labs** (Stanford HAI, FAccT community)
3. **Open science collectives** (BigScience, EleutherAI)

The work we're producing is:
- **Advocacy research** more than theory
- **Characterization** more than optimization
- **Fairness documentation** more than efficiency improvement

That's valuable—but it belongs in communities that value those things.

---

*Analysis completed: 2026-01-11*
