# Model Reference

## Currently Tested

### GPT-2 (gpt2)
- **HuggingFace**: `gpt2`
- **Parameters**: 124M
- **Architecture**: Decoder-only transformer
- **Layers**: 12
- **Heads**: 12
- **Hidden**: 768
- **Vocab**: 50,257
- **Training**: WebText (English-centric)
- **Status**: PRIMARY TEST MODEL

### OPT-125M
- **HuggingFace**: `facebook/opt-125m`
- **Parameters**: 125M
- **Architecture**: Decoder-only (OPT family)
- **Layers**: 12
- **Heads**: 12
- **Hidden**: 768
- **Vocab**: 50,272
- **Training**: Mixed web corpus
- **Status**: VALIDATION MODEL

### Pythia-160M
- **HuggingFace**: `EleutherAI/pythia-160m`
- **Parameters**: 160M
- **Architecture**: Decoder-only (GPT-NeoX)
- **Layers**: 12
- **Heads**: 12
- **Hidden**: 768
- **Vocab**: 50,304
- **Training**: The Pile
- **Status**: VALIDATION MODEL

---

## Planned Testing

### BLOOM-560M
- **HuggingFace**: `bigscience/bloom-560m`
- **Parameters**: 560M
- **Architecture**: Decoder-only
- **Layers**: 24
- **Heads**: 16
- **Hidden**: 1024
- **Vocab**: 250,880
- **Training**: 46 languages, 1.6TB
- **Status**: HIGH PRIORITY - truly multilingual

### mGPT
- **HuggingFace**: `ai-forever/mGPT`
- **Parameters**: 1.3B
- **Architecture**: Decoder-only
- **Layers**: 24
- **Heads**: 16
- **Hidden**: 2048
- **Vocab**: 100,000
- **Training**: 60 languages
- **Status**: MEDIUM PRIORITY

### Llama-2-7B
- **HuggingFace**: `meta-llama/Llama-2-7b-hf`
- **Parameters**: 7B
- **Architecture**: Decoder-only
- **Layers**: 32
- **Heads**: 32
- **Hidden**: 4096
- **Vocab**: 32,000
- **Training**: Mixed web corpus
- **Status**: REQUIRES GPU - scale validation

### Mistral-7B
- **HuggingFace**: `mistralai/Mistral-7B-v0.1`
- **Parameters**: 7B
- **Architecture**: Decoder-only with GQA
- **Layers**: 32
- **Heads**: 32
- **Hidden**: 4096
- **Vocab**: 32,000
- **Training**: Mixed web corpus
- **Status**: REQUIRES GPU

---

## Multilingual Encoders

### mBERT
- **HuggingFace**: `bert-base-multilingual-cased`
- **Parameters**: 110M
- **Architecture**: Encoder-only
- **Layers**: 12
- **Heads**: 12
- **Hidden**: 768
- **Vocab**: 119,547
- **Training**: 104 languages
- **Status**: ENCODER BASELINE

### XLM-RoBERTa
- **HuggingFace**: `xlm-roberta-base`
- **Parameters**: 278M
- **Architecture**: Encoder-only
- **Layers**: 12
- **Heads**: 12
- **Hidden**: 768
- **Vocab**: 250,002
- **Training**: 100 languages, CC-100
- **Status**: ENCODER COMPARISON

---

## Model Selection Rationale

1. **GPT-2**: Small, well-understood, many papers use it
2. **OPT/Pythia**: Independent validation of GPT-2 findings
3. **BLOOM**: First truly multilingual test
4. **7B models**: Scale validation (requires GPU)

---

## Known Quantization Behaviors

| Model | INT8 Status | INT4 Status | Notes |
|-------|-------------|-------------|-------|
| GPT-2 | Works | 52x disparity | Our finding |
| OPT-125M | Unknown | Unknown | To test |
| Pythia-160M | Unknown | Unknown | To test |
| BLOOM-560M | Unknown | Unknown | Priority |
| Llama-2-7B | AWQ works | GPTQ tested | Per literature |

---

*Last updated: 2026-01-04*
