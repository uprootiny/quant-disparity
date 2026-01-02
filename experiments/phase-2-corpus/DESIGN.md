# Phase 2: Multilingual Corpus Collection Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  DESIGN DOCUMENT                                           rev. 2026-01-03 │
│  Stylistic-Seed Corpus Collection for Quantization Research               │
└────────────────────────────────────────────────────────────────────────────┘
```

## Overview

Build a multilingual corpus (~500MB per language, 14 languages) using
stylistic seeds to find similar text across the web. The corpus will
support activation analysis, quantization evaluation, and fine-tuning.

## Design Principles

Following Unix philosophy:
1. Each tool does one thing well
2. Text streams as universal interface
3. Composable pipeline stages
4. Fail fast, log everything

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   SEEDS     │────▶│   EMBED     │────▶│   SEARCH    │────▶│   FILTER    │
│  (jsonl)    │     │  (vectors)  │     │ (candidates)│     │  (quality)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CORPUS    │◀────│   DEDUP     │◀────│   FETCH     │◀────│   RANK      │
│  (parquet)  │     │  (minhash)  │     │  (text)     │     │  (scores)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Component Specifications

### 1. Seeds Module

**Purpose:** Define stylistic exemplars per language.

**Input:** Manual curation of 5-10 seed texts per language.

**Output:** `seeds/{lang}.jsonl`
```json
{"id": "ara_001", "text": "...", "register": "philosophical", "source": "manual"}
{"id": "ara_002", "text": "...", "register": "literary", "source": "manual"}
```

**Seed selection criteria:**
- 200-500 words each
- Represent target register (philosophical, essay, literary)
- No proper nouns that would bias search
- Grammatically complex (tests model capability)

### 2. Embed Module

**Purpose:** Extract stylistic fingerprints from seeds.

**Method:**
1. Sentence-level embeddings via multilingual model (LaBSE, mUSE, or SONAR)
2. Aggregate to document embedding (mean pooling)
3. Extract stylistic features:
   - Mean sentence length
   - Vocabulary diversity (type-token ratio)
   - Punctuation density
   - Discourse marker frequency

**Output:** `embeddings/{lang}.npy` + `features/{lang}.json`

**Tool:** `embed.clj`
```clojure
;; Input: seeds/ara.jsonl
;; Output: embeddings/ara.npy, features/ara.json
(defn embed-seeds [lang]
  (->> (read-jsonl (str "seeds/" lang ".jsonl"))
       (map extract-embedding)
       (save-npy (str "embeddings/" lang ".npy"))))
```

### 3. Search Module

**Purpose:** Find candidate URLs matching seed style.

**Sources (prioritized):**
1. **CommonCrawl** — largest, most diverse
2. **Wikipedia** — clean, multilingual baseline
3. **News archives** — formal register
4. **Academic repositories** — essays, theses
5. **Literary sites** — Gutenberg equivalents per language

**Method:**
- Query CommonCrawl index with language filter
- Use embedding similarity to rank URLs
- Combine with keyword extraction from seeds

**Output:** `candidates/{lang}.jsonl`
```json
{"url": "...", "score": 0.87, "source": "cc", "lang": "ara"}
```

**Tool:** `search.clj`
```clojure
(defn search-candidates [lang n-candidates]
  (let [embedding (load-npy (str "embeddings/" lang ".npy"))
        features (read-json (str "features/" lang ".json"))]
    (->> (query-common-crawl lang)
         (score-by-embedding embedding)
         (take n-candidates)
         (save-jsonl (str "candidates/" lang ".jsonl")))))
```

### 4. Fetch Module

**Purpose:** Retrieve and extract text from URLs.

**Method:**
1. HTTP fetch with retry/backoff
2. HTML → text extraction (trafilatura or similar)
3. Language verification (fasttext lid)
4. Basic cleaning (normalize unicode, strip boilerplate)

**Output:** `raw/{lang}/{hash}.txt`

**Tool:** `fetch.clj`
```clojure
(defn fetch-url [url]
  (-> (http-get url)
      (extract-text)
      (verify-language)
      (clean-text)))
```

**Rate limiting:** 1 req/sec per domain, respect robots.txt

### 5. Filter Module

**Purpose:** Quality control on fetched text.

**Filters (sequential, fail-fast):**

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Length | 100-10000 chars | Too short = fragments, too long = noise |
| Language confidence | > 0.9 | Reject mixed-language |
| Perplexity (BLOOM) | < 100 | Reject gibberish |
| Embedding similarity | > 0.6 | Match seed style |
| Adult content | classifier | Safety |
| Repetition ratio | < 0.3 | Reject template spam |

**Output:** `filtered/{lang}.jsonl`

**Tool:** `filter.clj`
```clojure
(defn filter-text [text lang]
  (and (length-ok? text)
       (language-ok? text lang)
       (perplexity-ok? text)
       (style-ok? text lang)
       (safe? text)
       (not-repetitive? text)))
```

### 6. Dedup Module

**Purpose:** Remove near-duplicates.

**Method:**
1. MinHash signatures (128 permutations)
2. LSH for candidate pairs
3. Jaccard similarity > 0.8 → duplicate

**Output:** `deduped/{lang}.jsonl`

**Tool:** `dedup.clj`
```clojure
(defn deduplicate [texts]
  (let [signatures (map minhash texts)
        lsh-index (build-lsh signatures)]
    (remove-duplicates texts lsh-index)))
```

### 7. Corpus Assembly

**Purpose:** Package final corpus.

**Format:** Parquet for efficient storage and querying
```
corpus/
  ara.parquet  # ~500MB
  eng.parquet
  ...
  metadata.json
```

**Schema:**
```
id: string
text: string
source_url: string
fetch_date: timestamp
embedding_sim: float
perplexity: float
char_count: int
```

## Size Estimates

| Language | Target Size | Est. Documents | Est. Tokens |
|----------|-------------|----------------|-------------|
| ara | 500 MB | 100K | 50M |
| eng | 500 MB | 150K | 80M |
| zho | 500 MB | 80K | 40M |
| ... | ... | ... | ... |
| **Total** | **7 GB** | **~1.5M** | **~700M** |

## Infrastructure Requirements

**Compute:**
- Embedding generation: 1 GPU-hour per language
- Fetching: 10K URLs/hour (rate-limited)
- Filtering: CPU-bound, parallelizable

**Storage:**
- Raw fetched: ~50GB (temporary)
- Final corpus: ~7GB
- Embeddings/indices: ~1GB

**Dependencies:**
```
# Clojure
[com.taoensso/nippy "3.2.0"]     ; serialization
[clj-http/clj-http "3.12.3"]     ; http client
[techascent/tech.ml "7.0"]       ; ML utilities

# Python (for embedding)
sentence-transformers
trafilatura
fasttext
```

## Pipeline Execution

```bash
# Full pipeline for one language
./run.sh ara

# Which expands to:
clj -M:embed    ara          # seeds → embeddings
clj -M:search   ara 10000    # embeddings → candidates
clj -M:fetch    ara          # candidates → raw text
clj -M:filter   ara          # raw → filtered
clj -M:dedup    ara          # filtered → deduped
clj -M:assemble ara          # deduped → parquet
```

## Quality Metrics

Track per-language:
1. **Coverage:** % of target size achieved
2. **Style match:** mean embedding similarity to seeds
3. **Perplexity distribution:** should match seed distribution
4. **Diversity:** unique n-gram ratio
5. **Source diversity:** % from each source type

## Validation

Before using corpus for experiments:
1. Manual inspection of 100 random samples per language
2. Compare BLOOM perplexity on corpus vs Wikipedia
3. Verify activation patterns match EXP-007 expectations

## Seed Examples

### Arabic (philosophical register)

From user-provided sample:
```
تيارٌ فكريٌّ طويل، لا يُختزل ولا يُحبس في صيغةٍ نهائية، بل يمتدّ كما يمتدّ النهر
عبر تضاريس متغيّرة. يبدأ من منبعٍ خفيّ، من فكرةٍ أولى قد تكون غامضة، مرتبكة،
غير مكتملة، ثم يواصل التدفق...
```

Characteristics to match:
- Extended metaphors
- Abstract nouns
- Complex sentence structure
- Philosophical vocabulary

### Additional seeds needed for:
- eng, fra, deu (European philosophical tradition)
- heb (Talmudic/philosophical)
- jpn, zho, kor (East Asian essay tradition)
- hin (Sanskrit-influenced philosophical)
- rus (literary/philosophical)
- fin, tur, tha, vie (contemporary essay)

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Seed curation | 1 week | seeds/*.jsonl |
| Pipeline implementation | 2 weeks | working pipeline |
| Corpus collection | 1 week | raw corpus |
| Quality validation | 3 days | validated corpus |
| **Total** | **~4 weeks** | **7GB multilingual corpus** |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Low-resource languages have sparse web presence | Supplement with Wikipedia, parallel corpora |
| Style drift from seeds | Periodic embedding recalibration |
| Rate limiting / blocking | Distributed fetching, proxy rotation |
| Quality degradation at scale | Tighten filters, manual sampling |

## Decision Points

**D001:** Embedding model choice
- LaBSE (Google): best multilingual, 109 languages
- SONAR (Meta): newer, supports more languages
- mUSE: older but proven

Recommendation: Start with LaBSE, evaluate SONAR if coverage gaps.

**D002:** CommonCrawl vs custom crawl
- CC: faster, broader, but less control
- Custom: slower, targeted, better quality

Recommendation: CC first pass, custom for gaps.

---

*Next step: Implement seeds module with user-provided Arabic example.*
