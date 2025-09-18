# LLM Auditor (Search & Brand Analysis)

GEO brand & prompt monitoring workflow which:

1. (Optionally) expands an initial list of seed brands into competitor brands using an LLM + live search.
2. (Optionally) generates additional commercial / consumer search style prompts based on sector, market and (optionally) brands.
3. Executes every prompt across one or more LLM models (optionally with live web/search augmentation) concurrently.
4. (When search is enabled) analyses brand mention rank and URL placement across model responses and their cited references.
5. Returns a wide DataFrame (pushed as dataset items) with one row per prompt and columns per model for raw text and references, plus derived rank columns.

---

## When to use it

Use this actor to:

- Benchmark how different LLMs answer realistic commercial / comparison style queries in your sector.
- Track presence and relative ranking of your brand (and competitors) in both answer text and cited reference URLs.
- Generate a synthetic but realistic pool of prompts to stress-test multi-model performance.
- Analyse how inclusion / exclusion of brand names in queries affects model responses.

Not a good fit when you only need: simple single-model prompting, non-commercial Q&A, or full SERP scraping (see the separate SERPs actor for that).

---

## Input schema (summary)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| brands (required) | array[string] | (none) | Your own brand names or homepage URLs (canonicalized & lower‑cased). Must provide at least one. |
| competitors | array[string] | - | Optional seed competitor brand names or URLs. Automatically expanded (search + LLM) up to `competitors_max`. |
| competitors_max | integer | 10 | Maximum total competitors (including any seeds) to keep after discovery. |
| models | array[string] | openai/gpt-5, google/gemini-2.5-flash | LLM provider/model identifiers to evaluate (provider/model). If empty defaults are injected. |
| prompts | array[string] | - | Optional seed prompts to include directly. |
| prompts_max | integer | 10 (UI) / 20 (code default) | Target total prompt count (seed + generated). Additional prompts are LLM-generated if context available. |
| prompt_intents | array[string] | [commercial, transactional] | Intent categories to bias generation (broad topical diversity). |
| prompt_language | string | English | Language for generated prompts. |
| brands_in_prompt | enum | never | Whether generated prompts explicitly mention brand names (`never`, `sometimes`, `always`). |
| sector | string | - | Sector / industry context (improves prompt + competitor generation). |
| market | string | - | Geographic market (localises generation & competitor discovery). |
| use_search | boolean | true | Enable live web/search augmentation when querying models (recommended). |
| search_country | string | - | Optional country code for search localisation (e.g., 'us', 'uk', 'de') |

Notes:
1. `brands` is now required by the schema; generation & competitor discovery work best when at least `brands` or (`sector` + optionally `market`) are provided. 
2. If you omit `models`, internal defaults are inserted. 
3. All brand/competitor entries are normalised (lower‑cased; URLs reduced to host).
4. `search_country` localisation is currently supported only by OpenAI (GPT models), XAI (Grok), and Google's AI Overview (via hasdata API).
---

## Output

Dataset items (one row per prompt) with columns grouped into:

1. Core prompt/context
2. Per‑model raw outputs
3. Per‑model brand ranking details (when brands + search enabled)
4. Cross‑model summary indicators & counts

### 1. Core
* `prompt` – Original prompt text.

### 2. Per‑model raw outputs (model id appears literally, provider prefix removed)
* `answer_<model>` – Plain answer text returned by the model.
* `sources_<model>` – List of cited source objects (`{"title": ..., "url": ...}`) when live search enabled; absent / empty list otherwise.

### 3. Per‑model brand ranking detail columns (added only if `use_search` is true AND seed brands supplied)
Generated for each model:
* `answer_<model>_brand_ranking` – Ordered list of brand tokens by first text occurrence (earliest index first).
* `sources_<model>_brand_positions` – List of `{"name": <brand>, "position": <zero_based_pos>}` objects for first URL match per brand (sorted by position). Brands not found are omitted.
* `brand_mentioned_in_answer_<model>` – Boolean: at least one of own brands appears in answer text.
* `brand_position_in_answer_<model>` – 1-based position (rank order) of first own brand in `answer_<model>_brand_ranking` (None if not present).
* `competitor_mentioned_in_answer_<model>` – Boolean: any competitor brand mentioned in answer text.
* `competitor_position_in_answer_<model>` – 1-based position of first competitor brand in answer ranking.
* `brand_mentioned_in_sources_<model>` – Boolean: any own brand appears in cited sources URLs.
* `brand_position_in_sources_<model>` – 1-based first occurrence position for own brand in sources list.
* `competitor_mentioned_in_sources_<model>` – Boolean: any competitor brand in sources.
* `competitor_position_in_sources_<model>` – 1-based first occurrence position for competitor brand in sources list.

### 4. Cross‑model aggregated summary columns
* `brand_mentioned_in_answer_count` – Number of models where own brand appeared in answer text.
* `brand_mentioned_in_sources_count` – Number of models where own brand appeared in sources.
* `competitor_mentioned_in_answer_count` – Number of models mentioning any competitor in answer text.
* `competitor_mentioned_in_sources_count` – Number of models citing a competitor in sources.

### Notes
* Provider prefixes (`openai/`, `google/`, etc.) are stripped from column names.
* Non‑alphanumeric characters in model identifiers are normalised to underscores.
* If `use_search` is false, all `sources_*` and derived source‑based columns are omitted.
* If no brands were provided, brand ranking & summary columns (sections 3–4) are omitted.

---

## Brand ranking logic

Brand ranking is derived using simple positional heuristics:

- Text rank: first character index where each brand token appears (case-insensitive whole-word), sorted by earliest occurrence.
- Reference URL position: first index of a cited URL whose host contains a brand token (whole-word match on the URL field). Brands without a match are omitted (unless future modes include them as `None`).

This provides a lightweight relative prominence signal across models.

---

## Example minimal input

```json
{
  "brands": ["https://www.mapfre.es"],
  "sector": "insurance",
  "market": "Spain"
}
```

This will: (1) normalise your own brand list, (2) attempt competitor discovery (up to default `competitors_max=10`), (3) generate additional prompts (up to default `prompts_max`) using sector + market context, and (4) run evaluation with default models.

## Example with explicit prompts, intents, and custom models

```json
{
  "brands": ["mapfre"],
  "competitors": ["allianz", "generali"],
  "competitors_max": 12,
  "prompts": [
    "best small business liability insurance spain",
    "compare freelancer health policies"
  ],
  "prompt_intents": ["commercial", "informational", "navigational"],
  "prompts_max": 40,
  "prompt_language": "English",
  "brands_in_prompt": "sometimes",
  "models": ["openai/gpt-5", "google/gemini-2.5-flash"],
  "sector": "insurance",
  "market": "Spain",
  "use_search": true
}
```

## Example focusing on a local market with automatic competitor discovery only

```json
{
  "brands": ["mybrand"],
  "sector": "pet insurance",
  "market": "Germany",
  "prompts_max": 25,
  "brands_in_prompt": "never",
  "use_search": true,
  "search_country": de
}
```

In this case competitors are not seeded; the system discovers them (search + LLM) until `competitors_max` is reached.

---

## Failure modes & tips

- Missing required brands: `brands` must contain at least one entry (name or URL).
- Limited prompt generation: if neither `sector` nor any (own + competitor) brands give context, only provided `prompts` are used.
- Few or no competitors discovered: improve context with both `sector` and `market`, or seed some `competitors`.
- Search disabled: reference / ranking columns (`references_*`, *_in_txt_*, *_in_refs_* etc.) are omitted; enable `use_search` for richer comparative analysis.
