# LLM Monitor (Search & Brand Analysis)

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

Taken from `./.actor/input_schema.json` (see there for full metadata):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| models | array[string] | (internal defaults) | LLM models to evaluate. If empty: `openai/gpt-4.1`, `google/gemini-2.5-flash`. |
| prompts_seed | array[string] | - | Seed prompts to include directly. |
| prompts_max | integer | 10 | Max total prompts after generation. |
| prompts_language | string | English | Language for generated prompts. |
| brands_seed | array[string] | - | Seed brand names or homepage URLs. |
| brands_max | integer | 10 | Max competitor brands to discover. |
| brands_model | string | openai/gpt-4.1 | Model for competitor discovery. |
| brands_in_prompt | enum | never | Whether generated prompts include brands (never / sometimes / always). |
| sector | string | insurance | Sector context for generation & discovery. |
| market | string | Spain | Geographic market context. |
| use_search | boolean | true | Enable live search augmentation + ranking. |

Important: You must provide at least one of `prompts_seed`, `sector`, or `brands_seed`, otherwise no prompts can be generated and the run fails.

---

## Output

Dataset items (one per prompt) in a wide, merged structure. Core columns:

- `prompt` – The original prompt text.
- `text_<model>` – Raw answer text from each model.
- `references_<model>` – List (array) of cited references (URL + metadata) if search enabled.
- `text_<model>_ranks` – Ordered list of brand tokens by first occurrence rank in answer text (only if search enabled AND brands known).
- `references_<model>_url_pos` – List of tuples `(brand, position)` representing first URL mention position among references (only if search enabled AND brands known).

If `use_search` is false, all `references_...` columns are removed before output.

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
  "sector": "insurance",
  "market": "Spain",
  "prompts_max": 25,
  "brands_seed": ["https://www.mapfre.es"],
  "brands_in_prompt": "sometimes",
  "use_search": true
}
```

## Example with explicit prompts and multiple models

```json
{
  "prompts_seed": [
    "best small business liability insurance spain",
    "compare freelancer health policies"
  ],
  "models": ["openai/gpt-4.1", "google/gemini-2.5-flash"],
  "brands_seed": ["https://www.mapfre.es", "allianz"],
  "brands_max": 8,
  "prompts_max": 40,
  "brands_in_prompt": "never",
  "use_search": true
}
```

---

## Failure modes & tips

- No prompts generated: ensure at least one of `prompts_seed`, `sector`, or `brands_seed` is supplied.
- Empty competitor list: provide either `brands_seed` or `sector`; otherwise competitor discovery is skipped.
- Large prompt counts: generation requires brand or sector context; if missing, only seeds are used.
- Search disabled: brand ranking columns are skipped; enable `use_search` for richer analysis.
