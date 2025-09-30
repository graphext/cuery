# Brand Researcher

Discover competitors and fetch detailed brand information using LLM-powered search and analysis.

This actor automates competitive brand research by:

1. Identifying competitors for one or more seed brands using LLM + live search
2. Fetching detailed information for each brand including portfolios, market positions, and domains
3. Returning structured data about all discovered brands

---

## When to use it

Use this actor to:

- Quickly identify competitors in a specific market or sector
- Build comprehensive brand databases with detailed information
- Research brand portfolios and product offerings
- Analyze market positioning across multiple brands
- Gather official domains and brand assets (favicons) for further analysis

Not a good fit when you need: real-time SERP data, social media analysis, or financial metrics (those require specialized tools).

---

## Input schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| brand (required) | array[string] | (none) | Initial brand(s) to analyze and find competitors for. Can be names or URLs. |
| sector | string | - | Sector/industry (e.g., 'electric cars', 'insurance'). Helps identify relevant competitors. |
| market | string | - | Geographical market (e.g., 'Spain', 'global'). Focuses competitor discovery on relevant markets. |
| strict_competitors | boolean | true | If true, excludes competitors from the same parent company. |
| instructions | string | - | Optional additional guidance for competitor search (e.g., "focus on premium segment"). |

Notes:
1. At least one brand is required. Providing sector and/or market significantly improves competitor discovery quality.
2. Brand entries can be names (e.g., "Tesla") or URLs (e.g., "https://tesla.com") - they will be normalized.
3. The actor uses live web search to ensure up-to-date information.

---

## Output

Dataset items (one row per brand) with the following fields:

### Brand Identity
* `name` – Official name of the brand
* `short_name` – Common/canonical short name if different from official name (e.g., "Tesla" vs "Tesla, Inc.")
* `description` – Brief description of the brand and its main activities

### Brand Details
* `domain` – Official website domain (normalized to just the domain)
* `portfolio` – Array of products/services with `name` and `category` fields for each
* `market_position` – One of: "leader", "challenger", "niche", or "follower"
* `favicon` – URL of the brand's favicon (if available)

### Portfolio Structure
Each item in the `portfolio` array contains:
* `name` – Specific product or service name (e.g., "Model 3", not "electric cars")
* `category` – Product category (e.g., "sedan", "SUV")

---

## Example Use Cases

### 1. Competitive Intelligence
```json
{
  "brand": ["Mapfre"],
  "sector": "insurance",
  "market": "Spain",
  "strict_competitors": true
}
```
Discovers Spanish insurance competitors and their product portfolios.

### 2. Market Landscape Analysis
```json
{
  "brand": ["Tesla", "Rivian"],
  "sector": "electric vehicles",
  "market": "United States",
  "strict_competitors": false
}
```
Maps the entire EV market including subsidiary brands.

### 3. Focused Segment Research
```json
{
  "brand": ["Rolex"],
  "sector": "luxury watches",
  "market": "global",
  "instructions": "Focus on brands in the ultra-premium segment above $10,000 average price point"
}
```
Finds competitors in a specific market niche with custom criteria.

---

## How It Works

1. **Competitor Discovery**: Uses LLM with live search to identify relevant competitor brands based on your seed brand(s), sector, and market context.

2. **Information Gathering**: For each brand (seeds + competitors), performs targeted searches to gather:
   - Official brand information
   - Product/service portfolios
   - Market positioning
   - Web presence (domain, favicon)

3. **Data Structuring**: Normalizes and structures all information into consistent JSON records with validation.

The actor runs all brand information fetches concurrently with automatic retries for reliability.

---

## Tips

- **Provide context**: Including sector and market significantly improves competitor discovery accuracy
- **Use instructions**: Add specific criteria to narrow or broaden the competitor set
- **Strict mode**: Enable `strict_competitors` to exclude subsidiary brands of the same parent company
- **Batch processing**: You can provide multiple seed brands to analyze entire market segments at once

---

## API Keys Required

This actor requires the following API keys (configured as Apify secrets):
- `OPENAI_API_KEY` – For GPT models (primary search model)
- `GOOGLE_API_KEY` – For Gemini models (optional)
- `XAI_API_KEY` – For Grok models (optional)
- `DEEPSEEK_API_KEY` – For DeepSeek models (optional)
- `HASDATA_API_KEY` – For additional search capabilities (optional)

Only the OpenAI API key is strictly required for basic functionality.
