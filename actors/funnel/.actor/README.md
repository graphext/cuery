# Funnel Keyword Generator

Generate keyword ideas across all marketing funnel stages using LLM-powered seed generation and Google Keyword Planner integration.

This actor automates funnel keyword research by:

1. Generating seed keywords for each funnel stage and category using LLMs
2. Expanding those seeds into comprehensive keyword lists via Google Keyword Planner
3. Returning structured keyword data mapped to specific funnel stages

---

## When to use it

Use this actor to:

- Build complete funnel keyword strategies for SEO and content planning
- Map customer journey touchpoints to specific keyword groups
- Identify keyword opportunities across awareness, consideration, decision, conversion, and retention stages
- Generate contextually relevant keywords for specific brands, sectors, and markets
- Scale keyword research across multiple funnel categories automatically

Not a good fit when you need: manual keyword curation, paid advertising keyword data, or real-time search volume trends (those require different tools).

---

## Input schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| brand (required) | array[string] or string | (none) | Brand or list of brands to contextualize the funnel. |
| sector (required) | string | (none) | Sector to contextualize the funnel (e.g., 'running shoes', 'CRM software'). |
| language (required) | string | (none) | Language for keyword generation as 2-letter ISO code (e.g., 'en', 'es'). |
| country | string | null | Country to contextualize the funnel as 2-letter ISO code (e.g., 'us', 'gb'). Leave empty for global. |
| max_ideas_per_category | integer | 200 | Maximum number of keyword ideas to generate per funnel category (10-1000). |
| stages | array[string] | null | Optional list of specific funnel stage names to process. If empty, all stages are included. See available stages below. |

Notes:
1. Brand, sector, and language are required for meaningful keyword generation.
2. Higher `max_ideas_per_category` values provide more comprehensive results but take longer to process.
3. The `stages` parameter allows you to target specific funnel stages (e.g., only "Awareness / Discovery" and "Consideration / Research") for faster, more focused results.
4. The actor uses live Google Keyword Planner data via API (requires proper credentials).

---

## Output

Dataset items (one row per keyword) with the following fields:

### Funnel Context
* `funnel_stage` – Marketing funnel stage (e.g., "Awareness / Discovery", "Consideration / Research")
* `funnel_category` – Specific category within the stage (e.g., "Problem Identification", "Comparisons")
* `funnel_seed_keyword` – Original seed keyword that generated this keyword idea

### Keyword Details
* `keyword` – The actual keyword phrase
* `search_volume` – Average monthly search volume (from Google Keyword Planner)
* `competition` – Competition level (Low/Medium/High)
* `cpc` – Cost per click (if available)
* Additional Google Keyword Planner metrics as applicable

---

## Marketing Funnel Stages

The actor generates keywords across 5 funnel stages:

### 1. Awareness / Discovery
Goal: Problem recognition, education
- Problem Identification
- Category Education
- Trends & Inspiration

### 2. Consideration / Research
Goal: Compare options, evaluate solutions
- Features & Specifications
- Comparisons
- Suitability & Use Cases
- Social Proof & Reviews

### 3. Decision / Evaluation
Goal: Close to acting, still evaluating options
- Pricing & Packages
- Availability & Location
- Intent-to-Act Signals

### 4. Conversion / Action
Goal: Complete purchase or desired action
- Purchase / Signup
- Immediate Offers & Promotions

### 5. Post-Purchase / Retention & Advocacy
Goal: Support customers, encourage loyalty
- Usage & How-To
- Troubleshooting & Support
- Upgrades & Add-ons
- Community & Advocacy

---

## Example Use Cases

### 1. Full Funnel SEO Strategy
```json
{
  "brand": "Nike",
  "sector": "running shoes",
  "language": "en",
  "country": "us",
  "max_ideas_per_category": 200
}
```
Generates comprehensive keyword strategy across all funnel stages for Nike running shoes in the US market.

### 2. SaaS Content Planning
```json
{
  "brand": ["HubSpot", "Salesforce"],
  "sector": "CRM software",
  "language": "en",
  "max_ideas_per_category": 150
}
```
Maps keyword opportunities for CRM software considering multiple competitor contexts.

### 3. International Market Research
```json
{
  "brand": "Zara",
  "sector": "fast fashion",
  "language": "es",
  "country": "es",
  "max_ideas_per_category": 100
}
```
Generates Spanish-language keywords for the Spanish fashion market.

### 4. Quick Funnel Analysis
```json
{
  "brand": "Tesla",
  "sector": "electric cars",
  "language": "en",
  "max_ideas_per_category": 50
}
```
Faster execution with fewer keywords per category for rapid funnel analysis.

### 5. Targeted Stage Research
```json
{
  "brand": "Shopify",
  "sector": "e-commerce platform",
  "language": "en",
  "country": "us",
  "stages": ["Awareness / Discovery", "Consideration / Research"],
  "max_ideas_per_category": 200
}
```
Focus keyword generation on top-of-funnel stages only for content marketing strategy.
