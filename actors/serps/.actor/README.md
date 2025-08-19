# Google SERP Data Collection Actor

Collect and analyze Search Engine Results Page (SERP) data using Apify's Google Search Scraper with AI-powered topic extraction and search intent classification. Features comprehensive SERP analysis including organic results, paid results, AI overviews, and intelligent brand/competitor tracking for strategic SEO insights.

## What does Google SERP Data Collection Actor do?

This Actor connects to Apify's Google Search Scraper to provide comprehensive SERP analysis for SEO professionals:

- ✅ **Fetch organic search results** with titles, descriptions, URLs, and metadata for each keyword
- ✅ **Collect AI overviews** from Google's AI-powered search results with entity extraction
- ✅ **Track paid results** including ads and shopping results for competitive analysis
- ✅ **Analyze brand presence** by identifying brand mentions and rankings in SERPs
- ✅ **Monitor competitors** with automated competitor tracking and ranking analysis
- ✅ **Extract topics and intent** using AI models to classify search intent and extract semantic topics
- ✅ **Aggregate SERP features** including People Also Ask, related queries, and SERP features
- ✅ **Support multiple markets** with configurable language and country targeting
- ✅ **Generate structured data** ready for SEO analysis, competitive intelligence, and content strategy

**Perfect for**: SEO professionals, digital marketers, competitive analysts, content strategists, and businesses tracking their search presence and competitor performance.

## Input

Configure your SERP analysis with these comprehensive parameters:

### Example Input

By default, automatically includes AI-powered, topic and intent detection as well as entity extraction from AI overview.

```json
{
  "keywords": ["digital marketing", "seo tools", "content marketing"],
  "resultsPerPage": 100,
  "country": "us",
  "searchLanguage": "en",
  "top_n": 10,
  "brands": ["Ahrefs", "SEMrush", "Moz"],
  "competitors": ["HubSpot", "Screaming Frog", "Majestic"]
}
```

### Example without AI-Powered Analysis

```json
{
  "keywords": ["e-commerce platform", "online store builder", "shopify alternatives"],
  "resultsPerPage": 50,
  "country": "us",
  "searchLanguage": "en",
  "top_n": 15,
  "brands": ["Shopify", "WooCommerce"],
  "competitors": ["BigCommerce", "Squarespace", "Wix"],
  "topic_model": null,
  "entity_model": null,
}
```


### Input Parameters

| Field | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| `keywords` | Array | Keywords to fetch SERP data for | Optional* | - |
| `batch_size` | Integer | Number of keywords to process in a single batch | Optional | `100` |
| `resultsPerPage` | Integer | Number of search results to fetch per page | Optional | `100` |
| `maxPagesPerQuery` | Integer | Maximum number of pages to fetch per query | Optional | `1` |
| `countryCode` | String | Country code for SERP targeting (e.g., "us", "uk", "de") | Optional | - |
| `searchLanguage` | String | Search language (e.g., "en", "es", "fr") | Optional | - |
| `languageCode` | String | Language code for results (e.g., "en", "es") | Optional | - |
| `top_n` | Integer | Number of top organic results to analyze | Optional | `10` |
| `brands` | Array | Brand names to track in SERP results | Optional | - |
| `competitors` | Array | Competitor names to track in SERP results | Optional | - |
| `topic_max_samples` | Integer | Max samples for AI topic extraction | Optional | `500` |

*Keywords can be passed manually in calling functions if not provided in input.

### Quick Reference

**Most Common Country Codes:**
- `"us"` - United States
- `"uk"` - United Kingdom
- `"ca"` - Canada
- `"au"` - Australia
- `"de"` - Germany
- `"fr"` - France
- `"es"` - Spain
- `"it"` - Italy
- `"br"` - Brazil
- `"mx"` - Mexico
- `"jp"` - Japan

**Most Common Language Codes:**
- `"en"` - English
- `"es"` - Spanish  
- `"fr"` - French
- `"de"` - German
- `"it"` - Italian
- `"pt"` - Portuguese
- `"ja"` - Japanese
- `"zh"` - Chinese

**Brand and Competitor Tracking:**
- Provide exact brand names as they appear in search results
- System automatically calculates rankings in titles, descriptions, and domains
- Tracks first occurrence and specific rankings for each brand/competitor
- Works across organic results and AI overviews

**Batch Processing:**
- Use `batch_size` to control API rate limits and processing speed
- Larger batches (100-200) are more efficient but may hit rate limits
- Smaller batches (10-50) provide more granular control and error recovery

## Output

The Actor generates comprehensive SERP analysis data with detailed metrics for each keyword.

### Sample Output

```json
{
  "term": "digital marketing",
  "n_paidResults": 4,
  "n_paidProducts": 12,
  "relatedQueries": [
    "digital marketing strategy",
    "digital marketing course",
    "digital marketing jobs"
  ],
  "peopleAlsoAsk": [
    "What is digital marketing?",
    "How to start digital marketing?",
    "What are the types of digital marketing?"
  ],
  "aiOverview_content": "Digital marketing encompasses all marketing efforts that use electronic devices...",
  "aiOverview_source_titles": [
    "HubSpot Digital Marketing Guide",
    "Google Digital Marketing Courses"
  ],
  "num_results": 100,
  "num_has_date": 45,
  "num_has_views": 12,
  "titles": [
    "Digital Marketing Guide - Complete Beginner's Guide",
    "What is Digital Marketing? Types, Skills & Best Practices"
  ],
  "descriptions": [
    "Learn digital marketing fundamentals including SEO, social media, email marketing...",
    "Digital marketing uses digital channels to promote products and services..."
  ],
  "domains": [
    "hubspot.com",
    "semrush.com",
    "ahrefs.com"
  ],
  "emphasizedKeywords": [
    "digital marketing",
    "online marketing",
    "digital strategy"
  ],
  "title_rank_brand": 3,
  "domain_rank_brand": 1,
  "description_rank_brand": 2,
  "title_rank_competition": 5,
  "min_rank_HubSpot": 1,
  "min_rank_SEMrush": 2,
  "topic": "Digital Marketing Education",
  "subtopic": "Marketing Fundamentals",
  "intent": "Informational",
  "ai_overview_brand/company": [
    "HubSpot",
    "Google"
  ],
  "ai_overview_product/service": [
    "Google Ads",
    "Facebook Ads"
  ],
  "aiOverview_brand_mentions": [
    "HubSpot"
  ],
  "aiOverview_competitor_mentions": [
    "Mailchimp"
  ]
}
```

### Output Fields Explained

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `term` | String | The search keyword analyzed | Query identification |
| `n_paidResults` | Number | Count of paid ads in SERP | Ad competition level |
| `n_paidProducts` | Number | Count of shopping ads | E-commerce competition |
| `relatedQueries` | Array | Google's related search suggestions | Keyword expansion |
| `peopleAlsoAsk` | Array | Questions from "People Also Ask" section | Content ideas |
| `aiOverview_content` | String | AI overview text content | AI presence analysis |
| `aiOverview_source_titles` | Array | Sources cited in AI overview | Authority tracking |
| `num_results` | Number | Total organic results found | Search volume indicator |
| `num_has_date` | Number | Results with publication dates | Content freshness |
| `titles` | Array | Titles of top organic results | Content analysis |
| `descriptions` | Array | Meta descriptions of top results | SERP snippet analysis |
| `domains` | Array | Unique domains in top results | Domain diversity |
| `emphasizedKeywords` | Array | Keywords highlighted by Google | Relevance signals |
| `title_rank_brand` | Number | Brand's first appearance in titles | Brand visibility |
| `domain_rank_brand` | Number | Brand's first appearance in domains | Domain authority |
| `title_rank_competition` | Number | First competitor appearance | Competitive landscape |
| `min_rank_[Competitor]` | Number | Specific competitor's best ranking | Individual competitor tracking |
| `topic` | String | AI-extracted topic category | Content categorization |
| `subtopic` | String | AI-extracted subtopic | Detailed classification |
| `intent` | String | Search intent classification | User behavior insights |
| `ai_overview_brand/company` | Array | Companies mentioned in AI overview | Brand presence in AI |
| `aiOverview_brand_mentions` | Array | Your brands mentioned in AI overview | Brand AI visibility |
| `aiOverview_competitor_mentions` | Array | Competitors mentioned in AI overview | Competitive AI analysis |

### Real-World Examples

**🎯 Brand Monitoring and Competitive Analysis**
*Scenario: Track brand presence across key industry keywords*
```json
{
  "keywords": ["project management software", "team collaboration tools", "task management app"],
  "resultsPerPage": 50,
  "country": "us",
  "searchLanguage": "en",
  "top_n": 15,
  "brands": ["Asana", "Monday.com"],
  "competitors": ["Trello", "Notion", "ClickUp", "Jira"],
  "topic_model": "google/gemini-2.5-flash-preview-05-20",
  "assignment_model": "openai/gpt-4.1-mini"
}
```
*Expected output: Brand rankings, competitor positions, and topic classification for strategic positioning*

**🌍 Multi-Market SERP Analysis**  
*Scenario: Analyze search results across different countries for global SEO*
```json
{
  "keywords": ["ecommerce platform", "online shop builder", "create online store"],
  "resultsPerPage": 100,
  "country": "de",
  "searchLanguage": "de",
  "languageCode": "de",
  "top_n": 10,
  "brands": ["Shopify", "WooCommerce"],
  "competitors": ["Magento", "BigCommerce", "PrestaShop"]
}
```
*Expected output: German market SERP analysis with localized competitor intelligence*

**🤖 AI Overview and Entity Analysis**
*Scenario: Track how AI overviews mention your brand vs competitors*
```json
{
  "keywords": ["best crm software", "customer relationship management", "sales automation tools"],
  "resultsPerPage": 50,
  "country": "us",
  "searchLanguage": "en",
  "top_n": 10,
  "brands": ["Salesforce", "HubSpot"],
  "competitors": ["Pipedrive", "Zoho", "Freshsales"],
  "entity_model": "openai/gpt-4.1-mini",
  "topic_max_samples": 200
}
```
*Expected output: AI overview entity extraction and brand mention analysis*

**📱 Local SEO and Voice Search**
*Scenario: Analyze local search results for location-based queries*
```json
{
  "keywords": ["dentist near me", "best restaurant downtown", "plumber emergency service"],
  "resultsPerPage": 30,
  "country": "us",
  "searchLanguage": "en",
  "top_n": 10,
  "topic_model": "google/gemini-2.5-flash-preview-05-20",
  "assignment_model": "openai/gpt-4.1-mini"
}
```
*Expected output: Local SERP features analysis with intent classification for local SEO optimization*

### Tips for Best Results

- **Use targeted keywords**: Focus on keywords relevant to your industry and business goals
- **Enable brand tracking**: Add your brands and main competitors for comprehensive competitive analysis
- **Configure geographic targeting**: Set appropriate country and language for your target market
- **Optimize batch size**: Use 50-100 keywords per batch for efficient processing
- **Enable AI analysis**: Use topic and intent extraction for deeper SERP insights
- **Monitor AI overviews**: Track entity mentions in Google's AI-powered results
- **Analyze SERP features**: Use People Also Ask and related queries for content ideas
- **Track competitor rankings**: Monitor specific competitor positions across different SERP elements
- **Consider multiple markets**: Run separate analyses for different countries/languages
- **Use appropriate result depth**: Increase `top_n` for more comprehensive competitive analysis

