# SEO Keywords Analysis Actor

Analyze and expand keywords using the Google Ads API to generate comprehensive SEO keyword data. Features intelligent keyword expansion, SERP analysis, brand tracking, traffic data collection, and AI-powered insights for content planning and competitive analysis with detailed search volume and competition metrics.

## What does SEO Keywords Analysis Actor do?

This Actor connects to the Google Ads API to provide comprehensive keyword analysis for SEO professionals:

- ✅ **Expand seed keywords** into thousands of related keyword suggestions
- ✅ **Generate keyword ideas** with configurable limits for comprehensive research
- ✅ **Analyze search volumes** with up to 12 months of historical data  
- ✅ **Extract competition metrics** including average CPC and competition scores
- ✅ **SERP data analysis** with brand and competitor tracking
- ✅ **Traffic data collection** using Similarweb scraper for competitive insights
- ✅ **AI-powered insights** with topic extraction and intent classification
- ✅ **Support multiple markets** with configurable language and geographic targeting
- ✅ **Generate structured data** ready for SEO analysis, content planning, and reporting

**Perfect for**: SEO professionals, content marketers, digital agencies, PPC specialists, and businesses planning their content strategy or competitive analysis.

## Why SEO Keywords Analysis Actor?

- **Comprehensive data**: Get both current and historical search volume data
- **Keyword idea generation**: Automatically expand seed keywords into comprehensive lists
- **Multi-market support**: Analyze keywords across different countries and languages
- **Competition insights**: Understand keyword difficulty and cost-per-click data
- **SERP analysis**: Track brand visibility and competitor positioning in search results
- **AI-powered insights**: Advanced topic extraction and intent classification
- **Ready-to-use**: No API setup required - credentials are pre-configured
- **Export flexibility**: Download data in JSON, CSV, or XML formats
- **Scalable**: Analyze from a few keywords to thousands in a single run

## How much will it cost to run the Actor?

This Actor is **cost-effective** and transparent in its pricing:

| Component | Cost | Details |
|-----------|------|---------|
| **Google Ads API** | Free | Requires approved Developer Token (pre-configured) |
| **Apify platform** | ~$0.01-0.05 per 1000 keywords | Based on compute units consumed |
| **Typical run** | $0.50-2.00 | For 10,000-50,000 keyword analysis |

**Cost factors:**
- Number of seed keywords provided
- Historical data range requested (up to 12 months)
- Geographic and language targeting complexity
- Number of related keywords generated

**Example costs:**
- **Small project** (100 keywords): ~$0.10
- **Medium project** (5,000 keywords): ~$0.50
- **Large project** (50,000 keywords): ~$2.00

## Input

Configure your keyword analysis with these simple parameters:

### Example Input

```json
{
  "keywords": ["digital marketing", "seo", "web positioning"],
  "ideas": false,
  "language": "en",
  "geo_target": "us",
  "metrics_start": "2024-07",
  "metrics_end": "2025-07"
}
```

### Example with Keyword Ideas Generation

```json
{
  "keywords": ["digital marketing", "seo", "web positioning"],
  "ideas": true,
  "max_ideas": 500,
  "customer": "your-customer-id",
  "language": "en",
  "geo_target": "us",
  "metrics_start": "2024-01",
  "metrics_end": "2024-12"
}
```

### Example with SERP Data Analysis

```json
{
  "keywords": ["digital marketing", "seo", "web positioning"],
  "ideas": true,
  "max_ideas": 200,
  "language": "en",
  "geo_target": "us",
  "country": "us",
  "searchLanguage": "en",
  "languageCode": "en",
  "brands": ["your-brand", "known-brand"],
  "competitors": ["competitor1", "competitor2"],
  "topic_max_samples": 500,
  "metrics_start": "2024-07",
  "metrics_end": "2025-07"
}
```

### Example with Traffic Data Collection

```json
{
  "keywords": ["digital marketing", "seo", "web positioning"],
  "ideas": true,
  "max_ideas": 150,
  "language": "en",
  "geo_target": "us",
  "fetch_traffic": true,
  "country": "us",
  "searchLanguage": "en",
  "brands": ["your-brand"],
  "competitors": ["competitor1", "competitor2"],
  "metrics_start": "2024-07",
  "metrics_end": "2025-07"
}
```

### Input Parameters

| Field | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| `keywords` | Array | List of seed keywords to expand and analyze | ✅ **Required** | - |
| `ideas` | Boolean | Generate additional keyword ideas (only available for ≤20 keywords) | ❌ Optional | `false` |
| `max_ideas` | Integer | Maximum keyword ideas to generate (only if `ideas` enabled and ≤20 keywords) | ❌ Optional | - |
| `customer` | String | Google Ads Customer ID for API access | ❌ Optional | Uses environment variable |
| `language` | String | Language code for targeting (e.g., "en", "es", "fr") | ❌ Optional | - |
| `geo_target` | String | Geographic target code (e.g., "us", "es", "uk") | ❌ Optional | - |
| `metrics_start` | String | Start date for historical metrics (YYYY-MM format) | ❌ Optional | - |
| `metrics_end` | String | End date for historical metrics (YYYY-MM format) | ❌ Optional | - |
| `batch_size` | Integer | Number of keywords to process in each batch | ❌ Optional | `100` |
| `resultsPerPage` | Integer | Number of SERP results to fetch per page | ❌ Optional | `10` |
| `maxPagesPerQuery` | Integer | Maximum number of SERP pages to fetch per query | ❌ Optional | `1` |
| `country` | String | Country code for SERP data (e.g., "us", "uk") | ❌ Optional | - |
| `searchLanguage` | String | Search language for SERP data (e.g., "en", "es") | ❌ Optional | - |
| `languageCode` | String | Language code for SERP data processing | ❌ Optional | - |
| `params` | Object | Additional parameters for Google Search Scraper | ❌ Optional | `{}` |
| `brands` | Array | List of brand names to identify in SERP data | ❌ Optional | `[]` |
| `competitors` | Array | List of competitor names to identify in SERP data | ❌ Optional | `[]` |
| `topic_max_samples` | Integer | Maximum samples for topic extraction from SERP data | ❌ Optional | `500` |
| `topic_model` | String | AI model for topic extraction | ❌ Optional | `"google/gemini-2.5-flash-preview-05-20"` |
| `assignment_model` | String | AI model for intent classification | ❌ Optional | `"openai/gpt-4.1-mini"` |
| `entity_model` | String | AI model for entity extraction from AI overviews | ❌ Optional | `"openai/gpt-4.1-mini"` |
| `fetch_traffic` | Boolean | Fetch traffic data for keywords using Similarweb scraper | ❌ Optional | `false` |

### Quick Reference

**Most Common Language Codes:**
- `"en"` - English
- `"es"` - Spanish  
- `"fr"` - French
- `"de"` - German
- `"it"` - Italian
- `"pt"` - Portuguese
- `"ja"` - Japanese
- `"zh"` - Chinese

**Most Common Geographic Codes:**
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

**Historical Metrics Date Range:**
- `metrics_start` and `metrics_end` control the historical data period
- **Format**: YYYY-MM (e.g., "2024-07" for July 2024)
- **Limitations**: 
  - Maximum 2-year range
  - Cannot be more than 2 years in the past
  - End date cannot be in the future
- **Examples**:
  - Last 12 months: `"2024-07"` to `"2025-07"`
  - Calendar year: `"2024-01"` to `"2024-12"`
  - Recent 6 months: `"2024-12"` to `"2025-06"`

**Keyword Ideas Generation:**
- Set `ideas: true` to generate additional keyword suggestions
- **Important**: Only available when providing 20 or fewer seed keywords
- Use `max_ideas` to limit the number of generated keywords (recommended: 100-1000)
- If more than 20 keywords are provided, idea generation will be automatically disabled with a warning
- **Note**: Enabling ideas generation increases processing time but provides more comprehensive data

**SERP Data Analysis:**
- Configure `country`, `searchLanguage`, and `languageCode` for SERP analysis
- Add `brands` and `competitors` arrays to identify them in search results
- Adjust `resultsPerPage` and `maxPagesPerQuery` to control SERP data volume
- **AI Models**: Configure topic extraction and intent classification models for advanced analysis

### Customer ID Configuration

The `customer` parameter is **optional** and works as follows:

- **When provided**: Uses your specified Google Ads Customer ID to access keyword data from your Google Ads account
- **When omitted**: Automatically uses Graphext's internal customer ID, which provides access to Google Ads API data without requiring your own Google Ads account

💡 **Tip**: Most users can leave the customer field empty and use Graphext's internal access for comprehensive keyword data.

## Output

The Actor generates a comprehensive dataset with detailed keyword metrics for SEO analysis.

### Sample Output

```json
{
  "keyword": "marketing digital",
  "avg_monthly_searches": 14800,
  "competition": 3,
  "competition_index": 50,
  "low_top_of_page_bid_micros": 1964646,
  "high_top_of_page_bid_micros": 7035752,
  "search_volume": [12100, 12100, 18100, 18100, 14800, 12100, 18100, 18100, 18100, 14800, 14800],
  "search_volume_date": ["2024-07-01T00:00:00", "2024-08-01T00:00:00", "2024-09-01T00:00:00"],
  "search_volume_growth_3m": -18.23,
  "search_volume_trend": 0.016,
  "topic": "Digital Marketing",
  "subtopic": "Digital Marketing Fundamentals", 
  "intent": "informational",
  "globalRank_min": 8.0,
  "globalRank_max": 161101.0,
  "visits_min": 275036,
  "visits_max": 3803816246,
  "timeOnSite_min": 65.64,
  "timeOnSite_max": 497.60,
  "pagesPerVisit_min": 1.77,
  "pagesPerVisit_max": 10.01,
  "bounceRate_min": 0.24,
  "bounceRate_max": 0.69,
  "source_direct_min": 0.13,
  "source_direct_max": 0.62,
  "source_search_min": 0.08,
  "source_search_max": 0.81,
  "source_social_min": 0.002,
  "source_social_max": 0.020,
  "source_referrals_min": 0.010,
  "source_referrals_max": 0.381
}
```

### Output Fields Explained

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `keyword` | String | The keyword or phrase | Content targeting |
| `avg_monthly_searches` | Number | Average monthly search volume | Traffic potential estimation |
| `average_cpc` | Number | Average cost per click (local currency) | PPC budget planning |
| `competition_score` | Number | Competition level (1=Low, 2=Medium, 3=High) | Keyword difficulty assessment |
| `competition` | Number | Numeric competition score (33, 66, 100) | Detailed competition analysis |
| `search_volume_YYYY_MM` | Number | Historical monthly search volumes | Seasonal trend analysis |
| `globalRank_min/max` | Number | Global website ranking range | Site authority assessment |
| `visits_min/max` | Number | Monthly visits range | Traffic volume analysis |
| `timeOnSite_min/max` | Number | Time on site range (seconds) | User engagement analysis |
| `pagesPerVisit_min/max` | Number | Pages per visit range | Site stickiness metrics |
| `bounceRate_min/max` | Number | Bounce rate range (0-1) | Content quality indicator |
| `source_direct_min/max` | Number | Direct traffic percentage range | Brand awareness metrics |
| `source_search_min/max` | Number | Search traffic percentage range | SEO performance indicator |
| `source_social_min/max` | Number | Social traffic percentage range | Social media impact |
| `source_referrals_min/max` | Number | Referral traffic percentage range | Link building success |

### Export & Integration

**Available Formats:**
- **JSON** - Perfect for APIs and automated workflows
- **CSV** - Excel-compatible for analysis and reporting  
- **XML** - System integrations and enterprise workflows
- **RSS** - Feed-based integrations

**Dataset Features:**
- Up to 12 months of historical data per keyword
- Real-time search volume and competition metrics
- Website traffic and user behavior insights from Similarweb
- Competitive traffic analysis with min/max ranges
- Structured for immediate analysis and visualization
- Compatible with popular SEO and analytics tools

## Getting Started

### How to run SEO Keywords Analysis Actor

1. **📝 Enter your keywords**: Add your seed keywords in the input field
2. **🔧 Configure settings** (optional): 
   - **Customer ID**: Leave empty to use Graphext's internal customer, or provide your own Google Ads Customer ID
   - **Language & Geography**: Set your target language and geographic market
3. **▶️ Start the Actor**: Click "Start" and let it analyze your keywords
4. **📊 Download results**: Export your data in JSON, CSV, or XML format

### Real-World Examples

**🎯 SEO Content Strategy with Ideas Generation**
*Scenario: Blog content planning for a marketing agency*
```json
{
  "keywords": ["content marketing", "blog writing", "seo copywriting", "digital strategy"],
  "ideas": true,
  "max_ideas": 300,
  "language": "en",
  "geo_target": "us"
}
```
*Expected output: ~2,000-5,000 related keywords with search volumes*

**🏪 Local Business Research with SERP Analysis**  
*Scenario: Spanish restaurant chain expansion*
```json
{
  "keywords": ["restaurante madrid", "comida española", "tapas barcelona"],
  "ideas": true,
  "max_ideas": 200,
  "language": "es",
  "geo_target": "es",
  "country": "es",
  "searchLanguage": "es",
  "competitors": ["competitor-restaurant-1", "competitor-restaurant-2"]
}
```
*Expected output: ~1,500-3,000 location-based keywords with competitor analysis*

**🛒 E-commerce Optimization with Brand Tracking**
*Scenario: Online fashion store in the UK*
```json
{
  "keywords": ["buy shoes online", "fashion trends", "women clothing"],
  "ideas": true,
  "max_ideas": 500,
  "language": "en",
  "geo_target": "uk",
  "country": "uk",
  "searchLanguage": "en",
  "brands": ["your-fashion-brand"],
  "competitors": ["zara", "h&m", "asos"],
  "resultsPerPage": 20,
  "fetch_traffic": true
}
```
*Expected output: ~3,000-8,000 product-related keywords with brand visibility analysis and traffic data*

### Tips for Best Results

- **Use specific seed keywords**: More targeted seeds = better suggestions
- **Enable keyword ideas generation**: Set `ideas: true` for comprehensive keyword expansion (maximum 20 keywords)
- **Optimize idea limits**: Use `max_ideas: 100-1000` for balanced results and processing time
- **Stay within keyword limits**: Use 20 or fewer keywords for idea generation, unlimited for basic analysis
- **Try different language/geo combinations**: Discover market-specific opportunities  
- **Include brand and competitor terms**: Understand the competitive landscape with SERP analysis
- **Mix broad and specific terms**: Get both high-level and long-tail keyword data
- **Configure SERP analysis**: Add brands and competitors to track market positioning
- **Enable traffic data collection**: Set `fetch_traffic: true` for comprehensive competitive insights using Similarweb data
- **Adjust batch sizes**: Use smaller `batch_size` values for large keyword sets to avoid timeouts

## Connect SEO Keywords Analysis Actor to your workflows

### 🔌 Apify API Integration

```bash
curl -X POST https://api.apify.com/v2/acts/[ACTOR_ID]/runs \
  -H "Authorization: Bearer [YOUR_API_TOKEN]" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["digital marketing", "seo"],
    "ideas": true,
    "max_ideas": 200,
    "language": "en",
    "geo_target": "us"
  }'
```

### 🐍 Python SDK Integration

```python
from apify_client import ApifyClient

# Initialize the client
client = ApifyClient("[YOUR_API_TOKEN]")

# Prepare Actor input with keyword ideas generation
run_input = {
    "keywords": ["digital marketing", "seo", "content strategy"],
    "ideas": True,
    "max_ideas": 300,
    "language": "en", 
    "geo_target": "us",
    "country": "us",
    "searchLanguage": "en",
    "brands": ["your-brand"],
    "competitors": ["competitor1", "competitor2"],
    "fetch_traffic": True
}

# Run the Actor and wait for it to finish
run = client.actor("[ACTOR_ID]").call(run_input=run_input)

# Fetch results from the run's dataset
keywords_data = client.dataset(run["defaultDatasetId"]).list_items().items

# Process your keyword data
for keyword in keywords_data:
    print(f"{keyword['keyword']}: {keyword['avg_monthly_searches']} searches/month")
```

### 🔔 Webhook Automation

Automatically process results when the Actor finishes:

```json
{
  "eventTypes": ["ACTOR.RUN.SUCCEEDED"],
  "requestUrl": "https://your-website.com/webhook/keywords",
  "payloadTemplate": {
    "actorRunId": "{{resource.id}}",
    "datasetId": "{{resource.defaultDatasetId}}",
    "status": "{{resource.status}}"
  }
}
```

### 📊 Popular Integrations

- **Google Sheets**: Export to spreadsheets for team collaboration
- **Airtable**: Organize keyword data in structured databases  
- **Slack/Discord**: Get notifications when analysis completes
- **Zapier**: Connect to 5,000+ apps and services
- **Power BI/Tableau**: Create keyword performance dashboards

## Troubleshooting

### ❓ Common Questions

**"No results for my keywords"**
- ✅ Verify keywords are in the target language
- ✅ Try broader or more popular seed keywords
- ✅ Check if the language/geographic combination is valid

**"Keywords format error"**  
- ✅ Ensure keywords are provided as an array: `["keyword1", "keyword2"]`
- ✅ Check that at least one keyword is provided
- ✅ Remove special characters or excessive punctuation

**"Language/Geographic code not recognized"**
- ✅ Use standard ISO codes: `"en"` for English, `"us"` for United States
- ✅ Check that language and geo_target use consistent formatting
- ✅ Refer to the Quick Reference section above for valid codes
- ✅ Try common codes like `"en"` (English) or `"us"` (USA)

**"Ideas generation taking too long"**
- ✅ Reduce `max_ideas` to 100-500 for faster processing
- ✅ Use more specific seed keywords to narrow results
- ✅ Consider disabling ideas generation (`ideas: false`) for quick keyword analysis only

**"Ideas generation not working"**
- ✅ Ensure you have 20 or fewer seed keywords (idea generation is limited to 20 keywords)
- ✅ Check that `ideas` is set to `true` in your input
- ✅ Verify `max_ideas` is set to a reasonable number (100-1000)
- ✅ If you have more than 20 keywords, the system automatically disables idea generation

**"SERP data errors"**
- ✅ Ensure `country` and `searchLanguage` are properly configured
- ✅ Verify that geographic codes match between `geo_target` and `country`
- ✅ Use consistent language codes across all language-related fields

**"Too many results or timeout"**
- ✅ Reduce `batch_size` from default 100 to 50 or lower
- ✅ Decrease `max_ideas` to limit keyword expansion
- ✅ Lower `resultsPerPage` and `maxPagesPerQuery` for SERP analysis
- ✅ Process keywords in smaller batches by splitting your seed keywords

**"Actor run failed or timed out"**
- ✅ Reduce the number of seed keywords (try 10-50 keywords max)
- ✅ Try simpler, more common keywords first
- ✅ Wait a few minutes and try again

**"Should I provide my own Customer ID?"**
- ✅ **Leave empty** for most use cases - Graphext's internal customer provides full access
- ✅ **Provide your own** only if you need data from your specific Google Ads account
- ✅ **Ensure access** if using your own ID, make sure it has Google Ads API access enabled

**"Invalid date range error"**
- ✅ Use YYYY-MM format: `"2024-07"` not `"July 2024"`
- ✅ Ensure start date is before end date
- ✅ Keep within last 2 years (Google Ads API limitation)
- ✅ Don't use future dates for end date
- ✅ Maximum 2-year range between start and end dates

**"Google Ads API date error"**
- ✅ Try more recent dates (within last 12-18 months)
- ✅ Use current month or previous month as end date
- ✅ Check that date format is exactly YYYY-MM
- ✅ Verify month is valid (01-12, not 13 or 00)

### 📞 Need More Help?

1. **Check your input format** using the examples above
2. **Review the Getting Started section** for proper usage
3. **Try the provided example inputs** to test functionality
4. **Contact support** through Apify platform if issues persist

### 🔧 Advanced Troubleshooting

**For power users and developers:**
- Monitor the Actor run logs for detailed error messages
- Verify API quotas and rate limits haven't been exceeded
- Test with minimal input first, then scale up
- Check the dataset output even if the run shows warnings

**Performance optimization:**
- Use more specific seed keywords for faster processing
- Limit geographic scope when possible
- Consider running multiple smaller batches instead of one large batch

---

*Made with ❤️ by Graphext for the SEO community*

**Ready to supercharge your keyword research?** Start analyzing keywords now and discover new opportunities for your SEO strategy!
