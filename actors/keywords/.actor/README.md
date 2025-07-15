# Google Ads Keyword Planning Actor

Analyze and expand keywords using the Google Ads API to generate comprehensive keyword research data. Features intelligent keyword expansion from seed terms or URLs, historical search volume analysis with trend calculations, and competition metrics for strategic SEO and PPC planning.

## What does Google Ads Keyword Planning Actor do?

This Actor connects to the Google Ads API to provide comprehensive keyword planning data for SEO and PPC professionals:

- ✅ **Expand seed keywords** into thousands of related keyword suggestions using Google's keyword planner
- ✅ **Generate keyword ideas** from landing pages or websites with configurable limits
- ✅ **Analyze search volumes** with up to 4 years of historical data and monthly breakdowns
- ✅ **Calculate growth trends** including year-over-year, 3-month, and 1-month growth rates
- ✅ **Extract competition metrics** including average CPC, competition scores, and bid estimates
- ✅ **Perform trend analysis** using linear regression to identify keyword momentum
- ✅ **Support multiple markets** with configurable language and country targeting
- ✅ **Generate structured data** ready for SEO analysis, PPC planning, and content strategy

**Perfect for**: SEO professionals, PPC specialists, content marketers, digital agencies, and businesses planning their keyword strategy and content optimization.

## Input

Configure your keyword analysis with these simple parameters:

### Example Input

```json
{
  "keywords": ["digital marketing", "seo", "keyword research"],
  "ideas": false,
  "language": "en",
  "country": "us",
  "metrics_start": "2024-01",
  "metrics_end": "2025-01"
}
```

### Example with Keyword Ideas Generation

```json
{
  "keywords": ["content marketing", "blog writing"],
  "ideas": true,
  "max_ideas": 500,
  "language": "en",
  "country": "us",
  "metrics_start": "2023-06",
  "metrics_end": "2024-06"
}
```

### Example with URL-based Keyword Ideas

```json
{
  "url": "https://example.com/",
  "whole_site": true,
  "ideas": true,
  "max_ideas": 300,
  "language": "en",
  "country": "us",
  "metrics_start": "2024-01",
  "metrics_end": "2025-01"
}
```

### Input Parameters

| Field | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| `keywords` | Array | List of seed keywords to expand and analyze | Optional* | - |
| `url` | String | Landing page or website URL for keyword ideas generation | Optional* | - |
| `whole_site` | Boolean | Generate ideas for entire website (requires `url` to be domain only) | Optional | `false` |
| `ideas` | Boolean | Generate additional keyword ideas (limited to ≤20 seed keywords) | Optional | `false` |
| `max_ideas` | Integer | Maximum keyword ideas to generate (when `ideas` is enabled) | Optional | - |
| `customer` | String | Google Ads Customer ID for API access | Optional | Uses environment variable |
| `language` | String | Language code for targeting (e.g., "en", "es", "fr") | Optional | `"en"` |
| `country` | String | Geographic target code (e.g., "us", "es", "uk") | Optional | `"us"` |
| `metrics_start` | String | Start date for historical metrics (YYYY-MM format) | Optional | - |
| `metrics_end` | String | End date for historical metrics (YYYY-MM format) | Optional | - |

*Either `keywords` or `url` must be provided.

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
  - Maximum 4-year range
  - Cannot be more than 4 years in the past
  - End date cannot be in the future
- **Examples**:
  - Last 12 months: `"2024-01"` to `"2025-01"`
  - Calendar year: `"2024-01"` to `"2024-12"`
  - Two-year analysis: `"2023-01"` to `"2025-01"`

**Keyword Ideas Generation:**
- Set `ideas: true` to generate additional keyword suggestions from Google's keyword planner
- **Important**: Only available when providing 20 or fewer seed keywords
- Use `max_ideas` to limit the number of generated keywords (recommended: 100-1000)
- If more than 20 keywords are provided only the first 20 will be used
- **Alternative**: Use `url` parameter to generate ideas from a landing page or website
- **Whole Site Mode**: Set `whole_site: true` with domain-only URL to get site-wide keyword ideas

**URL-based Keyword Generation:**
- Provide a `url` parameter to generate keyword ideas from a specific landing page
- For whole-site analysis, use domain-only URL (e.g., "example.com") and set `whole_site: true`
- Works independently of the `keywords` parameter - you can use either or both

## Output

The Actor generates a comprehensive dataset with detailed keyword metrics for SEO analysis.

### Sample Output

```json
{
  "keyword": "digital marketing",
  "avg_monthly_searches": 14800,
  "competition": 3,
  "competition_index": 0.62,
  "average_cpc_micros": 7173598,
  "low_top_of_page_bid_micros": 1964646,
  "high_top_of_page_bid_micros": 7035752,
  "concepts": [
    "google keyword tool",
    "blog",
    "free keyword tool",
    "google",
    "word",
    "keyword planner",
    "used"
  ],
  "concept_groups": ["Site", "Tool"],
  "search_volume": [12100, 12100, 18100, 18100, 14800, 12100, 18100, 18100, 18100, 14800, 14800, 12100],
  "search_volume_date": ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00", "2024-04-01T00:00:00"],
  "search_volume_growth_yoy": -5.23,
  "search_volume_growth_3m": 8.16,
  "search_volume_trend": 0.016
}
```

### Output Fields Explained

| Field | Type | Description | Use Case |
|-------|------|-------------|----------|
| `keyword` | String | The keyword or phrase analyzed | Content targeting and SEO planning |
| `avg_monthly_searches` | Number | Average monthly search volume | Traffic potential estimation |
| `average_cpc_micros` | Number | Average cost per click | Ad placement difficulty |
| `competition` | Number | Competition level (1=Low, 2=Medium, 3=High) | Keyword difficulty assessment |
| `competition_index` | Number | Numeric competition score (0-1 scale) | Detailed competition analysis |
| `low_top_of_page_bid_micros` | Number | Lower bound of top-of-page bid estimate (micros) | PPC budget planning minimum |
| `high_top_of_page_bid_micros` | Number | Upper bound of top-of-page bid estimate (micros) | PPC budget planning maximum |
| `concepts` | Array | Semantic category of keyword | Keyword grouping |
| `concept_groups` | Array | Semantic category of keyword | Keyword grouping |
| `search_volume` | Array | Historical monthly search volumes (chronological) | Seasonal trend analysis |
| `search_volume_date` | Array | Corresponding dates for search volume data | Timeline correlation |
| `search_volume_growth_yoy` | Number | Year-over-year growth percentage | Annual trend assessment |
| `search_volume_growth_3m` | Number | 3-month growth percentage | Quarterly trend analysis |
| `search_volume_trend` | Number | Linear regression trend coefficient | Overall momentum direction |

### Export & Integration

**Available Formats:**
- **JSON** - Perfect for APIs and automated workflows
- **CSV** - Excel-compatible for analysis and reporting  
- **XML** - System integrations and enterprise workflows
- **RSS** - Feed-based integrations

**Dataset Features:**
- Up to 4 years of historical search volume data per keyword
- Real-time competition metrics and bid estimates from Google Ads
- Advanced trend calculations including growth rates and momentum analysis
- Monthly breakdowns with corresponding dates for timeline analysis
- Structured for immediate analysis and visualization
- Compatible with popular SEO and PPC tools

## Getting Started

### How to run Google Ads Keyword Planning Actor

1. **📝 Enter your keywords or URL**: Add your seed keywords or provide a landing page URL
2. **🔧 Configure settings** (optional): 
   - **Customer ID**: Leave empty to use Graphext's internal customer, or provide your own Google Ads Customer ID
   - **Language & Country**: Set your target language and country
   - **Historical Range**: Configure date range for trend analysis
3. **▶️ Start the Actor**: Click "Start" and let it analyze your keywords
4. **📊 Download results**: Export your data in JSON, CSV, or XML format

### Real-World Examples

**🎯 Content Strategy with Keyword Expansion**
*Scenario: Blog content planning for a marketing agency*
```json
{
  "keywords": ["content marketing", "blog writing", "seo copywriting"],
  "ideas": true,
  "max_ideas": 300,
  "language": "en",
  "country": "us",
  "metrics_start": "2023-06",
  "metrics_end": "2024-06"
}
```
*Expected output: ~300-500 related keywords with 12 months of trend data*

**🏪 Local Business Keyword Research**  
*Scenario: Spanish restaurant chain expansion*
```json
{
  "keywords": ["restaurante madrid", "comida española", "tapas barcelona"],
  "ideas": true,
  "max_ideas": 200,
  "language": "es",
  "country": "es",
  "metrics_start": "2024-01",
  "metrics_end": "2025-01"
}
```
*Expected output: ~200-400 location-based keywords with competition data*

**🛒 E-commerce PPC Planning**
*Scenario: Online fashion store keyword research*
```json
{
  "keywords": ["buy shoes online", "women fashion", "designer clothing"],
  "ideas": true,
  "max_ideas": 500,
  "language": "en",
  "country": "uk",
  "metrics_start": "2024-01",
  "metrics_end": "2025-01"
}
```
*Expected output: ~500-800 product-related keywords with bid estimates and trend analysis*

**🌐 URL-based Keyword Discovery**
*Scenario: Analyzing competitor landing pages*
```json
{
  "url": "https://competitor.com/services/digital-marketing",
  "ideas": true,
  "max_ideas": 200,
  "language": "en",
  "country": "us",
  "metrics_start": "2024-01",
  "metrics_end": "2025-01"
}
```
*Expected output: ~200-300 keywords related to the landing page content*

### Tips for Best Results

- **Use specific seed keywords**: More targeted seeds = better keyword suggestions
- **Enable keyword ideas generation**: Set `ideas: true` for comprehensive keyword expansion (maximum 20 seed keywords)
- **Optimize idea limits**: Use `max_ideas: 100-1000` for balanced results and processing time
- **Stay within keyword limits**: Use 20 or fewer keywords for idea generation, unlimited for basic analysis
- **Try different language/country combinations**: Discover market-specific opportunities  
- **Mix broad and specific terms**: Get both high-level and long-tail keyword data
- **Include historical data**: Set date ranges to analyze trends and seasonality
- **Use URL mode**: Generate ideas from competitor pages or your own landing pages
- **Consider whole-site analysis**: Use domain URLs with `whole_site: true` for comprehensive site analysis

## Connect Google Ads Keyword Planning Actor to your workflows

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
    "country": "us"
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
    "country": "us",
    "metrics_start": "2024-01",
    "metrics_end": "2025-01"
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
- ✅ Check if the language/country combination is valid

**"Keywords format error"**  
- ✅ Ensure keywords are provided as an array: `["keyword1", "keyword2"]`
- ✅ Check that at least one keyword is provided
- ✅ Remove special characters or excessive punctuation

**"Language/Geographic code not recognized"**
- ✅ Use standard ISO codes: `"en"` for English, `"us"` for United States
- ✅ Check that language and country use consistent formatting
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
- ✅ Try using `url` parameter instead for URL-based keyword generation

**"URL-based keyword generation not working"**
- ✅ Ensure the URL is accessible and returns a valid webpage
- ✅ Use complete URLs including https:// for landing pages
- ✅ For whole-site analysis, use domain-only format (e.g., "example.com") and set `whole_site: true`
- ✅ Check that the website has sufficient content for keyword extraction

**"No historical data returned"**
- ✅ Ensure both `metrics_start` and `metrics_end` are provided
- ✅ Use valid date range within the last 4 years
- ✅ Check that keywords have sufficient search volume for historical data
- ✅ Try broader or more popular keywords

**"Actor run failed or timed out"**
- ✅ Reduce the number of seed keywords (try 10-20 keywords max for idea generation)
- ✅ Decrease `max_ideas` to limit keyword expansion
- ✅ Try simpler, more common keywords first
- ✅ Wait a few minutes and try again

**"Should I provide my own Customer ID?"**
- ✅ **Leave empty** for most use cases - Graphext's internal customer provides full access
- ✅ **Provide your own** only if you need data from your specific Google Ads account
- ✅ **Ensure access** if using your own ID, make sure it has Google Ads API access enabled

**"Invalid date range error"**
- ✅ Use YYYY-MM format: `"2024-01"` not `"January 2024"`
- ✅ Ensure start date is before end date
- ✅ Keep within last 4 years (Google Ads API limitation)
- ✅ Don't use future dates for end date
- ✅ Maximum 2-year range between start and end dates

**"Google Ads API date error"**
- ✅ Try more recent dates (within last 18-24 months)
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
- Limit historical date ranges when possible  
- Consider running multiple smaller batches instead of one large batch
- Use URL-based generation for very specific keyword discovery

---

*Made with ❤️ by Graphext for the SEO and PPC community*

**Ready to supercharge your keyword research?** Start analyzing keywords now and discover new opportunities with Google's authoritative keyword data!
