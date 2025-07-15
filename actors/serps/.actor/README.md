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
| `country` | String | Country code for SERP targeting (e.g., "us", "uk", "de") | Optional | - |
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

### Export & Integration

**Available Formats:**
- **JSON** - Perfect for APIs and automated analysis
- **CSV** - Excel-compatible for reporting and visualization  
- **XML** - System integrations and enterprise workflows
- **RSS** - Feed-based monitoring and alerts

**Dataset Features:**
- Complete SERP feature analysis (organic, paid, AI overviews)
- Real-time brand and competitor tracking with ranking positions
- AI-powered topic and intent classification for semantic analysis
- Comprehensive metadata including SERP features and user queries
- Structured for immediate competitive analysis and SEO insights
- Compatible with popular SEO tools and analytics platforms

## Getting Started

### How to run Google SERP Data Collection Actor

1. **📝 Enter your keywords**: Add the keywords you want to analyze SERPs for
2. **🔧 Configure SERP settings**:
   - **Geographic targeting**: Set country and language preferences
   - **Results depth**: Configure number of results per page and pages per query
   - **Brand tracking**: Add your brands and competitors to monitor
3. **🤖 Enable AI analysis** (optional):
   - **Topic extraction**: Get semantic categorization of search results
   - **Intent classification**: Understand user search intent
   - **Entity extraction**: Identify entities in AI overviews
4. **▶️ Start the Actor**: Click "Start" and let it collect SERP data
5. **📊 Download results**: Export your SERP analysis in your preferred format

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

## Connect Google SERP Data Collection Actor to your workflows

### 🔌 Apify API Integration

```bash
curl -X POST https://api.apify.com/v2/acts/[ACTOR_ID]/runs \
  -H "Authorization: Bearer [YOUR_API_TOKEN]" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["seo tools", "keyword research", "serp analysis"],
    "resultsPerPage": 100,
    "country": "us",
    "searchLanguage": "en",
    "brands": ["Ahrefs", "SEMrush"],
    "competitors": ["Moz", "Screaming Frog"]
  }'
```

### 🐍 Python SDK Integration

```python
from apify_client import ApifyClient

# Initialize the client
client = ApifyClient("[YOUR_API_TOKEN]")

# Prepare Actor input with comprehensive SERP analysis
run_input = {
    "keywords": ["digital marketing", "seo strategy", "content optimization"],
    "resultsPerPage": 50,
    "country": "us",
    "searchLanguage": "en",
    "top_n": 15,
    "brands": ["HubSpot", "Salesforce"],
    "competitors": ["Marketo", "Pardot", "ActiveCampaign"],
    "topic_model": "google/gemini-2.5-flash-preview-05-20",
    "assignment_model": "openai/gpt-4.1-mini",
    "entity_model": "openai/gpt-4.1-mini"
}

# Run the Actor and wait for it to finish
run = client.actor("[ACTOR_ID]").call(run_input=run_input)

# Fetch results from the run's dataset
serp_data = client.dataset(run["defaultDatasetId"]).list_items().items

# Analyze SERP data
for result in serp_data:
    print(f"Keyword: {result['term']}")
    print(f"Brand rank: {result.get('title_rank_brand', 'Not found')}")
    print(f"Top competitor: {result.get('title_rank_competition', 'None')}")
    print(f"Topic: {result.get('topic', 'Unknown')}")
    print(f"Intent: {result.get('intent', 'Unknown')}")
    print("---")
```

### 🔔 Webhook Automation

Automatically process SERP analysis when the Actor finishes:

```json
{
  "eventTypes": ["ACTOR.RUN.SUCCEEDED"],
  "requestUrl": "https://your-website.com/webhook/serp-analysis",
  "payloadTemplate": {
    "actorRunId": "{{resource.id}}",
    "datasetId": "{{resource.defaultDatasetId}}",
    "status": "{{resource.status}}",
    "keywordCount": "{{resource.stats.requestsFinished}}"
  }
}
```

### 📊 Popular Integrations

- **Google Sheets**: Export SERP analysis to spreadsheets for team collaboration
- **Data Studio**: Create competitive intelligence dashboards
- **Slack/Teams**: Get alerts when competitor rankings change
- **Zapier**: Connect to 5,000+ apps for automated workflows
- **Power BI/Tableau**: Build advanced SERP analytics dashboards
- **CRM Systems**: Track brand mentions and competitor intelligence

## Troubleshooting

### ❓ Common Questions

**"No SERP results for my keywords"**
- ✅ Verify keywords are spelled correctly and in the target language
- ✅ Check if the country/language combination is valid
- ✅ Try more popular or broader keywords first
- ✅ Ensure keywords have sufficient search volume

**"Brand/competitor tracking not working"**  
- ✅ Use exact brand names as they appear in search results
- ✅ Check spelling and capitalization of brand names
- ✅ Try both short names ("Nike") and full names ("Nike Inc.")
- ✅ Ensure brands actually appear in the search results

**"AI analysis taking too long"**
- ✅ Reduce `topic_max_samples` to 100-300 for faster processing
- ✅ Use fewer keywords per batch (20-50 instead of 100+)
- ✅ Consider disabling AI models if not needed
- ✅ Try lighter models like "openai/gpt-4.1-mini"

**"Geographic targeting not working"**
- ✅ Use standard country codes: `"us"`, `"uk"`, `"de"`, etc.
- ✅ Match language with country when possible
- ✅ Check that the combination is supported by Google Search
- ✅ Try common combinations like `"us"` + `"en"` first

**"Too many paid results, not enough organic"**
- ✅ Try less commercial keywords for more organic results
- ✅ Increase `resultsPerPage` to get more total results
- ✅ Use informational rather than transactional keywords
- ✅ Consider long-tail keywords which typically have fewer ads

**"AI overviews not appearing"**
- ✅ Use informational keywords that typically trigger AI overviews
- ✅ Try question-based keywords ("what is...", "how to...")
- ✅ Not all keywords trigger AI overviews - this is normal
- ✅ Check if AI overviews are available in your target country

**"Entity extraction from AI overviews failing"**
- ✅ Ensure AI overviews are present in your results first
- ✅ Check that `entity_model` is set correctly
- ✅ Try different AI models if extraction is inconsistent
- ✅ Some AI overviews may have limited extractable entities

**"Batch processing too slow"**
- ✅ Reduce `batch_size` to 20-50 for faster individual processing
- ✅ Disable AI analysis features if not needed
- ✅ Use fewer `resultsPerPage` (30-50 instead of 100)
- ✅ Process keywords in multiple smaller runs

**"Inconsistent ranking data"**
- ✅ Rankings can vary by time, location, and personalization
- ✅ Use consistent geographic and language settings
- ✅ Consider running analysis multiple times for averages
- ✅ Remember that Google results are dynamic and personalized

### 📞 Need More Help?

1. **Check your input format** using the examples above
2. **Review the Getting Started section** for proper configuration
3. **Try the provided example inputs** to test functionality
4. **Monitor Actor run logs** for detailed error messages
5. **Contact support** through Apify platform if issues persist

### 🔧 Advanced Troubleshooting

**For power users and developers:**
- Monitor Apify actor logs for Google Search Scraper specific errors
- Verify that target websites are accessible and not blocking scrapers
- Check API quotas and rate limits for both Apify and AI model providers
- Test with minimal input first, then scale up gradually

**Performance optimization:**
- Use more specific, less competitive keywords for faster processing
- Batch similar keywords (same topic/intent) together
- Consider running analysis during off-peak hours
- Use caching by running identical queries infrequently

**Data quality tips:**
- Cross-reference results with manual Google searches
- Account for personalization and location differences
- Use multiple runs over time to identify ranking trends
- Combine with other SEO data sources for comprehensive analysis

---

*Made with ❤️ by Graphext for the SEO and competitive intelligence community*

**Ready to dominate your SERP analysis?** Start tracking competitor rankings, brand mentions, and search trends now with comprehensive AI-powered SERP intelligence!
