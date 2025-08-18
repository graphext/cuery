SEO Research and Analysis
=========================

The ``cuery.seo`` subpackage provides a comprehensive toolkit for SEO research and analysis, integrating data from multiple sources including Google Ads API, Apify web scraping actors, and AI-powered content analysis. This unified platform enables SEO professionals to perform end-to-end research workflows including keyword discovery, SERP analysis, traffic estimation, and competitive intelligence.

Overview
--------

The SEO subpackage consists of several interconnected modules:

- **Keywords** (``cuery.seo.keywords``): Google Ads API integration for keyword research and historical metrics
- **SERPs** (``cuery.seo.serps``): SERP data collection and analysis using Apify actors
- **Traffic** (``cuery.seo.traffic``): Domain traffic analysis using Similarweb data
- **Tasks** (``cuery.seo.tasks``): AI-powered topic extraction and search intent classification
- **SEO** (``cuery.seo.seo``): High-level orchestrator combining all components

Key Features
------------

🔍 **Keyword Research**
   - Generate keyword ideas from seed keywords or landing pages
   - Retrieve historical search volume and trend data
   - Geographic and language targeting
   - Competition analysis and cost-per-click metrics

📊 **SERP Analysis**
   - Real-time search results scraping for any keyword
   - Competitor presence tracking in search results
   - Organic result analysis with titles, URLs, and snippets
   - Brand monitoring across search results

🚦 **Traffic Intelligence**
   - Domain-level traffic estimation and trends
   - Traffic source breakdown (direct, search, social, referrals)
   - Engagement metrics (bounce rate, time on site, pages per visit)
   - Global ranking and competitive positioning

🤖 **AI-Powered Insights**
   - Automated topic extraction from SERP content
   - Search intent classification (informational, navigational, transactional, commercial)
   - Content gap analysis and opportunity identification
   - Hierarchical topic clustering for content strategy

Authentication Setup
--------------------

The SEO subpackage requires API credentials for Google Ads, Apify, and AI models. You can configure these using environment variables or configuration files.

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import json
    import cuery.utils

    # Google Ads API
    os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"] = "your_developer_token"
    os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"] = "your_login_customer_id"
    os.environ["GOOGLE_ADS_USE_PROTO_PLUS"] = "true"
    os.environ["GOOGLE_ADS_CUSTOMER_ID"] = "your_customer_id"
    
    # For service account authentication
    with open("path/to/service-account-key.json") as f:
        json_key = json.load(f)
    os.environ["GOOGLE_ADS_JSON_KEY"] = json.dumps(json_key)

    # Apify for SERP and traffic data
    os.environ["APIFY_TOKEN"] = "your_apify_token"

    # AI model API keys
    cuery.utils.set_api_keys({
        "OpenAI": "your_openai_key",
        "Google": "your_google_key",
    })

Configuration Files
~~~~~~~~~~~~~~~~~~~

Alternatively, you can pass credential file paths directly in the configuration:

.. code-block:: python

    from cuery.seo import SeoConfig

    config = SeoConfig(
        kwd_cfg={
            "google_ads_config": "path/to/google-ads-config.yaml",
            "keywords": ["your keywords"],
        },
        serp_cfg={
            "apify_token": "path/to/apify-token.txt",
        },
        traffic_cfg={
            "apify_token": "path/to/apify-token.txt",
        }
    )

Quick Start
-----------

Here's a simple example to get started with SEO analysis:

.. code-block:: python

    from cuery.seo import SeoConfig, seo

    # Configure the SEO analysis
    config = SeoConfig(
        kwd_cfg={
            "keywords": ["machine learning", "data science"],
            "language": "en",
            "country": "us",
            "ideas": True,
            "max_ideas": 50,
        },
        serp_cfg={
            "resultsPerPage": 10,
            "country": "us",
            "brands": ["your_brand"],
            "competitors": ["competitor1", "competitor2"],
        },
        traffic_cfg={
            "batch_size": 25,
        }
    )

    # Run the complete SEO analysis
    result = await seo.seo_data(config)
    
    # The result contains keyword data, SERP results, and traffic insights
    print(result.head())

Keyword Research
----------------

The keywords module provides access to Google Ads keyword planning data:

.. code-block:: python

    from cuery.seo.keywords import GoogleKwdConfig, keywords

    # Configure keyword research
    kwd_config = GoogleKwdConfig(
        keywords=["SEO tools", "keyword research"],
        language="en",
        country="us",
        ideas=True,
        max_ideas=100,
        metrics_start="2023-01",
        metrics_end="2024-12",
    )

    # Get keyword ideas and historical metrics
    keyword_data = await keywords(kwd_config)
    print(keyword_data.columns)
    # Output: ['keyword', 'avg_monthly_searches', 'competition', 'low_bid', 'high_bid', ...]

Features:
^^^^^^^^^

- **Keyword Ideas**: Generate related keywords from seed terms
- **Historical Metrics**: Monthly search volume over time
- **Competition Data**: Competition level and bid estimates
- **Geographic Targeting**: Country and language specific data
- **Trend Analysis**: Search volume trends and seasonality

SERP Analysis
-------------

The serps module collects and analyzes search engine results:

.. code-block:: python

    from cuery.seo.serps import SerpConfig, serps

    # Configure SERP analysis
    serp_config = SerpConfig(
        resultsPerPage=20,
        country="us",
        searchLanguage="en",
        brands=["your_brand"],
        competitors=["competitor1", "competitor2"],
        topic_model="google/gemini-2.5-flash-preview-05-20",
    )

    # Analyze SERPs for keywords
    keywords_list = ["SEO analysis", "SERP tracking"]
    serp_data = await serps(keywords_list, serp_config)

Features:
^^^^^^^^^

- **Real-time SERP Data**: Fresh search results for any keyword
- **Competitor Tracking**: Monitor competitor presence in results
- **Brand Monitoring**: Track your brand's search visibility
- **AI Topic Analysis**: Automated topic extraction from SERP content
- **Intent Classification**: Categorize search intent automatically

Traffic Analysis
----------------

The traffic module provides domain-level traffic insights:

.. code-block:: python

    from cuery.seo.traffic import TrafficConfig, keyword_traffic

    # Configure traffic analysis
    traffic_config = TrafficConfig(
        batch_size=50,
    )

    # Get traffic data for domains from SERP results
    keywords_series = pd.Series(["keyword1", "keyword2"])
    domain_lists = [["example.com", "competitor.com"], ["another.com"]]
    
    traffic_data = await keyword_traffic(keywords_series, domain_lists, traffic_config)

Features:
^^^^^^^^^

- **Traffic Estimation**: Monthly visitor estimates for domains
- **Source Breakdown**: Direct, search, social, and referral traffic
- **Engagement Metrics**: Bounce rate, time on site, pages per visit
- **Global Rankings**: Worldwide traffic rankings
- **Competitive Analysis**: Compare traffic across multiple domains

AI-Powered Analysis
-------------------

The tasks module provides intelligent analysis of SERP data:

.. code-block:: python

    from cuery.seo.tools import SerpTopicExtractor, SerpTopicAndIntentAssigner

    # Extract topics from SERP data
    topic_extractor = SerpTopicExtractor(
        model="google/gemini-2.5-flash-preview-05-20",
        n_topics=10,
        n_subtopics=5,
    )

    # Assign topics and intent to keywords
    intent_assigner = SerpTopicAndIntentAssigner(
        model="openai/gpt-4.1-mini",
    )

Features:
^^^^^^^^^

- **Topic Extraction**: Hierarchical topic identification from SERP content
- **Intent Classification**: Automatic categorization into informational, navigational, transactional, or commercial intent
- **Content Analysis**: Analysis of page titles, domains, and breadcrumbs
- **Semantic Understanding**: AI-powered understanding of search context

Complete Workflow Example
-------------------------

Here's a comprehensive example combining all SEO components:

.. code-block:: python

    import pandas as pd
    from cuery.seo import SeoConfig, seo

    # Define your research parameters
    target_keywords = [
        "content marketing strategy",
        "SEO best practices",
        "digital marketing tools"
    ]

    # Configure the complete SEO analysis
    seo_config = SeoConfig(
        kwd_cfg={
            "keywords": target_keywords,
            "ideas": True,
            "max_ideas": 200,
            "language": "en",
            "country": "us",
            "metrics_start": "2023-01",
            "metrics_end": "2024-12",
        },
        serp_cfg={
            "resultsPerPage": 20,
            "country": "us",
            "searchLanguage": "en",
            "brands": ["your_company"],
            "competitors": [
                "hubspot",
                "semrush",
                "ahrefs",
                "moz"
            ],
            "topic_model": "google/gemini-2.5-flash-preview-05-20",
            "assignment_model": "openai/gpt-4.1-mini",
        },
        traffic_cfg={
            "batch_size": 50,
        }
    )

    # Run the complete analysis
    results = await seo.seo_data(seo_config)

    # Analyze the results
    print("Keyword Analysis:")
    print(f"Total keywords analyzed: {len(results)}")
    print(f"Average monthly searches: {results['avg_monthly_searches'].mean():.0f}")
    
    print("\nTop performing competitors:")
    competitor_presence = results.groupby('competitor_domains')['avg_monthly_searches'].sum().sort_values(ascending=False)
    print(competitor_presence.head())

    print("\nSearch intent distribution:")
    intent_dist = results['search_intent'].value_counts()
    print(intent_dist)

    # Save results for further analysis
    results.to_csv("seo_analysis_results.csv", index=False)

Data Export and Integration
---------------------------

Results from SEO analysis can be easily exported and integrated with other tools:

.. code-block:: python

    # Export to various formats
    results.to_csv("seo_data.csv", index=False)
    results.to_parquet("seo_data.parquet", index=False)
    results.to_excel("seo_data.xlsx", index=False)

    # Integration with visualization tools
    import plotly.express as px

    # Search volume trends
    monthly_data = results.groupby('metrics_month')['avg_monthly_searches'].sum().reset_index()
    fig = px.line(monthly_data, x='metrics_month', y='avg_monthly_searches', 
                  title='Search Volume Trends')
    fig.show()

    # Competitor analysis
    competitor_data = results.groupby('competitor_domains')['traffic_visits_max'].sum().reset_index()
    fig = px.bar(competitor_data.head(10), x='competitor_domains', y='traffic_visits_max',
                 title='Top Competitors by Traffic')
    fig.show()

Best Practices
--------------

Rate Limiting and Quotas
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Google Ads API**: Respect daily quota limits and implement exponential backoff
- **Apify**: Monitor credit usage and implement batch processing for large datasets
- **AI Models**: Use appropriate models for different tasks (fast models for classification, powerful models for content generation)

Data Quality
~~~~~~~~~~~~

- **Keyword Validation**: Clean and normalize keywords before analysis
- **Domain Cleaning**: Use the built-in domain normalization functions
- **Result Filtering**: Filter out irrelevant or low-quality results

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

- **Batch Processing**: Use appropriate batch sizes for your use case
- **Concurrent Requests**: Leverage async processing for faster execution
- **Caching**: Implement caching for repeated analyses
- **Data Storage**: Use efficient formats like Parquet for large datasets

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

    try:
        results = await seo.seo_data(config)
    except Exception as e:
        logger.error(f"SEO analysis failed: {e}")
        # Implement fallback or retry logic

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Authentication Errors**
   - Verify API credentials are correctly set
   - Check quota limits and billing status
   - Ensure service accounts have proper permissions

**Rate Limiting**
   - Reduce batch sizes
   - Implement delays between requests
   - Use exponential backoff for retries

**Data Quality Issues**
   - Validate input keywords for special characters
   - Check geographic and language settings
   - Filter results based on relevance scores

**Performance Issues**
   - Optimize batch sizes for your infrastructure
   - Use async processing for I/O bound operations
   - Consider data sampling for large datasets

API Reference
-------------

For detailed API documentation, see the auto-generated documentation:

- :doc:`autoapi/cuery/seo/index`
