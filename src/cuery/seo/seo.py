"""High-level API to fetch SEO data from various sources and to enrich with AI.

Authentication
    To use all features and data sources, i.e. to have access to Google Ads, Apify, and AI models,
    you can either:

    - pass paths to credential files in the configuration
    - pass dictionaries/strings with already loaded credentials
    - or set the respective environment variables


    Google Ads
        Google Ads variables should be prefixed with `GOOGLE_ADS_` and can be set as follows::

            os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"] = "..."
            os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"] = "..."
            os.environ["GOOGLE_ADS_USE_PROTO_PLUS"] = "true"
            os.environ["GOOGLE_ADS_JSON_KEY"] = json.dumps(json_key)
            os.environ["GOOGLE_ADS_CUSTOMER_ID"] = "..."


    Apify
        For Apify, you can set the `APIFY_TOKEN` environment variable::

            os.environ["APIFY_TOKEN"] = "..."

    LLMs
        For AI use, set the corresponding API keys like this::

            llm_keys = {
                "OpenAI": "...",
                "Google": "...",
            }
            cuery.utils.set_api_keys(llm_keys)
"""

from pathlib import Path
from warnings import warn

from pandas import DataFrame
from pydantic import BaseModel, Field

from ..utils import LOG
from .keywords import fetch_keywords, process_keywords
from .serps import (
    add_brand_mentions,
    add_ranks,
    aggregate_organic_results,
    fetch_serps,
    process_ai_overviews,
    process_serps,
    topic_and_intent,
)
from .traffic import add_keyword_traffic


class HashableConfig(BaseModel):
    """Base class for configurations. Hashable so we can cache API calls using them."""

    def __hash__(self) -> int:
        return self.model_dump_json().__hash__()


class GoogleKwdConfig(HashableConfig):
    """Configuration for Google Ads API access."""

    keywords: tuple[str, ...] | None = None
    """The (initial) keywords to fetch data for."""
    page: str | None = None
    """The page to fetch data for (if applicable)."""
    ideas: bool = False
    """Whether to expand initial keywords with Google Keyword Planner's idea generator."""
    max_ideas: int | None = None
    """Maximum number of additional keyword ideas to fetch (if `ideas` is True)."""
    language: str = "en"
    """The language to use for keyword data (e.g., 'en' for English)."""
    geo_target: str = "us"
    """The geographical target for keyword data (e.g., 'us' for United States)."""
    metrics_start: str | None = None
    """Start date (year and month) for metrics in YYYY-MM format (e.g., '2023-01')."""
    metrics_end: str | None = None
    """End date (year and month) for metrics in YYYY-MM format (e.g., '2023-12')."""
    credentials: str | Path | dict | None = None
    """Path to Google Ads API credentials file or a dictionary with credentials.
    If not provided, will look for environment variables with prefix `GOOGLE_ADS_`."""
    customer: str | None = None
    """Google Ads customer ID to use for API requests.
    If not provided, will use the `GOOGLE_ADS_CUSTOMER_ID` or `GOOGLE_ADS_LOGIN_CUSTOMER_ID`
    environment variable."""


class SerpConfig(HashableConfig):
    """Configuration for fetching SERP data using Apify Google Search Scraper actor."""

    batch_size: int = 100
    """Number of keywords to fetch in a single batch."""
    resultsPerPage: int = 100
    """Number of results to fetch per page."""
    maxPagesPerQuery: int = 1
    """Maximum number of pages to fetch per query."""
    country: str | None = None
    """Country code for SERP data (e.g., 'us' for United States)."""
    searchLanguage: str | None = None
    """Search language for SERP data (e.g., 'en' for English)."""
    languageCode: str | None = None
    """Language code for SERP data (e.g., 'en' for English)."""
    params: dict | None = Field(default_factory=dict)
    """Additional parameters to pass to the Apify actor."""
    apify_token: str | Path | None = None
    """Path to Apify API token file.
    If not provided, will use the `APIFY_TOKEN` environment variable."""


class SeoConfig(HashableConfig):
    """Configuration for complete keyword data extraction (historical metrics, SERPs, traffic)."""

    kwd_cfg: GoogleKwdConfig
    """Configuration for Google Ads API keyword data extraction."""
    serp_cfg: SerpConfig | None = None
    """Configuration for SERP data extraction using Apify Google Search Scraper actor."""
    brands: str | list[str] | None = None
    """List of brand names to identify in SERP data."""
    competitors: str | list[str] | None = None
    """List of competitor names to identify in SERP data."""
    topic_max_samples: int = 500
    """Maximum number of samples to use for topic and intent extraction from SERP data."""
    topic_model: str | None = "google/gemini-2.5-flash-preview-05-20"
    """Model to use for topic extraction from SERP organic results."""
    assignment_model: str | None = "openai/gpt-4.1-mini"
    """Model to use for intent classification from SERP organic results."""
    entity_model: str | None = "openai/gpt-4.1-mini"
    """Model to use for entity extraction from AI overviews."""
    fetch_traffic: bool = False
    """Whether to fetch traffic data for keywords using Similarweb scraper."""


async def fetch_data(cfg: SeoConfig) -> DataFrame:
    """Fetch all supported SEO data types for a given set of keywords."""

    LOG.info(f"Starting SEO data extraction with configuration:\n{cfg.model_dump_json(indent=2)}")

    LOG.info("Fetching and processing keywords from Google Ads API")
    kwd_cfg = cfg.kwd_cfg.model_dump()
    kwds = fetch_keywords(**kwd_cfg)
    if kwds is None or len(kwds.results) == 0:
        LOG.error(
            "No keywords were fetched from Google Ads API! "
            "Check your configuration, credentials, and network connection."
        )
        return DataFrame()

    df = process_keywords(kwds, collect_volumes=True)
    LOG.info(f"Got keyword dataframe:\n{df.head()}")

    LOG.info("Fetching and processing SERP data")
    if cfg.serp_cfg is not None:
        serp_cfg = cfg.serp_cfg.model_dump()
        serp_params = serp_cfg.pop("params", {})
        serps = await fetch_serps(keywords=tuple(df.keyword), **serp_cfg, **serp_params)

        if serps is None:
            LOG.warning(
                "The SERP actor has failed! Check your logs, configuration etc. "
                "Will return keyword metrics only."
            )
            return df

        if len(serps) == 0:
            LOG.warning("Got 0 SERP results! Will return keyword metrics only.")
            return df

        features, org, paid, ads = process_serps(serps, copy=True)
        LOG.info(f"Got SERP dataframes\nFeatures:\n{features.head()}")
        LOG.info(f"Organic:\n{org.head()}")

        if set(features.term) != set(df.keyword):
            warn("SERP terms do not match keywords!", stacklevel=2)

        LOG.info("Aggregating organic results")
        orgagg = aggregate_organic_results(org, top_n=10)

        LOG.info("Calculating brand and competitor ranks in SERP data")
        if cfg.brands or cfg.competitors:
            orgagg = add_ranks(orgagg, brands=cfg.brands, competitors=cfg.competitors)

        df = df.merge(features, left_on="keyword", right_on="term", how="left")
        df = df.merge(orgagg, on="term", how="left")

        if cfg.topic_model is not None and cfg.assignment_model is not None:
            LOG.info("Extracting topics and intents from keywords SERP data")
            clf_df = await topic_and_intent(
                df=orgagg,
                max_samples=cfg.topic_max_samples,
                topic_model=cfg.topic_model,
                assignment_model=cfg.assignment_model,
                max_retries=6,
            )
            if clf_df is not None:
                df = df.merge(clf_df, on="term", how="left")

        if cfg.entity_model is not None:
            LOG.info("Processing AI overviews from SERP data")
            ai_df = await process_ai_overviews(features, entity_model=cfg.entity_model)
            if ai_df is not None:
                df = df.merge(ai_df, on="term", how="left")

        if cfg.brands or cfg.competitors:
            df = add_brand_mentions(df, brands=cfg.brands, competitors=cfg.competitors)

    if cfg.fetch_traffic:
        LOG.info("Fetching and processing traffic data for keywords")
        df = await add_keyword_traffic(df)

    return df
