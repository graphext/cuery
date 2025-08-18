"""Unified SEO research platform integrating keyword data, SERP analysis, and traffic insights.

This module serves as the high-level orchestrator for comprehensive SEO research workflows,
combining data from Google Ads API, Apify web scraping actors, and AI-powered analysis
tools. It provides a streamlined interface for conducting end-to-end SEO research that
includes keyword discovery, search volume analysis, SERP competitor research, traffic
estimation, and content strategy insights.

The module integrates multiple data sources and analysis capabilities: Google Ads keyword
planning for search volume and competition data, Apify actors for real-time SERP scraping
and traffic analysis, and large language models for topic extraction and search intent
classification. This unified approach enables comprehensive SEO audits, competitive
analysis, and content planning in a single workflow, making it ideal for SEO professionals,
content strategists, and digital marketers who need actionable insights at scale.

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

from pandas import DataFrame

from ..utils import LOG, Configurable
from .keywords import GoogleKwdConfig, keywords
from .serps import SerpConfig, serps
from .traffic import TrafficConfig, keyword_traffic


class SeoConfig(Configurable):
    """Configuration for complete keyword data extraction (historical metrics, SERPs, traffic)."""

    kwd_cfg: GoogleKwdConfig
    """Configuration for Google Ads API keyword data extraction."""
    serp_cfg: SerpConfig | None = None
    """Configuration for SERP data extraction using Apify Google Search Scraper actor."""
    traffic_cfg: TrafficConfig | None = None
    """Whether and how to fetch traffic data for keywords using Similarweb scraper."""


async def seo_data(cfg: SeoConfig) -> DataFrame:
    """Fetch all supported SEO data types for a given set of keywords."""
    df = keywords(cfg.kwd_cfg)
    if df is None or len(df) == 0:
        raise ValueError("No keywords were fetched from Google Ads API!")

    if cfg.serp_cfg:
        srp = await serps(cfg=cfg.serp_cfg, keywords=df.keyword)
        if srp is None or len(srp) == 0:
            LOG.warning("The SERP actor has failed! Will return keyword metrics only.")
            return df

        df = df.merge(srp, how="left", left_on="keyword", right_on="term")

        if cfg.traffic_cfg:
            LOG.info("Fetching and processing traffic data for keywords")
            trf = await keyword_traffic(df["term"], df["domains"], cfg.traffic_cfg)
            if trf is None or len(trf) == 0:
                LOG.warning(
                    "Traffic actor has failed! Will return SERPs and keyword metrics only."
                )
            else:
                LOG.info(f"Got traffic dataframe:\n{trf}")
                df = df.merge(trf, how="left", on="term")

    if "term" in df.columns and "keyword" in df.columns:
        df = df.drop(columns=["term"])

    return df
