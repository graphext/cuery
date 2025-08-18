"""Domain traffic analysis and aggregation using Similarweb data via Apify actors.

This module provides comprehensive website traffic analysis capabilities by integrating
with Similarweb data through Apify's web scraping infrastructure. It enables large-scale
collection of domain-level traffic metrics including visitor counts, engagement metrics,
traffic sources, and global rankings. The module is particularly useful for competitive
analysis, market research, and understanding traffic patterns across multiple domains.

Key features include batch processing of domain URLs for efficient data collection,
automatic domain extraction and normalization from various URL formats, traffic source
breakdown (direct, search, social, referrals), and aggregation functions for keyword-based
traffic analysis. The module handles rate limiting and error recovery to ensure reliable
data collection, making it suitable for analyzing hundreds or thousands of domains
in SEO and competitive intelligence workflows.
"""

import asyncio
import os
import urllib.parse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from apify_client import ApifyClientAsync
from async_lru import alru_cache
from pandas import DataFrame, NamedAgg, Series

from ..utils import LOG, Configurable


class TrafficConfig(Configurable):
    """Configuration for fetching SERP data using Apify Google Search Scraper actor."""

    batch_size: int = 100
    """Number of keywords to fetch in a single batch."""
    apify_token: str | Path | None = None
    """Path to Apify API token file.
    If not provided, will use the `APIFY_TOKEN` environment variable.
    """


def domain(url: str) -> str | None:
    """Clean domain name."""
    if not url:
        return None

    dot_coms = ["X", "youtube", "reddit", "facebook", "instagram", "twitter", "linkedin", "tiktok"]
    if url.lower() in dot_coms:
        return url.lower() + ".com"

    if not url.startswith("http"):
        url = "https://" + url

    domain = urllib.parse.urlparse(str(url)).netloc
    if domain.startswith("www."):
        domain = domain[4:]

    return domain


async def fetch_batch(urls: list[str], client: ApifyClientAsync, **kwargs):
    """Process a single batch of keywords."""
    run_input = {"websites": urls, **kwargs}
    actor = client.actor("tri_angle/fast-similarweb-scraper")
    run = await actor.call(run_input=run_input)
    if run is None:
        LOG.error(f"Actor run failed for batch: {urls}... ")
        return None

    dataset_client = client.dataset(run["defaultDatasetId"])
    return await dataset_client.list_items()


@alru_cache(maxsize=3)
async def fetch_domain_traffic(urls: tuple[str, ...], cfg: TrafficConfig) -> DataFrame:
    """Fetch traffic data for a DataFrame of organic SERP results.

    Note that free similarweb crawlers only fetch data at the domain level, not for specific URLs!

    Actor: https://apify.com/tri_angle/fast-similarweb-scraper
    """
    if isinstance(cfg.apify_token, str | Path):
        with open(cfg.apify_token) as f:
            token = f.read().strip()
    else:
        token = os.environ["APIFY_TOKEN"]

    client = ApifyClientAsync(token)

    # Get clean unique domain names from URLs
    # Maintain original order despite deduplication and removal of empty domains
    domains = Series(urls, name="domain").apply(domain)
    unique = domains.dropna().drop_duplicates().reset_index(drop=True)
    batches = [unique.iloc[i : i + cfg.batch_size] for i in range(0, len(unique), cfg.batch_size)]
    tasks = [fetch_batch(list(batch), client) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    records = []
    for result in batch_results:
        if result is not None:
            records.extend(result.items)

    data = DataFrame.from_records(records)
    data = data.drop(columns=["url"])  # This is the url to Similarweb

    # Reconstruct the original order of urls/domains, including duplicates and empty domains
    result = DataFrame({"url": urls, "domain": domains})
    result = result.merge(data, left_on="domain", right_on="name", how="left")
    return result.drop(columns=["name"])


def normalize_traffic(df: DataFrame) -> DataFrame:
    """Process traffic data into flat DataFrame with relevant data only."""
    df["globalRank"] = pd.json_normalize(df.globalRank)

    engagements = pd.json_normalize(df.pop("engagements"))
    df = pd.concat([df, engagements], axis=1)

    sources = pd.json_normalize(df.pop("trafficSources"))
    sources.columns = [f"source_{col}" for col in sources.columns]
    df = pd.concat([df, sources], axis=1)

    df["category"] = df.category.apply(lambda x: x.split("/") if isinstance(x, str) else None)

    drop_columns = [
        "countryRank",
        "categoryRank",
        "description",
        "estimatedMonthlyVisits",
        "globalCategoryRank",
        "icon",
        "previewDesktop",
        "previewMobile",
        "scrapedAt",
        "snapshotDate",
        "title",
        "topCountries",
    ]

    return df.drop(columns=drop_columns)


def aggregate_traffic(df: DataFrame, by: str) -> DataFrame:
    """Aggregate traffic data for each keyword's top domains.

    Note: for now we don't keep similarweb's categorization of domains or top keyword data.
    """
    aggs = {
        "urls": NamedAgg("url", list),
        "globalRank_min": NamedAgg("globalRank", "min"),
        "globalRank_max": NamedAgg("globalRank", "max"),
        "visits_min": NamedAgg("visits", "min"),
        "visits_max": NamedAgg("visits", "max"),
        "timeOnSite_min": NamedAgg("timeOnSite", "min"),
        "timeOnSite_max": NamedAgg("timeOnSite", "max"),
        "pagesPerVisit_min": NamedAgg("pagePerVisit", "min"),
        "pagesPerVisit_max": NamedAgg("pagePerVisit", "max"),
        "bounceRate_min": NamedAgg("bounceRate", "min"),
        "bounceRate_max": NamedAgg("bounceRate", "max"),
        "source_direct_min": NamedAgg("source_direct", "min"),
        "source_direct_max": NamedAgg("source_direct", "max"),
        "source_search_min": NamedAgg("source_search", "min"),
        "source_search_max": NamedAgg("source_search", "max"),
        "source_social_min": NamedAgg("source_social", "min"),
        "source_social_max": NamedAgg("source_social", "max"),
        "source_referrals_min": NamedAgg("source_referrals", "min"),
        "source_referrals_max": NamedAgg("source_referrals", "max"),
    }

    return df.groupby(by).agg(**aggs).reset_index()


async def keyword_traffic(
    kwds: Series | Iterable[str],
    urls: Iterable[list | None],
    cfg: TrafficConfig,
) -> DataFrame | None:
    """Fetch and aggregate traffic data for lists of urls associated with given keywords."""
    try:
        df = DataFrame({"keyword": Series(kwds), "url": Series(urls)})  # type: ignore
        df = df.explode(column="url")
        trf = await fetch_domain_traffic(tuple(df.url), cfg)
        trf = normalize_traffic(trf)

        kwd_trf = df.merge(trf, on="url", how="left")
        result = aggregate_traffic(kwd_trf, by="keyword")

        if isinstance(kwds, Series):
            result = result.rename(columns={"keyword": kwds.name})

        return result
    except Exception as exc:
        LOG.error(f"Failed to fetch traffic data for keywords: {kwds}. Error: {exc}")
        return None
