"""Use Apify actors to fetch domain traffic data from Similarweb and other sources."""

import asyncio
import os
import urllib.parse
from pathlib import Path

import pandas as pd
from apify_client import ApifyClientAsync
from async_lru import alru_cache
from pandas import DataFrame, NamedAgg

from ..utils import LOG


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


@alru_cache(maxsize=3)
async def fetch_domain_traffic(
    urls: tuple[str, ...],  # type: ignore
    batch_size: int = 100,
    apify_token: str | Path | None = None,
    **kwargs,
) -> DataFrame:
    """Fetch traffic data for a DataFrame of organic SERP results.

    Note that free similarweb crawlers only fetch data at the domain level, not for specific URLs!

    Actor: https://apify.com/tri_angle/fast-similarweb-scraper
    """
    if isinstance(apify_token, str | Path):
        with open(apify_token) as f:
            token = f.read().strip()
    else:
        token = os.environ["APIFY_TOKEN"]

    client = ApifyClientAsync(token)

    domains = [domain(url) for url in urls]
    domains_unq = list({d for d in domains if d})
    batches = [domains_unq[i : i + batch_size] for i in range(0, len(domains_unq), batch_size)]

    async def process_batch(batch):
        """Process a single batch of keywords."""
        run_input = {"websites": batch, **kwargs}

        actor = client.actor("tri_angle/fast-similarweb-scraper")
        run = await actor.call(run_input=run_input)
        if run is None:
            LOG.error(f"Actor run failed for batch: {batch}... ")
            return None

        dataset_client = client.dataset(run["defaultDatasetId"])
        return await dataset_client.list_items()

    tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    result = []
    for batch_result in batch_results:
        if batch_result is not None:
            result.extend(batch_result.items)

    df = DataFrame.from_records(result)
    idx = DataFrame({"url": urls, "domain": domains})
    df = idx.merge(df.drop(columns=["url"]), left_on="domain", right_on="name", how="left")
    return df.drop(columns=["name"])


def process_traffic(df: DataFrame) -> DataFrame:
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


async def add_keyword_traffic(kwds: DataFrame) -> DataFrame:
    """Fetch and add traffic data to keywords DataFrame.

    Note: each keyword has multiple top organic websites/domains in SERP results.
    """
    required_columns = ["keyword", "domains"]
    if any(col not in kwds for col in required_columns):
        LOG.warning(
            f"Keywords DF must contain at least these columns: {required_columns}! "
            "Will return original DataFrame without traffic data."
        )
        return kwds

    try:
        kwds_expl = kwds[["keyword", "domains"]].explode(column="domains")

        trf = await fetch_domain_traffic(tuple(kwds_expl.domains))
        trf = process_traffic(trf)

        kwds_trf = kwds_expl.merge(trf, left_on="domains", right_on="url", how="left")
        agg_trf = aggregate_traffic(kwds_trf, by="keyword")

        return kwds.merge(agg_trf, on="keyword", how="left")
    except Exception as exc:
        LOG.warning(
            f"Failed to fetch traffic data for keywords: {exc}. "
            "Will return original DataFrame without traffic data."
        )
        return kwds
