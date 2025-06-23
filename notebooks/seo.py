import json
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
from apify_client import ApifyClientAsync
from google.ads.googleads.client import GoogleAdsClient
from pandas import DataFrame, NamedAgg

LANGUAGE = "1003"  # Español
GEO_TARGET = "2724"  # España
CUSTOMER = "6560490700"


def ads_client(config_path: str | Path):
    return GoogleAdsClient.load_from_storage(config_path)


def fetch_keywords(
    ads_config: str | Path,
    keywords: list[str],
    customer: str = CUSTOMER,
    language: str = LANGUAGE,
    geo_target: str = GEO_TARGET,
    metrics_start: str = "2023-03",
    metrics_end: str = "2024-02",
) -> Iterable:
    client = GoogleAdsClient.load_from_storage(ads_config)

    kwd_service = client.get_service("KeywordPlanIdeaService")
    ads_service = client.get_service("GoogleAdsService")

    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer

    # Configurar palabras clave de semilla
    keyword_seed = client.get_type("KeywordSeed")
    keyword_seed.keywords.extend(keywords)
    request.keyword_seed = keyword_seed

    # Añadir idioma
    request.language = ads_service.language_constant_path(language)

    # Añadir ubicación geográfica
    request.geo_target_constants.append(ads_service.geo_target_constant_path(geo_target))

    # Configurar métricas históricas
    start = datetime.strptime(metrics_start, "%Y-%m")
    end = datetime.strptime(metrics_end, "%Y-%m")
    request.historical_metrics_options.year_month_range.start.year = start.year
    request.historical_metrics_options.year_month_range.start.month = start.month
    request.historical_metrics_options.year_month_range.end.year = end.year
    request.historical_metrics_options.year_month_range.end.month = end.month
    request.historical_metrics_options.include_average_cpc = True

    return kwd_service.generate_keyword_ideas(request=request)


def process_keywords(response: Iterable) -> DataFrame:
    keywords = []
    for idea in response.results:
        info = {
            "keyword": idea.text,
        }

        if metrics := getattr(idea, "keyword_idea_metrics", None):
            info["avg_monthly_searches"] = getattr(metrics, "avg_monthly_searches", None)

            if avg_cpc := getattr(metrics, "average_cpc_micros", None):
                info["average_cpc"] = round(avg_cpc / 1_000_000, 2)

            if competition := getattr(metrics, "competition", None):
                competition_map = {"LOW": 33, "MEDIUM": 66, "HIGH": 100}
                info["competition_score"] = competition
                info["competition"] = competition_map.get(metrics.competition.name, None)

            if volumes := getattr(metrics, "monthly_search_volumes", None):
                for volume in volumes:
                    year = volume.year
                    month = volume.month
                    date = datetime.strptime(f"{year}-{month.name.capitalize()}", "%Y-%B")
                    volume = volume.monthly_searches
                    info[f"search_volume_{date.year}_{date.month:02}"] = volume

        keywords.append(info)

    return pd.DataFrame(keywords)


async def fetch_serps(token_path: str | Path, keywords: Iterable[str], **kwargs):
    """Fetch SERP data for a list of keywords using the Apify Google Search Scraper actor.

    For parameters see: https://apify.com/apify/google-search-scraper/input-schema

    TODO: concurrency

        # Split keywords into batches
        keyword_batches = [
            ["keyword1", "keyword2"],
            ["keyword3", "keyword4"],
            ["keyword5", "keyword6"]
        ]

        # Launch concurrent tasks
        tasks = [
            client.actor("actor-id").call(run_input={"keywords": batch})
            for batch in keyword_batches
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        return results
    """
    with open(token_path) as f:
        token = f.read().strip()

    client = ApifyClientAsync(token)

    run_input = {"queries": "\n".join(keywords), **kwargs}

    run = await client.actor("apify/google-search-scraper").call(run_input=run_input)
    if run is None:
        raise Exception("Actor run failed. Check the Apify console for details.")

    dataset_client = client.dataset(run["defaultDatasetId"])
    return await dataset_client.list_items()


def process_toplevel_keys(row: dict):
    rm = [
        "#debug",
        "#error",
        "htmlSnapshotUrl",
        "url",
        "hasNextPage",
        "resultsTotal",
        "serpProviderCode",
        "customData",
        "suggestedResults",
    ]
    for k in rm:
        if k in row:
            del row[k]


def process_search_query(row: dict):
    """Everything here except the term is as originally configured in Apify."""
    keep = ["term"]
    sq = row.pop("searchQuery", {})
    row.update(**{k: sq[k] for k in keep if k in sq})


def process_related_queries(row: dict):
    """Only keep titles for now, we don't need the corresponding url."""
    rq = row.pop("relatedQueries", [])
    rq = [q["title"] for q in rq]
    row["relatedQueries"] = rq


def process_also_asked(row: dict):
    """Only keep question for now, e.g. to extend original keywords."""
    paa = row.pop("peopleAlsoAsk", [])
    paa = [q["question"] for q in paa]
    row["peopleAlsoAsk"] = paa


def process_ai_overview(row: dict):
    """Keep only content and source titles."""
    aio = row.pop("aiOverview", {})
    items = {
        "aiOverview_content": aio.get("content", None),
        "aiOverview_source_titles": [s["title"] for s in aio.get("sources", [])] or None,
    }
    row.update(**items)


def parse_displayed_url(url: str) -> tuple[str, list[str] | None]:
    """Parse the displayed URL into domain and breadcrumb."""
    parts = [part.strip() for part in url.split("›")]
    domain = parts[0]
    breadcrumb = [part for part in parts[1:] if part != "..."] if len(parts) > 1 else None
    return domain, breadcrumb


def extract_organic_results(data: list[dict]) -> list[dict]:
    """Extract organic results and return as a list of dictionaries."""
    results = []
    for row in data:
        ores = row.pop("organicResults", [])
        for res in ores:
            domain, breadcrumb = parse_displayed_url(res.pop("displayedUrl", ""))

            drop = [
                "siteLinks",  # seems to be present only in paid results
                "productInfo",  # probably present only in paid products
            ]
            for k in drop:
                res.pop(k, None)

            results.append({"term": row["term"], "domain": domain, "breadcrumb": breadcrumb} | res)

    return results


def extract_paid_results(data: list[dict]) -> list[dict]:
    """Extract organic results and return as a list of dictionaries."""
    results = []
    for row in data:
        pres = row.pop("paidResults", [])
        row["n_paidResults"] = len(pres)  # Add count of paid results
        for res in pres:
            results.append({"term": row["term"]} | res)

    return results


def extract_paid_products(data: list[dict]) -> list[dict]:
    """Extract organic results and return as a list of dictionaries."""
    results = []
    for row in data:
        prods = row.pop("paidProducts", [])
        row["n_paidProducts"] = len(prods)
        for res in prods:
            results.append({"term": row["term"]} | res)

    return results


def process_serps(serps, copy=True) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    if not isinstance(serps, list):
        serps = serps.items

    pages = [deepcopy(page) for page in serps] if copy else serps

    # Process these in place to save memory
    for page in pages:
        process_toplevel_keys(page)
        process_search_query(page)
        process_related_queries(page)
        process_also_asked(page)
        process_ai_overview(page)

    org_res = extract_organic_results(pages)
    paid_res = extract_paid_results(pages)
    paid_prods = extract_paid_products(pages)

    pages = DataFrame(pages)
    return DataFrame(pages), DataFrame(org_res), DataFrame(paid_res), DataFrame(paid_prods)


def flatten(lists: Iterable[list | None]) -> list:
    """Flatten list of lists into a single list, elements can be None."""
    return [
        item for sublist in lists if sublist is not None for item in sublist if item is not None
    ]


def aggregate_organic_results(df: DataFrame, top_n=10) -> DataFrame:
    """Aggregate organic results by term and apply aggregation functions."""

    def num_notna(ser):
        return ser.notna().sum()

    # These apply to all results
    agg_funcs = {
        "num_results": NamedAgg("title", "count"),
        "num_has_date": NamedAgg("date", lambda ser: num_notna(ser)),
        "num_has_views": NamedAgg("views", lambda ser: num_notna(ser)),
        "num_has_ratings": NamedAgg("averageRating", lambda ser: num_notna(ser)),
        "num_has_reviews": NamedAgg("numberOfReviews", lambda ser: num_notna(ser)),
        "num_has_comments": NamedAgg("commentsAmount", lambda ser: num_notna(ser)),
        "num_has_reactions": NamedAgg("reactions", lambda ser: num_notna(ser)),
        "num_has_channel": NamedAgg("channelName", lambda ser: num_notna(ser)),
        "num_has_reel": NamedAgg("reelLength", lambda ser: num_notna(ser)),
        "num_has_followers": NamedAgg("followersAmount", lambda ser: num_notna(ser)),
        "num_has_personal_info": NamedAgg("personalInfo", lambda ser: num_notna(ser)),
        "num_has_tweet": NamedAgg("tweetCards", lambda ser: num_notna(ser)),
    }

    # These apply to only the top N results
    top_agg_funcs = {
        "titles": NamedAgg("title", list),
        "descriptions": NamedAgg("description", list),
        "domains": NamedAgg("domain", lambda ser: list(set(ser))),
        "breadcrumbs": NamedAgg("breadcrumb", lambda ser: list(set(flatten(ser)))),
        "emphasizedKeywords": NamedAgg("emphasizedKeywords", lambda ser: list(set(flatten(ser)))),
    }

    agg = df.groupby("term").agg(**agg_funcs).reset_index()

    top = df.groupby("term").head(top_n)
    topagg = top.groupby("term").agg(**top_agg_funcs).reset_index()

    return agg.merge(topagg, on="term", how="left")
