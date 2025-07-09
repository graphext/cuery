"""Fetch SERP results using Apify actors."""

import asyncio
import json
import os
import re
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import pandas as pd
from apify_client import ApifyClientAsync
from async_lru import alru_cache
from pandas import DataFrame, NamedAgg, Series

from ..utils import LOG
from .tasks import EntityExtractor, SerpTopicAndIntentAssigner, SerpTopicExtractor


@alru_cache(maxsize=3)
async def fetch_serps(
    keywords: tuple[str, ...],
    batch_size: int = 100,
    apify_token: str | Path | None = None,
    **kwargs,
):
    """Fetch SERP data for a list of keywords using the Apify Google Search Scraper actor."""
    if isinstance(apify_token, str | Path):
        with open(apify_token) as f:
            token = f.read().strip()
    else:
        token = os.environ["APIFY_TOKEN"]

    client = ApifyClientAsync(token)

    keywords_list = list(keywords)
    keyword_batches = [
        keywords_list[i : i + batch_size] for i in range(0, len(keywords_list), batch_size)
    ]

    async def process_batch(batch):
        """Process a single batch of keywords."""
        run_input = {"queries": "\n".join(batch), **kwargs}

        actor = client.actor("apify/google-search-scraper")
        run = await actor.call(run_input=run_input)
        if run is None:
            LOG.error(f"Actor run failed for batch: {batch}... ")
            return None

        dataset_client = client.dataset(run["defaultDatasetId"])
        return await dataset_client.list_items()

    tasks = [process_batch(batch) for batch in keyword_batches]
    batch_results = await asyncio.gather(*tasks)

    result = []
    for batch_result in batch_results:
        if batch_result is not None:
            result.extend(batch_result.items)

    return result


def process_toplevel_keys(row: dict):
    """Process top-level keys in a SERP result row (single keyword)."""
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

    agg_funcs = {k: v for k, v in agg_funcs.items() if v.column in df.columns}

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


def token_rank(tokens: str | list[str], texts: list[str] | None) -> int | None:
    """Find position of first occurrence of a token in a list of texts."""
    if isinstance(texts, list):
        if isinstance(tokens, str):
            tokens = [tokens]

        for i, text in enumerate(texts):
            if any(token.lower() in text.lower() for token in tokens):
                return i + 1

    return None


def add_ranks(
    df: DataFrame,
    brands: str | list[str] | None,
    competitors: str | list[str] | None,
) -> DataFrame:
    """Calculate brand and competitor ranks in organic search results."""
    if brands is not None:
        df["title_rank_brand"] = df.titles.apply(lambda x: token_rank(brands, x))
        df["domain_rank_brand"] = df.domains.apply(lambda x: token_rank(brands, x))
        df["description_rank_brand"] = df.descriptions.apply(lambda x: token_rank(brands, x))

    if competitors is not None:
        # First position of any(!) competitor
        df["title_rank_competition"] = df.titles.apply(lambda x: token_rank(competitors, x))
        df["description_rank_competition"] = df.descriptions.apply(
            lambda x: token_rank(competitors, x)
        )
        df["domain_rank_competition"] = df.domains.apply(lambda x: token_rank(competitors, x))

        # Specific ranks for each individual competitor
        for name in competitors:
            c_ranks = []
            for col in ("titles", "descriptions", "domains"):
                rank = df[col].apply(lambda x, name=name: token_rank(name, x))
                c_ranks.append(rank)

            c_ranks = pd.concat(c_ranks, axis=1)
            df[f"min_rank_{name}"] = c_ranks.min(axis=1)

    return df


async def topic_and_intent(
    df: DataFrame,
    max_samples: int,
    topic_model: str,
    assignment_model: str,
    max_retries: int = 5,
) -> DataFrame | None:
    """Classify keywords and their top N organic results into topics and intent."""
    n_samples_max = min(max_samples, len(df))

    try:
        extractor = SerpTopicExtractor()
        topic_intent = await extractor(
            df=df.sample(n=n_samples_max),
            model=topic_model,
            max_retries=max_retries,
        )
        LOG.info("Extracted topic hierarchy")
        LOG.info(json.dumps(topic_intent.to_dict(), indent=2, ensure_ascii=False))

        assigner = SerpTopicAndIntentAssigner(topic_intent)
        classified = await assigner(df=df, model=assignment_model, n_concurrent=100)
        clf = classified.to_pandas()
        return clf[["term", "topic", "subtopic", "intent"]]
    except Exception as exc:
        LOG.error(f"Error during topic and intent extraction: {exc}")
        LOG.exception("Stack trace:")
        return None


async def process_ai_overviews(
    df: DataFrame,
    entity_model: str = "openai/gpt-4.1-mini",
) -> DataFrame | None:
    """Process AI overviews in SERP data and extract entities."""
    if "aiOverview_content" in df.columns and df["aiOverview_content"].notna().any():
        try:
            # Todo: extract brand and competitor ranks
            ai_df = df[df.aiOverview_content.notna()].copy().reset_index()
            entity_extractor = EntityExtractor()
            entities = await entity_extractor(df=ai_df, model=entity_model, n_concurrent=100)
            ent_df = entities.to_pandas(explode=False)

            for kind in ("brand/company", "product/service", "technology"):
                ent_df[f"ai_overview_{kind}"] = ent_df.entities.apply(
                    lambda es, kind=kind: [
                        e.name
                        for e in es
                        if e is not None
                        and hasattr(e, "type")
                        and hasattr(e, "name")
                        and e.type == kind
                    ]
                    if es is not None
                    else None
                )

            ent_df["term"] = ai_df["term"]
            return ent_df.drop(
                columns=["aiOverview_content", "aiOverview_source_titles", "entities"]
            )
        except Exception as exc:
            LOG.error(f"Error processing AI overviews: {exc}")
            return None

    return None


def mentioned_in_string(
    words: str | list[str],
    text: str | None,
    whole_word: bool = True,
) -> list[str] | None:
    """Check if the brand is mentioned in the text."""
    if not text:
        return None

    if isinstance(words, str):
        words = [words]

    if whole_word:
        return [
            word for word in words if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
        ]

    return [word for word in words if word.lower() in text.lower()]


def mentioned_in_list(
    words: str | list[str],
    texts: list[str] | None,
    whole_word: bool = True,
) -> list[str] | None:
    """Check if the brand is mentioned in any of the strings in the list."""
    if not texts:
        return None

    mentions = []
    for txt in texts:
        if txt_words := mentioned_in_string(words, txt, whole_word):
            mentions.extend(txt_words)

    return mentions if mentions else None


def mentioned_brands(
    ser: Series,
    brands: str | list[str],
    whole_word: bool = True,
) -> Series:
    valid_idx = ser.first_valid_index()
    if valid_idx is None:
        return Series([False] * len(ser), index=ser.index)

    someval = ser[valid_idx]
    if isinstance(someval, str):
        return ser.apply(lambda txt: mentioned_in_string(brands, txt, whole_word))

    if isinstance(someval, list):
        return ser.apply(lambda lst: mentioned_in_list(brands, lst, whole_word))

    raise TypeError(f"Unsupported type {type(someval)} in Series.")


def add_brand_mentions(
    df,
    brands: str | list[str] | None = None,
    competitors: str | list[str] | None = None,
    whole_word: bool = True,
) -> DataFrame:
    """Process brand mentions in SERP data."""
    if "aiOverview_content" in df.columns:
        if brands:
            df["aiOverview_brand_mentions"] = mentioned_brands(
                df["aiOverview_content"],
                brands=brands,
                whole_word=whole_word,
            )
        if competitors:
            df["aiOverview_competitor_mentions"] = mentioned_brands(
                df["aiOverview_content"],
                brands=competitors,
                whole_word=whole_word,
            )

    if "aiOverview_source_titles" in df.columns:
        if brands:
            df["aiOverview_source_mentions"] = mentioned_brands(
                df["aiOverview_source_titles"],
                brands=brands,
                whole_word=True,
            )
        if competitors:
            df["aiOverview_source_competitor_mentions"] = mentioned_brands(
                df["aiOverview_source_titles"],
                brands=competitors,
                whole_word=True,
            )

    return df
