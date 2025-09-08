"""SERP data collection and analysis using Apify web scraping actors.

This module provides comprehensive tools for fetching and analyzing Search Engine Results
Page (SERP) data through Apify's Google Search Scraper actors. It enables large-scale
SERP data collection with features like batch processing, geographic targeting, and
intelligent result aggregation. The module also integrates AI-powered topic extraction
and search intent classification to provide deeper insights into SERP content patterns.

Key capabilities include fetching organic search results with metadata (titles, URLs,
snippets), identifying brand and competitor presence in SERPs, extracting topics and
search intent using language models, and aggregating results for keyword analysis.
The module handles rate limiting, error recovery, and data normalization to ensure
reliable SERP data collection at scale.
"""

import asyncio
import json
import os
import re
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from apify_client import ApifyClientAsync
from async_lru import alru_cache
from pandas import DataFrame, NamedAgg, Series
from pydantic import Field

from ..tools.flex.classify import Classifier
from ..tools.flex.entities import EntityExtractor
from ..tools.flex.topics import TopicAssigner, TopicExtractor
from ..utils import LOG, Configurable, dedent

TOPIC_INSTRUCTIONS = dedent("""
Data records represent Google search keywords and associated SERP data.
Make sure to create topics relevant in the context of SEO keywords research,
focusing on the semantic meaning of keywords and SERPS, commercial intent etc.
""")

INTENT_INSTRUCTIONS = dedent("""
Data records represent Google search keywords and associated SERP data.
Categorize their search intent in the context of a SEO keywords analysis,
focusing on the semantic meaning and intent.
""")

INTENTS = {
    "informational": "The search intent is to find information or learn about a topic.",
    "navigational": "The search intent is to navigate to a specific website or page.",
    "transactional": "The search intent is to make a purchase or complete a transaction.",
    "commercial": "The search intent is to compare products or services before making a decision.",
    "local": "The search intent is to find local businesses or services in a specific area.",
}

ENTITY_INSTRUCTIONS = dedent("""
Data records represent AI Overviews from SERP data associated with keyword searches.
Focus on extracting entities that are relevant to SEO analysis.
""")

ENTITIES = {
    "brand_company": "A brand or company name, typically a well-known entity in the market.",
    "product_service": "A specific product or service offered by a brand or company.",
    "technology": "A technology or platform related to the product or service.",
    "other": "Any other relevant entity that does not fit into the above categories.",
}


class SerpConfig(Configurable):
    """Configuration for fetching SERP data using Apify Google Search Scraper actor."""

    keywords: tuple[str, ...] | None = None
    """Keywords to fetch SERP data for. If None, pass keywords manually in calling functions."""
    batch_size: int = 100
    """Number of keywords to fetch in a single batch."""
    resultsPerPage: int = 100
    """Number of results to fetch per page."""
    maxPagesPerQuery: int = 1
    """Maximum number of pages to fetch per query."""
    countryCode: str = "us"
    """Specifies the country used for the search and the Google Search domain (e.g. google.es for
    Spain). By default, the actor uses United States (google.com).
    """
    searchLanguage: str = ""
    """Restricts search results to pages in a specific language. For example, choosing 'German'
    results in pages only in German. Passed to Google Search as the lr URL query parameter.
    """
    languageCode: str = ""
    """Language of the Google Search interface, not the search results themselves. Passed to
    Google Search as the hl URL query parameter.
    """
    params: dict[str, Any] | None = Field(default_factory=dict)
    """Additional parameters to pass to the Apify actor."""
    top_n: int = 10
    """Number of top organic results to consider for aggregation per keyword."""
    brands: str | list[str] | None = None
    """List of brand names to identify in SERP data."""
    competitors: str | list[str] | None = None
    """List of competitor names to identify in SERP data."""
    topic_max_samples: int = 500
    """Maximum number of samples to use for topic and intent extraction from SERP data."""
    topic_model: str | None = "google/gemini-2.5-flash-preview-05-20"
    """Model to use for topic extraction from SERP organic results."""
    topic_min_ldist: int = 2
    """Minimum Levenshtein distance between topic labels."""
    assignment_model: str | None = "openai/gpt-4.1-mini"
    """Model to use for intent classification from SERP organic results."""
    entity_model: str | None = "openai/gpt-4.1-mini"
    """Model to use for entity extraction from AI overviews."""
    apify_token: str | Path | None = None
    """Path to Apify API token file.
    If not provided, will use the `APIFY_TOKEN` environment variable.
    """


async def fetch_batch(keywords: list[str], client: ApifyClientAsync, **params):
    """Process a single batch of keywords."""
    run_input = {"queries": "\n".join(keywords), **params}
    actor = client.actor("apify/google-search-scraper")
    run = await actor.call(run_input=run_input)
    if run is None:
        LOG.error(f"Actor run failed for batch: {keywords}... ")
        return None

    dataset_client = client.dataset(run["defaultDatasetId"])
    return await dataset_client.list_items()


@alru_cache(maxsize=3)
async def fetch_serps(
    cfg: SerpConfig,
    keywords: tuple[str, ...] | None = None,
) -> list[dict]:
    """Fetch SERP data for a list of keywords using the Apify Google Search Scraper actor."""
    if isinstance(cfg.apify_token, str | Path):
        with open(cfg.apify_token) as f:
            token = f.read().strip()
    else:
        token = os.environ["APIFY_TOKEN"]

    actor_param_names = (
        "resultsPerPage",
        "maxPagesPerQuery",
        "countryCode",
        "searchLanguage",
        "languageCode",
    )
    actor_params = {p: getattr(cfg, p) for p in actor_param_names} | (cfg.params or {})

    if cfg.keywords and keywords:
        LOG.warning("Both cfg.keywords and keywords are provided, using cfg.keywords.")

    keywords = cfg.keywords or keywords
    if not keywords:
        raise ValueError("No keywords provided for SERP data fetching!")

    keywords_list = list(keywords)
    LOG.info(f"Fetching SERP data for {len(keywords)} keywords")

    keyword_batches = [
        keywords_list[i : i + cfg.batch_size] for i in range(0, len(keywords_list), cfg.batch_size)
    ]

    client = ApifyClientAsync(token)
    tasks = [fetch_batch(batch, client, **actor_params) for batch in keyword_batches]
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
        "aio_content": aio.get("content", None),
        "aio_source_titles": [s["title"] for s in aio.get("sources", [])] or None,
        "aio_source_descs": [s["description"] for s in aio.get("sources", [])] or None,
        "aio_source_urls": [s["url"] for s in aio.get("sources", [])] or None,
    }
    items["aio_sources"] = [
        s["title"] + " – " + s["description"] + " " + s["url"] for s in aio.get("sources", [])
    ] or None

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


def serps_to_pandas(serps, copy=True) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
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


def unique(lst: list) -> list:
    """Return a list of unique elements, preserving order."""
    return list(dict.fromkeys(lst))


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
        "urls": NamedAgg("url", list),
        "descriptions": NamedAgg("description", list),
        "domains": NamedAgg("domain", lambda ser: list),
        "breadcrumbs": NamedAgg("breadcrumb", lambda ser: unique(flatten(ser))),
        "emphasizedKeywords": NamedAgg("emphasizedKeywords", lambda ser: unique(flatten(ser))),
    }

    agg = df.groupby("term").agg(**agg_funcs).reset_index()  # type: ignore[call-overload]

    top = df.groupby("term").head(top_n)
    topagg = top.groupby("term").agg(**top_agg_funcs).reset_index()  # type: ignore[call-overload]

    return agg.merge(topagg, on="term", how="left")


def token_rank(
    tokens: str | list[str],
    texts: list[str] | None,
    whole_word: bool = True,
) -> int | None:
    """Find position of first occurrence of a token in a list of texts."""
    if whole_word:
        has_substr = lambda text, word: re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)  # noqa: E731
    else:
        has_substr = lambda text, word: word.lower() in text.lower()  # noqa: E731

    if isinstance(texts, list):
        if isinstance(tokens, str):
            tokens = [tokens]

        for i, text in enumerate(texts):
            if any(has_substr(text, token) for token in tokens):
                return i + 1

    return None


def add_ranks(
    df: DataFrame,
    brands: str | list[str] | None,
    competitors: str | list[str] | None,
    columns: tuple[str, ...] | list[str] = ("titles", "domains", "descriptions"),
) -> DataFrame:
    """Calculate brand and competitor ranks in organic search results."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        LOG.warning(f"Columns {missing} not found in DataFrame! Will not add ranks for these.")
        columns = [col for col in columns if col not in missing]

    if brands is not None:
        for col in columns:
            df[f"{col}_rank_brand"] = df[col].apply(lambda x: token_rank(brands, x))

    if competitors is not None:
        # First position of any(!) competitor
        for col in columns:
            df[f"{col}_rank_competition"] = df[col].apply(lambda x: token_rank(competitors, x))

        # Specific ranks for each individual competitor
        for name in competitors:
            c_ranks = []
            for col in columns:
                rank = df[col].apply(lambda x, name=name: token_rank(name, x))
                c_ranks.append(rank)

            c_ranks = pd.concat(c_ranks, axis=1)
            df[f"min_rank_{name}"] = c_ranks.min(axis=1)

    return df


async def topic_and_intent(  # noqa: PLR0913
    df: DataFrame,
    max_samples: int,
    topic_model: str = "google/gemini-2.5-flash-preview-05-20",
    assignment_model: str = "openai/gpt-4.1-mini",
    max_retries: int = 5,
    text_column: str = "term",
    extra_columns: list[str] | None = None,
    topics_instructions: str = "",
    min_ldist: int = 2,
) -> DataFrame | None:
    """Classify keywords and their top N organic results into topics and intent."""
    columns = [text_column] + (extra_columns or ["titles", "domains", "breadcrumbs"])
    topic_instructions = TOPIC_INSTRUCTIONS + f"\n\n{topics_instructions}".strip()

    try:
        extractor = TopicExtractor(
            model=topic_model,
            records=df,
            attrs=columns,
            n_topics=10,
            n_subtopics=5,
            instructions=topic_instructions,
            min_ldist=min_ldist,
            max_samples=max_samples,
        )
        topics = await extractor(max_retries=max_retries)
        LOG.info("Extracted topic hierarchy")
        LOG.info(json.dumps(topics.to_dict(), indent=2, ensure_ascii=False))

        assigner = TopicAssigner(
            model=assignment_model,
            records=df,
            attrs=columns,
            topics=topics,
        )
        term_topics = await assigner(n_concurrent=100)
        term_topics = term_topics[[text_column, "topic", "subtopic"]]

        classifier = Classifier(
            model=assignment_model,
            records=df,
            attrs=columns,
            categories=INTENTS,
            instructions=INTENT_INSTRUCTIONS,
        )
        term_intents = await classifier(n_concurrent=100)
        term_intents = term_intents[[text_column, "label"]]
        term_intents = term_intents.rename(columns={"label": "intent"})

        return term_topics.merge(term_intents, on=text_column, how="left")
    except Exception as exc:
        LOG.error(f"Error during topic and intent extraction: {exc}")
        LOG.exception("Stack trace:")
        return None


async def extract_aio_entities(
    df: DataFrame,
    entity_model: str = "openai/gpt-4.1-mini",
    id_column: str = "term",
) -> DataFrame | None:
    """Process AI overviews in SERP data and extract entities."""
    if ("aio_content" not in df.columns) or df["aio_content"].isna().all():
        LOG.warning("No AI overview content found in DataFrame, skipping entity extraction.")
        return None

    aio_columns = [
        "aio_content",
        "aio_source_titles",
        "aio_source_descs",
        "aio_source_urls",
    ]
    aio_columns = [col for col in aio_columns if col in df.columns]
    LOG.info(f"Extracting entities from columns: {aio_columns}")

    try:
        # Todo: extract brand and competitor ranks
        ai_df = df[df.aio_content.notna()].copy().reset_index()
        extractor = EntityExtractor(
            model=entity_model,
            records=ai_df,
            attrs=aio_columns,
            entities=ENTITIES,
        )
        ent_df = await extractor(n_concurrent=100)

        ent_df = ent_df[[col for col in ent_df.columns if col.startswith("entities_")]]
        ent_df.columns = [f"aio_{col}" for col in ent_df.columns]
        ent_df[id_column] = ai_df[id_column]
        return ent_df
    except Exception as exc:
        LOG.error(f"Error processing AI overviews: {exc}")
        return None


def mentioned_in_string(
    words: str | list[str],
    text: str | None,
    whole_word: bool = True,
) -> list[str] | None:
    """Return those words (brands) that are mentioned in the text."""
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
    """Identify brand mentions in AI overviews using regex."""
    for col in ("aio_content", "aio_sources"):
        if col in df:
            for tag, names in zip(("brand", "competitor"), (brands, competitors), strict=True):
                if names:
                    df[f"{col}_{tag}_mentions"] = mentioned_brands(
                        df[col],
                        brands=names,
                        whole_word=whole_word,
                    )
    return df


async def process_serps(
    response,
    cfg: SerpConfig,
    copy: bool = True,
) -> DataFrame:
    """Process SERP results and return dataframes for features, organic, paid, and ads."""

    # Separate SERP sections into DataFrames
    features, org, paid, ads = serps_to_pandas(response, copy=copy)

    # Process organic results
    orgagg = aggregate_organic_results(org, top_n=cfg.top_n)
    if cfg.brands or cfg.competitors:
        LOG.info("Identifying brand and competitor ranks in organic results")
        orgagg = add_ranks(orgagg, brands=cfg.brands, competitors=cfg.competitors)

    df = features.merge(orgagg, on="term", how="left")

    if cfg.brands or cfg.competitors:
        LOG.info("Identifying brand and competitor mentions in AI overviews")
        df = add_brand_mentions(df, brands=cfg.brands, competitors=cfg.competitors)

    # AI topics and intent
    if cfg.topic_model is not None and cfg.assignment_model is not None:
        LOG.info("Extracting topics and intents from keywords and top organic results")
        topics = await topic_and_intent(
            orgagg,
            max_samples=cfg.topic_max_samples,
            topic_model=cfg.topic_model,
            assignment_model=cfg.assignment_model,
            max_retries=6,
            text_column="term",
            extra_columns=["titles", "descriptions"],
        )
        if topics is not None:
            df = df.merge(topics, on="term", how="left")

    # AI overview entities
    if cfg.entity_model is not None:
        LOG.info("Extracting entities from AI overviews")
        entities = await extract_aio_entities(features, entity_model=cfg.entity_model)
        if entities is not None:
            df = df.merge(entities, on="term", how="left")

    return df


async def serps(cfg: SerpConfig, keywords: Iterable[str] | None = None) -> DataFrame | None:
    """Fetch and process SERP data for a list of keywords."""
    LOG.info(f"Fetching SERP data with config:\n{cfg}")
    keywords = tuple(keywords) if keywords is not None else None
    response = await fetch_serps(cfg=cfg, keywords=keywords)

    if response is None or len(response) == 0:
        LOG.warning("No SERP results found.")
        return None

    LOG.info("Processing SERP data")
    df = await process_serps(response, cfg)
    LOG.info(f"Got SERPs dataframe:\n{df}")
    return df
