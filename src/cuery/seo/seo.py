"""High-level API to fetch SEO data from various sources and to enrich with AI."""

from pathlib import Path
from warnings import warn

from pandas import DataFrame
from pydantic import BaseModel, Field

from ..utils import LOG
from .keywords import fetch_keywords, process_keywords
from .serps import (
    add_ranks,
    aggregate_organic_results,
    fetch_serps,
    process_ai_overviews,
    process_serps,
    topic_and_intent,
)


class HashableCfg(BaseModel):
    def __hash__(self) -> int:
        return self.model_dump_json().__hash__()


class GoogleKwdConfig(HashableCfg):
    """Configuration for Google Ads API access."""

    credentials: str | Path
    customer: str
    keywords: tuple[str, ...]
    ideas: bool = False
    max_ideas: int | None = None
    language: str = "en"
    geo_target: str = "us"
    metrics_start: str | None = None
    metrics_end: str | None = None


class SerpConfig(HashableCfg):
    token_path: str | Path
    batch_size: int = 100
    resultsPerPage: int = 100
    maxPagesPerQuery: int = 1
    country: str | None = None
    searchLanguage: str | None = None
    languageCode: str | None = None
    params: dict | None = Field(default_factory=dict)


class SeoConfig(HashableCfg):
    """Configuration for keyword extraction and assignment."""

    kwd_cfg: GoogleKwdConfig
    serp_cfg: SerpConfig | None = None
    brands: str | list[str] | None = None
    competitors: str | list[str] | None = None
    topic_max_samples: int = 500
    topic_model: str | None = "google/gemini-2.5-flash-preview-05-20"
    assignment_model: str | None = "openai/gpt-4.1-mini"
    entity_model: str | None = "openai/gpt-4.1-mini"


async def fetch_data(cfg: SeoConfig) -> DataFrame:
    """Fetch all supported SEO data types for a given set of keywords."""

    LOG.info("Fetching and processing keywords from Google Ads API")
    kwd_cfg = cfg.kwd_cfg.model_dump()
    kwds = fetch_keywords(**kwd_cfg)
    df = process_keywords(kwds, collect_volumes=True)

    LOG.info("Fetching and processing SERP data")
    if cfg.serp_cfg is not None:
        serp_cfg = cfg.serp_cfg.model_dump()
        serp_params = serp_cfg.pop("params", {})
        serps = await fetch_serps(keywords=tuple(df.keyword), **serp_cfg, **serp_params)

        features, org, paid, ads = process_serps(serps, copy=True)

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
            )
            df = df.merge(clf_df, on="term", how="left")

        if cfg.entity_model is not None:
            LOG.info("Processing AI overviews from SERP data")
            ai_df = await process_ai_overviews(features, entity_model=cfg.entity_model)
            if ai_df is not None:
                df = df.merge(ai_df, on="term", how="left")

    # Todo: add traffic data ...

    return df
