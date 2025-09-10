"""Google Ads API integration for comprehensive keyword research and analysis.

This module provides a streamlined interface to the Google Ads API for keyword planning
and research. It enables users to generate keyword ideas, retrieve historical search
volume data, and analyze keyword performance metrics across different geographic regions
and time periods. The module handles authentication, batching, and data processing to
deliver clean, structured keyword data for SEO and content strategy development.

Key features include keyword idea generation from seed keywords or landing pages,
historical metrics retrieval with monthly breakdowns, geographic and language targeting,
and automated data cleaning and aggregation for analysis workflows.

Useful documentation:
    - Keyword ideas:
        - https://developers.google.com/google-ads/api/docs/keyword-planning/generate-keyword-ideas
        - https://developers.google.com/google-ads/api/samples/generate-keyword-ideas
        - https://developers.google.com/google-ads/api/reference/rpc/v20/GenerateKeywordIdeasRequest
    - Historical metrics:
        - https://developers.google.com/google-ads/api/docs/keyword-planning/generate-historical-metrics
        - https://developers.google.com/google-ads/api/reference/rpc/v20/GenerateKeywordHistoricalMetricsRequest
    - ID/Code references:
        - https://developers.google.com/google-ads/api/data/codes-formats#expandable-7
        - https://developers.google.com/google-ads/api/data/geotargets

"""

import base64
import json
import os
import tempfile
import urllib.parse
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.v20.enums.types import MonthOfYearEnum
from google.ads.googleads.v20.services import (
    GenerateKeywordHistoricalMetricsRequest,
    GenerateKeywordHistoricalMetricsResponse,
    GenerateKeywordIdeasRequest,
)
from google.ads.googleads.v20.services.services.keyword_plan_idea_service.pagers import (
    GenerateKeywordIdeasPager,
)
from numpy import ndarray
from pandas import DataFrame, Series
from pydantic import Field, ValidationInfo, field_validator, model_validator

from .. import resources, utils
from ..context import AnyContext
from ..prompt import Prompt
from ..response import Response, ResponseSet
from ..task import Task
from ..utils import LOG, Configurable, dedent


class GoogleKwdConfig(Configurable):
    """Configuration for Google Ads API access."""

    keywords: tuple[str, ...] | None = None
    """The (initial) keywords to fetch data for.
    When generating keyword ideas, only the first 20 keywords will be used.
    Will be ignored in whole-site mode.
    """
    url: str | None = None
    """The page to fetch data for (if applicable). For whole-site mode,
    provide a pure domain URL (e.g., 'example.com').
    """
    whole_site: bool = False
    """Whether to fetch keyword ideas for the whole site (if `url` is provided)."""
    ideas: bool = False
    """Whether to expand initial keywords with Google Keyword Planner's idea generator.
    Otherwise, will fetch historical metrics for the provided keywords only.
    """
    max_ideas: int = Field(100, le=10_000)
    """Maximum number of additional keyword ideas to fetch (if `ideas` is True)."""
    language: str = ""
    """The resource name of the language to target. Each keyword belongs to some set of languages;
    a keyword is included if language is one of its languages. If not set, all keywords will be
    included. (e.g., 'en' for English).
    """
    country: str = ""
    """The geographical target for keyword data (e.g., 'us' for United States). If not set,
    keywords from all locations will be included."""
    metrics_start: str | None = None
    """Start date (year and month) for metrics in YYYY-MM format (e.g., '2023-01').
    Either provide both `metrics_start` and `metrics_end` or neither.
    """
    metrics_end: str | None = None
    """End date (year and month) for metrics in YYYY-MM format (e.g., '2023-12').
    Either provide both `metrics_start` and `metrics_end` or neither.
    """
    credentials: str | Path | dict | None = None
    """Path to Google Ads API credentials file or a dictionary with credentials.
    If not provided, will look for environment variables with prefix `GOOGLE_ADS_`.
    """
    customer: str | None = None
    """Google Ads customer ID to use for API requests.
    If not provided, will use the `GOOGLE_ADS_CUSTOMER_ID` or `GOOGLE_ADS_LOGIN_CUSTOMER_ID`
    environment variable.
    """

    @field_validator("ideas")
    @classmethod
    def validate_ideas(cls, ideas, info: ValidationInfo):
        if not ideas and not info.data["keywords"] and info.data["url"]:
            LOG.warning(
                "Idea generation is disabled, no keywords are provided, but a URL is set. "
                "Will enable idea generation automatically (ideas: true)."
            )
            return True

        return ideas

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if not v:
            return v
        try:
            return resources.google_lang_id(v)
        except ValueError as e:
            raise ValueError(f"Invalid language code: {v}") from e

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if not v:
            return v
        try:
            return resources.google_country_id(v)
        except ValueError as e:
            raise ValueError(f"Invalid country code: {v}") from e

    @model_validator(mode="after")
    def validate(self):
        # Consistent dates
        if (self.metrics_start is not None and self.metrics_end is None) or (
            self.metrics_start is None and self.metrics_end is not None
        ):
            raise ValueError("Either provide both metrics_end and metrics_start or neither!")

        if self.metrics_start and self.metrics_end:
            try:
                start = datetime.strptime(self.metrics_start, "%Y-%m")
                end = datetime.strptime(self.metrics_end, "%Y-%m")
            except ValueError as e:
                raise ValueError(
                    "Metrics start and end dates must be in YYYY-MM format (e.g., '2023-01')."
                ) from e

            if start > end:
                raise ValueError("Metrics start date must be before or equal to the end date.")

        # Limit keyword seed
        seed_max = 20
        whole_site = self.url and self.whole_site and is_domain(self.url)
        if self.ideas and self.keywords and not whole_site and (len(self.keywords) > seed_max):
            self.keywords = self.keywords[:seed_max]
            LOG.warning(
                "Google only supports up to 20 initial keywords for keyword ideas generation. "
                "Will use the first 20 keywords and ignore rest."
            )

        return self


def encode_json_b64(value):
    """Encode a JSON key as a base64 string.

    Can be used e.g. to store complex objects in environment variables.
    """
    b64 = base64.b64encode(json.dumps(value).encode("utf-8"))
    return b64.decode("ascii")


def decode_json_b64(value):
    """Decode a base64-encoded JSON key string."""
    str_val = value.encode("ascii")
    return json.loads(base64.b64decode(str_val).decode("utf-8"))


def config_from_env() -> dict:
    """Load Google Ads API configuration from environment variables."""
    vars = (
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        "GOOGLE_ADS_USE_PROTO_PLUS",
        "GOOGLE_ADS_JSON_KEY",
        "GOOGLE_ADS_JSON_KEY_FILE_PATH",
    )
    return {
        var.replace("GOOGLE_ADS_", "").lower(): os.environ[var]
        for var in vars
        if var in os.environ
    }


def connect_ads_client(config: str | Path | dict | None = None) -> GoogleAdsClient:
    """Load Google Ads client from credentials."""
    if config is None:
        config = config_from_env()

    if isinstance(config, dict):
        if json_key := config.pop("json_key", None):
            try:
                json_key = json.loads(json_key)
            except json.JSONDecodeError:
                LOG.warning("Failed to decode JSON key. Will try base64 decoding.")
                json_key = decode_json_b64(json_key)
            with tempfile.NamedTemporaryFile("w", suffix=".json") as fp:
                json.dump(json_key, fp)
                fp.flush()
                config["json_key_file_path"] = fp.name
                safe_config = {
                    k: v for k, v in config.items() if k not in ("json_key", "developer_token")
                }
                LOG.info(f"Connecting to Google Ads API with credentials dict:\n{safe_config}")
                client = GoogleAdsClient.load_from_dict(config)

            return client  # noqa: RET504

        LOG.info(f"Connecting to Google Ads API with credentials dict:\n{config}")
        return GoogleAdsClient.load_from_dict(config)

    if isinstance(config, str | Path):
        return GoogleAdsClient.load_from_storage(config)

    raise ValueError(f"Invalid config type: {type(config)}. Need PathLike, dict or None.")


def year_month_from_date(date: str | datetime) -> tuple[int, MonthOfYearEnum.MonthOfYear]:
    """Convert a datetime object to a YearMonth string.

    Month enum values 0 and 1 are "UNSPECIFIED" and "UNKNOWN". January is 2 etc.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m")

    y = date.year
    m = MonthOfYearEnum.MonthOfYear(date.month + 1)
    return y, m


def is_domain(url: str) -> bool:
    """Check if the URL is a domain only, without path."""
    if not url:
        return False

    if not url.startswith("http"):
        url = "https://" + url

    parsed = urllib.parse.urlparse(str(url))
    return not parsed.path.strip("/")


@lru_cache(maxsize=3)
def fetch_keywords(
    cfg: GoogleKwdConfig,
) -> GenerateKeywordIdeasPager | GenerateKeywordHistoricalMetricsResponse:
    """Fetch metrics for a fixed list of keywords or generate keyword ideas from Google Ads API."""
    client = connect_ads_client(cfg.credentials)
    ads_service = client.get_service("GoogleAdsService")
    kwd_service = client.get_service("KeywordPlanIdeaService")

    request: GenerateKeywordIdeasRequest | GenerateKeywordHistoricalMetricsRequest
    url, keywords, whole_site = cfg.url, cfg.keywords, cfg.whole_site

    if cfg.ideas:
        request = GenerateKeywordIdeasRequest()
        request.page_size = cfg.max_ideas or 100
        request.include_adult_keywords = False
        request.keyword_annotation.append(
            client.enums.KeywordPlanKeywordAnnotationEnum.KEYWORD_CONCEPT
        )

        if url and whole_site and is_domain(url):
            LOG.info(f"Fetching keyword ideas for whole site {url}.")
            request.site_seed.site = url
            if keywords:
                LOG.warning("Manually specified keywords will be ignored!")
        elif url and not keywords:
            LOG.info(f"Fetching keyword ideas for page {url}.")
            request.url_seed.url = url
        elif keywords and not url:
            LOG.info(f"Fetching keyword ideas for {len(keywords)} seed keyword(s).")
            request.keyword_seed.keywords.extend(keywords)
        elif keywords and url:
            LOG.info(f"Fetching ideas for {len(keywords)} seed keyword(s) and page: {url}.")
            request.keyword_and_url_seed.url = url
            request.keyword_and_url_seed.keywords.extend(keywords)
        else:
            raise ValueError(
                "Either 'keywords' or 'url' must be provided when 'ideas' is True. "
                "Provide a list of keywords or a url URL to fetch ideas from."
            )
    else:
        if not keywords:
            raise ValueError(
                "No keywords provided. Please provide keywords to fetch historical metrics for."
            )
        LOG.info(f"Fetching historical metrics for {len(keywords)} keywords.")
        request = GenerateKeywordHistoricalMetricsRequest()
        request.keywords = list(keywords)

    request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

    request.customer_id = cfg.customer or os.environ.get(
        "GOOGLE_ADS_CUSTOMER_ID", os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "")
    )

    if cfg.language:
        request.language = ads_service.language_constant_path(cfg.language)

    if cfg.country:
        request.geo_target_constants.append(ads_service.geo_target_constant_path(cfg.country))

    request.historical_metrics_options.include_average_cpc = True

    if cfg.metrics_start is not None:
        y, m = year_month_from_date(cfg.metrics_start)
        request.historical_metrics_options.year_month_range.start.year = y
        request.historical_metrics_options.year_month_range.start.month = m

    if cfg.metrics_end is not None:
        y, m = year_month_from_date(cfg.metrics_end)
        request.historical_metrics_options.year_month_range.end.year = y
        request.historical_metrics_options.year_month_range.end.month = m

    if cfg.ideas:
        return kwd_service.generate_keyword_ideas(request=request)

    return kwd_service.generate_keyword_historical_metrics(request=request)


def collect_columns(df: DataFrame, columns: list[str]) -> Series:
    """Collects values in specified columns into a Series of lists."""
    matrix = df[columns].values
    return Series([row for row in matrix])


def collect_volume_columns(df: DataFrame):
    """Mutates monthly search volume columns into two list columns containing values and dates."""
    vol_cols = [col for col in df.columns if "search_volume_" in col]
    df["search_volume"] = collect_columns(df, vol_cols)

    def col_to_date(col):
        """Convert column name to datetime."""
        dt = datetime(*map(int, col.split("_")[-2:]), 1) if "_" in col else None
        return dt.isoformat() if dt else None

    sv_dt = [col_to_date(col) for col in vol_cols]
    df["search_volume_date"] = [sv_dt] * len(df)
    return df.drop(columns=vol_cols)


def calculate_trend_pct(volumes: list[float] | ndarray | None, n_months: int):
    """Calculate trend based on monthly search volumes provided as list."""
    if not isinstance(volumes, list | ndarray) or volumes is None or len(volumes) < n_months:
        return None

    end_volume = volumes[-1]
    start_volume = volumes[-n_months]
    return 100 * (end_volume - start_volume) / (start_volume or 1)


def linreg_trend(y: list | ndarray | None) -> float | None:
    """Calculate linear regression slope for a list of values."""
    if not isinstance(y, list | ndarray) or y is None or len(y) < 3:
        return None

    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    X = np.vstack([np.ones_like(x), x]).T

    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    intercept, slope = theta
    # y_pred = X @ theta  # noqa: ERA001

    return slope / y.mean()


def add_trend_columns(df: DataFrame) -> DataFrame:
    """Add trend columns to the DataFrame based on the specified trend type."""
    if "search_volume" in df.columns:
        valid_idx = df["search_volume"].first_valid_index()
        if valid_idx is not None:
            some_value = df["search_volume"][valid_idx]
            n_months = len(some_value)
            if n_months >= 12:  # noqa: PLR2004
                df["search_volume_growth_yoy"] = df["search_volume"].apply(
                    lambda x: calculate_trend_pct(x, 12)
                )

            if n_months >= 3:  # noqa: PLR2004
                df["search_volume_growth_3m"] = df["search_volume"].apply(
                    lambda x: calculate_trend_pct(x, 3)
                )
                df["search_volume_trend"] = df["search_volume"].apply(lambda x: linreg_trend(x))
            elif n_months > 1:
                df["search_volume_growth_1m"] = df["search_volume"].apply(
                    lambda x: calculate_trend_pct(x, 2)
                )

    return df


def process_keywords(  # noqa: PLR0912
    response: GenerateKeywordIdeasPager | GenerateKeywordHistoricalMetricsResponse,
    collect_volumes: bool = True,
    zeros_to_nans: bool = True,
) -> DataFrame:
    """Process Google Ads API keyword response into a DataFrame."""

    # Check which metrics and attributes we have extracted
    if hasattr(response.results[0], "keyword_idea_metrics"):
        metrics_attr = "keyword_idea_metrics"
    else:
        metrics_attr = "keyword_metrics"

    some_metrics = getattr(response.results[0], metrics_attr, None)
    fields = [
        "avg_monthly_searches",
        "competition",
        "competition_index",
        "average_cpc_micros",
        "low_top_of_page_bid_micros",
        "high_top_of_page_bid_micros",
    ]
    fields = [f for f in fields if hasattr(some_metrics, f)]

    records = []
    for kwd in response.results:
        record = {
            "keyword": kwd.text,
        }

        if (metrics := getattr(kwd, metrics_attr, None)) is not None:
            # Single value metrics
            for field in fields:
                record[field] = getattr(metrics, field, None)

            # Monthly search volumes
            if volumes := getattr(metrics, "monthly_search_volumes", None):
                for volume in volumes:
                    year = volume.year
                    month = volume.month
                    date = datetime.strptime(f"{year}-{month.name.capitalize()}", "%Y-%B")
                    record[f"search_volume_{date.year}_{date.month:02}"] = volume.monthly_searches

        # Concept annotations
        if (annotations := getattr(kwd, "keyword_annotations", None)) is not None:  # noqa: SIM102
            if (concepts := getattr(annotations, "concepts", None)) is not None:
                concept_names, concept_groups = set(), set()
                for concept in concepts:
                    concept_names.add(concept.name)
                    if hasattr(concept, "concept_group"):
                        concept_groups.add(concept.concept_group.name)

                concept_names -= {"Others", "Non-Brands"}
                concept_groups.discard("Others")
                record["concepts"] = list(concept_names) or None
                record["concept_groups"] = list(concept_groups) or None

        records.append(record)

    df = DataFrame(records)
    if collect_volumes:
        df = collect_volume_columns(df)
        df = add_trend_columns(df)

    # Convert micros to dollars/euros, and 0s to NaNs
    for col in (
        "average_cpc_micros",
        "low_top_of_page_bid_micros",
        "high_top_of_page_bid_micros",
    ):
        if col in df.columns:
            name = col.replace("_micros", "")
            df[name] = df[col] / 1_000_000
            if zeros_to_nans:
                df[name] = df[name].replace(0, np.nan)
            df = df.drop(columns=col)

    return df


def keywords(cfg: GoogleKwdConfig) -> DataFrame:
    """Fetch and process keywords from Google Ads API based on the provided configuration."""
    LOG.info(f"Fetching keywords with\n\n{cfg}")
    response = fetch_keywords(cfg)  # type: ignore

    if response is None or len(response.results) == 0:
        raise ValueError(
            "No keywords were fetched from Google Ads API! "
            "Check your configuration, credentials, and network connection."
        )

    LOG.info("Processing metrics response into DataFrame.")
    df = process_keywords(response, collect_volumes=True)
    LOG.info(f"Got keyword dataframe:\n{df}")
    return df


SYSTEM_PROMPT = dedent("""
You're an expert SEO specialist analyzing google keyword searches for a specific domain.

Your task is to simplify a list of search keywords (short phrases) into a smaller group of clean
keywords that make sense to later group, aggregate and analyze together. The idea is to remove
duplicate keywords that are identical in meaning but are spelled differently
(misspelling, singular vs. plural etc.), while preserving different search intents and
meaningful variations.

The keywords come from a dataset of '%(domain)s'. %(extra)s
""")

USER_PROMPT = dedent("""
Extract a clean, deduplicated list of search keywords of no more than %(n_max)s items
from the following list.

# Keywords

{{keywords}}
""")

ASSIGNMENT_PROMPT_SYSTEM = dedent("""
You're task is to use the following list of clean keywords,
and select and return the best semantically matching keyword for a given input phrase.

# Keywords

%(keywords)s
""")

ASSIGNMENT_PROMPT_USER = dedent("""
Assign the correct keyword to the following phrase: {{text}}.
""")


class KeywordCleaner:
    """A class to clean and deduplicate search keywords from a list of texts."""

    def __init__(
        self,
        domain: str,
        n_max: int = 10,
        extra: str | None = None,
    ):
        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT % {"domain": domain, "extra": extra},
                },
                {
                    "role": "user",
                    "content": USER_PROMPT % {"n_max": n_max},
                },
            ],  # type: ignore
            required=["keywords"],
        )

        class Keywords(Response):
            keywords: list[str] = Field(
                ...,
                description="A list of clean google search keywords.",
                max_length=n_max,
            )

        self.task = Task(prompt=prompt, response=Keywords)

    async def __call__(
        self,
        keywords: Iterable[str],
        model: str,
        max_dollars: float,
        max_tokens: float | None = None,
        max_texts: float | None = None,
    ) -> Response:
        """Extracts a two-level topic hierarchy from a list of texts."""
        text = utils.concat_up_to(
            keywords,
            model=model,
            max_dollars=max_dollars,
            max_tokens=max_tokens,
            max_texts=max_texts,
            separator="\n",
        )
        responses = await self.task.call(context={"keywords": text}, model=model)
        return responses[0]


class KeywordAssigner:
    """Enforce correct clean keyword assignment."""

    def __init__(self, keywords: Response):
        keywords = keywords.to_dict()["keywords"]
        prompt = Prompt(
            messages=[
                {"role": "system", "content": ASSIGNMENT_PROMPT_SYSTEM % {"keywords": keywords}},
                {"role": "user", "content": ASSIGNMENT_PROMPT_USER},
            ],  # type: ignore
            required=["text"],
        )

        class Match(Response):
            keyword: Literal[*keywords]

        self.task = Task(prompt=prompt, response=Match)

    async def __call__(self, texts: AnyContext, model: str, **kwds) -> ResponseSet:
        return await self.task(context=texts, model=model, **kwds)
