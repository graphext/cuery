"""Functions to identify competitors and fetch brand information."""

import asyncio
from typing import Literal

from pandas import DataFrame
from pydantic import Field
from tqdm.auto import tqdm

from .. import asy
from ..response import Response, ResponseSet
from ..search import search_with_format
from ..templates import load_template
from ..utils import LOG, Configurable, dedent, extract_domain, render_template

DEFAULT_SEARCH_MODEL = "openai/gpt-4.1"


COMPETITORS_PROMPT = dedent("""
You are an expert in market analysis and competitive intelligence.
Given the following brand(s) information, find and return an exhaustive list of competitors.
Consider competitors to be brands that offer similar products or services and target
the same or similar customer segments. {strictness_clause}
Provide a list of competitors giving their name and a brief description of their main activity.
The output should be a JSON array of objects with name and description fields. It's more important
to be comprehensive than selective, so include more competitors rather than less.
Try to order them by relevance, with the most direct and bigger competitors first.

{instructions}

# Brand(s) information

{record_template}
""")
"""Used in first step identifying competitors of a brand."""

BRAND_PROMPT = dedent("""
You are an expert in market analysis and competitive intelligence.
Given the below brand information, provide a detailed overview of the brand.
Include the following information:
- Name: The official name of the brand.
- Short name: A common or canonical short name for the brand, if different from the official name.
- Description: A brief description of the brand.
- Domain: The official website of the brand.
- Portfolio: A list of main products or services offered by the brand, each with its name and
  category. Make sure the names refer to specific product or service, not broad categories like
  "smartphones", "cloud computing", "electric cars").
- Market Position: Classify the brand's market position as one of the following:
  "leader", "challenger", "niche", or "follower".
- Favicon: The URL of the brand's favicon, if available.

Use search and make sure that the searches include the kinds of information requested above.

# Brand

{record_template}
""")
"""Prompt to extract detailed information for a single brand."""


class Brand(Response):
    """Identifier for a brand."""

    name: str
    """Name of the brand."""
    description: str
    """Brief description of the brand."""


class Brands(Response):
    """List of brands."""

    brands: list[Brand]
    """List of brands."""


class Product(Response):
    """Represents a product or service offered by a brand."""

    name: str
    """Name of the product or service."""
    category: str | None
    """Category of the product or service."""


class BrandInfo(Response):
    """Represents a brand and its attributes."""

    name: str
    """Name of the brand."""
    short_name: str | None
    """Short, common/canonical name of the brand, if different from the official name. E.g.
    "Tesla" instead of "Tesla, Inc.", or "Peugoet" instead of "Automobiles Peugeot."""
    description: str
    """Description of the brand."""
    domain: str = Field(..., min_length=1)
    """Official website of the brand."""
    portfolio: list[Product]
    """List of products or services offered by the brand."""
    market_position: Literal["leader", "challenger", "niche", "follower"]
    """Market position of the brand."""
    favicon: str | None
    """URL of the brand's favicon, if available."""


async def find_competitors(
    brand: str | list[str],
    sector: str | None,
    market: str | None,
    strict: bool = True,
    instructions: str | None = None,
    **kwds,
) -> list[Brands]:
    """Identify main competitors for a given brand."""
    instructions = instructions or ""

    if strict:
        strictness_clause = (
            "Only consider as competitors those that do NOT belong to the same parent company as "
            "the original brand(s) (e.g. Fanta is NOT a competitor of Coca-Cola in this sense)."
        )
    else:
        strictness_clause = (
            "Consider all relevant competitors, including those from the same "
            "parent company (e.g. Fanta IS a competitor of Coca-Cola in this sense)."
        )

    record = {"brand": brand, "sector": sector, "market": market}
    prompt = COMPETITORS_PROMPT.format(
        record_template=load_template("record_to_text"),
        strictness_clause=strictness_clause,
        instructions=instructions,
    )
    prompt = render_template(prompt, record=record)

    response = await search_with_format(
        prompt=prompt,
        model=DEFAULT_SEARCH_MODEL,
        response_format=Brands,  # type: ignore
        **kwds,
    )

    return response.brands


async def fetch_brand_info(brand: Brand, **kwds) -> BrandInfo:
    """Fetch detailed information for a given brand."""
    record = brand.to_dict()
    if not record.get("description"):
        record.pop("description", None)

    prompt = BRAND_PROMPT.format(record_template=load_template("record_to_text"))
    prompt = render_template(prompt, record=record)

    return await search_with_format(
        prompt=prompt,
        model=DEFAULT_SEARCH_MODEL,
        response_format=BrandInfo,  # type: ignore
        **kwds,
    )


class BrandSearchConfig(Configurable):
    """Configuration for GEO analysis (LLM brand mentions and ranks)."""

    brand: str | list[str]
    """Initial brand or list of brands to find competitors for."""
    sector: str | None
    """Sector or industry the brand operates in, e.g. 'electric cars', 'soft drinks'."""
    market: str | None
    """Geographical market or region, e.g. 'Spain', 'global'."""
    strict_competitors: bool = True
    """Whether to exclude competitors from the same parent company."""
    instructions: str | None = None
    """Additional instructions to guide the competitor search."""
    search_kwargs: dict | None = None
    """Additional keyword arguments to pass to the search function."""


async def search_brands(
    cfg: BrandSearchConfig,
    to_pandas: bool = False,
) -> ResponseSet | DataFrame:
    """Fetch detailed information for a list of brands."""
    search_kwargs = cfg.search_kwargs or {}
    brands = [cfg.brand] if isinstance(cfg.brand, str) else cfg.brand

    LOG.info(f"Finding competitors for {brands} in sector='{cfg.sector}', market='{cfg.market}'.")
    competitors = await find_competitors(
        brand=brands,
        sector=cfg.sector,
        market=cfg.market,
        strict=cfg.strict_competitors,
        instructions=cfg.instructions,
        **search_kwargs,
    )
    LOG.info(f"Found {len(competitors)} competitors for {brands}: {[c.name for c in competitors]}")

    brands = [Brand(name=b, description="") for b in brands]
    all_brands = brands + competitors

    pbar = tqdm(total=len(all_brands), desc="Fetching brand info")
    policies = {
        "timeout": 120,
        "n_concurrent": 100,
        "retries": 3,
        "fallback": BrandInfo.fallback(),
        "timer": True,
        "pbar": pbar,
    }

    coros = asy.all_with_policies(
        func=fetch_brand_info,
        kwds=[{**search_kwargs, "brand": brand} for brand in all_brands],
        policies=policies,
        labels="fetch_brand_info",
    )
    responses = await asyncio.gather(*coros)

    for response in responses:
        if response.domain:
            clean_domain = extract_domain(response.domain)
            response.domain = clean_domain or response.domain

    rs = ResponseSet(responses, context=None, required=None)  # type: ignore

    if to_pandas:
        df = rs.to_pandas()
        df["is_competitor"] = [False] * len(brands) + [True] * len(competitors)
        return df

    return rs
