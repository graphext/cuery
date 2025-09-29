import asyncio
import json
from collections.abc import Iterable
from typing import Literal

from .. import search
from ..prompt import Prompt

from ..response import Response
from ..task import Task
from ..utils import dedent

DEFAULT_SEARCH_MODEL = "openai/gpt-5-mini"


COMPETITORS_PROMPT = dedent("""
You are an expert in market analysis and competitive intelligence.
Given the following brand(s) information, identify and list the main competitors.
Consider competitors to be brands that offer similar products or services and target
the same customer segments. Provide a list of competitors giving their name and a brief
description of their main activity. The output should be a JSON array of objects with
name and description fields.

# Brand(s) information

${brand}
""")
"""Used in first step identifying competitors of a brand."""

BRAND_PROMPT = dedent("""
You are an expert in market analysis and competitive intelligence.
Given the below brand information, provide a detailed overview of the brand.
Include the following attributes:
- Name: The official name of the brand.
- Description: A brief description of the brand's history, mission, and values.
- Domain: The official website of the brand.
- Portfolio: A list of main products or services offered by the brand.
- Market Position: Classify the brand's market position as one of the following:
  "leader", "challenger", "niche", or "follower".

# Brand

Name: ${name}
Description: ${description}
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


class BrandInfo(Response):
    """Represents a brand and its attributes."""

    name: str
    """Name of the brand."""
    description: str
    """Description of the brand."""
    domain: str
    """Official website of the brand."""
    portfolio: list[str]
    """List of products or services offered by the brand."""
    market_position: Literal["leader", "challenger", "niche", "follower"]


async def find_competitors(
    brand: str | list[str],
    sector: str | None,
    market: str | None,
) -> list[Brand]:
    """Identify main competitors for a given brand."""
    ...


async def fetch_brand_info(brand: Brand | list[Brand]) -> Brand | list[Brand]:
    """Fetch detailed information for a given brand."""
    ...
