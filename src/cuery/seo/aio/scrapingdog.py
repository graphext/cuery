"""API helpers to access Google AI Overview in Google SERP results via ScrapingDog.

Scraping Dog
-------------

Implements a convenience function :func:`query_google` that mirrors the shape of
``cuery.search`` provider helpers by returning a ``SearchResult`` (answer + sources)
extracted from Google AI Overview (aka AI Overviews / AI Summary) when available.

The ScrapingDog API exposes (at least) two relevant endpoints:

* ``https://api.scrapingdog.com/google`` – standard SERP results. In some cases
  the AI Overview content may be embedded directly in the JSON payload (future
  proofing – not currently documented in retrieved snippets but handled here).
* ``https://api.scrapingdog.com/google/ai_overview`` – dedicated endpoint for
  AI Overview content when Google requires a secondary fetch.
"""

import json
import os
from collections.abc import Iterable
from typing import Any

import requests

from ...search import SearchResult, Source


def flatten_text_blocks(blocks: Iterable[dict[str, Any]] | None) -> str:
    """Convert list of ``text_blocks`` to a single answer string.

    Supported block types (based on docs sample): ``paragraph`` and ``list``.
    A ``list`` block contains a ``list`` key with items each having ``snippet``.
    Unknown types are ignored (future proof).
    """
    if not blocks:
        return ""

    parts: list[str] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "paragraph" and (text := block.get("snippet", block.get("text"))):
            parts.append(text.strip())
        elif btype == "list":
            for item in block.get("list", []) or []:
                if text := item.get("snippet", item.get("text")):
                    parts.append(f"- {text.strip()}")
    return "\n".join(parts).strip()


def parse_ai_overview(aio) -> SearchResult:
    """Extract AI Overview into a ``SearchResult``.

    Expected structure (subset):
    {
        "ai_overview": {
            "text_blocks": [...],
            "references": [ {"title": str, "link": str, ...}, ... ]
        }
    }
    """
    text_blocks = aio.get("text_blocks") or []
    references = aio.get("references") or []

    answer = flatten_text_blocks(text_blocks)
    sources = []
    for ref in references:
        link = ref.get("link")
        if link:
            title = ref.get("title") or ref.get("source") or ""
            sources.append(Source(title=title, url=link))

    return SearchResult(answer=answer, sources=sources)


def aio_api_url(aio) -> str | None:
    """Extract the API URL from the aio dict, if available."""
    if "text_blocks" in aio or "references" in aio:
        return None

    if "ai_overview_api_url" in aio:
        return aio["ai_overview_api_url"]

    return None


def query_scraping_dog(
    prompt: str,
    country: str | None = None,  # 2-letter country code, e.g. "us"
    language: str | None = None,  # 2-letter language code, e.g. "en"
    n_results: int = 10,
    validate: bool = True,
    timeout: int = 10,
) -> SearchResult | dict[str, Any]:
    """Execute a Google search via ScrapingDog and extract AI Overview."""
    api_key = os.environ["SCRAPINGDOG_API_KEY"]

    params = {"api_key": api_key, "query": prompt, "results": n_results}
    if country:
        params["country"] = country
    if language:
        params["language"] = language

    resp = requests.get("https://api.scrapingdog.com/google", params=params, timeout=timeout)
    resp.raise_for_status()
    print(f"Request URL: {resp.request.url}")
    content = resp.json()
    print("Response:")
    print(json.dumps(content, indent=2))
    aio = content.get("ai_overview") or {}

    if aio_url := aio_api_url(aio):
        params = {"api_key": api_key, "url": aio_url}
        resp = requests.get(
            "https://api.scrapingdog.com/google/ai_overview",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()
        aio = content.get("ai_overview") or {}

    if not validate:
        return aio

    result = parse_ai_overview(aio)
    result._raw_response = aio
    return result
