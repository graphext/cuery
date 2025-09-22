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

import asyncio
import json
import os
from collections.abc import Coroutine, Iterable, Sequence
from functools import partial
from typing import Any

import aiohttp

from ...asy import all_with_policies
from ...search import SearchResult, Source
from ...utils import LOG


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


def parse_aio(aio) -> SearchResult:
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


async def query(
    prompt: str,
    country: str | None = None,  # 2-letter country code, e.g. "us"
    language: str | None = None,  # 2-letter language code, e.g. "en"
    validate: bool = True,
    log: bool = False,
    session: aiohttp.ClientSession | None = None,
) -> SearchResult | dict[str, Any]:
    """Execute a Google search via ScrapingDog and extract AI Overview."""
    api_key = os.environ["SCRAPINGDOG_API_KEY"]
    serp_endpoint = "https://api.scrapingdog.com/google"
    aio_endpoint = "https://api.scrapingdog.com/google/ai_overview"

    params = {"api_key": api_key, "query": prompt, "results": 10}
    if country:
        params["country"] = country
    if language:
        params["language"] = language

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.get(serp_endpoint, params=params) as resp:
            resp.raise_for_status()
            content = await resp.json()
            if log:
                LOG.info(f"Request URL: {str(resp.url)}")
                LOG.info("Response:")
                LOG.info(json.dumps(content, indent=2))

        aio = content.get("ai_overview") or {}

        if aio_url := aio_api_url(aio):
            LOG.warning(f"Need second request to get AI overview for prompt: {prompt}")
            async with session.get(
                aio_endpoint, params={"api_key": api_key, "url": aio_url}
            ) as resp:
                resp.raise_for_status()
                content = await resp.json()

            aio = content.get("ai_overview") or {}

        if not validate:
            return aio

        result = parse_aio(aio)
        result._raw_response = aio
        return result
    finally:
        if close_session:
            await session.close()


async def gather(  # noqa: PLR0913
    prompts: Sequence[str] | Iterable[str],
    country: str | None = None,
    language: str | None = None,
    validate: bool = True,
    log: bool = False,
    session: aiohttp.ClientSession | None = None,
    policies: dict[str, Any] | None = None,
    execute: bool = True,
) -> list[Coroutine] | list[SearchResult | dict[str, Any]]:
    """Create zero-argument coroutine factories (with policies) for many prompts."""
    policies = policies or {}
    func = partial(
        query,
        country=country,
        language=language,
        validate=validate,
        log=log,
        session=session,
    )

    kwds = [{"prompt": p} for p in prompts]
    coros = all_with_policies(func, kwds=kwds, policies=policies, labels="hasdata-aio")

    if not execute:
        return coros

    return await asyncio.gather(*coros)
