"""Async helpers for grounded / live web search via OpenAI, Google Gemini and xAI Grok.

Provides a convenience ``search`` function that executes multiple web searches
concurrently (rate limited via a semaphore) and preserves input order. Logic
mirrors :func:`cuery.call.gather_calls` but is simplified for plain string prompts.

Currently supported providers (``client`` argument):

* ``"openai"`` – Uses the OpenAI Responses API with the ``web_search_preview`` tool.
* ``"gemini"`` – Uses the Google Gemini (google-genai) API with Google Search grounding
    (``google_search`` for Gemini 2.x, ``google_search_retrieval`` for 1.5 models).

Both return a :class:`SearchResult` with extracted plain text plus a list of
``Reference`` objects (title + URL) when available. If parsing fails a ``ValueError``
is raised (no silent fallbacks here – upstream caller can decide how to handle).
"""

from __future__ import annotations

import asyncio
from asyncio import Semaphore
from collections.abc import Callable
from io import StringIO
from typing import Any, Literal

import instructor
import requests
from google import genai as gai
from google.genai import types as gaitypes
from google.genai.types import GenerateContentResponse as GGResponse
from markdown import Markdown
from openai import AsyncOpenAI
from openai import types as oaitypes
from pydantic import BaseModel
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm
from xai_sdk import AsyncClient as XaiAsyncClient
from xai_sdk.chat import Response as XAIResponse
from xai_sdk.chat import user as xai_user
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters, news_source, web_source, x_source

from .call import TQDM_POSITION
from .response import Response, ResponseSet

OAIResponse = oaitypes.responses.response.Response

VALID_MODELS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
        "o4-mini",
        "o3",
        "gpt-5 with reasoning levels low, medium and high",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ],
    "xai": [
        "grok-4",
        "grok-3",
        "grok-3-mini",  # Maybe doesn't support citations
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
}


# https://stackoverflow.com/a/54923798
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
Markdown.output_formats["plain"] = unmark_element  # type: ignore
__md = Markdown(output_format="plain")  # type: ignore
__md.stripTopLevelTags = False


def unmark(text):
    """Convert Markdown text to plain text, stripping all formatting."""
    return __md.convert(text)


class Reference(Response):
    """Single search reference with title and URL."""

    title: str
    url: str


class SearchResult(Response):
    """Search result with extracted text and references."""

    text: str
    references: list[Reference]


def resolve_redirect(redirect_url: str, timeout: int = 2) -> str:
    """Resolve a URL redirect to its final destination URL.

    Gemini grounding chunk URIs are things like:
    https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFXCboQq0LRg6hUOP...
    """
    try:
        return requests.head(redirect_url, allow_redirects=True, timeout=timeout).url
    except requests.RequestException:
        return redirect_url


def validate_openai(response, plain: bool = False) -> SearchResult:
    """Convert a raw web search response into a ``SearchResult`` instance."""
    output = response.output
    text = ""
    refs = []

    if len(output) < 2:  # noqa: PLR2004, without search
        text = output[0].content[0].text
    else:
        if output[0].type != "web_search_call":
            raise ValueError("First output must be of type 'web_search_call'.")

        content = output[1].content[0]
        if content.type == "output_text":
            text = content.text
            if hasattr(content, "annotations"):
                refs = [Reference(title=ann.title, url=ann.url) for ann in content.annotations]

    if plain:
        text = unmark(text)

    result = SearchResult(text=text, references=refs)
    result._raw_response = response  # type: ignore
    return result


def validate_gemini(response: BaseModel, plain: bool = False) -> SearchResult:
    """Convert a Gemini grounded search raw response into ``SearchResult``."""
    if not isinstance(response, BaseModel):
        raise ValueError("Gemini response is not a Pydantic model.")

    candidate = response.candidates[0]  # type: ignore

    # Text content
    content = candidate.content
    texts = [getattr(p, "text", None) for p in content.parts]
    text = "\n".join([t for t in texts if t])
    if plain:
        text = unmark(text)

    # References
    try:
        chunks = candidate.grounding_metadata.grounding_chunks or []
        refs = [Reference(title=c.web.title, url=resolve_redirect(c.web.uri)) for c in chunks]
    except Exception:
        refs = []

    result = SearchResult(text=text, references=refs)
    result._raw_response = response
    return result


def validate_xai(response, plain: bool = False) -> SearchResult:
    """Convert an xAI Grok Live Search chat completion into ``SearchResult``.

    Citations (if ``return_citations`` enabled – default true) are exposed as a
    list of URL strings on ``response.citations``. We map each into a ``Reference``
    using the URL as both title and URL (title data not currently provided).
    """
    text = getattr(response, "content", "") or ""
    if plain:
        text = unmark(text)

    citations = getattr(response, "citations", None) or []
    refs = [Reference(title="", url=url) for url in citations]

    result = SearchResult(text=text, references=refs)
    result._raw_response = response  # type: ignore[attr-defined]
    return result


async def query_openai(
    prompt: str,
    country: str | None = None,  # e.g. "US"
    city: str | None = None,  # e.g. "Madrid"
    context_size: Literal["low", "medium", "high"] | str = "high",
    model: str = "gpt-5",
    use_search: bool = True,
    validate: bool = True,
) -> SearchResult | OAIResponse:
    """Call OpenAI Responses API with Web Search enabled (async).

    API Docs:
    - https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses
    """
    client = AsyncOpenAI()

    params: dict = {
        "model": model,
        "input": prompt,
    }

    if use_search:
        tool: dict = {
            "type": "web_search",
            "search_context_size": context_size,
        }
        if country:
            tool["user_location"] = {
                "type": "approximate",
                "country": country,
            }
            if city:
                tool["user_location"]["city"] = city

        params["tools"] = [tool]
        params["tool_choice"] = "required"

    response = await client.responses.create(**params)
    if validate:
        return validate_openai(response)

    return response


async def query_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    thinking_budget: int = -1,
    use_search: bool = True,
    validate: bool = True,
) -> SearchResult | GGResponse:
    """Call Gemini with Google Search grounding (supports Gemini > 2.0 only).

    API Docs:
    - https://ai.google.dev/gemini-api/docs/google-search
    - https://ai.google.dev/gemini-api/docs/thinking
    """
    client = gai.client.Client()  # type: ignore

    params = {
        "model": model,
        "contents": prompt,
    }

    config: dict = {}

    if use_search:
        search_tool = gaitypes.Tool(google_search=gaitypes.GoogleSearch())
        config["tools"] = [search_tool]

    if "2.5" in model:
        config["thinking_config"] = gaitypes.ThinkingConfig(thinking_budget=thinking_budget)

    params = {
        "model": model,
        "contents": prompt,
        "config": gaitypes.GenerateContentConfig(**config),
    }

    response = await client.aio.models.generate_content(**params)
    if validate:
        return validate_gemini(response)

    return response


async def query_xai(  # noqa: PLR0913
    prompt: str,
    model: str = "grok-4",
    timeout: int | None = None,
    mode: Literal["auto", "on", "off"] = "on",
    max_search_results: int = 15,
    sources: list[Literal["web", "news", "x"]] | None = None,  # type: ignore
    country: str | None = None,
    use_search: bool = True,
    validate: bool = True,
) -> SearchResult | XAIResponse:
    """Call xAI Grok with Live Search (async).

    API Docs:
    - https://docs.x.ai/docs/guides/live-search
    """
    params: dict = {"model": model}

    if use_search:
        if sources is None:
            sources: list[chat_pb2.Source] = [web_source(country=country)]

        if isinstance(sources, list) and all(isinstance(s, str) for s in sources):
            sources: list[chat_pb2.Source] = []
            for src in sources:
                if src == "web":
                    sources.append(web_source(country=country))
                elif src == "news":
                    sources.append(news_source(country=country))
                elif src == "x":
                    sources.append(x_source())

        sp = SearchParameters(
            mode=mode,
            return_citations=mode in ("auto", "on"),
            max_search_results=max_search_results,
            sources=sources,  # type: ignore
        )

        params["search_parameters"] = sp

    client = XaiAsyncClient(timeout=timeout)
    chat = client.chat.create(**params)
    chat.append(xai_user(prompt))
    response = await chat.sample()
    if validate:
        return validate_xai(response)

    return response


async def query_deepseek(
    prompt: str,
    model: str = "deepseek-chat",
    validate: bool = True,
    use_search: bool = True,
    **kwds,
) -> Any:
    """DeepSeek doesn't support search yet, so we simply execute the prompt as-is."""
    client = instructor.from_provider(
        f"deepseek/{model}",
        async_client=True,
        base_url="https://api.deepseek.com",
    )
    answer = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_model=str,
        **kwds,
    )
    if validate:
        return SearchResult(text=answer, references=[])

    return answer


LLMS = {
    "openai": query_openai,
    "google": query_gemini,
    "xai": query_xai,
    "deepseek": query_deepseek,
}


async def query(
    prompts: list[str],
    model: str = "openai/gpt-5",
    use_search: bool = True,
    max_concurrent: int = 2,
    progress_callback: Callable | None = None,
    return_tasks: bool = False,
    plain: bool = False,
    **kwds,
) -> ResponseSet | Any:
    """Execute multiple web searches concurrently for a given provider."""

    client, model = model.split("/", 1)
    if client not in VALID_MODELS:
        raise ValueError(f"Unsupported client '{client}'. Supported: {list(VALID_MODELS.keys())}.")

    if model not in VALID_MODELS[client]:
        raise ValueError(
            f"Unsupported model '{model}' for client '{client}'. Supported: {VALID_MODELS[client]}"
        )

    if not prompts:
        raise ValueError("No prompts provided.")

    query = LLMS[client]
    sem = Semaphore(max_concurrent)
    total = len(prompts)
    min_iters = max(1, int(total / 20))

    async def rate_limited_search(prompt: str):
        async with sem:
            return await query(
                prompt=prompt,
                model=model,
                use_search=use_search,
                **kwds,
            )

    if return_tasks:
        return [rate_limited_search(p) for p in prompts]

    if progress_callback is None:
        tasks = [rate_limited_search(p) for p in prompts]
        results = await async_tqdm.gather(
            *tasks,
            desc="Gathering searches",
            total=len(tasks),
            position=TQDM_POSITION,
            miniters=min_iters,
        )
    else:
        pbar = tqdm(
            desc="Gathering searches",
            total=total,
            position=TQDM_POSITION,
            miniters=min_iters,
        )

        async def with_progress(prompt: str):
            result = await rate_limited_search(prompt)
            pbar.update()
            if (pbar.n % min_iters == 0) or (pbar.n == total):
                await progress_callback(pbar.format_dict)  # type: ignore
            return result

        tasks = [with_progress(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    context = [{"prompt": p} for p in prompts]
    return ResponseSet(responses=results, context=context, required=["prompt"])  # type: ignore
