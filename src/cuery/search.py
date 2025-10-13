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
from collections.abc import Coroutine
from functools import cached_property, partial
from io import StringIO
from typing import Any, Literal

import instructor
import requests
from async_lru import alru_cache
from google import genai as gai
from google.genai import types as gaitypes
from google.genai.types import GenerateContentResponse as GGResponse
from markdown import Markdown
from openai import AsyncOpenAI
from openai import types as oaitypes
from pydantic import BaseModel, computed_field
from xai_sdk import AsyncClient as XaiAsyncClient
from xai_sdk.chat import Response as XAIResponse
from xai_sdk.chat import user as xai_user
from xai_sdk.proto import chat_pb2
from xai_sdk.search import SearchParameters, news_source, web_source, x_source

from . import ask
from .asy import all_with_policies
from .resources import country_coords
from .response import Response, ResponseSet
from .utils import dedent, extract_domain

OAIResponse = oaitypes.responses.response.Response  # type: ignore[attr-defined]

VALID_MODELS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
        "o4-mini",
        "o3",
        "gpt-5",
        "gpt-5-mini",
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


class Source(Response):
    """Single search reference with title and URL."""

    title: str
    url: str

    @computed_field
    @cached_property
    def domain(self) -> str | None:
        """Extract domain from URL."""
        return extract_domain(self.url)


class SearchResult(Response):
    """Search result with extracted text and references."""

    answer: str
    sources: list[Source]


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
    answer: str = ""
    sources: list[Source] = []

    if len(output) < 2:  # noqa: PLR2004, without search tool output
        # Only model response
        answer = output[0].content[0].text
    else:
        if output[0].type != "web_search_call":
            raise ValueError("First output must be of type 'web_search_call'.")

        content = output[1].content[0]
        if content.type == "output_text":
            answer = content.text
            if hasattr(content, "annotations"):
                sources = [Source(title=ann.title, url=ann.url) for ann in content.annotations]

    if plain:
        answer = unmark(answer)

    result = SearchResult(answer=answer, sources=sources)
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
    answer = "\n".join([t for t in texts if t])
    if plain:
        answer = unmark(answer)

    # References
    try:
        chunks = candidate.grounding_metadata.grounding_chunks or []
        sources = [Source(title=c.web.title, url=resolve_redirect(c.web.uri)) for c in chunks]
    except Exception:
        sources = []

    result = SearchResult(answer=answer, sources=sources)
    result._raw_response = response
    return result


def validate_xai(response, plain: bool = False) -> SearchResult:
    """Convert an xAI Grok Live Search chat completion into ``SearchResult``.

    Citations (if ``return_citations`` enabled – default true) are exposed as a
    list of URL strings on ``response.citations``. We map each into a ``Reference``
    using the URL as both title and URL (title data not currently provided).
    """
    answer = getattr(response, "content", "") or ""
    if plain:
        answer = unmark(answer)

    citations = getattr(response, "citations", None) or []
    sources = [Source(title="", url=url) for url in citations]

    result = SearchResult(answer=answer, sources=sources)
    result._raw_response = response  # type: ignore[attr-defined]
    return result


@alru_cache(maxsize=1000)
async def query_openai(
    prompt: str,
    country: str | None = None,  # 2-letter code, e.g. "US"
    city: str | None = None,  # text string, e.g. "Madrid"
    context_size: Literal["low", "medium", "high"] | str = "medium",
    reasoning_effort: Literal["low", "medium", "high"] | str = "low",
    model: str = "gpt-4.1-mini",
    use_search: bool = True,
    validate: bool = True,
    response_format: Response | None = None,
) -> SearchResult | OAIResponse:
    """Call OpenAI Responses API with Web Search enabled (async).

    API Docs:
    - https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses
    """
    client = AsyncOpenAI()

    params: dict = {"model": model, "input": prompt}
    if "-5" in model:
        params["reasoning"] = {"effort": reasoning_effort}

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

    if response_format is None:
        response = await client.responses.create(**params)

        if validate:
            return validate_openai(response)

        return response

    # Use .parse endpoint to get structured response
    params["response_format"] = response_format
    params["messages"] = [{"role": "user", "content": params.pop("input")}]
    response = await client.chat.completions.parse(**params)
    message = response.choices[0].message
    return message.parsed


def coords(country: str) -> gaitypes.LatLng | None:
    latlng = country_coords(country)
    if latlng is None:
        return None
    return gaitypes.LatLng(latitude=latlng[0], longitude=latlng[1])


async def query_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    country: str | None = None,  # 2-letter code, e.g. "US"
    language: str | None = None,  # 2-letter code, e.g. "EN"
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

    lat_lng = coords(country) if country else None
    if lat_lng or language:
        tool_config = gaitypes.ToolConfig(
            retrieval_config=gaitypes.RetrievalConfig(
                lat_lng=lat_lng,
                language_code=language,
            ),
        )
        config["tool_config"] = tool_config

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
    country: str | None = None,  # 2-letter code, e.g. "US"
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
        return SearchResult(answer=answer, sources=[])

    return answer


LLMS = {
    "openai": query_openai,
    "google": query_gemini,
    "xai": query_xai,
    "deepseek": query_deepseek,
}


def validate_model(client: str, model: str) -> None:
    """Validate if the given client and model are supported."""
    if client not in VALID_MODELS:
        raise ValueError(f"Unsupported client '{client}'. Supported: {list(VALID_MODELS.keys())}.")
    if model not in VALID_MODELS[client]:
        raise ValueError(
            f"Unsupported model '{model}' for client '{client}'. Supported: {VALID_MODELS[client]}"
        )


FORMATTED_SEARCH_PROMPT = dedent("""
Based on the following search results, please format the information according
to the requested structure:

# Search Query

{prompt}

# Search Results

{answer}

# Sources

{sources}

Please extract and format the relevant information from these search results.
""")


async def search_with_format(
    prompt: str,
    model: str,
    response_format: Response,
    **kwds,
) -> Response:
    """Perform a web search using the specified model and return structured response."""

    client, model = model.split("/", 1)
    validate_model(client, model)

    search_func = LLMS[client]
    search_result = await search_func(
        prompt=prompt,
        model=model,
        use_search=True,
        validate=True,
        **kwds,
    )

    sources = "\n".join([f"- {source.title}: {source.url}" for source in search_result.sources])

    prompt = FORMATTED_SEARCH_PROMPT.format(
        prompt=prompt,
        answer=search_result.answer,
        sources=sources,
    )

    return await ask(
        prompt=prompt,
        model="openai/gpt-4.1-mini",
        response_model=response_format,
    )


SUPPORT_COUNTRY = ["openai", "xai", "google"]


async def gather(  # noqa: PLR0913
    prompts: str | list[str],
    model: str = "openai/gpt-4.1-mini",
    use_search: bool = True,
    country: str | None = None,  # 2-letter code, e.g. "US"
    validate: bool = True,
    policies: dict[str, Any] | None = None,
    execute: bool = True,
    **kwds,
) -> list[Coroutine] | Coroutine | ResponseSet | SearchResult:
    """Simplified gather mirroring ``hasdata.gather`` using ``all_with_policies``.

    Creates one coroutine per prompt for the selected model/provider with optional
    policies (timeout, retries, semaphore, fallback, progress). When ``execute`` is
    False the wrapped coroutine objects are returned for the caller to schedule.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
        return_single = True
    else:
        return_single = False

    if not prompts:
        raise ValueError("No prompts provided.")

    client, model = model.split("/", 1)
    validate_model(client, model)

    if use_search and country and client in SUPPORT_COUNTRY:
        kwds["country"] = country

    # Partial of async function is a "coroutine factory" returning a coroutine when called
    func = partial(
        LLMS[client],
        model=model,
        use_search=use_search,
        validate=validate,
        **kwds,
    )

    kwds = [{"prompt": p} for p in prompts]
    coros = all_with_policies(func, kwds=kwds, policies=policies, labels=model)

    if not execute:
        return coros if not return_single else coros[0]

    responses = await asyncio.gather(*coros)
    if return_single:
        return responses[0]

    return ResponseSet(responses=responses, context=kwds, required=["prompt"])  # type: ignore
