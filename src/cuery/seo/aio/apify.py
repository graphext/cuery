"""AI Overview collection using Apify SERP actor."""

from collections.abc import Iterable

from ...search import SearchResult, Source
from ...utils import LOG
from ..serps import ApifySerpConfig, fetch_serps


async def gather(
    prompts: Iterable[str],
    country: str = "us",
    language: str = "en",
    batch_size: int = 10,
) -> list[SearchResult]:
    """Gather AI Overview results for a list of prompts using Apify SERP actor"""

    cfg = ApifySerpConfig(
        keywords=tuple(prompts),
        resultsPerPage=10,
        countryCode=country,
        searchLanguage=language,
        batch_size=batch_size,
    )

    serps = await fetch_serps(cfg)

    # Extract AI Overviews and convert to SearchResult format
    results = []
    terms = []
    for serp in serps:
        terms.append(serp.get("searchQuery", {}).get("term"))
        aio = serp.get("aiOverview", {})
        answer = aio.get("content", "")
        sources = aio.get("sources", [])
        sources = [
            Source(
                title=s.get("title", ""),
                url=s.get("url", ""),
            )
            for s in sources
        ]
        results.append(SearchResult(answer=answer, sources=sources))

    # Restore original order of results based on input prompts
    # Find the index of each term in the original prompts list
    prompt_list = list(prompts)
    term_indices = [prompt_list.index(term) if term in prompt_list else -1 for term in terms]
    print(f"term_indices for ordering: {term_indices}")
    # Sort results based on these indices
    results = [res for _, res in sorted(zip(term_indices, results), key=lambda x: x[0])]

    LOG.info(f"Extracted {len(results)} AI Overviews from {len(serps)} SERP results")
    return results
