"""Tools to apply SEO to LLM responses."""

import asyncio
import json
import re
from collections.abc import Coroutine
from contextlib import suppress
from functools import partial
from typing import Any, Literal

import instructor
import tldextract
from pandas import DataFrame
from pydantic import model_validator
from tqdm.auto import tqdm

from .. import Prompt, Response, ResponseSet, ask, search
from ..call import call
from ..search import SearchResult
from ..utils import LOG, Configurable, dedent
from . import sources
from .aio import hasdata

DEFAULT_MODELS = ["openai/gpt-4.1-mini", "google/gemini-2.5-flash"]


class Brands(Response):
    names: list[str]
    """List of brand names."""


def normalize_brand(url: str) -> str:
    """Extract the brand name (domain) from a URL."""
    with suppress(Exception):
        url = tldextract.extract(url).domain or url

    return url.lower().strip()


async def find_competitors(
    brands: list[str] | None = None,
    sector: str | None = None,
    market: str | None = None,
    max_count: int = 5,
    model: str = "openai/gpt-4.1",
    known: list[str] | None = None,
) -> list[str]:
    """Find a list of competitor brands using LLM with live search."""
    brands = brands or []
    known = known or []

    if not brands and not sector:
        raise ValueError("At least one of 'brands' or 'sector' must be provided.")

    prompt = f"Which are the {max_count} most important competing brands"
    prompt += f" for {brands}" if brands else ""
    prompt += f" in the '{sector}' sector" if sector else ""
    prompt += f" in the '{market}' market" if market else ""
    if known:
        prompt += f". Do NOT include these known competitors: {known}."

    LOG.info(f"Searching for competitor brands with prompt:\n{prompt}")
    search_result: SearchResult = await search.gather(prompts=prompt, model=model)  # type: ignore
    search_str = (
        search_result.answer + "\n\n" + "\n".join(c.url for c in search_result.sources or [])
    )

    extraction_prompt = dedent(f"""
    Extract a list of up to {max_count} competitor brand names from the Search Result section
    below, which contains a text summary of a search for competitors of original brands {brands}
    as well as a list of references used in the search.

    Do NOT include any of these already known competitors: {known}, nor any of the original brands.

    Return the new competitors in the order of importance (e.g. by approx. market share or company
    size).

    # Search Result

    {search_str}
    """)

    LOG.info("Extracting competitor names from search result")
    client = instructor.from_provider("openai/gpt-4.1", async_client=True)
    competitors = await call(
        client=client,
        prompt=Prompt.from_string(extraction_prompt),
        context=None,
        response_model=Brands,
    )

    competitors = list({name.lower() for name in competitors.names})  # type: ignore
    competitors = [c for c in competitors if c not in brands and c not in known][:max_count]
    LOG.info(f"Found these new competitors brands: {competitors}")
    return competitors


async def generate_prompts(
    n: int,
    intents: list[str] | None = None,
    language: str = "English",
    sector: str | None = None,
    market: str | None = None,
    brands: list[str] | None = None,
    include_brands: Literal["never", "sometimes", "always"] = "sometimes",
    seed_prompts: list[str] | None = None,
) -> list[str]:
    """Generate N realistic commercial/consumer search queries using an LLM meta-instruction."""
    if not brands and not sector:
        raise ValueError("At least one of 'brands' or 'sector' must be provided.")

    intents = intents or ["commercial", "transactional"]

    prompt = dedent(f"""
    Generate {n} unique, concise LLM prompts with one or more of the following intents:
    {intents}. Cover realistic user intentions like comparisons, alternatives, trust/regulatory,
    location specific queries etc. The prompts should be similar to Google search queries
    but adapted to how users would ask an LLM.
    """)

    prompt += f" Generate prompts in the '{language}' language."
    if sector:
        prompt += f" Focus on the '{sector}' sector."
    if market:
        prompt += f" Focus on the '{market}' market."

    if brands:
        prompt += f" If brand context helps, consider these brands: {brands}."

        if include_brands == "always":
            prompt += " Always include at least one of the brand names explicitly in every query."
        elif include_brands == "never":
            prompt += " Do not mention any brand names explicitly in the queries though."

    prompt += " Strictly return a JSON array of strings. No numbering, no prose, no code fences."

    if seed_prompts:
        prompt += (
            "Do NOT generate prompts that are semantically equivalent to these initial "
            f"seed prompts:\n\n{json.dumps(seed_prompts, indent=2)}"
        )

    LOG.info(f"Generating search queries with prompt:\n{json.dumps(prompt, indent=2)}")
    queries = await ask(prompt, model="openai/gpt-4.1", response_model=list[str])
    LOG.info(
        f"Generated {len(queries)} commercial search queries: {json.dumps(queries, indent=2)}"
    )
    return queries


async def query_ais(
    prompts: list[str],
    models: list[str],
    use_search: bool = True,
    search_country: str | None = None,
    progress_callback: Coroutine | None = None,
    to_pandas: bool = True,
) -> DataFrame | ResponseSet:
    """Run a list of prompts through a list of models and return a combined DataFrame.

    Gathers all model and prompt comnbinations concurrently.
    """
    coros = []
    n_total = len(prompts) * len(models)
    pbar = tqdm(total=n_total)
    policies = {
        "n_concurrent": 100,
        "timeout": 300,
        "retries": 3,
        "wait_max": 60,
        "fallback": SearchResult.fallback(),
        "pbar": pbar,
        "progress_callback": progress_callback,
        "timer": True,
        "min_iters": max(1, int(n_total / 20)),
    }

    for model in models:
        if model == "google/ai-overview":
            coros.extend(
                await hasdata.gather(
                    prompts=prompts,
                    country=search_country,
                    policies=policies.copy() | {"n_concurrent": 14},
                    execute=False,
                )
            )
        else:
            coros.extend(
                await search.gather(
                    prompts=prompts,
                    model=model,
                    use_search=use_search,
                    country=search_country,
                    policies=policies,
                    execute=False,
                )  # type: ignore
            )

    responses = await asyncio.gather(*coros)
    responses = ResponseSet(responses=responses, context=None, required=None)
    if not to_pandas:
        return responses

    df = responses.to_pandas()
    df["prompt"] = prompts * len(models)
    df["model"] = [model for model in models for _ in range(len(prompts))]

    # Pivot the dataframe so that unique values in "prompt" become rows, and "text" and "reference"
    # columns are prefixed with the model name
    df_pivot = df.pivot(index="prompt", columns="model", values=["answer", "sources"])  # noqa: PD010
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    return df_pivot.reset_index()


def token_rank_in_text(
    text: str,
    tokens: list[str],
    whole_word: bool = True,
) -> list[str] or None:
    """Find mention of first(!) token in text, returning list of (token, rank) tuples."""
    if not text or not tokens:
        return None

    flags = re.IGNORECASE
    pattern = r"\b{token}\b" if whole_word else "({token})"
    matches = []
    for token in tokens:
        match = re.search(pattern.format(token=re.escape(token)), text, flags=flags)
        if match:
            matches.append((token, match.start()))
        else:
            matches.append((token, None))

    matches = sorted(matches, key=lambda x: float("inf") if x[1] is None else x[1])
    matches = [token for token, pos in matches if pos is not None]
    return matches or None


def token_pos_in_list(
    items: list[dict],
    tokens: list[str],
    key: str = "url",
    whole_word: bool = True,
    include_none: bool = False,
) -> list[dict] | None:
    """Find mention of first token in list of strings, returning list of (token, position) tuples."""
    if not items or not tokens:
        return None

    flags = re.IGNORECASE
    pattern = r"\b{token}\b" if whole_word else "({token})"
    matches = []
    for token in tokens:
        pos = None
        for i, item in enumerate(items):
            if re.search(pattern.format(token=re.escape(token)), item.get(key, ""), flags=flags):
                pos = i
                break
        matches.append((token, pos))

    matches = sorted(matches, key=lambda x: float("inf") if x[1] is None else x[1])
    if include_none:
        return matches or None

    return [{"name": token, "position": pos} for token, pos in matches if pos is not None] or None


def add_brand_ranks(search_result: DataFrame, brands: list[str]) -> DataFrame:
    """Add brand rank columns to a search result DataFrame."""
    for col in search_result.columns:
        if col.startswith("answer_"):
            search_result[f"{col}_brand_ranking"] = search_result[col].apply(
                partial(token_rank_in_text, tokens=brands)
            )
        elif col.startswith("sources_"):
            search_result[f"{col}_brand_positions"] = search_result[col].apply(
                partial(token_pos_in_list, tokens=brands, key="url")
            )

    return search_result


def in_strings(values: list[str], lst: list[str] | None) -> bool:
    """Check if any of values is among list items."""
    if lst is None:
        return False

    return any(val.lower() in [item.lower() for item in lst] for val in values)


def pos_in_strings(values: list[str], lst: list[str] | None) -> int | None:
    """Find first position of any of values among list items. To Do: optimize."""
    if lst is None:
        return None

    values = [v.lower() for v in values]
    lst = [item.lower() for item in lst]
    positions = [lst.index(val) for val in values if val in lst]
    return min(positions) + 1 if positions else None


def in_dicts(values: list, lst: list[dict] | None, key: str) -> bool:
    """Check if any of values is in any dict under the specified key."""
    if lst is None:
        return False

    return any(d.get(key, "").lower() in [v.lower() for v in values] for d in lst)


def pos_in_dicts(values: list, lst: list[dict] | None, key: str) -> int | None:
    """Find first position of any of values in any dict under the specified key"""
    if lst is None:
        return None

    values = [v.lower() for v in values]
    positions = [idx for idx, d in enumerate(lst) if d.get(key, "").lower() in values]
    return min(positions) + 1 if positions else None


def summarize_ranks(
    df: DataFrame,
    own: list[str],
    competitors: list[str],
    models: list[str],
) -> DataFrame:
    """Summarize brand ranks in a results DataFrame."""
    own = [b.lower() for b in own]
    competitors = [b.lower() for b in competitors]
    for model in models:
        source_col = f"answer_{model}_brand_ranking"
        if source_col in df.columns:
            # Text mentions and rank for own brand and competitors
            df[f"brand_mentioned_in_answer_{model}"] = df[source_col].apply(
                lambda x: in_strings(own, x)
            )
            df[f"brand_position_in_answer_{model}"] = df[source_col].apply(
                lambda x: pos_in_strings(own, x)
            )
            df[f"competitor_mentioned_in_answer_{model}"] = df[source_col].apply(
                lambda x: in_strings(competitors, x)
            )
            df[f"competitor_position_in_answer_{model}"] = df[source_col].apply(
                lambda x: pos_in_strings(competitors, x)
            )

        source_col = f"sources_{model}_brand_positions"
        if source_col in df.columns:
            # URL mentions and rank for own brand and competitors
            df[f"brand_mentioned_in_sources_{model}"] = df[source_col].apply(
                lambda x: in_dicts(own, x, "name")
            )
            df[f"brand_position_in_sources_{model}"] = df[source_col].apply(
                lambda x: pos_in_dicts(own, x, "name")
            )
            df[f"competitor_mentioned_in_sources_{model}"] = df[source_col].apply(
                lambda x: in_dicts(competitors, x, "name")
            )
            df[f"competitor_position_in_sources_{model}"] = df[source_col].apply(
                lambda x: pos_in_dicts(competitors, x, "name")
            )

    # Sum of own/competitor mentions across models per row/prompt
    own_ans_cols = [f"brand_mentioned_in_answer_{model}" for model in models]
    df["brand_mentioned_in_answer_count"] = df[own_ans_cols].sum(axis=1)

    cmp_ans_cols = [f"competitor_mentioned_in_answer_{model}" for model in models]
    df["competitor_mentioned_in_answer_count"] = df[cmp_ans_cols].sum(axis=1)

    own_src_cols = [col for m in models if (col := f"brand_mentioned_in_sources_{m}") in df]
    df["brand_mentioned_in_sources_count"] = df[own_src_cols].sum(axis=1)

    cmp_src_cols = [col for m in models if (col := f"competitor_mentioned_in_sources_{m}") in df]
    df["competitor_mentioned_in_sources_count"] = df[cmp_src_cols].sum(axis=1)

    return df


class GeoConfig(Configurable):
    """Configuration for GEO analysis (LLM brand mentions and ranks)."""

    brands: list[str] | None = None
    """List of own(!) brand names or URLs."""
    models: list[str] | None = None
    """List of LLM models to evaluate."""
    prompts: list[str] | None = None
    """List of seed prompts."""
    prompts_max: int = 20
    """Maximum number of prompts to generate using LLM."""
    prompt_intents: list[str] | None = None
    """List of user intents to focus on in generated prompts."""
    prompt_language: str = "English"
    """Language for generated prompts."""
    brands_in_prompt: Literal["never", "sometimes", "always"] = "never"
    """Whether to include brand names in generated prompts."""
    competitors: list[str] | None = None
    """List of seed brand names or URLs."""
    competitors_max: int = 10
    """Maximum number of competitor brands to identify using LLM."""
    competitors_model: str = "openai/gpt-4.1"
    """LLM model to use for competitor brand identification."""
    sector: str | None = None
    """Sector to focus on."""
    market: str | None = None
    """Market to focus on."""
    use_search: bool = True
    """Whether to enable web/live search when evaluating LLMs."""
    search_country: str | None = None
    """Country code for search localisation, e.g. 'us', 'uk', 'de '."""

    @model_validator(mode="after")
    def check_params(self):
        if self.models is None or not self.models:
            self.models = DEFAULT_MODELS
        return self


async def analyse(cfg: GeoConfig, progress_callback: Coroutine | None = None) -> DataFrame | Any:
    """Run a list of prompts through a list of models and return a combined DataFrame."""
    LOG.info(f"Querying LLMs with\n\n{cfg}")

    # Prepare brands
    brands = list({normalize_brand(brand) for brand in cfg.brands or []})
    competitors = list({normalize_brand(comp) for comp in cfg.competitors or []})
    LOG.info(f"Using these seed brands: {brands}, and these competitors: {competitors}")

    # Find competitors
    n_gen_comps = max(0, cfg.competitors_max - len(competitors))
    if (brands or cfg.sector) and (n_gen_comps > 0):
        competitors += await find_competitors(
            brands=brands,
            sector=cfg.sector,
            market=cfg.market,
            max_count=n_gen_comps,
            model=cfg.competitors_model,
            known=competitors,
        )

    all_brands = brands + competitors

    # Generate prompts
    prompts = cfg.prompts.copy() if cfg.prompts else []

    n_gen_prompts = max(0, cfg.prompts_max - len(prompts))
    if n_gen_prompts > 0 and (all_brands or cfg.sector):
        gen_prompts = await generate_prompts(
            n=n_gen_prompts,
            intents=cfg.prompt_intents,
            language=cfg.prompt_language,
            sector=cfg.sector,
            market=cfg.market,
            brands=all_brands,
            include_brands=cfg.brands_in_prompt,
            seed_prompts=cfg.prompts,
        )
        prompts += gen_prompts

    if not prompts:
        raise ValueError("No prompts to analyse")

    # Execute searches
    LOG.info(f"Running {len(prompts)} prompts through {len(cfg.models)} models")  # type: ignore
    df: DataFrame = await query_ais(
        prompts=prompts,
        models=cfg.models,  # type: ignore
        use_search=cfg.use_search,
        search_country=cfg.search_country,
        progress_callback=progress_callback,
        to_pandas=True,
    )

    # Analyse results
    if brands:
        LOG.info("Analysing brand ranks")
        try:
            df = add_brand_ranks(df, brands=all_brands)
        except Exception as e:
            LOG.error(f"Error analysing brand ranks: {e}")
        else:
            if cfg.brands:
                LOG.info("Summarising own brand and competitor ranks")
                try:
                    df = summarize_ranks(
                        df,
                        own=cfg.brands,
                        competitors=competitors,
                        models=cfg.models,  # type: ignore
                    )
                except Exception as e:
                    LOG.error(f"Error summarizing brand ranks: {e}")

    # Clean up dataframe
    def clean_name(name):
        providers = search.VALID_MODELS.keys()
        for provider in providers:
            name = name.replace(f"{provider}/", "")

        return re.sub(r"[^a-zA-Z0-9]", "_", name)

    df.columns = [clean_name(col) for col in df.columns]
    LOG.info(f"Cleaned dataframe columns: {df.columns.tolist()}")

    # Extract and categorize sources
    LOG.info("Processing and categorizing sources cited by LLMs")
    df = await sources.process_sources(df, models=cfg.models)  # type: ignore

    # Column order: prompt, answers, sources, summary columns, other columns
    answer_cols = [f"answer_{clean_name(model)}" for model in cfg.models]  # type: ignore
    source_cols = [f"sources_{clean_name(model)}" for model in cfg.models]  # type: ignore
    count_cols = [col for col in df.columns if re.search(r"(brand|competitor)_.*_count", col)]
    own_in_cols = [
        col
        for col in df.columns
        if re.search(r"brand_mentioned_in_|brand_position_in_", col) and col not in count_cols
    ]
    cmp_in_cols = [
        col
        for col in df.columns
        if re.search(r"competitor_mentioned_in_|competitor_position_in_", col)
        and col not in count_cols
    ]
    sorted_cols = answer_cols + source_cols + count_cols + own_in_cols + cmp_in_cols
    other_cols = [col for col in df.columns if col not in (["prompt"] + sorted_cols)]
    df = df[["prompt"] + sorted_cols + other_cols]

    LOG.info(f"Got results dataframe:\n{df}")
    return df
