"""Tools to apply SEO to LLM responses."""

import json
import re
from collections.abc import Coroutine
from functools import partial
from typing import Any, Literal

import instructor
import tldextract
from pandas import DataFrame
from pydantic import model_validator

from .. import Prompt, Response, ResponseSet, ask
from ..call import call
from ..search import query
from ..utils import LOG, Configurable, dedent, gather_with_progress

DEFAULT_MODELS = ["openai/gpt-4.1", "google/gemini-2.5-flash"]


class Brands(Response):
    names: list[str]
    """List of brand names."""


def brand_from_url(url: str) -> str:
    """Extract the brand name (domain) from a URL."""
    try:
        return tldextract.extract(url).domain or url
    except Exception:
        return url


async def find_competitors(
    brands: list[str] | None = None,
    sector: str | None = None,
    market: str | None = None,
    max_count: int = 5,
    model: str = "openai/gpt-4.1",
) -> list[str]:
    """Find a list of competitor brands using LLM with live search."""
    brands = brands or []

    if not brands and not sector:
        raise ValueError("At least one of 'brands' or 'sector' must be provided.")

    prompt = f"Identify up to {max_count} real competing brands"
    prompt += f" for {brands}" if brands else ""
    prompt += f" in the '{sector}' sector" if sector else ""
    prompt += f" in the '{market}' market" if market else ""

    LOG.info(f"Searching for competitor brands with prompt:\n{prompt}")
    search_result = await query(prompts=[prompt], model=model)
    search_result = search_result[0]
    search_str = (
        search_result.text + "\n\n" + "\n".join(c.url for c in search_result.references or [])
    )

    extraction_prompt = dedent(f"""
    Extract a list of competitor brand names from the search result below, which contains a
    text summary of a search for competitors as well as a list of references used in the
    search.

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
    LOG.info(f"Found these competitors brands: {competitors}")
    return competitors


async def commercial_prompts(
    n: int,
    language: str = "English",
    sector: str | None = None,
    market: str | None = None,
    brands: list[str] | None = None,
    include_brands: Literal["never", "sometimes", "always"] = "sometimes",
) -> list[str]:
    """Generate N realistic commercial/consumer search queries using an LLM meta-instruction."""
    if not brands and not sector:
        raise ValueError("At least one of 'brands' or 'sector' must be provided.")

    prompt = dedent(f"""
    Generate {n} unique, concise search queries for consumer/commercial intent. Cover realistic
    user intents like comparisons, transactional queries, alternatives, trust/regulatory, and
    location nuances.
    """)

    prompt += f" Generate prompts in the '{language}' language."
    if sector:
        prompt += f" Focus on the '{sector}' sector."
    if market:
        prompt += f" Focus on the '{market}' market."

    if brands:
        prompt += f" If brand context helps, consider these brands: {brands}."

        if include_brands == "always":
            prompt += " Always include at least one of the competitor brand names explicitly in every query."
        elif include_brands == "never":
            prompt += " Do not mention any brand names explicitly in the queries though."

    prompt += " Strictly return a JSON array of strings. No numbering, no prose, no code fences."

    LOG.info(f"Generating commercial search queries with prompt:\n{prompt}")
    queries = await ask(prompt, model="openai/gpt-4.1", response_model=list[str])
    LOG.info(
        f"Generated {len(queries)} commercial search queries: {json.dumps(queries, indent=2)}"
    )
    return queries


async def query_with_models_iter(
    prompts: list[str],
    models: list[str],
    use_search: bool = True,
) -> DataFrame | Any:
    """Run a list of prompts through a list of models and return a combined DataFrame.

    Iterates over models one by one.
    """
    dfs = []
    for i, model in enumerate(models):
        LOG.info(f"[{i + 1}/{len(models)}] Running queries through {model}")
        model_results = await query(
            prompts=prompts,
            model=model,
            use_search=use_search,
            max_concurrent=10,
        )
        model_df = model_results.to_pandas()
        model_df = model_df.rename(
            columns={
                "text": f"text_{model}",
                "references": f"references_{model}",
            }
        )
        LOG.info(f"Model search results:\n{model_df}")
        dfs.append(model_df)

    df = dfs[0]
    for other_df in dfs[1:]:
        df = df.merge(other_df, on="prompt", how="outer")

    return df


async def query_with_models(
    prompts: list[str],
    models: list[str],
    use_search: bool = True,
    progress_callback: Coroutine | None = None,
) -> DataFrame | Any:
    """Run a list of prompts through a list of models and return a combined DataFrame.

    Gathers all model and prompt comnbinations concurrently.
    """
    coros = []
    for model in models:
        coros.extend(
            await query(
                prompts=prompts,
                model=model,
                use_search=use_search,
                max_concurrent=10,
                return_coros=True,
            )
        )

    responses = await gather_with_progress(
        coros,  # type: ignore
        min_iters=max(1, int(len(coros) / 20)),
        progress_callback=progress_callback,
    )

    df = ResponseSet(responses=responses, context=None, required=None).to_pandas()
    df["prompt"] = prompts * len(models)
    df["model"] = [model for model in models for _ in range(len(prompts))]

    # Pivot the dataframe so that unique values in "prompt" become rows, and "text" and "reference"
    # columns are prefixed with the model name
    df_pivot = df.pivot(index="prompt", columns="model", values=["text", "references"])  # noqa: PD010
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    return df_pivot.reset_index()


def token_rank_in_text(
    text: str,
    tokens: list[str],
    whole_word: bool = True,
) -> list[str] or None:
    """Find mention of first(!) token in text, returning list of (token, rank) tuples."""
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

    return [{"brand": token, "position": pos} for token, pos in matches if pos is not None] or None


def add_brand_ranks(search_result: DataFrame, brands: list[str]) -> DataFrame:
    """Add brand rank columns to a search result DataFrame."""
    for col in search_result.columns:
        if col.startswith("text_"):
            search_result[f"{col}_ranks"] = search_result[col].apply(
                partial(token_rank_in_text, tokens=brands)
            )
        elif col.startswith("references_"):
            search_result[f"{col}_url_pos"] = search_result[col].apply(
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
    emoji_flags: bool = False,
) -> DataFrame:
    """Summarize brand ranks in a results DataFrame."""
    own = [b.lower() for b in own]
    competitors = [b.lower() for b in competitors]
    for model in models:
        if f"text_{model}_ranks" in df.columns:
            # Text mentions and rank for own brand and competitors
            df[f"own_in_txt_{model}"] = df[f"text_{model}_ranks"].apply(
                lambda x: in_strings(own, x)
            )
            df[f"own_in_txt_pos_{model}"] = df[f"text_{model}_ranks"].apply(
                lambda x: pos_in_strings(own, x)
            )
            df[f"cmp_in_txt_{model}"] = df[f"text_{model}_ranks"].apply(
                lambda x: in_strings(competitors, x)
            )
            df[f"cmp_in_txt_pos_{model}"] = df[f"text_{model}_ranks"].apply(
                lambda x: pos_in_strings(competitors, x)
            )

        if f"references_{model}_url_pos" in df.columns:
            # URL mentions and rank for own brand and competitors
            df[f"own_in_refs_{model}"] = df[f"references_{model}_url_pos"].apply(
                lambda x: in_dicts(own, x, "brand")
            )
            df[f"own_in_refs_pos_{model}"] = df[f"references_{model}_url_pos"].apply(
                lambda x: pos_in_dicts(own, x, "brand")
            )
            df[f"cmp_in_refs_{model}"] = df[f"references_{model}_url_pos"].apply(
                lambda x: in_dicts(competitors, x, "brand")
            )
            df[f"cmp_in_refs_pos_{model}"] = df[f"references_{model}_url_pos"].apply(
                lambda x: pos_in_dicts(competitors, x, "brand")
            )

    # Sum of own/competitor mentions across models per row/prompt
    own_in_txt_cols = [f"own_in_txt_{model}" for model in models]
    df["own_in_txt_count"] = df[own_in_txt_cols].sum(axis=1)

    own_in_refs_cols = [f"own_in_refs_{model}" for model in models]
    df["own_in_refs_count"] = df[own_in_refs_cols].sum(axis=1)

    cmp_in_txt_cols = [f"cmp_in_txt_{model}" for model in models]
    df["cmp_in_txt_count"] = df[cmp_in_txt_cols].sum(axis=1)

    cmp_in_refs_cols = [f"cmp_in_refs_{model}" for model in models]
    df["cmp_in_refs_count"] = df[cmp_in_refs_cols].sum(axis=1)

    if emoji_flags:
        for col in df.columns:
            if (
                (col.startswith(("own_in_", "cmp_in_")))
                and ("_pos" not in col)
                and ("_count" not in col)
            ):
                df[col] = df[col].replace({True: "✅", False: "❌"})
    return df


class GeoConfig(Configurable):
    """Configuration for GEO analysis (LLM brand mentions and ranks)."""

    models: list[str] | None = None
    """List of LLM models to evaluate."""
    prompts_seed: list[str] | None = None
    """List of seed prompts."""
    prompts_max: int = 100
    """Maximum number of prompts to generate using LLM."""
    prompts_language: str = "English"
    """Language for generated prompts."""
    brands_seed: list[str] | None = None
    """List of seed brand names or URLs."""
    brands_max: int = 10
    """Maximum number of competitor brands to identify using LLM."""
    brands_model: str = "openai/gpt-4.1"
    """LLM model to use for competitor brand identification."""
    brands_in_prompt: Literal["never", "sometimes", "always"] = "never"
    """Whether to include brand names in generated prompts."""
    own_brands: list[str] | None = None
    """List of own brand names or URLs."""
    sector: str | None = None
    """Sector to focus on."""
    market: str | None = None
    """Market to focus on."""
    use_search: bool = True
    """Whether to enable web/live search when evaluating LLMs."""

    @model_validator(mode="after")
    def check_models(self):
        if self.models is None or not self.models:
            self.models = DEFAULT_MODELS
        return self


async def analyse(cfg: GeoConfig, progress_callback: Coroutine | None = None) -> DataFrame | Any:
    """Run a list of prompts through a list of models and return a combined DataFrame."""
    LOG.info(f"Querying LLMs with\n\n{cfg}")

    # Generate brands
    brands = cfg.brands_seed.copy() if cfg.brands_seed else []
    if cfg.own_brands:
        brands += cfg.own_brands

    brands = list({brand_from_url(brand) for brand in brands})
    LOG.info(f"Using these seed brands: {brands}")

    if brands or cfg.sector:
        competitors = await find_competitors(
            brands=brands,
            sector=cfg.sector,
            market=cfg.market,
            max_count=cfg.brands_max,
            model=cfg.brands_model,
        )
        brands += competitors

    # Generate prompts
    prompts = cfg.prompts_seed.copy() if cfg.prompts_seed else []

    n_gen_prompts = max(0, cfg.prompts_max - len(prompts))
    if n_gen_prompts > 0 and (brands or cfg.sector):
        gen_prompts = await commercial_prompts(
            n=n_gen_prompts,
            language=cfg.prompts_language,
            sector=cfg.sector,
            market=cfg.market,
            brands=brands,
            include_brands=cfg.brands_in_prompt,
        )
        prompts += gen_prompts

    if not prompts:
        raise ValueError("No prompts to analyse")

    # Execute searches
    LOG.info(f"Running {len(prompts)} prompts through {len(cfg.models)} models")  # type: ignore
    df = await query_with_models(
        prompts=prompts,
        models=cfg.models,  # type: ignore
        use_search=cfg.use_search,
        progress_callback=progress_callback,
    )

    # Analyse results
    if cfg.use_search and brands:
        LOG.info("Analysing brand ranks")
        try:
            df = add_brand_ranks(df, brands=brands)
        except Exception as e:
            LOG.error(f"Error analysing brand ranks: {e}")
        else:
            if cfg.own_brands:
                LOG.info("Summarising own brand and competitor ranks")
                try:
                    df = summarize_ranks(
                        df,
                        own=cfg.own_brands,
                        competitors=[b for b in brands if b not in (cfg.own_brands or [])],
                        models=cfg.models,  # type: ignore
                        emoji_flags=True,
                    )
                except Exception as e:
                    LOG.error(f"Error summarizing brand ranks: {e}")
    elif not cfg.use_search:
        df = df.drop(columns=[col for col in df.columns if col.startswith("references_")])

    LOG.info(f"Got results dataframe:\n{df}")
    return df
