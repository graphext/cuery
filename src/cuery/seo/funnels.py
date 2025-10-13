import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Iterator
from copy import deepcopy

import pandas as pd
from pandas import DataFrame
from pydantic import Field, field_validator
from tqdm.auto import tqdm

from .. import Response, ask, asy
from ..templates import load_template
from ..tools import TopicAssigner
from ..utils import LOG, Configurable, dedent, render_template, set_log_level
from . import keywords

FUNNEL = [
    {
        "stage": "Awareness / Discovery",
        "goal": "Problem recognition, education.",
        "categories": [
            {
                "name": "Problem Identification",
                "description": "User searches to understand or define their problem or need.",
                "keyword_patterns": ["questions", "how-to", "why", "tips", "guides"],
                "examples": [
                    "why does my back hurt when running",
                    "how to organize customer data",
                ],
                "intent": "Informational",
            },
            {
                "name": "Category Education",
                "description": "Exploring broad product/service categories without specific brands.",
                "keyword_patterns": ["types of", "what is", "overview", "guide to"],
                "examples": ["types of running shoes", "what is CRM software"],
                "intent": "Informational",
            },
            {
                "name": "Trends & Inspiration",
                "description": "Looking for ideas, new trends, or general inspiration.",
                "keyword_patterns": ["trends", "ideas", "inspiration", "popular", "latest"],
                "examples": ["latest running shoe trends 2025", "popular small business tools"],
                "intent": "Informational",
            },
        ],
    },
    {
        "stage": "Consideration / Research",
        "goal": "Compare options, evaluate solutions.",
        "categories": [
            {
                "name": "Features & Specifications",
                "description": "Interest in specific attributes or capabilities.",
                "keyword_patterns": ["feature", "specifications", "capabilities", "functions"],
                "examples": ["running shoes with arch support", "CRM with email automation"],
                "intent": "Commercial / Research",
            },
            {
                "name": "Comparisons",
                "description": "Directly comparing brands, products, or categories.",
                "keyword_patterns": ["vs", "compare", "alternatives", "best of"],
                "examples": ["Nike vs Adidas", "HubSpot vs Salesforce"],
                "intent": "Commercial / Research",
            },
            {
                "name": "Suitability & Use Cases",
                "description": "Evaluating how well a solution fits specific needs or contexts.",
                "keyword_patterns": ["best for", "ideal for", "use case", "fit for"],
                "examples": ["best shoes for marathon training", "CRM for freelancers"],
                "intent": "Commercial / Research",
            },
            {
                "name": "Social Proof & Reviews",
                "description": "Looking for opinions, ratings, testimonials.",
                "keyword_patterns": ["review", "rating", "top-rated", "customer feedback"],
                "examples": ["best-rated running shoes", "HubSpot reviews"],
                "intent": "Commercial / Research",
            },
        ],
    },
    {
        "stage": "Decision / Evaluation",
        "goal": "Prospect is close to acting but still evaluating options.",
        "categories": [
            {
                "name": "Pricing & Packages",
                "description": "Researching cost, plans, discounts, promotions.",
                "keyword_patterns": ["price", "pricing", "cost", "plan", "tier", "discount"],
                "examples": ["Nike Pegasus price", "HubSpot CRM pricing tiers"],
                "intent": "Commercial / Research",
            },
            {
                "name": "Availability & Location",
                "description": "Where or how to obtain the product/service.",
                "keyword_patterns": ["buy near me", "availability", "store", "online purchase"],
                "examples": ["buy running shoes near me", "best CRM free trial"],
                "intent": "Commercial / Research",
            },
            {
                "name": "Intent-to-Act Signals",
                "description": "Keywords showing strong intent to act soon but still evaluating options.",
                "keyword_patterns": [
                    "sign up trial",
                    "get started demo",
                    "order sample",
                    "try now",
                ],
                "examples": ["sign up for HubSpot demo", "get started with CRM trial"],
                "intent": "Commercial / Research",
            },
        ],
    },
    {
        "stage": "Conversion / Action",
        "goal": "Prospect decides to purchase or take desired action (checkout, demo, signup).",
        "categories": [
            {
                "name": "Purchase / Signup",
                "description": "Final action: completing a purchase, signing up, or starting a trial.",
                "keyword_patterns": ["buy", "checkout", "signup", "register", "demo"],
                "examples": ["buy Nike Pegasus online", "HubSpot CRM demo signup"],
                "intent": "Transactional",
            },
            {
                "name": "Immediate Offers & Promotions",
                "description": "Using discounts, coupon codes, or limited-time deals to convert.",
                "keyword_patterns": ["discount", "promo code", "deal", "offer", "coupon"],
                "examples": ["Nike Pegasus 20% off", "HubSpot CRM free trial code"],
                "intent": "Transactional",
            },
        ],
    },
    {
        "stage": "Post-Purchase / Retention & Advocacy",
        "goal": "Support existing customers, encourage loyalty or advocacy.",
        "categories": [
            {
                "name": "Usage & How-To",
                "description": "Guides, tutorials, setup instructions.",
                "keyword_patterns": ["how to", "tutorial", "setup", "guide", "instructions"],
                "examples": ["how to break in running shoes", "HubSpot CRM tutorial"],
                "intent": "Retention / Post-Purchase",
            },
            {
                "name": "Troubleshooting & Support",
                "description": "Fixing problems, maintenance, FAQs.",
                "keyword_patterns": ["help", "troubleshoot", "issue", "problem", "FAQ"],
                "examples": ["Nike Pegasus sizing issues", "HubSpot login help"],
                "intent": "Retention / Post-Purchase",
            },
            {
                "name": "Upgrades & Add-ons",
                "description": "Expanding or enhancing existing purchase.",
                "keyword_patterns": ["upgrade", "add-on", "extension", "premium features"],
                "examples": ["best insoles for running shoes", "HubSpot premium features"],
                "intent": "Retention / Post-Purchase",
            },
            {
                "name": "Community & Advocacy",
                "description": "Engagement, referrals, sharing experiences.",
                "keyword_patterns": ["forum", "community", "refer", "share", "testimonial"],
                "examples": ["running shoe user forum", "refer a friend HubSpot discount"],
                "intent": "Retention / Post-Purchase",
            },
        ],
    },
]


CUSTOM_PROMPT = dedent("""
You are a marketing expert specializing in digital marketing funnels. Your task is to adapt a
generic marketing funnel structure to a specific industrial sector and geographic market.
Given the generic funnel structure and categories (section below), modify and customize them to
fit the context of the sector '{sector}' and market '{market}'. Create a tailored marketing funnel
that reflects the unique needs, behaviors, and preferences of customers in this sector and market.
Consider factors such as common customer pain points, industry terminology, cultural nuances,
and market-specific trends. Ensure that the adapted funnel stages and categories are relevant and
actionable for SEO experts operating in this sector and market, i.e. make sure the categories
are suitable for keyword research and content marketing.

IMPORTANT: KEEP ALL FUNNEL *STAGES* (only translate them to the requested language).
Only customize the *categories* within each stage to be relevant to the specified sector and
market. I.e. adapt the category name and descriptions, keyword_patterns and examples. If not all
categories are relevant within the given sector and market context, remove or replace them with
more suitable ones. But only change the structure if really necessary.

"keyword_patterns" should be short labels describing common keyword types associated with the
category, e.g. ["questions", "how-to", "vs", "compare", "buy", "discount"] for the category
"Problem Identification". The "examples" field, on the other hand, should contain actual example
keywords a person may use in that category fitting one of the keyword patterns, e.g.
["why does my back hurt when running", "how to organize customer data"].

Make sure the example keywords are short and broad enough to be used in the Google Keyword Planner
tool to generate 100s to 1000s of derived keyword ideas (1-3 words).

Generate all funnel information in the language '{language}'.

# Generic marketing funnel

{funnel}
""")
"""A prompts to customize the generic marketing funnel and categories,
given a specific industrial sector and geographic market."""


EXAMPLES_PROMPT = dedent("""
You task is to generate 3 to 10 initial Google search keyword examples for a particular
marketing funnel stage and category (details below). I.e. keywords a person may use in that stage
and category of a product or service search. The keywords should be broad enough to be used in the
Google Keyword Planner tool to generate a significant number (100s to 1000s) of derived keyword
ideas (1-3 words).

At the same time, the keywords should be specific to the market '{market}', the sector '{sector}',
as well as the brand(s) '{brand}' and its competitors. Create keywords in the language '{language}'.

Do NOT use the examples provided in the category details below, but use them as inspiration to
create new keywords in the specific market, sector and brand context. Do NOT include brand names
in the keywords. The keywords should be in the form of a list of strings.

IMPORTANT: all keywords should consist of 1 to 3 words (not longer). Avoid long-tail keywords.

# Funnel stage and category details

{record_template}
""")


PARAPHRASE_PROMPT = dedent("""
Convert the phrase "{phrase}" to singular if it is plural and to plural if
it is singular. Then also generate a couple of paraphrases for both versions.
Do NOT make the phrases more specific or more general. They should be grammatical
variations essentially. Return a list of all variations. Do NOT pre- or postfix them
with anything, just the plain variations.
""")


class FunnelCategory(Response):
    """A category of Google search keywords within a marketing funnel stage."""

    name: str
    """The name of a keyword category within a marketing funnel stage."""
    description: str
    """A brief description of the category and its purpose."""
    keyword_patterns: list[str]
    """Common keyword patterns or phrases associated with this category."""
    intent: str
    """The primary search intent for this category
    (e.g., Informational, Commercial, Transactional, Navigational).
    """
    examples: list[str]
    """Example Google search keywords that fit within this category."""


class FunnelStage(Response):
    """A stage in the marketing funnel containing multiple keyword categories."""

    stage: str
    """The name of the marketing funnel stage (e.g., Awareness, Consideration)."""
    goal: str
    """The primary goal or objective of this funnel stage."""
    categories: list[FunnelCategory]
    """A list of keyword categories within this funnel stage."""


class Funnel(Response):
    """A complete marketing funnel with multiple stages."""

    stages: list[FunnelStage]


class Seeds(Response):
    """A list of seed keywords for a particular funnel category."""

    seeds: list[str] = Field(..., min_length=3, max_length=10)


async def custom(
    sector: str,
    language: str,
    country: str = "global",
    model: str = "openai/gpt-4.1",
    funnel: Funnel | list[dict] = FUNNEL,
) -> Funnel:
    """ "Customize a generic marketing funnel to a specific sector and market using an LLM."""
    funnel = funnel.to_dict() if isinstance(funnel, Funnel) else funnel
    prompt = CUSTOM_PROMPT.format(
        sector=sector,
        market=country,
        language=language,
        funnel=json.dumps(funnel, indent=2),
    )

    return await ask(prompt, model=model, response_model=Funnel)


def flatten_level(stage: dict, category: dict) -> dict:
    """Merge stage and category dictionaries into a single flat dictionary."""
    result = deepcopy(stage)
    result.pop("categories")
    result = result | category
    result["category"] = result.pop("name")

    field_order = [
        "stage",
        "goal",
        "category",
        "intent",
        "description",
        "keyword_patterns",
        "examples",
    ]
    field_order += [k for k in result if k not in field_order]
    return {k: result[k] for k in field_order if k in result}


async def paraphrase(phrase: str) -> list[str]:
    """"""
    prompt = PARAPHRASE_PROMPT.format(phrase=phrase)
    return await ask(prompt, model="openai/gpt-4.1", response_model=list[str])


class KeywordFunnel(Configurable):
    """A class representing a marketing funnel with stages and categories."""

    brand: str | list[str]
    """Brand or list of brands to contextualize the funnel."""
    sector: str
    """Sector to contextualize the funnel."""
    language: str
    """Language for keyword generation as 2-letter ISO code, e.g. 'en'."""
    country: str | None = None
    """Country to contextualize the funnel as 2-letter ISO code, e.g. 'us'."""
    max_ideas_per_category: int = 10_000
    """Maximum number of keyword ideas to generate per category."""
    stages: list[str] | None = None
    """List of stage names to filter keyword generation. If None, all stages are processed."""
    funnel: list[dict] = FUNNEL
    """List of funnel stages and their categories."""
    forced_seeds: str | list[str] | None = None
    """Additional keywords to always include in the keyword generation."""
    sector_seed: bool = True
    """Whether to include the sector as a keyword seed."""
    brand_seed: bool = True
    """Whether to include the brand(s) as keyword seeds."""

    @field_validator("funnel", mode="before")
    @classmethod
    def deep_copy_funnel(cls, v: list[dict]) -> list[dict]:
        """Deep copy the funnel to prevent mutation of the original."""
        return deepcopy(v)

    def __len__(self) -> int:
        """Return the total number of categories across all stages in the funnel."""
        return sum(len(stage["categories"]) for stage in self.funnel)

    def __iter__(self) -> Iterator[dict]:
        """Make the Funnel iterable over all stages and categories."""
        for stage in self.funnel:
            for category in stage["categories"]:
                yield flatten_level(stage, category)

    def enumerate(self) -> Iterator[tuple[int, int, dict]]:
        """Make the Funnel iterable over all stages and categories, yielding index and item."""
        for i, stage in enumerate(self.funnel):
            for j, category in enumerate(stage["categories"]):
                yield i, j, flatten_level(stage, category)

    def get(self, state: int | str, category: str | int | None) -> dict:
        """Get funnel subcategory details by stage index or name and category name."""
        if isinstance(state, int):
            stage = self.funnel[state]
        else:
            stage = next(s for s in self.funnel if s["stage"] == state)

        if category is None:
            return stage

        if isinstance(category, int):
            cat = stage["categories"][category]
        else:
            cat = next(c for c in stage["categories"] if c["name"] == category)

        return flatten_level(stage, cat)

    def __getitem__(self, key: str | int | tuple[int | str, str | int | None]) -> dict:
        """Get funnel subcategory details by stage index or name and category name."""
        state, category = key if isinstance(key, tuple) else (key, None)
        return self.get(state, category)

    def to_pandas(self) -> DataFrame:
        """Convert the funnel structure to a pandas DataFrame for analysis."""
        df = pd.DataFrame.from_dict(self.funnel)
        df = df.explode("categories").reset_index(drop=True)
        df = pd.concat([df, pd.json_normalize(df["categories"])], axis=1)
        return df.drop(columns=["categories"]).rename(columns={"name": "category"})

    async def _seeds(self, level: dict) -> list[str]:
        """Generate initial keyword examples for a particular funnel stage and category"""
        record_template = load_template("record_to_text")
        prompt = EXAMPLES_PROMPT.format(
            market=self.country or "global",
            sector=self.sector,
            language=self.language,
            brand=", ".join(self.brand) if isinstance(self.brand, list) else self.brand,
            record_template=record_template,
        )
        prompt = render_template(prompt, record=level)
        seeds = await ask(
            prompt=prompt,
            model="openai/gpt-4.1-mini",
            response_model=Seeds,
        )  # type: ignore
        return seeds.seeds

    async def seed(self) -> "KeywordFunnel":
        """Generate initial keyword seeds for all funnel categories."""
        pbar = tqdm(total=len(self), desc="Seeding funnel keywords")
        policies = {
            "timeout": 120,
            "n_concurrent": 100,
            "retries": 3,
            "fallback": [],
            "timer": True,
            "pbar": pbar,
        }

        # Collect all levels with their stage and category indices
        tasks_data = [
            (stage, cat, level)
            for stage, cat, level in self.enumerate()
            if self.stages is None or level["stage"] in self.stages
        ]

        coros = asy.all_with_policies(
            func=self._seeds,
            kwds=[{"level": level} for _, _, level in tasks_data],
            policies=policies,
            labels="seed_keywords",
        )
        responses = await asyncio.gather(*coros)

        # Assign the results back to the funnel structure
        for (stage, cat, _), seed_keywords in zip(tasks_data, responses, strict=True):
            self.funnel[stage]["categories"][cat]["seed"] = seed_keywords

        return self

    def iter_seeds(self) -> Iterator[tuple[str, str, str]]:
        """Iterate over all funnel categories and their seed keywords."""
        for _, _, info in self.enumerate():
            if "seed" in info and info["seed"]:
                for seed in info["seed"]:
                    yield seed, info["stage"], info["category"]

    async def explicit_seeds(self) -> list[str]:
        seeds = []
        if self.forced_seeds:
            if isinstance(self.forced_seeds, str):
                seeds.append(self.forced_seeds)
            else:
                seeds.extend(self.forced_seeds)

        if self.brand_seed:
            if isinstance(self.brand, str):
                seeds.append(self.brand)
            else:
                seeds.extend(self.brand)

        if self.sector_seed:
            seeds.append(self.sector)

            try:
                variations = await paraphrase(self.sector)
                seeds.extend(variations)
            except Exception as e:
                LOG.error(f"Error generating sector paraphrases for '{self.sector}': {e}")

        return list(set(seeds))

    def keyword_ideas(
        self,
        seed: str,
        stage: str | None = None,
        category: str | None = None,
    ) -> DataFrame | None:
        """Generate keyword ideas for a given seed keyword using Google Keyword Planner."""
        time.sleep(1)  # Avoid hitting API rate limits
        cfg = keywords.GoogleKwdConfig(
            keywords=(seed,),
            ideas=True,
            max_ideas=self.max_ideas_per_category,
            language=self.language,
            country=self.country or "",
        )
        try:
            with set_log_level(LOG, "WARNING"):
                df = keywords.keywords(cfg)

            df["funnel_stage"] = stage
            df["funnel_category"] = category
            df["funnel_seed_keyword"] = seed
            LOG.info(f"Generated {len(df)} keywords for '{seed}' in {stage}/{category}")
            return df
        except Exception as e:
            LOG.error(f"Error generating keywords for '{seed}': {e}")
            return None

    async def keywords(self) -> DataFrame:
        """Generate keyword ideas for all funnel categories using the seed keywords."""

        # Collect all seeds from the funnel and explicit seeds
        funnel_seeds = list(self.iter_seeds())
        explicit_seeds = [(s, None, None) for s in await self.explicit_seeds()]
        all_seeds = funnel_seeds + explicit_seeds

        # Fetch keyword ideads for all seeds
        dfs = []
        fetched = set()
        for seed, stage, category in tqdm(all_seeds):
            if seed in fetched:
                continue
            kwd_df = self.keyword_ideas(seed, stage, category)
            if kwd_df is not None and not kwd_df.empty:
                dfs.append(kwd_df)
                fetched.add(seed)

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["keyword"]).reset_index(drop=True)

        # Reorder columns to put funnel info first
        funnel_cols = ["funnel_stage", "funnel_category", "funnel_seed_keyword"]
        other_cols = [col for col in df if col not in funnel_cols]
        return df[funnel_cols + other_cols]

    def hierarchy(self) -> dict:
        """Return the funnel hierarchy as a nested dictionary."""
        stages = defaultdict(list)
        for level in self:
            stages[level["stage"]].append(level["category"])
        return dict(stages)

    async def categorize(self, keywords: DataFrame) -> DataFrame:
        """Categorize keywords into funnel stages and categories using an LLM."""
        prompt = dedent("""
        You'll receive Google search keywords, along with a marketing funnel stage and category.
        Some of the keywords may not fit in the provided funnel/category. Your task is to assign
        each keyword to the most appropriate funnel stage (topic) and category (subtopic).
        """)

        topics = self.hierarchy()
        assigner = TopicAssigner(
            model="openai/gpt-4.1",
            topics=topics,
            records=keywords,
            attrs=["keyword", "funnel_stage", "funnel_category"],
            instructions=prompt,
        )

        responses = await assigner(n_concurrent=100)

        df = keywords.merge(
            responses,
            on=["keyword", "funnel_stage", "funnel_category"],
            how="left",
        )
        df = df.drop(columns=["funnel_stage", "funnel_category", "funnel_seed_keyword"])
        return df.rename(columns={"topic": "funnel_stage", "subtopic": "funnel_category"})

    async def run(self):
        """Customize the funnel and generate keyword ideas."""
        LOG.info("Customizing funnel...")
        customized = await custom(
            funnel=self.funnel,
            sector=self.sector,
            country=self.country or "global",
            language=self.language,
        )
        self.funnel = customized.to_dict()["stages"]
        LOG.info("Got funnel hierarchy:")
        LOG.info(json.dumps(self.hierarchy(), indent=2, ensure_ascii=False))

        LOG.info("Seeding funnel...")
        await self.seed()

        LOG.info("Generating keywords...")
        df = await self.keywords()
        LOG.info(f"Generated a total of {len(df)} unique keywords.")

        LOG.info("Categorizing keywords...")
        df = await self.categorize(df)

        return df
