import asyncio
import time
from collections.abc import Iterator
from copy import deepcopy

import pandas as pd
from pandas import DataFrame
from pydantic import field_validator
from tqdm.auto import tqdm

from cuery import ask, asy
from cuery.seo import keywords
from cuery.templates import load_template
from cuery.utils import LOG, Configurable, dedent, render_template

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

SIMPLE_FUNNEL = {
    "Awareness / Discovery": [
        "Problem Identification",
        "Category Education",
        "Trends & Inspiration",
    ],
    "Consideration / Research": [
        "Features & Specifications",
        "Comparisons",
        "Suitability & Use Cases",
        "Social Proof & Reviews",
    ],
    "Decision / Evaluation": [
        "Pricing & Packages",
        "Availability & Location",
        "Intent-to-Act Signals",
    ],
    "Conversion / Action": ["Purchase / Signup", "Immediate Offers & Promotions"],
    "Post-Purchase / Retention & Advocacy": [
        "Usage & How-To",
        "Troubleshooting & Support",
        "Upgrades & Add-ons",
        "Community & Advocacy",
    ],
}

EXAMPLES_PROMPT = dedent("""
You task is to generate 2 to 5 initial Google search keyword examples for a particular
marketing funnel stage and category (details below). I.e. keywords a person may use in that stage
and category of a product or service search. The keywords should be broad enough to be used in the
Google Keyword Planner tool to generate a significant number of derived keyword ideas (1-3 words).

At the same time, the keywords should be specific to the market '{market}', the sector '{sector}',
as well as the brand(s) '{brand}' and its competitors. Create keywords in the language '{language}'.

Do NOT use the examples provided in the category details below, but use them as inspiration to
create new keywords in the specific market, sector and brand context. Do NOT include brand names
in the keywords. The keywords should be in the form of a list of strings.

# Funnel stage and category details

{record_template}
""").strip()


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


class Funnel(Configurable):
    """A class representing a marketing funnel with stages and categories."""

    brand: str | list[str]
    """Brand or list of brands to contextualize the funnel."""
    sector: str
    """Sector to contextualize the funnel."""
    language: str
    """Language for keyword generation as 2-letter ISO code, e.g. 'en'."""
    country: str | None = None
    """Country to contextualize the funnel as 2-letter ISO code, e.g. 'us'."""
    max_ideas_per_category: int = 200
    """Maximum number of keyword ideas to generate per category."""
    stages: list[str] | None = None
    """List of stage names to filter keyword generation. If None, all stages are processed."""
    funnel: list[dict] = FUNNEL
    """List of funnel stages and their categories."""

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

    async def seed_keywords(self, level: dict) -> list[str]:
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
        return await ask(
            prompt=prompt,
            model="openai/gpt-4.1-mini",
            response_model=list[str],
        )  # type: ignore

    async def seed(self) -> "Funnel":
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
            func=self.seed_keywords,
            kwds=[{"level": level} for _, _, level in tasks_data],
            policies=policies,
            labels="seed_keywords",
        )
        responses = await asyncio.gather(*coros)

        # Assign the results back to the funnel structure
        for (stage, cat, _), seed_keywords in zip(tasks_data, responses, strict=True):
            self.funnel[stage]["categories"][cat]["seed"] = seed_keywords

        return self

    def keywords(self) -> DataFrame:
        """Generate keyword ideas for all funnel categories using the seed keywords."""
        dfs = []
        for stage, cat, info in tqdm(self.enumerate(), total=len(self)):
            if self.stages is not None and info["stage"] not in self.stages:
                continue

            if "seed" not in info or not info["seed"]:
                LOG.warning(f"No seed keywords in {info}. Run seed() first.")
                continue

            seed_keywords = info["seed"]
            cat_dfs = []

            for seed_kwd in seed_keywords:
                cfg = keywords.GoogleKwdConfig(
                    keywords=(seed_kwd,),
                    ideas=True,
                    max_ideas=self.max_ideas_per_category,
                    language=self.language,
                    country=self.country or "",
                )

                try:
                    kwd_df = keywords.keywords(cfg)
                    kwd_df["funnel_stage"] = info["stage"]
                    kwd_df["funnel_category"] = info["category"]
                    kwd_df["funnel_seed_keyword"] = seed_kwd
                    cat_dfs.append(kwd_df)

                    # Avoid hitting rate limits
                    time.sleep(1)
                except Exception as e:
                    LOG.error(f"Error generating keywords for '{seed_kwd}' in {stage}/{cat}: {e}")

            if cat_dfs:
                cat_df = pd.concat(cat_dfs, axis=0)
                dfs.append(cat_df)

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Deduplicate keywords
        df = df.drop_duplicates(subset=["keyword"]).reset_index(drop=True)

        # Reorder columns to put funnel info first
        funnel_cols = ["funnel_stage", "funnel_category", "funnel_seed_keyword"]
        other_cols = [col for col in df if col not in funnel_cols]
        return df[funnel_cols + other_cols]
