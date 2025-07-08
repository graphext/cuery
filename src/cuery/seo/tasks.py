"""Keyword-SERP topic and intent analyzer.

Analyzes Google SERP (Search Engine Results Page) data to extract topics and
assign search intent. Each input row contains a keyword with its corresponding
SERP results (domains, breadcrumbs, page titles).

Two-level analysis is performed:

1. Topic extraction and assignment using SERP content
   - Extract a hierarchy of topics and subtopics from aggregated SERP data
   - Assign the correct topic and subtopic to each keyword based on SERP results

2. Search intent classification
   - Classify each keyword into one of 4 fixed categories: informational,
     navigational, transactional, commercial
   - Based on SERP content patterns and domain analysis
"""

import json
from string import Template
from typing import Literal

from pandas import DataFrame
from pydantic import Field

from ..prompt import Prompt
from ..response import Response, ResponseSet
from ..task import Task
from ..topics.oneshot import TopicAssignment, Topics, make_assignment_model, make_topic_model
from ..utils import dedent

SERP_TOPICS_PROMPT = dedent("""
From the keyword SERP data below, extract a two-level nested list of topics.
Each entry contains a keyword and its top search results (domains, breadcrumbs, page titles).
The output should be a JSON object with top-level topics as keys and lists of subtopics as values.
The top-level should not contain more than $n_topics topics, and each top-level
should not contain more than $n_subtopics subtopics.

Focus on the semantic meaning and commercial intent behind the keywords based on:
- Domain patterns (e.g., e-commerce sites, informational sites, brand sites)
- Page title content and structure
- Breadcrumb navigation patterns

Make sure top-level topics are generalizable and capture broad search themes.
Subtopics should represent more specific search categories within each theme.
$extra

# Keyword SERP Data

{% for keyword in keywords %}
**Keyword:** {{ keyword["term"] }}

**Top Domains:**

{% for domain in keyword["domains"] %}
{{ domain }}, 
{% endfor %}

**Page Titles:**

{% for title in keyword["titles"] %}
{{ title }}, 
{% endfor %}

**Breadcrumbs:**

{% for breadcrumb in keyword["breadcrumbs"] %}
{{ breadcrumb }}, 
{% endfor %}

{% endfor %}
""")

SERP_ASSIGNMENT_PROMPT_SYSTEM = dedent("""
You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign the correct topic and subtopic to each keyword based on its SERP results.

Additionally, classify the search intent using these definitions:
- **Informational**: User seeks information, answers, or knowledge (how-to, what is, tutorials)
- **Navigational**: User wants to find a specific website or page (brand names, specific sites)
- **Transactional**: User intends to complete an action or purchase (buy, download, sign up)
- **Commercial**: User researches products/services before purchasing (reviews, comparisons, best)

Consider the domains, page titles, and breadcrumbs to understand both the search context
and intent.

# Topics

%(topics)s
""")

SERP_ASSIGNMENT_PROMPT_USER = dedent("""
Assign the correct topic and subtopic to the following keyword based on its SERP results.
Also classify the search intent as one of: informational, navigational, transactional,
or commercial.

# Keyword: {{term}}

## Top Domains:
{{domains}}

## Page Titles:
{{titles}}

## Breadcrumbs:
{{breadcrumbs}}
""")

ENTITY_EXTRACTION_PROMPT = dedent("""
From the Google SERP AI Overview data and Source Titles below, extract entities that are relevant
to SEO analysis. Focus on identifying 3 kinds of entities: brand mentions and company names;
products and services; and technologies, tools, and platforms. Categorize other entities as "other".

For each entity, provide the entity name/text as it appears, and the type/category of entity.
Pay special attention to URLs etc. in the source titles, which may refer to brands, companies or
products. Ensure to report the names of entities always in lowercase and singular form, even if
they appear in plural or uppercase in the source titles, to avoid inconsistencies in the output.

# AI Overview Data

{{ aiOverview_content }}

## Source Titles:

{% for title in aiOverview_source_titles %}
- {{ title }}  

{% endfor %}
""")


class TopicAndIntent(TopicAssignment):
    """Topic assignment with additional intent classification."""

    intent: Literal["informational", "navigational", "transactional", "commercial"] = Field(
        ..., description="The primary search intent category"
    )


class Entity(Response):
    """Individual entity extracted from AI Overview content."""

    name: str = Field(..., description="The entity name or text as it appears")
    type: Literal[
        "brand/company",
        "product/service",
        "technology",
        "other",
    ] = Field(..., description="The category/type of the entity")


class SerpTopicExtractor:
    """Extract topics from keyword SERP data."""

    def __init__(
        self,
        n_topics: int = 10,
        n_subtopics: int = 5,
        extra: str | None = None,
    ):
        prompt_args = {
            "n_topics": n_topics,
            "n_subtopics": n_subtopics,
            "extra": extra or "",
        }
        prompt = Template(SERP_TOPICS_PROMPT).substitute(prompt_args)
        prompt = Prompt.from_string(prompt)
        response = make_topic_model(n_topics, n_subtopics)
        self.task = Task(prompt=prompt, response=response)

    def _format_serp_data(self, df: DataFrame, n_keywords: int) -> list[dict]:
        """Format DataFrame of SERP data for topic extraction prompt."""

        def flatten(s):
            return [
                item
                for sublist in s
                if sublist is not None
                for item in sublist
                if item is not None
            ]

        grouped = (
            df.groupby("term")
            .agg({"title": list, "domain": list, "breadcrumb": flatten})
            .reset_index()
        )

        return grouped.iloc[:n_keywords].to_dict(orient="records")

    async def __call__(self, df: DataFrame, model: str, **kwds) -> Topics:
        """Extract topics from SERP data DataFrame."""
        context = {"keywords": df.to_dict(orient="records")}
        responses = await self.task.call(context=context, model=model, **kwds)
        return responses[0]  # type: ignore


class SerpTopicAndIntentAssigner:
    """Assign topics and classify intent for keywords based on their SERP results."""

    def __init__(self, topics: Topics | dict[str, list[str]]):
        if isinstance(topics, Topics):
            topics = topics.to_dict()

        topics_json = json.dumps(topics, indent=2)
        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": SERP_ASSIGNMENT_PROMPT_SYSTEM % {"topics": topics_json},
                },
                {"role": "user", "content": SERP_ASSIGNMENT_PROMPT_USER},
            ],  # type: ignore
            required=["term", "domains", "titles", "breadcrumbs"],
        )
        response = make_assignment_model(topics, TopicAndIntent)  # type: ignore
        self.task = Task(prompt=prompt, response=response)  # type: ignore

    async def __call__(self, df: DataFrame, model: str, **kwds) -> ResponseSet:
        """Assign topics and classify intent for each keyword in the DataFrame."""
        return await self.task(context=df, model=model, **kwds)


class Entities(Response):
    """Result containing all extracted entities from AI Overview data."""

    entities: list[Entity] = Field(
        default=[], description="List of extracted SEO-relevant entities"
    )


class EntityExtractor:
    """Extract SEO-relevant entities from Google SERP AI Overview data."""

    def __init__(self, **kwds):
        """Initialize the EntityExtractor with predefined prompt and response model."""
        prompt = Prompt.from_string(ENTITY_EXTRACTION_PROMPT)
        self.task = Task(prompt=prompt, response=Entities, **kwds)

    async def __call__(self, df: DataFrame, model: str, **kwds) -> ResponseSet:
        return await self.task(context=df, model=model, **kwds)
