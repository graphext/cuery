"""AI-powered SERP topic extraction and search intent classification.

This module provides sophisticated analysis tools for Search Engine Results Page (SERP)
data using large language models. It performs two-level analysis to extract meaningful
insights from SERP content: hierarchical topic extraction that identifies themes and
subtopics from aggregated SERP data, and search intent classification that categorizes
keywords into informational, navigational, transactional, or commercial intent types.

The analysis leverages domain patterns, page titles, and breadcrumb navigation to
understand the semantic meaning and commercial context behind search queries. This
enables SEO professionals to better understand search landscapes, identify content
opportunities, and optimize for user intent. The module uses configurable language
models and provides structured outputs suitable for further analysis and reporting.
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
from ..utils import HashableConfig, dedent

SERP_TOPICS_PROMPT = dedent("""
From the keyword SERP data below, extract a two-level nested list of topics.
Each entry contains a keyword and its associated search results data.
The output should be a JSON object with top-level topics as keys and lists of subtopics as values.
The top-level should not contain more than $n_topics topics, and each top-level
should not contain more than $n_subtopics subtopics.

Focus on the semantic meaning and commercial intent behind the keywords based on:
- Domain patterns (e.g., e-commerce sites, informational sites, brand sites)
- Page title content and structure
- Breadcrumb navigation patterns
- Any other available search result attributes

Make sure top-level topics are generalizable and capture broad search themes.
Subtopics should represent more specific search categories within each theme.
$extra

# Keyword SERP Data

## Keywords

{% for keyword in keywords -%}
{{ keyword["term"] }}{% if not loop.last %}, {% endif %}
{%- endfor %}

{% if keywords %}
{% for attr_name in keywords[0].keys() | list %}
{% if attr_name != "term" %}

## {{ attr_name.replace('_', ' ').title() }}

{% for keyword in keywords -%}
{% if keyword[attr_name] is iterable and keyword[attr_name] is not string -%}
{%- for item in keyword[attr_name] -%}
{{ item | trim }} |
{% endfor -%}
{% elif keyword[attr_name] -%}
{{ keyword[attr_name] }}
{% endif -%}
{% endfor -%}
{% endif %}
{% endfor %}
{% endif %}
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


class SerpTopicExtractor(HashableConfig):
    """Extract topics from keyword SERP data."""

    n_topics: int = Field(10, ge=1, le=20)
    """Maximum number of top-level topics to extract (maximum 20)."""
    n_subtopics: int = Field(5, ge=2, le=10)
    """Maximum number of subtopics per top-level topic (At least 2, maximum 10)."""
    extra: str = ""
    """Additional use-case specific instructions or context for the topic extraction."""
    max_samples: int = 500
    """Maximum number of samples to use for topic extraction."""
    model: str = Field("google/gemini-2.5-flash-preview-05-20", pattern=r"^[\w\.-]+/[\w\.-]+$")  # type: ignore
    """Model to use for topic extraction."""

    _task: Task | None = None

    async def __call__(self, df: DataFrame, model: str | None = None, **kwds) -> Topics:
        """Extract topics from SERP data DataFrame."""
        model = model or self.model

        prompt_args = {
            "n_topics": self.n_topics,
            "n_subtopics": self.n_subtopics,
            "extra": self.extra or "",
        }

        # Configure the prompt and task
        prompt = Template(SERP_TOPICS_PROMPT).substitute(prompt_args)
        prompt = Prompt.from_string(prompt)
        response_cls = make_topic_model(self.n_topics, self.n_subtopics)
        self._task = Task(prompt=prompt, response=response_cls)

        # Configure the context
        samples_max = min(self.max_samples, len(df))
        df = df.sample(n=samples_max, random_state=42)
        context = {"keywords": df.to_dict(orient="records")}

        responses = await self._task.call(context=context, model=model, **kwds)
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
