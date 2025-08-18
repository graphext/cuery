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

import pandas as pd
from pandas import DataFrame
from pydantic import Field

from ..prompt import Prompt
from ..response import Response, ResponseSet
from ..task import Task
from ..tools.topics import (
    Topics,
    make_label_model,
    make_multi_label_model,
    make_topic_model,
)
from ..utils import LOG, Configurable, dedent

TOPICS_PROMPT = dedent("""
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

TOPIC_ASSIGNMENT_PROMPT_SYSTEM = dedent("""
You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign the correct topic and subtopic to each keyword based on its SERP results.

Consider the keyword term itself, as well as any other provided attribute associated with
the keyword.

# Topics

%(topics)s
""")

TOPIC_ASSIGNMENT_PROMPT_USER = dedent("""
Assign the correct topic and subtopic to the following keyword based on its SERP results.

# Keyword: "{{keyword["term"]}}"

{% for attr_name in keyword.keys() | list -%}
{% if attr_name != "term" -%}

{{ attr_name }}:
{% if keyword[attr_name] is iterable and keyword[attr_name] is not string -%}
{%- for item in keyword[attr_name] -%}
{{ item | trim }}{% if not loop.last %} | {% endif %}
{%- endfor %}
{% elif keyword[attr_name] -%}
{{ keyword[attr_name] }}
{% endif %}

{% endif -%}
{% endfor -%}
""")

MULTI_TOPIC_ASSIGNMENT_PROMPT_SYSTEM = dedent("""
You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign one or more relevant topics and subtopics to each keyword based on its SERP results.
A keyword can be assigned multiple topics and subtopics if its SERP results cover multiple themes.

Consider the keyword term itself, as well as any other provided attribute associated with
the keyword.

# Topics

%(topics)s
""")

MULTI_TOPIC_ASSIGNMENT_PROMPT_USER = dedent("""
Assign one or more relevant topics and subtopics to the following keyword based on its
SERP results. Consider all themes covered in the SERP data and assign appropriate
topic-subtopic pairs.

# Keyword: "{{keyword["term"]}}"

{% for attr_name in keyword.keys() | list -%}
{% if attr_name != "term" -%}

{{ attr_name }}:
{% if keyword[attr_name] is iterable and keyword[attr_name] is not string -%}
{%- for item in keyword[attr_name] -%}
{{ item | trim }}{% if not loop.last %} | {% endif %}
{%- endfor %}
{% elif keyword[attr_name] -%}
{{ keyword[attr_name] }}
{% endif %}

{% endif -%}
{% endfor -%}
""")

INTENT_ASSIGNMENT_PROMPT_SYSTEM = dedent("""
You're task is to classify the search intent for each keyword based on its SERP results.

Use these search intent definitions:
- **Informational**: User seeks information, answers, or knowledge (how-to, what is, tutorials)
- **Navigational**: User wants to find a specific website or page (brand names, specific sites)
- **Transactional**: User intends to complete an action or purchase (buy, download, sign up)
- **Commercial**: User researches products/services before purchasing (reviews, comparisons, best)

Consider the keyword term itself, as well as any other provided attribute associated with
the keyword.
""")

INTENT_ASSIGNMENT_PROMPT_USER = dedent("""
Classify the search intent for the following keyword based on its SERP results.
The intent should be one of: informational, navigational, transactional, or commercial.

# Keyword: "{{keyword["term"]}}"

{% for attr_name in keyword.keys() | list -%}
{% if attr_name != "term" -%}

{{ attr_name }}:
{% if keyword[attr_name] is iterable and keyword[attr_name] is not string -%}
{%- for item in keyword[attr_name] -%}
{{ item | trim }}{% if not loop.last %} | {% endif %}
{%- endfor %}
{% elif keyword[attr_name] -%}
{{ keyword[attr_name] }}
{% endif %}

{% endif -%}
{% endfor -%}
""")


PURCHASE_PROBABILITY_PROMPT = dedent("""
Assign a purchase probability score (integer between 0 and 10) to the (Google) search keyword and
associated SERP data below (Keyword section). The score should reflect the likelihood of a user
completing a purchase or taking a commercial action based on the keyword within the next 30 days.

# Keyword: "{{keyword["term"]}}"

{% for attr_name in keyword.keys() | list -%}
{% if attr_name != "term" -%}

{{ attr_name }}:
{% if keyword[attr_name] is iterable and keyword[attr_name] is not string -%}
{%- for item in keyword[attr_name] -%}
{{ item | trim }}{% if not loop.last %} | {% endif %}
{%- endfor %}
{% elif keyword[attr_name] -%}
{{ keyword[attr_name] }}
{% endif %}

{% endif -%}
{% endfor -%}
""")

ENTITY_EXTRACTION_PROMPT = """
# Instructions
From the "AI Overview Data" section below, which contains AI overviews from Google SERPs,
extract entities that are relevant to SEO analysis. Focus on identifying 3 kinds of entities:

- brand mentions and company names (label "brand_company")
- products and services (label "product_service")
- technologies and tools (label "technology")

Categorize other entities as "other".

For each entity, provide the entity name/text as it appears, and the type/category of entity.
Pay special attention to URLs, which may refer to brands, companies or
products. Ensure to report the names of entities always in lowercase and singular form, even if
they appear in plural or uppercase in the source titles, to avoid inconsistencies in the output.

# AI Overview Data

{% for attr_name in aio.keys() | list -%}

## {{ attr_name }}

{% if aio[attr_name] is iterable and aio[attr_name] is not string -%}
{%- for item in aio[attr_name] -%}
- {{ item | trim }}

{% endfor %}
{% elif aio[attr_name] -%}
{{ aio[attr_name] }}
{% endif %}

{% endfor -%}
"""


class SerpIntentAssignment(Response):
    """Intent classification for SERP keyword data."""

    intent: Literal["informational", "navigational", "transactional", "commercial"] = Field(
        ..., description="The primary search intent category"
    )


class PurchaseProbability(Response):
    """Purchase probability score for SERP keyword data."""

    score: int = Field(
        ..., ge=0, le=10, description="Probability score (0-10) of a purchase action"
    )


class Entity(Response):
    """Individual entity extracted from AI Overview content."""

    name: str = Field(..., description="The entity name or text as it appears")
    type: Literal[
        "brand_company",
        "product_service",
        "technology",
        "other",
    ] = Field(..., description="The category/type of the entity")


class Entities(Response):
    """Result containing all extracted entities from AI Overview data."""

    entities: list[Entity] = Field(
        default=[], description="List of extracted SEO-relevant entities"
    )


class SerpTopicExtractor(Configurable):
    """Extract topics from keyword SERP data."""

    text_column: str = "term"
    """Column name in the DataFrame containing the main texts."""
    extra_columns: list[str] = Field(default_factory=list)
    """List of additional columns to include in the context for topic extraction."""
    n_topics: int = Field(10, ge=1, le=20)
    """Maximum number of top-level topics to extract (maximum 20)."""
    n_subtopics: int = Field(5, ge=2, le=10)
    """Maximum number of subtopics per top-level topic (At least 2, maximum 10)."""
    min_ldist: int = Field(2, ge=1)
    """Minimum Levenshtein distance between topic labels."""
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
        prompt = Template(TOPICS_PROMPT).substitute(prompt_args)
        prompt = Prompt.from_string(prompt)
        response_cls = make_topic_model(self.n_topics, self.n_subtopics, min_ldist=self.min_ldist)
        self._task = Task(prompt=prompt, response=response_cls)

        # Configure the context
        df = df.sample(n=min(self.max_samples, len(df)), random_state=42)
        df = df.rename(columns={self.text_column: "term"})
        columns = ["term"]
        if self.extra_columns:
            columns.extend(col for col in self.extra_columns if col in df.columns)
        df = df[columns]
        context = {"keywords": df.to_dict(orient="records")}

        LOG.info(f"Extracting topics with columns: {columns}")
        responses = await self._task.call(context=context, model=model, **kwds)
        return responses[0]  # type: ignore


class SerpTopicAssigner(Configurable):
    """Assign topics for keywords based on their SERP results.

    Returns a DataFrame containing the configured input columns along with the
    assigned topics and subtopics.
    """

    topics: Topics | dict[str, list[str]]
    """Topics and subtopics to use for assignment, either as a Topics object or a dict."""
    text_column: str = "term"
    """Column name in the DataFrame containing the main texts."""
    extra_columns: list[str] = Field(default_factory=list)
    """List of additional columns to include in the context for topic extraction."""
    model: str = Field("openai/gpt-4.1-mini", pattern=r"^[\w\.-]+/[\w\.-]+$")  # type: ignore
    """Model to use for topic extraction."""

    _task: Task | None = None

    async def __call__(self, df: DataFrame, model: str | None = None, **kwds) -> DataFrame:
        """Assign topics for each keyword in the DataFrame."""
        model = model or self.model

        topics = self.topics
        if isinstance(topics, Topics):
            topics = topics.to_dict()

        topics_json = json.dumps(topics, indent=2)

        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": TOPIC_ASSIGNMENT_PROMPT_SYSTEM % {"topics": topics_json},
                },
                {"role": "user", "content": TOPIC_ASSIGNMENT_PROMPT_USER},
            ],  # type: ignore
            required=["keyword"],
        )
        response = make_label_model(topics)
        self._task = Task(prompt=prompt, response=response)

        # Keep only configured columns
        columns = [self.text_column] + [col for col in self.extra_columns if col in df.columns]
        df = df[columns]
        df = df.rename(columns={self.text_column: "term"})

        # Convert DataFrame to context format (one dict per row)
        context = [{"keyword": record} for record in df.to_dict(orient="records")]

        LOG.info(f"Assigning topics row-wise with columns: {columns}")
        response = await self._task(context=context, model=model, **kwds)

        result = response.to_pandas(explode=False)
        result = pd.concat(
            [
                result.drop(columns="keyword"),
                pd.json_normalize(result["keyword"]),  # type: ignore
            ],
            axis=1,
        )
        return result.rename(columns={"term": self.text_column})


class SerpMultiTopicAssigner(Configurable):
    """Assign multiple topics for keywords based on their SERP results.

    Returns a DataFrame containing the configured input columns along with the
    assigned topics and subtopics. Unlike SerpTopicAssigner, this class can assign
    multiple topic-subtopic pairs to each keyword when SERP data covers multiple themes.
    """

    topics: Topics | dict[str, list[str]]
    """Topics and subtopics to use for assignment, either as a Topics object or a dict."""
    text_column: str = "term"
    """Column name in the DataFrame containing the main texts."""
    extra_columns: list[str] = Field(default_factory=list)
    """List of additional columns to include in the context for topic extraction."""
    model: str = Field("openai/gpt-4.1-mini", pattern=r"^[\w\.-]+/[\w\.-]+$")  # type: ignore
    """Model to use for topic extraction."""
    max_labels: int = Field(3, ge=1, le=10)
    """Maximum number of topic-subtopic pairs to assign per keyword (default 3)."""

    _task: Task | None = None
    _response: ResponseSet | None = None

    async def __call__(self, df: DataFrame, model: str | None = None, **kwds) -> DataFrame:
        """Assign multiple topics for each keyword in the DataFrame."""
        model = model or self.model

        topics = self.topics
        if isinstance(topics, Topics):
            topics = topics.to_dict()

        topics_json = json.dumps(topics, indent=2)

        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": MULTI_TOPIC_ASSIGNMENT_PROMPT_SYSTEM % {"topics": topics_json},
                },
                {"role": "user", "content": MULTI_TOPIC_ASSIGNMENT_PROMPT_USER},
            ],  # type: ignore
            required=["keyword"],
        )
        response = make_multi_label_model(topics, max_assignments=self.max_labels)  # type: ignore
        self._task = Task(prompt=prompt, response=response)  # type: ignore

        # Keep only configured columns
        columns = [self.text_column] + [col for col in self.extra_columns if col in df.columns]
        df = df[columns]
        df = df.rename(columns={self.text_column: "term"})

        # Convert DataFrame to context format (one dict per row)
        context = [{"keyword": record} for record in df.to_dict(orient="records")]

        LOG.info(f"Assigning multi-topics row-wise with columns: {columns}")
        self._response = await self._task(context=context, model=model, **kwds)

        result = self._response.to_pandas(explode=False, normalize=True)
        result = pd.concat(
            [
                result.drop(columns="keyword"),
                pd.json_normalize(result["keyword"]),  # type: ignore
            ],
            axis=1,
        )
        return result.rename(columns={"term": self.text_column})


class SerpIntentAssigner(Configurable):
    """Classify intent for keywords based on their SERP results."""

    text_column: str = "term"
    """Column name in the DataFrame containing the main texts."""
    extra_columns: list[str] = Field(default_factory=list)
    """List of additional columns to include in the context for topic extraction."""
    model: str = Field("openai/gpt-4.1-mini", pattern=r"^[\w\.-]+/[\w\.-]+$")  # type: ignore
    """Model to use for topic extraction."""

    _task: Task | None = None

    async def __call__(self, df: DataFrame, model: str | None = None, **kwds) -> DataFrame:
        """Classify intent for each keyword in the DataFrame."""
        model = model or self.model

        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": INTENT_ASSIGNMENT_PROMPT_SYSTEM,
                },
                {"role": "user", "content": INTENT_ASSIGNMENT_PROMPT_USER},
            ],  # type: ignore
            required=["keyword"],
        )
        self._task = Task(prompt=prompt, response=SerpIntentAssignment)  # type: ignore

        # Keep only configured columns
        columns = [self.text_column] + [col for col in self.extra_columns if col in df.columns]
        df = df[columns]
        df = df.rename(columns={self.text_column: "term"})

        # Convert DataFrame to context format (one dict per row)
        context = [{"keyword": record} for record in df.to_dict(orient="records")]

        LOG.info(f"Assigning intents row-wise with columns: {columns}")
        response = await self._task(context=context, model=model, **kwds)

        result = response.to_pandas(explode=False)
        result = pd.concat(
            [
                result.drop(columns="keyword"),
                pd.json_normalize(result["keyword"]),  # type: ignore
            ],
            axis=1,
        )

        return result.rename(columns={"term": self.text_column})


class EntityExtractor(Configurable):
    """ "Extract SEO-relevant entities from Google SERP AI Overview data."""

    columns: list[str]
    """List of columns to include in the context for entity extraction."""
    model: str = Field("openai/gpt-4.1-mini", pattern=r"^[\w\.-]+/[\w\.-]+$")  # type: ignore
    """Model to use for topic extraction."""

    _task: Task | None = None

    async def __call__(self, df: DataFrame, model: str | None = None, **kwds) -> ResponseSet:
        model = model or self.model

        prompt = Prompt.from_string(ENTITY_EXTRACTION_PROMPT)
        self._task = Task(prompt=prompt, response=Entities)

        df = df[[col for col in self.columns if col in df.columns]]
        context = [{"aio": record} for record in df.to_dict(orient="records")]

        LOG.info(f"Assigning entities row-wise with columns: {df.columns.tolist()}")
        return await self._task(context=context, model=model, **kwds)


class PurchaseProbAssigner(Configurable):
    """Classify intent for keywords based on their SERP results."""

    text_column: str = "term"
    """Column name in the DataFrame containing the main texts."""
    extra_columns: list[str] = Field(default_factory=list)
    """List of additional columns to include in the context."""
    model: str = Field("openai/gpt-4.1-mini", pattern=r"^[\w\.-]+/[\w\.-]+$")
    """Model to use for purchase probability estimation."""

    _task: Task | None = None

    async def __call__(self, df: DataFrame, model: str | None = None, **kwds) -> DataFrame:
        """Estimate purchase probability for each keyword in the DataFrame."""
        model = model or self.model
        prompt = Prompt.from_string(PURCHASE_PROBABILITY_PROMPT)
        self._task = Task(prompt=prompt, response=PurchaseProbability)

        # Keep only configured columns
        columns = [self.text_column] + [col for col in self.extra_columns if col in df.columns]
        df = df[columns]
        df = df.rename(columns={self.text_column: "term"})

        # Convert DataFrame to context format (one dict per row)
        context = [{"keyword": record} for record in df.to_dict(orient="records")]

        LOG.info(f"Assigning purchase probabilities row-wise with columns: {columns}")
        response = await self._task(context=context, model=model, **kwds)

        result = response.to_pandas(explode=False)
        result = pd.concat(
            [
                result.drop(columns="keyword"),
                pd.json_normalize(result["keyword"]),  # type: ignore
            ],
            axis=1,
        )

        return result.rename(columns={"term": self.text_column})
