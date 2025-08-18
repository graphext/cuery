"""Tools that work with flexible input contexts.

I.e. these tools accept context objects that have arbitrary numbers and types of fields.
The Jinja prompt templates auto-convert these fields into the appropriate format for LLMs.
"""

import json
from functools import cached_property
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from ... import Prompt, Response, ResponseClass, templates
from ...utils import dedent
from ..topics import Topics, make_label_model, make_multi_label_model, make_topic_model
from .base import FlexTool, preprocess_records

TOPICS_PROMPT = dedent("""
# Instructions

From the data records below, extract a two-level nested list of topics.
The output should be a JSON object with top-level topics as keys and lists of subtopics as values.
The top-level should not contain more than ${n_topics} topics, and each top-level
should not contain more than ${n_subtopics} subtopics.

Make sure top-level topics are generalizable and capture broad themes.
Subtopics should represent more specific categories within each theme.

${instructions}

# Data Records

${records_template}
""")


LABEL_PROMPT_SYSTEM = dedent("""
# Instructions

You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign the correct topic and subtopic to the data record you will receive.

Make sure to consider all the attributes of the data record, and assign the most appropriate
topic and subtopic based on the content of the attributes.

# Topics

${topics}
""")


LABEL_PROMPT_USER = dedent("""
Assign the correct topic and subtopic to the following data record.

# Data Record

${record_template}
""")

MULTI_LABEL_PROMPT_SYSTEM = dedent("""
# Instructions

You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign one or more relevant topics and subtopics to the data record you will receive.

Make sure to consider all the attributes of the data record, and assign the most appropriate
topics and subtopics based on the content of the attributes.

A data record can be assigned multiple topics and subtopics if it covers multiple themes.

# Topics

${topics}
""")

MULTI_LABEL_PROMPT_USER = dedent("""
Assign one or more relevant topics and subtopics to the following data record.

# Data Record

${record_template}
""")


class TopicExtractor(FlexTool):
    """Extract topics from records with arbitrary attributes."""

    n_topics: int = Field(10, ge=1, le=20)
    """Approximate number of top-level topics to extract (maximum 20)."""
    n_subtopics: int = Field(5, ge=1, le=10)
    """Approximate number of subtopics per top-level topic (At least 2, maximum 10)."""
    instructions: str = ""
    """Additional use-case specific instructions or context for the topic extraction."""
    min_ldist: int = Field(2, ge=1)
    """Minimum Levenshtein distance between topic labels."""
    max_samples: int = 500
    """Maximum number of samples to use for topic extraction."""
    record_format: Literal["attr_wise", "rec_wise"] = "attr_wise"
    """Format of the records in the prompt."""

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_topic_model(self.n_topics, self.n_subtopics, min_ldist=self.min_ldist)

    @cached_property
    def prompt(self) -> Prompt:
        prompt_args = {
            "n_topics": self.n_topics,
            "n_subtopics": self.n_subtopics,
            "instructions": self.instructions,
            "records_template": templates.load_template(f"records_{self.record_format}"),
        }

        return Prompt.from_string(TOPICS_PROMPT).substitute(**prompt_args)

    @cached_property
    def context(self) -> dict:
        """Override FlexTool base implementation.

        This tool is different because it doesn't iterate over records,
        but rather processes them all at once to extract topics.
        """
        df = preprocess_records(self.records, self.attrs, self.max_samples)
        return {"records": df.to_dict(orient="records")}

    async def __call__(self, **kwargs) -> Topics:
        responses = await self.task.call(context=self.context, **kwargs)
        return responses[0]  # type: ignore


class TopicAssigner(FlexTool):
    """Assign topics to records with arbitrary attributes."""

    topics: Topics
    """Topics and subtopics to use for assignment, either as a Topics object or a dict."""

    SYSTEM_PROMPT: ClassVar[str] = LABEL_PROMPT_SYSTEM
    USER_PROMPT: ClassVar[str] = LABEL_PROMPT_USER

    @field_validator("topics", mode="before", json_schema_input_type=dict[str, list[str]] | Topics)
    @classmethod
    def validate_topics(cls, topics) -> Topics:
        if isinstance(topics, dict):
            topics = Topics(topics=topics)  # type: ignore

        return topics

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT},
            ],  # type: ignore
        )

        prompt_args = {
            "topics": json.dumps(self.topics.to_dict(), indent=2),
            "record_template": self.template,
        }
        return prompt.substitute(**prompt_args)

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_label_model(self.topics.to_dict())


class MultiTopicAssigner(TopicAssigner):
    """Enforce correct multi-topic-subtopic assignment via a Pydantic model."""

    SYSTEM_PROMPT: ClassVar[str] = MULTI_LABEL_PROMPT_SYSTEM
    USER_PROMPT: ClassVar[str] = MULTI_LABEL_PROMPT_USER

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_multi_label_model(self.topics.to_dict())  # type: ignore
