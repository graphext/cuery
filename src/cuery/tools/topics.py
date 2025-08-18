"""Higher-level API for extracting topics from texts using a one-shot prompt.

Two-level topic extraction is performed using two steps:

1. Extract a hierarchy of topics and subtopics from a list of texts.
  - Dynamicaly construct a Pydantic response model with the desired number of topics and subtopics
  - Use a one-shot prompt to extract the topics and subtopics from a concatenated list of texts
    limited by a desired token count, dollar cost, or number of texts.
2. Assign the correct topic and subtopic to each text using the extracted hierarchy
  - Dynamically construct a Pydantic response model for the topics and subtopics with custom
    validation to ensure that the subtopic belongs to the topic.
  - Iterate over the texts and use prompt to assign the correct topic and subtopic
"""

import json
from collections.abc import Iterable
from functools import cached_property
from typing import ClassVar, Literal, Self

from Levenshtein import distance as ldist
from pydantic import field_validator, model_validator

from .. import AnyContext, Field, Prompt, Response, ResponseClass, Tool, utils
from ..utils import customize_fields, dedent

TOPICS_PROMPT = dedent("""
From the list of texts below (separated by line breaks), extract a two-level nested list of topics.
The output should be a JSON object with top-level topics as keys and lists of subtopics as values.
The top-level should not contain more than ${n_topics} topics, and each top-level
should not contain more than ${n_subtopics} subtopics. Make sure top-level topics are
generalizable and not too specific, so they can be used as a hierarchy for the subtopics. Make
sure also that subtopics are not redundant (no similar ones within the the same top-level topic).
Create fewer topics and subtopics if needed, i.e. when otherwise top-level categories or subtopics
would be too similar.

${instructions}

# Texts

{{texts}}
""")

LABEL_PROMPT_SYSTEM = dedent("""
You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign the correct topic and subtopic to each text in the input.

# Topics

${topics}
""")

LABEL_PROMPT_USER = dedent("""
Assign the correct topic and subtopic to the following text.

# Text

{{text}}
""")

MULTI_LABEL_PROMPT_SYSTEM = dedent("""
You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign one or more relevant topics and subtopics to each text in the input.
A text can be assigned multiple topics and subtopics if it covers multiple themes.

# Topics

${topics}
""")

MULTI_LABEL_PROMPT_USER = dedent("""
Assign one or more relevant topics and subtopics to the following text.
Consider all themes covered in the text and assign appropriate topic-subtopic pairs.

# Text

{{text}}
""")


class Topic(Response):
    """A response containing a topic and its subtopics.

    Validates that subtopics are sufficiently distinct from each other and from the parent topic.
    """

    topic: str
    """The top-level topic."""
    subtopics: list[str]
    """A list of subtopics under the top-level topic."""

    _MIN_LDIST: ClassVar[int] = 2
    """Minimum Levenshtein distance between subtopics and subtopics->parent topic."""

    @model_validator(mode="after")
    def validate_subtopics(self) -> Self:
        # Topic titles should be at least N character edits apart

        subtopics = [st.lower() for st in self.subtopics]
        errors = []

        sim_err = "Subtopic '{}' too similar to other subtopic '{}'.".format
        perm_err = "Subtopic '{}' is a duplicate (permutation) of subtopic '{}'.".format

        for i, st in enumerate(subtopics):
            # Subtopics should not be too similar to their parent topic
            if ldist(st, self.topic.lower()) < self._MIN_LDIST:
                errors.append(f"Subtopic '{st}' too similar to parent topic '{self.topic}'.")

            # Subtopics should not be too similar to each other
            for j in range(i + 1, len(subtopics)):
                other = subtopics[j]

                # Check Levenshtein distance for similarity
                if ldist(st.replace(" ", ""), other.replace(" ", "")) < self._MIN_LDIST:
                    errors.append(sim_err(st, other))

                # Check for permutations of words
                if set(st.split()) == set(other.split()):
                    errors.append(perm_err(st, other))

        if errors:
            raise ValueError("Invalid subtopics:\n" + "\n".join(errors))

        return self


class Topics(Response):
    """A response containing a two-level nested list of topics."""

    topics: list[Topic]
    """A list of top-level topics with their subtopics."""

    @field_validator("topics", mode="before")
    @classmethod
    def validate_topics(cls, topics) -> list:
        """Validate that the topics are a list of dictionaries with topic and subtopics."""
        if isinstance(topics, dict):
            topics = [
                {"topic": topic, "subtopics": subtopics} for topic, subtopics in topics.items()
            ]

        return topics

    def to_dict(self) -> dict[str, list[str]]:
        """Convert the response to a dictionary."""
        return {t.topic: t.subtopics for t in self.topics}


class TopicLabel(Response):
    """Base class for topic and subtopic assignment(!) with validation of correspondence."""

    topic: str
    """A specific top-level label."""
    subtopic: str
    """A specific subtopic label."""

    mapping: ClassVar[dict[str, list]] = {}
    """The allowed topic hierarchy."""

    @model_validator(mode="after")
    def is_subtopic(self) -> Self:
        allowed = self.mapping.get(self.topic, [])
        if self.subtopic not in allowed:
            raise ValueError(
                f"Subtopic '{self.subtopic}' is not a valid subtopic for topic '{self.topic}'."
                f" Allowed subtopics are: {allowed}."
            )
        return self


class MultiTopicLabels(Response):
    """Base class for multiple topic and subtopic assignment with validation of correspondence."""

    labels: list[TopicLabel] = Field(..., min_length=1)
    """A list of topic-subtopic assignments for the text."""


def make_topic_model(n_topics: int, n_subtopics: int, min_ldist: int = 2) -> type[Topics]:
    """Create a specific response model for a list of N topics with M subtopics."""
    TopicK = customize_fields(Topic, "TopicK", subtopics={"max_length": n_subtopics})
    TopicK._MIN_LDIST = min_ldist

    return customize_fields(
        Topics,
        "TopicsN",
        topics={"max_length": n_topics, "annotation": list[TopicK]},
    )


def make_label_model(topics: dict[str, list[str]]) -> type[TopicLabel]:
    """Create a Pydantic model class for topics and subtopic assignment."""
    tops = list(topics)
    subs = [topic for subtopics in topics.values() for topic in subtopics]

    cls = utils.customize_fields(
        TopicLabel,
        "CustomTopicLabel",
        topic={"annotation": Literal[*tops]},
        subtopic={"annotation": Literal[*subs]},
    )

    cls.mapping = topics
    return cls


def make_multi_label_model(
    topics: dict[str, list[str]],
    max_assignments: int = 3,
) -> type[MultiTopicLabels]:
    """Create a Pydantic model class for multiple topics and subtopic assignment."""
    single_assignment_cls = make_label_model(topics)

    return utils.customize_fields(
        MultiTopicLabels,
        "CustomMultiTopicLabels",
        labels={"annotation": list[single_assignment_cls], "max_length": max_assignments},
    )


class TopicExtractor(Tool):
    """Enforce the topic-subtopic hierarchy directly via response model."""

    n_topics: int = 10
    """The number of top-level topics to extract."""
    n_subtopics: int = 5
    """The number of subtopics to extract for each topic."""
    instructions: str = ""
    """Prompt instructions to add with details for the topic extraction."""
    texts: Iterable[str | float | None]
    """The texts to extract topics from."""
    max_dollars: float | None = None
    """The maximum to spend on the query."""
    max_tokens: float | None = None
    """The maximum number of tokens to spend."""
    max_texts: float | None = None
    """The maximum number of texts to process."""

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_topic_model(self.n_topics, self.n_subtopics)

    @cached_property
    def prompt(self) -> Prompt:
        prompt_args = {
            "n_topics": self.n_topics,
            "n_subtopics": self.n_subtopics,
            "instructions": self.instructions,
        }
        return Prompt.from_string(TOPICS_PROMPT).substitute(**prompt_args)

    @cached_property
    def context(self) -> dict:
        text = utils.concat_up_to(
            self.texts,
            model=self.model,
            max_dollars=self.max_dollars,
            max_tokens=self.max_tokens,
            max_texts=self.max_texts,
            separator="\n",
        )
        return {"texts": text}

    async def __call__(self, **kwds) -> Topics:
        """Extracts a two-level topic hierarchy from a list of texts."""
        responses = await super().__call__(**kwds)
        return responses[0]  # type: ignore


class TopicAssigner(Tool):
    """Enforce correct topic-subtopic assignment via a Pydantic model."""

    topics: Topics
    """The topic hierarchy to assign topics from."""
    texts: Iterable[str | float | None]
    """The texts to assign topics to."""

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
        topics_json = json.dumps(self.topics.to_dict(), indent=2)
        return prompt.substitute(topics=topics_json)

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_label_model(self.topics.to_dict())

    @cached_property
    def context(self) -> AnyContext:
        return [{"text": text} for text in self.texts]


class MultiTopicAssigner(TopicAssigner):
    """Enforce correct multi-topic-subtopic assignment via a Pydantic model."""

    SYSTEM_PROMPT: ClassVar[str] = MULTI_LABEL_PROMPT_SYSTEM
    USER_PROMPT: ClassVar[str] = MULTI_LABEL_PROMPT_USER

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_multi_label_model(self.topics.to_dict())  # type: ignore
