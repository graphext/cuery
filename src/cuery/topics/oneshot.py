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
from typing import ClassVar, Literal, Self

from Levenshtein import distance as ldist
from pydantic import model_validator

from .. import utils
from ..context import AnyContext
from ..prompt import Prompt
from ..response import Field, Response, ResponseSet
from ..task import Task

TOPICS_PROMPT = """
From the list of texts below (separated by line breaks), extract a two-level nested list of topics.
The output should be a JSON object with top-level topics as keys and lists of subtopics as values.
The top-level should not contain more than <<n_topics>> topics, and each top-level
should not contain more than <<n_subtopics>> subtopics. The texts come from a dataset of
'<<domain>>', so the topics should be relevant to that domain. Make sure top-level topics are
generalizable and not too specific, so they can be used as a hierarchy for the subtopics. Make
sure also that subtopics are not redundant (no similar ones within the the same top-level topic).
Create fewer topics and subtopics if needed, i.e. when otherwise top-level categories or subtopics
would be too similar.
<<extra>>

# Texts

{{texts}}
"""

ASSIGNMENT_PROMPT_SYSTEM = """
You're task is to use the following hierarchy of topics and subtopics (in json format),
to assign the correct topic and subtopic to each text in the input.

# Topics

<<topics>>
"""

ASSIGNMENT_PROMPT_USER = """
Assign the correct topic and subtopic to the following text.

# Text

{{text}}
"""


def format_prompt(
    prompt: str,
    n_topics: int,
    n_subtopics: int,
    domain: str,
    extra: str | None = None,
) -> str:
    """Format the prompt with the given number of topics and subtopics."""
    extra = extra or ""
    prompt = prompt.replace("<<n_topics>>", str(n_topics))
    prompt = prompt.replace("<<n_subtopics>>", str(n_subtopics))
    prompt = prompt.replace("<<domain>>", domain)
    prompt = prompt.replace("<<extra>>", extra)
    return utils.dedent(prompt)


class TopicExtractor:
    """Enforce the topic-subtopic hierarchy directly via response model."""

    def __init__(
        self,
        domain: str,
        n_topics: int = 10,
        n_subtopics: int = 5,
        extra: str | None = None,
    ):
        prompt = format_prompt(TOPICS_PROMPT, n_topics, n_subtopics, domain, extra)
        prompt = Prompt.from_string(prompt)

        class Topic(Response):
            topic: str = Field(..., description="The top-level topic.")
            subtopics: list[str] = Field(
                ...,
                description="A list of subtopics under the top-level topic.",
                max_length=n_subtopics,
            )

            @model_validator(mode="after")
            def validate_subtopics(self) -> Self:
                # Topic titles should be at least N character edits apart
                min_ldist = 2

                subtopics = [st.lower() for st in self.subtopics]
                errors = []

                sim_err = "Subtopic '{}' too similar to other subtopic '{}'.".format
                perm_err = "Subtopic '{}' is a duplicate (permutation) of subtopic '{}'.".format

                for i, st in enumerate(subtopics):
                    # Subtopics should not be too similar to their parent topic
                    if ldist(st, self.topic.lower()) < min_ldist:
                        errors.append(
                            f"Subtopic '{st}' too similar to parent topic '{self.topic}'."
                        )

                    # Subtopics should not be too similar to each other
                    for j in range(i + 1, len(subtopics)):
                        other = subtopics[j]

                        # Check Levenshtein distance for similarity
                        if ldist(st.replace(" ", ""), other.replace(" ", "")) < min_ldist:
                            errors.append(sim_err(st, other))

                        # Check for permutations of words
                        if set(st.split()) == set(other.split()):
                            errors.append(perm_err(st, other))

                if errors:
                    raise ValueError("Invalid subtopics:\n" + "\n".join(errors))

                return self

        class Topics(Response):
            """A response containing a two-level nested list of topics."""

            topics: list[Topic] = Field(
                ...,
                description="A list of top-level topics with their subtopics.",
                max_length=n_topics,
            )

        self.task = Task(prompt=prompt, response=Topics)

    async def __call__(
        self,
        texts: Iterable[str],
        model: str,
        max_dollars: float | None = None,
        max_tokens: float | None = None,
        max_texts: float | None = None,
        **kwds,
    ) -> Response:
        """Extracts a two-level topic hierarchy from a list of texts."""
        text = utils.concat_up_to(
            texts,
            model=model,
            max_dollars=max_dollars,
            max_tokens=max_tokens,
            max_texts=max_texts,
            separator="\n",
        )
        context = {"texts": text}
        responses = await self.task.call(context=context, model=model, **kwds)
        return responses[0]


def make_topic_class(topics: dict[str, list[str]]) -> type:
    """Create a Pydantic model class for topics and subtopics."""
    tops = list(topics.keys())
    subs = [sub for sublist in topics.values() for sub in sublist]

    class Topic(Response):
        topic: Literal[*tops]
        subtopic: Literal[*subs]

        mapping: ClassVar[dict[str, list]] = topics

        @model_validator(mode="after")
        def is_subtopic(self) -> Self:
            allowed = self.mapping.get(self.topic, [])
            if self.subtopic not in allowed:
                raise ValueError(
                    f"Subtopic '{self.subtopic}' is not a valid subtopic for topic '{self.topic}'."
                    f" Allowed subtopics are: {allowed}."
                )
            return self

    return Topic


class TopicAssigner:
    """Enforce correct topic-subtopic assignment via a Pydantic model."""

    def __init__(self, hierarchy: Response):
        topics = hierarchy.to_dict()["topics"]
        topics = {t["topic"]: t["subtopics"] for t in topics}
        topics_json = json.dumps(topics, indent=2)
        sys_msg = utils.dedent(ASSIGNMENT_PROMPT_SYSTEM).replace("<<topics>>", topics_json)
        usr_msg = utils.dedent(ASSIGNMENT_PROMPT_USER)
        prompt = Prompt(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ],  # type: ignore
            required=["text"],
        )
        response = make_topic_class(topics)
        self.task = Task(prompt=prompt, response=response)

    async def __call__(self, texts: AnyContext, model: str, **kwds) -> ResponseSet:
        return await self.task(context=texts, model=model, **kwds)
