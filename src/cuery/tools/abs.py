"""Higher-level API for extracting entities from texts using one-shot prompts.

Some examples of LLM-based methods:

- Evaluating Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models
  https://arxiv.org/abs/2412.12564
- Structured Sentiment Analysis with Large Language Models: A Winning Solution for RuOpinionNE-2024
  https://dialogue-conf.org/wp-content/uploads/2025/04/VatolinA.104.pdf
- https://simmering.dev/blog/absa-with-dspy/
  https://github.com/psimm/website/blob/master/blog/absa-with-dspy/configs/manual_prompt.json
"""

from collections.abc import Iterable
from functools import cached_property
from typing import ClassVar, Literal

from .. import AnyContext, Prompt, Response, ResponseClass, Tool
from ..utils import dedent

ABS_PROMPT_SYSTEM = dedent("""
You're an expert in Aspect-Based Sentiment Analysis (ABSA). Your task involves identifying specific
entities mentioned in a text (e.g. a person, product, service, or experience) and determining the
polarity of the sentiment expressed toward each.

Specifically:

1. Identify entities in the text that have either a positive or negative sentiment expressed toward them.
2. Ignore(!) all entities that do not have a sentiment associated with them or where the sentiment is neutral.
3. Output a list of objects, where each object contains
    a. the entity as it occurs in the text (key "entity")
    b. the sentiment label as either "positive" or "negative" (key "sentiment")
    c. the reason for the sentiment assignment as a short text (key "reason")
4. If there are no sentiment-bearing entities in the text, the output should be an empty list

Example Output format:
[{"entity": "<entity>", "sentiment": "<polarity>", "reason": "<reason>"}, ...]

Only extract entities that have an explicitly expressed sentiment associated with them, i.e.
subjective opinions, feelings, or evaluations. Do not infer sentiment from factual statements,
e.g. just because a feature is mentioned as "new", a product or service is mentioned as having
a certain feature, or because something is mentioned as "modern", "efficient" etc. it shouldn't be
considered a sentiment. Look for explicit expressions of positive or negative feelings, especially
adjectives or adverbs that indicate a sentiment.

${instructions}
""")

ABS_PROMPT_USER = dedent("""
Return the entities and their sentiments with reasons from the following text section.

# Text

{{text}}
""")


class AspectEntity(Response):
    """Represents an entity with its sentiment and reason for assignment."""

    entity: str
    """The entity mentioned in the text."""
    sentiment: Literal["positive", "negative"]
    """The sentiment associated with the entity (positive or negative)."""
    reason: str
    """The reason for the sentiment assignment."""


class AspectEntities(Response):
    """Represents a collection of entities with their sentiments and reasons for assignment."""

    entities: list[AspectEntity]
    """A list of entities with their sentiments and reasons."""


class AspectSentimentExtractor(Tool):
    """Extract entities with sentiments from texts."""

    texts: Iterable[str | float | None]
    """The texts to extract entities from."""
    instructions: str = ""
    """Further instructions from the user for the entity extraction task."""

    response_model: ClassVar[ResponseClass] = AspectEntities

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt(
            messages=[
                {"role": "system", "content": ABS_PROMPT_SYSTEM},
                {"role": "user", "content": ABS_PROMPT_USER},
            ],  # type: ignore
        )

        return prompt.substitute(instructions=self.instructions)

    @cached_property
    def context(self) -> AnyContext:
        return [{"text": text} for text in self.texts]
