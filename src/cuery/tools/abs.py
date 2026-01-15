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
aspects (entities/features) mentioned in a text and determining the polarity of the sentiment expressed
toward each.

Specifically:

1. Identify aspects in the text that have either a positive or negative sentiment expressed toward them.
2. Ignore(!) all aspects that do not have a sentiment associated with them or where the sentiment is neutral.
3. Output a list of objects, where each object contains:
    a. "entity": the normalized, SPECIFIC aspect being evaluated (see normalization rules below)
    b. "sentiment": the sentiment label as either "positive" or "negative"
    c. "reason": a normalized "<adjective> <entity>" phrase capturing the sentiment (see rules below)
4. If there are no sentiment-bearing aspects in the text, the output should be an empty list.

Example Output format:
[{"entity": "<entity>", "sentiment": "<polarity>", "reason": "<adjective> <entity>"}, ...]

### Entity Normalization Rules ###
- Use lowercase, remove articles ("the", "a"), remove extra whitespace
- CRITICAL: Be SPECIFIC about what the entity refers to. Avoid generic terms.
  - BAD: "queues", "lines", "wait" (queues for what?)
  - GOOD: "ride queues", "food queues", "ticket lines", "ride wait times"
  - BAD: "prices", "cost" (prices of what?)
  - GOOD: "food prices", "ticket prices", "merchandise prices", "parking cost"
  - BAD: "service", "staff" (which service/staff?)
  - GOOD: "restaurant service", "ride operators", "customer support", "park staff"
- Standardize common variations: "battery-life" → "battery life", "wait time" → "wait times"
- Use the most common/canonical form of an entity across texts

### Reason (Aspect Phrase) Normalization Rules ###
- Format: "<normalized_adjective> <specific_entity>" (e.g., "expensive food", "long ride queues")
- Use STANDARDIZED adjectives from this preference list if applicable:
  - Price/cost: "expensive" or "affordable" (not "overpriced", "pricey", "costs a fortune", "cheap")
  - Wait/time: "long" or "short" (not "endless", "quick", "forever")
  - Quality: "poor" or "excellent" (not "terrible", "amazing", "awful", "great")
  - Size/amount: "crowded" or "spacious" (not "packed", "empty", "busy")
  - Cleanliness: "dirty" or "clean" (not "filthy", "spotless", "gross")
  - Temperature: "hot" or "cold" (not "freezing", "boiling", "warm")
  - Staff/service: "friendly" or "unfriendly", "helpful" or "unhelpful"
- The entity in the reason MUST match the normalized entity field exactly
- Examples:
  - "costs you a fortune" + "food" → reason: "expensive food"
  - "super long wait" + "ride queues" → reason: "long ride queues"
  - "staff were so rude" + "park staff" → reason: "unfriendly park staff"

### What NOT to extract ###
- Factual statements without sentiment ("the park has 5 rides")
- Neutral descriptions ("new", "modern", "efficient" alone aren't sentiments)
- Inferred sentiments - only explicit expressions of positive/negative feelings

### Aspect Categories (if provided) ###
If aspect_categories are provided below, map each entity to one of these categories.
This helps with grouping and filtering. Use the category field to indicate which category
the entity belongs to. If no categories are provided, leave category as null.

${instructions}
""")

ABS_PROMPT_USER = dedent("""
Extract aspects and their sentiments from the following text. Remember:
- Entity must be SPECIFIC (e.g., "ride queues" not just "queues")
- Reason must use standardized adjectives + the exact entity (e.g., "long ride queues")
- If aspect categories were provided, assign each entity to the most appropriate category

# Text

{{text}}
""")


class AspectEntity(Response):
    """Represents an aspect with its sentiment and qualified phrase."""

    entity: str
    """The specific, normalized aspect being evaluated (e.g., 'ride queues', 'food prices')."""
    sentiment: Literal["positive", "negative"]
    """The sentiment associated with the aspect (positive or negative)."""
    reason: str
    """The normalized aspect phrase: '<standardized_adjective> <entity>' (e.g., 'long ride queues')."""
    category: str | None = None
    """Optional category the entity belongs to (e.g., 'food', 'service', 'pricing')."""


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
    aspect_categories: list[str] | None = None
    """Optional list of aspect categories to map entities to (e.g., ['food', 'service', 'pricing'])."""

    response_model: ClassVar[ResponseClass] = AspectEntities

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt(
            messages=[
                {"role": "system", "content": ABS_PROMPT_SYSTEM},
                {"role": "user", "content": ABS_PROMPT_USER},
            ],  # type: ignore
        )

        # Build category instruction if categories provided
        category_instruction = ""
        if self.aspect_categories:
            cats = ", ".join(f'"{c}"' for c in self.aspect_categories)
            category_instruction = f"\n\nAspect Categories to use: [{cats}]"

        combined_instructions = self.instructions + category_instruction
        return prompt.substitute(instructions=combined_instructions)

    @cached_property
    def context(self) -> AnyContext:
        return [{"text": text} for text in self.texts]
