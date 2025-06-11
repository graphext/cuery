from collections.abc import Iterable
from typing import Literal

from .. import utils
from ..context import AnyContext
from ..prompt import Prompt
from ..response import Field, Response, ResponseSet
from ..task import Task
from ..utils import dedent

SYSTEM_PROMPT = dedent("""
You're an expert SEO specialist analyzing google keyword searches for a specific domain.

Your task is to simplify a list of search keywords (short phrases) into a smaller group of clean
keywords that make sense to later group, aggregate and analyze together. The idea is to remove
duplicate keywords that are identical in meaning but are spelled differently
(misspelling, singular vs. plural etc.), while preserving different search intents and
meaningful variations.

The keywords come from a dataset of '%(domain)s'. %(extra)s
""")

USER_PROMPT = dedent("""
Extract a clean, deduplicated list of search keywords of no more than %(n_max)s items
from the following list.

# Keywords

{{keywords}}
""")

ASSIGNMENT_PROMPT_SYSTEM = dedent("""
You're task is to use the following list of clean keywords,
and select and return the best semantically matching keyword for a given input phrase.

# Keywords

%(keywords)s
""")

ASSIGNMENT_PROMPT_USER = dedent("""
Assign the correct keyword to the following phrase: {{text}}.
""")


class KeywordCleaner:
    """A class to clean and deduplicate search keywords from a list of texts."""

    def __init__(
        self,
        domain: str,
        n_max: int = 10,
        extra: str | None = None,
    ):
        prompt = Prompt(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT % {"domain": domain, "extra": extra},
                },
                {
                    "role": "user",
                    "content": USER_PROMPT % {"n_max": n_max},
                },
            ],  # type: ignore
            required=["keywords"],
        )

        class Keywords(Response):
            keywords: list[str] = Field(
                ...,
                description="A list of clean google search keywords.",
                max_length=n_max,
            )

        self.task = Task(prompt=prompt, response=Keywords)

    async def __call__(
        self,
        keywords: Iterable[str],
        model: str,
        max_dollars: float,
        max_tokens: float | None = None,
        max_texts: float | None = None,
    ) -> Response:
        """Extracts a two-level topic hierarchy from a list of texts."""
        text = utils.concat_up_to(
            keywords,
            model=model,
            max_dollars=max_dollars,
            max_tokens=max_tokens,
            max_texts=max_texts,
            separator="\n",
        )
        responses = await self.task.call(context={"keywords": text}, model=model)
        return responses[0]


class KeywordAssigner:
    """Enforce correct clean keyword assignment."""

    def __init__(self, keywords: Response):
        keywords = keywords.to_dict()["keywords"]
        prompt = Prompt(
            messages=[
                {"role": "system", "content": ASSIGNMENT_PROMPT_SYSTEM % {"keywords": keywords}},
                {"role": "user", "content": ASSIGNMENT_PROMPT_USER},
            ],  # type: ignore
            required=["text"],
        )

        class Match(Response):
            keyword: Literal[*keywords]

        self.task = Task(prompt=prompt, response=Match)

    async def __call__(self, texts: AnyContext, model: str, **kwds) -> ResponseSet:
        return await self.task(context=texts, model=model, **kwds)
