from collections.abc import Iterable

import instructor
from instructor import Instructor
from openai import AsyncOpenAI
from pandas import DataFrame
from pydantic import BaseModel

from . import prompt
from .prompt import Prompt

BaseModelClass = type[BaseModel]


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def context_is_iterable(context: dict | list[dict] | DataFrame) -> bool:
    """Check if context is iterable."""
    if isinstance(context, DataFrame):
        return True
    if isinstance(context, dict):
        return all(is_iterable(v) for v in context.values())

    return isinstance(context, list) and all(isinstance(d, dict) for d in context)


class Task:
    def __init__(
        self,
        prompt: Prompt,
        response: BaseModelClass,
        client: str | None = None,
        model: str | None = None,
    ):
        self.prompt = prompt
        self.response = response
        self.client = client
        self.model = model

        if self.client is None:
            self.client = instructor.from_openai(AsyncOpenAI())

        if self.model is None:
            self.model = "gpt-3.5-turbo"

    async def call(
        self,
        context: dict | list[dict] | DataFrame,
        client: Instructor | None = None,
        model: str | None = None,
        **kwds,
    ) -> list[BaseModel]:
        client = client or self.client
        model = model or self.model
        return await prompt.call(
            client=client,
            model=model,
            prompt=self.prompt,
            context=context,
            response_model=self.response,
            **kwds,
        )

    async def iter(
        self,
        context: dict | list[dict] | DataFrame,
        client: Instructor | None = None,
        model: str | None = None,
        **kwds,
    ) -> list[BaseModel]:
        client = client or self.client
        model = model or self.model
        return await prompt.iter_calls(
            client=client,
            model=model,
            prompt=self.prompt,
            context=context,
            response_model=self.response,
            **kwds,
        )

    async def gather(
        self,
        context: dict | list[dict] | DataFrame,
        client: Instructor | None = None,
        model: str | None = None,
        n_concurrent: int = 1,
        **kwds,
    ) -> list[BaseModel]:
        client = client or self.client
        model = model or self.model
        return await prompt.gather_calls(
            client=client,
            model=model,
            prompt=self.prompt,
            context=context,
            response_model=self.response,
            max_concurrent=n_concurrent,
            **kwds,
        )

    async def __call__(
        self,
        context: dict | list[dict] | DataFrame,
        client: Instructor | None = None,
        model: str | None = None,
        n_concurrent: int = 1,
        **kwds,
    ) -> BaseModel | list[BaseModel]:
        """Auto-dispatch to the appropriate method based on context type."""
        client = client or self.client
        model = model or self.model

        if context_is_iterable(context):
            if n_concurrent > 1:
                return await self.gather(context, client, model, n_concurrent, **kwds)

            return await self.iter(context, client, model, **kwds)

        return await self.call(context, client, model)
