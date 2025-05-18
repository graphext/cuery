from collections.abc import Iterable
from pathlib import Path

import instructor
from instructor import Instructor
from openai import AsyncOpenAI
from pandas import DataFrame
from pydantic import BaseModel

from . import prompt, utils
from .prompt import Prompt
from .response import ResponseClass

AnyCfg = str | Path | dict


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def context_is_iterable(context: dict | list[dict] | DataFrame) -> bool:
    """Check if context is iterable."""
    if isinstance(context, DataFrame):
        return True
    if isinstance(context, dict):
        return all(is_iterable(v) for v in context.values())

    return isinstance(context, list) and all(isinstance(d, dict) for d in context)


class ErrorCounter:
    def __init__(self) -> None:
        self.count = 0

    def count_error(self, error: Exception) -> None:
        self.count += 1


class Task:
    def __init__(
        self,
        prompt: Prompt,
        response: ResponseClass,
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

        self.error_counter = ErrorCounter()
        client.on("parse:error", self.error_counter.count_error)

        result = await prompt.iter_calls(
            client=client,
            model=model,
            prompt=self.prompt,
            context=context,
            response_model=self.response,
            **kwds,
        )

        if self.error_counter.count > 0:
            print(f"Encountered: {self.error_counter.count} response parsing errors!")

        return result

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

        self.error_counter = ErrorCounter()
        client.on("parse:error", self.error_counter.count_error)

        result = await prompt.gather_calls(
            client=client,
            model=model,
            prompt=self.prompt,
            context=context,
            response_model=self.response,
            max_concurrent=n_concurrent,
            **kwds,
        )

        if self.error_counter.count > 0:
            print(f"Encountered: {self.error_counter.count} response parsing errors!")

        return result

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

    def explode_responses(
        self,
        responses: Iterable[BaseModel],
        context_df: DataFrame,
        to_pandas: bool = True,
    ):
        """Flatten a list of pydantic models containing lists into a flat list of records."""
        is_multi, field = utils.is_multi_output(self.response)
        if not is_multi:
            raise ValueError(
                "Responses don't seem to be multi-output "
                "(1:N or having a single list/array field)."
            )

        records = []
        for i, response in enumerate(responses):
            context = context_df.iloc[i].to_dict()
            for item in getattr(response, field):
                rec = context | dict(item)
                records.append(rec)

        if to_pandas:
            return DataFrame.from_records(records)

        return records

    def from_config(prompt: AnyCfg, response: AnyCfg) -> "Task":
        prompt = Prompt.from_config(prompt)
        response = ResponseClass.from_config(response)
        return Task(prompt=prompt, response=response)
