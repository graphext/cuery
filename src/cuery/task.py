from collections.abc import Iterable
from pathlib import Path

import instructor
from instructor import Instructor
from openai import AsyncOpenAI
from pandas import DataFrame
from pydantic import BaseModel

from . import prompt
from .context import AnyContext, context_is_iterable, iter_context
from .pretty import Console, ConsoleOptions, Group, Padding, Panel, RenderResult
from .prompt import Prompt
from .response import ResponseClass
from .utils import LOG

AnyCfg = str | Path | dict


class ErrorCounter:
    def __init__(self) -> None:
        self.count = 0

    def count_error(self, error: Exception) -> None:
        self.count += 1


class Task:
    def __init__(
        self,
        prompt: str | Path | Prompt,
        response: ResponseClass,
        client: str | None = None,
        model: str | None = None,
    ):
        self.prompt = prompt
        self.response = response
        self.client = client
        self.model = model

        if isinstance(prompt, str | Path):
            self.prompt = Prompt.from_config(prompt)

        if self.client is None:
            self.client = instructor.from_openai(AsyncOpenAI())

        if self.model is None:
            self.model = "gpt-3.5-turbo"

    async def call(
        self,
        context: AnyContext | None = None,
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
        context: AnyContext | None = None,
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
            LOG.warning(f"Encountered: {self.error_counter.count} response parsing errors!")

        return result

    async def gather(
        self,
        context: AnyContext | None = None,
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
            LOG.warning(f"Encountered: {self.error_counter.count} response parsing errors!")

        return result

    async def __call__(
        self,
        context: AnyContext | None = None,
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

    def from_config(prompt: AnyCfg, response: AnyCfg) -> "Task":
        prompt = Prompt.from_config(prompt)
        response = ResponseClass.from_config(response)
        return Task(prompt=prompt, response=response)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        group = [
            Padding(self.prompt, (1, 0, 0, 0)),
            Padding(self.response.fallback(), (1, 0, 0, 0)),
        ]
        yield Panel(Group(*group), title="Task")

    def to_pandas(
        self,
        responses: BaseModel | Iterable[BaseModel],
        context: AnyContext | None = None,
        to_pandas: bool = True,
        explode: bool = True,
    ) -> list[dict] | DataFrame:
        """Output as DataFrame, optionally with original context merged in"""
        if isinstance(responses, BaseModel):
            responses = [responses]

        if isinstance(context, dict):
            context = [context]

        if context is not None:
            contexts, _ = iter_context(context, self.prompt.required)
        else:
            contexts = ({} for _ in responses)

        is_multi, field = self.response.is_multi_output()

        records = []
        if is_multi and explode:
            for ctx, response in zip(contexts, responses, strict=True):
                for item in getattr(response, field):
                    records.append(ctx | dict(item))
        else:
            for ctx, response in zip(contexts, responses, strict=True):
                records.append(ctx | dict(response))

        return DataFrame.from_records(records) if to_pandas else records


class Chain:
    """Chain multiple tasks together.

    The output of each task is auto-converted to a DataFrame and passed to the next task as
    input context.
    """

    def __init__(self, *tasks: list[Task]):
        self.tasks = tasks

    async def __call__(self, context: AnyContext | None = None, **kwds) -> DataFrame:
        n = len(self.tasks)
        for i, task in enumerate(self.tasks):
            LOG.info(f"[{i + 1}/{n}] Running task '{task.response.__name__}'")
            result = await task(context, **kwds)
            context = task.to_pandas(result, context)

        return context
