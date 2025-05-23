from pathlib import Path

import instructor
import pandas as pd
from instructor import Instructor
from openai import AsyncOpenAI
from pandas import DataFrame

from . import call
from .context import AnyContext, context_is_iterable
from .pretty import Console, ConsoleOptions, Group, Padding, Panel, RenderResult
from .prompt import Prompt
from .response import ResponseClass, ResponseModel, ResponseSet
from .utils import LOG

AnyCfg = str | Path | dict


class ErrorCounter:
    def __init__(self) -> None:
        self.count = 0

    def count_error(self, error: Exception) -> None:
        self.count += 1


class Task:
    registry: dict[str, "Task"] = {}

    def __init__(
        self,
        name: str,
        prompt: str | Path | Prompt,
        response: ResponseClass,
        client: Instructor | None = None,
        model: str | None = None,
    ):
        self.name = name
        self.response = response
        self.client = client
        self.model = model
        self.prompt = prompt

        if isinstance(prompt, str | Path):
            self.prompt = Prompt.from_config(prompt)

        if self.client is None:
            self.client = instructor.from_openai(AsyncOpenAI())

        if self.model is None:
            self.model = "gpt-3.5-turbo"

        Task.registry[name] = self

    async def call(
        self,
        context: AnyContext | None = None,
        client: Instructor | None = None,
        model: str | None = None,
        **kwds,
    ) -> ResponseSet:
        client = client or self.client
        model = model or self.model

        if client is None:
            raise ValueError("Client cannot be None")

        response = await call.call(
            client=client,
            model=model,
            prompt=self.prompt,  # type: ignore
            context=context,  # type: ignore
            response_model=self.response,
            **kwds,
        )

        return ResponseSet(response, context, self.prompt.required)  # type: ignore

    async def iter(
        self,
        context: AnyContext | None = None,
        client: Instructor | None = None,
        model: str | None = None,
        **kwds,
    ) -> ResponseSet:
        client = client or self.client
        model = model or self.model

        if client is None:
            raise ValueError("Client cannot be None")

        self.error_counter = ErrorCounter()
        client.on("parse:error", self.error_counter.count_error)

        responses = await call.iter_calls(
            client=client,
            model=model,
            prompt=self.prompt,  # type: ignore
            context=context,  # type: ignore
            response_model=self.response,
            **kwds,
        )

        if self.error_counter.count > 0:
            LOG.warning(f"Encountered: {self.error_counter.count} response parsing errors!")

        return ResponseSet(responses, context, self.prompt.required)  # type: ignore

    async def gather(
        self,
        context: AnyContext | None = None,
        client: Instructor | None = None,
        model: str | None = None,
        n_concurrent: int = 1,
        **kwds,
    ) -> ResponseSet:
        client = client or self.client
        model = model or self.model

        if client is None:
            raise ValueError("Client cannot be None")

        self.error_counter = ErrorCounter()
        client.on("parse:error", self.error_counter.count_error)

        responses = await call.gather_calls(
            client=client,
            model=model,
            prompt=self.prompt,  # type: ignore
            context=context,  # type: ignore
            response_model=self.response,
            max_concurrent=n_concurrent,
            **kwds,
        )

        if self.error_counter.count > 0:
            LOG.warning(f"Encountered: {self.error_counter.count} response parsing errors!")

        return ResponseSet(responses, context, self.prompt.required)  # type: ignore

    async def __call__(
        self,
        context: AnyContext | None = None,
        client: Instructor | None = None,
        model: str | None = None,
        n_concurrent: int = 1,
        **kwds,
    ) -> ResponseSet:
        """Auto-dispatch to the appropriate method based on context type."""
        client = client or self.client
        model = model or self.model

        if context_is_iterable(context):
            if n_concurrent > 1:
                return await self.gather(context, client, model, n_concurrent, **kwds)

            return await self.iter(context, client, model, **kwds)

        return await self.call(context, client, model)

    @classmethod
    def from_config(cls, prompt: AnyCfg, response: AnyCfg) -> "Task":
        prompt = Prompt.from_config(prompt)  # type: ignore
        response = ResponseModel.from_config(response)  # type: ignore
        return Task(prompt=prompt, response=response)  # type: ignore

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        group = [
            Padding(self.prompt, (1, 0, 0, 0)),  # type: ignore
            Padding(self.response.fallback(), (1, 0, 0, 0)),
        ]
        yield Panel(Group(*group), title="Task")


class Chain:
    """Chain multiple tasks together.

    The output of each task is auto-converted to a DataFrame and passed to the next task as
    input context.
    """

    def __init__(self, *tasks: list[Task]):
        self.tasks = tasks

    async def __call__(self, context: AnyContext | None = None, **kwds) -> DataFrame:
        n = len(self.tasks)
        self._responses = []
        for i, task in enumerate(self.tasks):
            LOG.info(f"[{i + 1}/{n}] Running task '{task.response.__name__}'")  # type: ignore
            response = await task(context, **kwds)  # type: ignore
            context = response.to_pandas()  # type: ignore
            self._responses.append(response)

        usages = [response.usage() for response in self._responses]
        task_names = [task.response.__name__ for task in self.tasks]  # type: ignore
        for i, usage in enumerate(usages):
            usage["task_index"] = i
            usage["task"] = task_names[i]

        self._usage = pd.concat(usages, axis=0)
        return context  # type: ignore
