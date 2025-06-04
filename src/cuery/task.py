from collections.abc import Callable
from pathlib import Path

import instructor
import pandas as pd
from instructor import Instructor
from pandas import DataFrame

from . import call
from .context import AnyContext, context_is_iterable
from .pretty import Console, ConsoleOptions, Group, Padding, Panel, RenderResult
from .prompt import Prompt
from .response import Response, ResponseClass, ResponseSet
from .utils import LOG

AnyCfg = str | Path | dict


def check_model_name(model: str) -> None:
    """Check if the model name is valid."""
    if "/" not in model:
        raise ValueError(
            f"Invalid model name: {model}. It should be in the format 'provider/model'."
        )


class ErrorLogger:
    def __init__(self) -> None:
        self.count = 0

    def log(self, error: Exception) -> None:
        self.count += 1


class QueryLogger:
    def __init__(self) -> None:
        self.queries = []

    def log(self, *args, **kwargs) -> None:
        """Log a query to the internal list."""
        self.queries.append(kwargs)


class Task:
    registry: dict[str, "Task"] = {}

    def __init__(
        self,
        prompt: str | Path | Prompt,
        response: ResponseClass,
        name: str | None = None,
        model: str | None = None,
        log_prompt: bool = False,
        log_response: bool = False,
    ):
        self.name = name
        self.response = response
        self.prompt = prompt
        self.log_prompt = log_prompt
        self.log_response = log_response

        if isinstance(prompt, str | Path):
            self.prompt = Prompt.from_config(prompt)

        if model is None:
            self.client = instructor.from_provider("openai/gpt-3.5-turbo", async_client=True)
        else:
            check_model_name(model)
            self.client = instructor.from_provider(model, async_client=True)

        if name:
            Task.registry[name] = self

    def _select_client(self, model: str | None = None) -> Instructor:
        if model is None:
            return self.client

        check_model_name(model)
        return instructor.from_provider(model, async_client=True) or self.client

    async def call(
        self,
        context: AnyContext | None = None,
        model: str | None = None,
        **kwds,
    ) -> ResponseSet:
        client = self._select_client(model)

        response = await call.call(
            client=client,
            prompt=self.prompt,  # type: ignore
            context=context,  # type: ignore
            response_model=self.response,
            log_prompt=self.log_prompt,
            log_response=self.log_response,
            **kwds,
        )

        return ResponseSet(response, context, self.prompt.required)  # type: ignore

    async def iter(
        self,
        context: AnyContext | None = None,
        model: str | None = None,
        callback: Callable[[Response, Prompt, dict], None] | None = None,
        **kwds,
    ) -> ResponseSet:
        client = self._select_client(model)

        self.error_log = ErrorLogger()
        self.query_log = QueryLogger()
        client.on("parse:error", self.error_log.log)
        client.on("completion:kwargs", self.query_log.log)

        responses = await call.iter_calls(
            client=client,
            prompt=self.prompt,  # type: ignore
            context=context,  # type: ignore
            response_model=self.response,
            callback=callback,
            log_prompt=self.log_prompt,
            log_response=self.log_response,
            **kwds,
        )

        if self.error_log.count > 0:
            LOG.warning(f"Encountered: {self.error_log.count} response parsing errors!")

        return ResponseSet(responses, context, self.prompt.required)  # type: ignore

    async def gather(
        self,
        context: AnyContext | None = None,
        model: str | None = None,
        n_concurrent: int = 1,
        **kwds,
    ) -> ResponseSet:
        client = self._select_client(model)

        self.error_log = ErrorLogger()
        self.query_log = QueryLogger()
        client.on("parse:error", self.error_log.log)
        client.on("completion:kwargs", self.query_log.log)

        responses = await call.gather_calls(
            client=client,
            prompt=self.prompt,  # type: ignore
            context=context,  # type: ignore
            response_model=self.response,
            max_concurrent=n_concurrent,
            log_prompt=self.log_prompt,
            log_response=self.log_response,
            **kwds,
        )

        if self.error_log.count > 0:
            LOG.warning(f"Encountered: {self.error_log.count} response parsing errors!")

        return ResponseSet(responses, context, self.prompt.required)  # type: ignore

    async def __call__(
        self,
        context: AnyContext | None = None,
        model: str | None = None,
        n_concurrent: int = 1,
        **kwds,
    ) -> ResponseSet:
        """Auto-dispatch to the appropriate method based on context type."""
        if context_is_iterable(context):
            if n_concurrent > 1:
                return await self.gather(context, model, n_concurrent, **kwds)

            return await self.iter(context, model, **kwds)

        return await self.call(context, model)

    @classmethod
    def from_config(cls, prompt: AnyCfg, response: AnyCfg) -> "Task":
        prompt = Prompt.from_config(prompt)  # type: ignore
        response = Response.from_config(response)  # type: ignore
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
