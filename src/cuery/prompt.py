from asyncio import Semaphore
from collections.abc import Callable
from functools import partial
from pathlib import Path

from instructor.client import Instructor
from pandas import DataFrame
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.text import Text
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from .context import check_context_iterable
from .response import ResponseModel
from .utils import get_config

ROLE_STYLES = {
    "system": "bold cyan",
    "user": "bold green",
    "assistant": "bold yellow",
    "function": "bold magenta",
}


class Message(BaseModel):
    """Message class for chat completions."""

    role: str
    content: str

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        style = ROLE_STYLES.get(self.role, "bold")
        text = Syntax(
            self.content,
            "django",
            # code_width=84,
            word_wrap=True,
            theme="friendly",
            padding=1,
        )
        title = f"[{style}]{self.role.upper()}"
        yield Panel(text, title=title, expand=True)


class Prompt(BaseModel):
    """Prompt class for chat completions."""

    messages: list[Message] = Field(min_items=1)
    required: list[str] = Field(default_factory=list)

    def __iter__(self):
        yield from (dict(message) for message in self.messages)

    @classmethod
    def from_config(cls, source: str | Path | dict) -> "Prompt":
        config = get_config(source)
        return cls(**config)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        group = []
        if self.required:
            group.append(
                Padding(
                    Group(
                        Text("Required: ", end=""),
                        Pretty(self.required),
                    ),
                    1,
                )
            )

        for message in self.messages:
            group.append(message)

        yield Panel(Group(*group), title="Prompt", expand=False)


async def call(
    client: Instructor,
    prompt: Prompt,
    context: dict | None,
    response_model: ResponseModel,
    model: str | None = None,
    fallback: bool = True,
    **kwds,
) -> BaseModel:
    """Prompt once with the given context (validated)."""
    if prompt.required and not context:
        raise ValueError("Context is required for prompt but wasn't provided!")

    missing = [k for k in prompt.required if k not in context]
    if missing:
        raise ValueError(
            f"Missing required keys in context: {', '.join(missing)}\nContext:\n{context}"
        )

    if model is None:
        try:
            return await client.chat.completions.create(
                messages=list(prompt),
                response_model=response_model,
                context=context,
                **kwds,
            )
        except Exception as exception:
            if not fallback:
                raise

            print(f"Error: {exception}")
            print("Falling back to default response.")
            return response_model.fallback()

    try:
        return await client.chat.completions.create(
            model=model,
            messages=list(prompt),
            response_model=response_model,
            context=context,
            **kwds,
        )
    except Exception as exception:
        if not fallback:
            raise

        print(f"Error: {exception}")
        print("Falling back to default response.")
        return response_model.fallback()


async def iter_calls(
    client: Instructor,
    prompt: Prompt,
    context: dict | list[dict] | DataFrame,
    response_model: ResponseModel,
    model: str | None = None,
    **kwds,
) -> list[BaseModel]:
    """Sequential iteration of prompt over iterable contexts."""

    context, total = check_context_iterable(context, prompt.required)

    results = []
    with tqdm(desc="Iterating context", total=total) as pbar:
        for c in context:
            result = await call(
                client,
                model=model,
                prompt=prompt,
                context=c,
                response_model=response_model,
                **kwds,
            )
            results.append(result)
            pbar.update(1)

    return results


async def rate_limited(func: Callable, sem: Semaphore, **kwds):
    async with sem:
        return await func(**kwds)


async def gather_calls(
    client: Instructor,
    prompt: Prompt,
    context: dict | list[dict] | DataFrame,
    response_model: ResponseModel,
    model: str | None = None,
    max_concurrent: int = 2,
    **kwds,
) -> list[BaseModel]:
    """Async iteration of prompt over iterable contexts."""
    sem = Semaphore(max_concurrent)
    context, _ = check_context_iterable(context, prompt.required)
    func = partial(
        rate_limited,
        func=call,
        sem=sem,
        client=client,
        model=model,
        prompt=prompt,
        response_model=response_model,
        **kwds,
    )
    tasks = [func(context=c) for c in context]
    return await async_tqdm.gather(*tasks, desc="Gathering responses", total=len(tasks))
