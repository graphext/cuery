from asyncio import Semaphore, gather
from collections.abc import Callable
from functools import partial
from pathlib import Path

from instructor.client import Instructor
from pandas import DataFrame
from pydantic import BaseModel, Field

from .context import check_context_iterable
from .utils import load_yaml


class Message(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    messages: list[Message] = Field(
        description="A list of messages to be used in the prompt",
        min_items=1,
    )
    required: list[str] = Field(
        description="A list of required keys that must be present in the context",
        default_factory=list,
    )

    def __iter__(self):
        yield from (dict(message) for message in self.messages)


def load(relpath: str | Path) -> dict:
    """Load prompts from a YAML file."""
    configs = load_yaml(relpath)
    return {k: Prompt(**v) for k, v in configs.items()}


async def call(
    client: Instructor | None,
    model: str | None,
    prompt: Prompt,
    context: dict | None,
    response_model: BaseModel,
    **kwds,
) -> BaseModel:
    """Prompt once with the given context (validated)."""
    if prompt.required and not context:
        raise ValueError("Context is required for prompt but wasn't provided!")

    missing = [k for k in prompt.required if k not in context]
    if missing:
        raise ValueError(f"Missing required keys in context: {', '.join(missing)}")

    if model is None:
        return await client.chat.completions.create(
            messages=list(prompt),
            response_model=response_model,
            context=context,
            **kwds,
        )

    return await client.chat.completions.create(
        model=model,
        messages=list(prompt),
        response_model=response_model,
        context=context,
        **kwds,
    )


async def iter_calls(
    client: Instructor | None,
    model: str | None,
    prompt: Prompt,
    context: dict | list[dict] | DataFrame,
    response_model: BaseModel,
    **kwds,
) -> list[BaseModel]:
    """Sequential iteration of prompt over iterable contexts."""

    context = check_context_iterable(context, prompt.required)

    results = []
    for c in context:
        result = await call(
            client,
            model,
            prompt=prompt,
            context=c,
            response_model=response_model,
            **kwds,
        )
        results.append(result)

    return results


async def rate_limited(func: Callable, sem: Semaphore, **kwds):
    async with sem:
        return await func(**kwds)


async def gather_calls(
    client: Instructor | None,
    model: str | None,
    prompt: Prompt,
    context: dict | list[dict] | DataFrame,
    response_model: BaseModel,
    max_concurrent: int = 2,
    **kwds,
) -> list[BaseModel]:
    """Async iteration of prompt over iterable contexts."""
    sem = Semaphore(max_concurrent)
    context = check_context_iterable(context, prompt.required)
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
    return await gather(*tasks)
