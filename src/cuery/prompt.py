from asyncio import Semaphore
from collections.abc import Callable
from functools import partial
from pathlib import Path

from instructor.client import Instructor
from pandas import DataFrame
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from .context import check_context_iterable
from .utils import load_yaml


class ResponseModel(BaseModel):
    """Base class for all response models."""

    @classmethod
    def fallback(cls) -> "ResponseModel":
        return cls.model_construct(**dict.fromkeys(cls.model_fields, None))


ResponseClass = type[ResponseModel]


class Message(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    messages: list[Message] = Field(min_items=1)
    required: list[str] = Field(default_factory=list)

    def __iter__(self):
        yield from (dict(message) for message in self.messages)


def load(relpath: str | Path) -> dict:
    configs = load_yaml(relpath)
    return {k: Prompt(**v) for k, v in configs.items()}


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
