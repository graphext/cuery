"""Wrappers to call instructor with a prompt and context."""

from asyncio import Semaphore
from collections.abc import Callable
from functools import partial

from instructor.client import Instructor
from pandas import DataFrame
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm

from .context import iter_context
from .prompt import Prompt
from .response import ResponseClass, ResponseModel
from .utils import LOG


async def call(
    client: Instructor,
    prompt: Prompt,
    context: dict | None,
    response_model: ResponseClass,
    model: str | None = None,
    fallback: bool = True,
    **kwds,
) -> ResponseModel:
    """Prompt once with the given context (validated)."""
    if prompt.required:
        if not context:
            raise ValueError("Context is required for prompt but wasn't provided!")

        if missing := [k for k in prompt.required if k not in context]:
            raise ValueError(
                f"Missing required keys in context: {', '.join(missing)}\nContext:\n{context}"
            )

    if model is not None:
        kwds["model"] = model

    try:
        response, completion = await client.chat.completions.create_with_completion(
            messages=list(prompt),  # type: ignore
            response_model=response_model,
            context=context,
            **kwds,
        )
        response._raw_response = completion
        return response
    except Exception as exception:
        if not fallback:
            raise

        LOG.error(f"{exception}")
        LOG.error("Falling back to default response.")
        return response_model.fallback()


async def iter_calls(
    client: Instructor,
    prompt: Prompt,
    context: dict | list[dict] | DataFrame,
    response_model: ResponseClass,
    model: str | None = None,
    **kwds,
) -> list[ResponseModel]:
    """Sequential iteration of prompt over iterable contexts."""

    context, total = iter_context(context, prompt.required)  # type: ignore

    results = []
    with tqdm(desc="Iterating context", total=total) as pbar:
        for c in context:
            result = await call(
                client,
                model=model,
                prompt=prompt,
                context=c,  # type: ignore
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
    response_model: ResponseClass,
    model: str | None = None,
    max_concurrent: int = 2,
    **kwds,
) -> list[ResponseModel]:
    """Async iteration of prompt over iterable contexts."""
    sem = Semaphore(max_concurrent)
    context, _ = iter_context(context, prompt.required)  # type: ignore
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
