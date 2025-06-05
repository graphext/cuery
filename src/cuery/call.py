"""Wrappers to call instructor with a cuery.Prompt, cuery.Response and context."""

from asyncio import Semaphore
from collections.abc import Callable
from functools import partial

from instructor.client import Instructor
from pandas import DataFrame
from rich import print as pprint
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm

from .context import iter_context
from .prompt import Prompt
from .response import Response, ResponseClass
from .utils import LOG


async def call(
    client: Instructor,
    prompt: Prompt,
    context: dict | None,
    response_model: ResponseClass,
    fallback: bool = True,
    log_prompt: bool = False,
    log_response: bool = False,
    **kwds,
) -> Response:
    """Prompt once with the given Prompt and context (validated).

    If fallback is True, will return result of response_model.fallback() if the call fails.
    """
    if prompt.required:
        if not context:
            raise ValueError("Context is required for prompt but wasn't provided!")

        if missing := [k for k in prompt.required if k not in context]:
            raise ValueError(
                f"Missing required keys in context: {', '.join(missing)}\nContext:\n{context}"
            )

    if log_prompt:
        pprint(prompt)

    try:
        response, completion = await client.chat.completions.create_with_completion(
            messages=list(prompt),  # type: ignore
            response_model=response_model,
            context=context,
            **kwds,
        )
        response._raw_response = completion
    except Exception as exception:
        if not fallback:
            raise

        LOG.error(f"{exception}")
        LOG.error("Falling back to default response.")
        response = response_model.fallback()

    if log_response:
        pprint(response.to_dict())

    return response


async def iter_calls(
    client: Instructor,
    prompt: Prompt,
    context: dict | list[dict] | DataFrame,
    response_model: ResponseClass,
    callback: Callable[[Response, Prompt, dict], None] | None = None,
    **kwds,
) -> list[Response]:
    """Sequential iteration of prompt over iterable contexts."""

    context, total = iter_context(context, prompt.required)  # type: ignore

    results = []
    with tqdm(desc="Iterating context", total=total) as pbar:
        for c in context:
            result = await call(
                client,
                prompt=prompt,
                context=c,  # type: ignore
                response_model=response_model,
                **kwds,
            )
            results.append(result)

            if callback is not None:
                callback(result, prompt, c)  # type: ignore

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
    max_concurrent: int = 2,
    **kwds,
) -> list[Response]:
    """Async iteration of prompt over iterable contexts."""
    sem = Semaphore(max_concurrent)
    context, _ = iter_context(context, prompt.required)  # type: ignore
    func = partial(
        rate_limited,
        func=call,
        sem=sem,
        client=client,
        prompt=prompt,
        response_model=response_model,
        **kwds,
    )
    tasks = [func(context=c) for c in context]
    return await async_tqdm.gather(*tasks, desc="Gathering responses", total=len(tasks))
