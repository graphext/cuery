"""Wrappers to call instructor with a cuery.Prompt, cuery.Response and context."""

import asyncio
from asyncio import Semaphore
from collections.abc import Callable
from functools import partial

from instructor import Instructor
from pandas import DataFrame
from rich import print as pprint
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm

from .context import iter_context
from .prompt import Prompt
from .response import Response, ResponseClass
from .utils import LOG, on_apify, progress

TQDM_POSITION = -1 if on_apify() else None
"""On Apify log each tqdm update on a separate line (line breaks trigger log updates)."""


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
    progress_callback: Callable | None = None,
    **kwds,
) -> list[Response]:
    """Sequential iteration of prompt over iterable contexts."""

    context, total = iter_context(context, prompt.required)  # type: ignore

    results = []
    for c in progress(
        context,
        desc="Iterating context",
        total=total,
        position=TQDM_POSITION,
        miniters=total / 100,
        callback=progress_callback,
    ):
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
    progress_callback: Callable | None = None,
    **kwds,
) -> list[Response]:
    """Async iteration of prompt over iterable contexts."""
    sem = Semaphore(max_concurrent)
    context, total = iter_context(context, prompt.required)  # type: ignore

    min_iters = int(total / 20)
    rate_limited_call = partial(
        rate_limited,
        func=call,
        sem=sem,
        client=client,
        prompt=prompt,
        response_model=response_model,
        **kwds,
    )

    if progress_callback is None:
        tasks = [rate_limited_call(context=c) for c in context]
        return await async_tqdm.gather(
            *tasks,
            desc="Gathering responses",
            total=len(tasks),
            position=TQDM_POSITION,
            miniters=min_iters,
        )

    async_method = "gather"  # "gather or "as_completed"

    pbar = tqdm(
        desc="Iterating context",
        total=total,
        position=TQDM_POSITION,
        miniters=min_iters,
    )

    if async_method == "gather":

        async def with_callback(**kwds):
            result = await rate_limited_call(**kwds)
            pbar.update()
            if (pbar.n % min_iters == 0) or (pbar.n == total):
                await progress_callback(pbar.format_dict)
            return result

        tasks = [with_callback(context=c) for c in context]
        return await asyncio.gather(*tasks)

    # As completed: usually slower, and dpesn't preserve order
    results = []
    tasks = [rate_limited_call(context=c) for c in context]
    for f in asyncio.as_completed(tasks):
        results.append(await f)
        pbar.update()
        if (pbar.n % min_iters == 0) or (pbar.n == total):
            await progress_callback(pbar.format_dict)

    return results
