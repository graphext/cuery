"""cuery.asy
=================

Composable async "policy" wrappers (timeout, semaphore, retries, fallback, timer, progress)
for coroutine *factories*.

Why factories instead of raw coroutines?
---------------------------------------
Each policy wrapper calls the provided zero‑argument callable to obtain a *fresh*
coroutine object every time it needs to run (initial execution, each retry, etc.).
This avoids the common pitfall of trying to re‑await an already awaited coroutine
and enables clean layering of cross‑cutting concerns.

Acceptable inputs ("coro factories"):
    * An ``async def`` function with all arguments pre‑bound via ``functools.partial``
    * A ``lambda: some_async_fn(...)`` returning a coroutine
    * Any ``Callable[[], Awaitable[Any]]`` fulfilling the ``CoroFactory`` protocol

NOT acceptable:
    * An already *created* coroutine object (e.g. ``some_async_fn(...)``) because it
        is not callable and cannot be re‑created for retries
    * A sync function unless it returns an awaitable (would raise ``TypeError`` when awaited)

Provided wrappers
-----------------
``with_timeout``      Apply an ``asyncio.wait_for`` timeout (per attempt)
``with_semaphore``    Limit concurrency with an ``asyncio.Semaphore``
``with_retries``      Exponential backoff retries using ``tenacity`` (returns after first success)
``with_fallback``     Provide a fallback value or callable if the wrapped factory raises
``with_timer``        Time a single execution attempt and log the elapsed seconds
``with_progress``     Increment a ``tqdm`` progress bar and optionally call an
                       async progress callback
``with_policies``     Convenience function composing any subset of the above in a predictable order

Composition order in ``with_policies``
-------------------------------------
The order is: timeout -> semaphore -> retries -> fallback -> timer -> progress.
This means, for example, that retries encompass timeout and semaphore acquisition,
and that fallback is only engaged after retries are exhausted. The timer (if enabled)
measures only the final, successful execution path (after fallback if provided) and
excludes progress update overhead. Progress is only updated on *successful* completions
of the wrapped call.

Tenacity retry semantics
------------------------
``with_retries`` uses ``AsyncRetrying`` with exponential backoff capped by ``wait_max``.
It returns the first successful result. If all attempts fail, it raises the last exception.

Partial objects "just work"
---------------------------
``functools.partial`` of an ``async def`` is itself a zero‑arg callable producing a new
coroutine each invocation, so it satisfies the factory contract even if static type
checkers cannot perfectly infer ``Callable[[], Awaitable[Any]]``. This is why passing
``partial(fetch, url, client)`` to ``with_policies`` behaves correctly.

Example
-------
.. code-block:: python

        import asyncio
        from functools import partial
        from asyncio import Semaphore
        from tqdm.auto import tqdm
        from cuery.asy import with_policies

        async def fetch(idx: int) -> int:
                await asyncio.sleep(0.1)
                if idx % 5 == 0:
                        raise RuntimeError("boom")
                return idx

        async def main():
                sem = Semaphore(10)
                pbar = tqdm(total=100)

                async def progress_hook(d: dict):  # optional
                        # d contains tqdm's format_dict (e.g. n, total, rate, etc.)
                        pass

                factories = [partial(fetch, i) for i in range(100)]
                wrapped = [
                        with_policies(
                                f,
                                timeout=1.0,
                                semaphore=sem,
                                retries=2,
                                wait_max=5,
                                fallback=lambda: -1,  # return -1 after all retries fail
                                pbar=pbar,
                                progress_callback=progress_hook,
                                min_iters=10,
                        )
                        for f in factories
                ]

                results = await asyncio.gather(*[w() for w in wrapped])
                pbar.close()
                return results

        if __name__ == "__main__":
                asyncio.run(main())

Design goals
------------
    * Small, orthogonal wrappers easy to unit test
    * Deferred coroutine creation for safe retries / timeouts
    * Minimal runtime overhead when a policy is unused (simple identity composition)
    * Clear logging hooks (see retry ``after=`` callback and fallback warning)

"""

from asyncio import Semaphore, wait_for
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from time import perf_counter
from typing import Any

from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

from .utils import LOG

CoroFactory = Callable[[], Awaitable]
"""A callable that produces an awaitable coroutine when called."""
AsyncFunc = Callable[..., Awaitable]
"""An async function type alias for better readability."""


def with_timeout(coro_factory: CoroFactory, timeout: float) -> CoroFactory:
    """Wrap a coroutine factory with a per-run timeout."""

    async def wrapper():
        try:
            return await wait_for(coro_factory(), timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError("Awaitable timed out!") from exc

    return wrapper


def with_semaphore(coro_factory: CoroFactory, semaphore: Semaphore) -> CoroFactory:
    """Wrap a coroutine factory with concurrency control using a semaphore."""

    async def wrapper():
        async with semaphore:
            return await coro_factory()

    return wrapper


def with_retries(
    coro_factory: CoroFactory,
    attempts: int = 3,
    wait_max: int = 60,
) -> Callable[[], Awaitable]:
    """Wrap a coroutine factory with retry logic (tenacity)."""

    async def wrapper():
        def _after(state):
            if getattr(state, "outcome", None) and state.outcome.failed:
                exc = state.outcome.exception()
                LOG.info(f"Retry attempt {state.attempt_number} failed: {exc}")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_random_exponential(multiplier=1, max=wait_max),
            after=_after,
            reraise=False,
        ):
            with attempt:
                return await coro_factory()
        raise Exception("Retry loop ended without success!")

    return wrapper


def with_fallback(coro_factory: CoroFactory, fallback: Callable | Any) -> CoroFactory:
    """Wrap a coroutine factory with fallback if all attempts fail."""

    async def wrapper():
        try:
            return await coro_factory()
        except Exception as exc:
            LOG.warning(f"Coroutine fallback: {exc}")
            if fallback is None:
                raise
            if callable(fallback):
                return fallback()
            return fallback

    return wrapper


def with_progress(
    coro_factory: Callable[[], Awaitable],
    pbar: tqdm,
    progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    min_iters: int = 1,
) -> Callable[[], Awaitable]:
    """
    Wrap a coroutine factory to update a tqdm progress bar after completion,
    and optionally call an async progress_callback every `min_iters` steps or on final step.
    """

    async def wrapper():
        result = await coro_factory()
        pbar.update()
        if progress_callback is not None:  # noqa: SIM102
            if (pbar.n % min_iters == 0) or (pbar.total is not None and pbar.n == pbar.total):
                await progress_callback(pbar.format_dict)  # type: ignore
        return result

    return wrapper


def with_timer(coro_factory: CoroFactory, label: str | None = None) -> CoroFactory:
    """Wrap a coroutine factory with a simple elapsed time logger.

    Logs the wall-clock seconds the coroutine took to resolve (successful or via fallback
    if used outside). If the coroutine raises an exception the timing is still logged
    before the exception is propagated.

    Args:
        coro_factory: Zero-arg coroutine factory to wrap.
        label: Optional label to include in the log line for identification.
    """
    name = label or getattr(coro_factory, "__name__", "<factory>")

    async def wrapper():
        start = perf_counter()
        try:
            return await coro_factory()
        except Exception:
            raise
        finally:
            elapsed = perf_counter() - start
            LOG.info(f"q({name}): t={elapsed:.2f}s")

    return wrapper


def with_policies(  # noqa: PLR0913
    coro_factory: CoroFactory,
    timeout: float | None = None,
    semaphore: Semaphore | None = None,
    retries: int = 0,
    wait_max: int = 60,
    fallback: Callable | Any = None,
    timer: bool = False,
    pbar: tqdm | None = None,
    progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    min_iters: int = 1,
    label: str | None = None,
) -> CoroFactory:
    """Wrap a coroutine factory with multiple policies."""
    wrapped = coro_factory

    if timeout is not None:
        wrapped = with_timeout(wrapped, timeout)
    if timer:
        wrapped = with_timer(wrapped, label=label)
    if semaphore is not None:
        wrapped = with_semaphore(wrapped, semaphore)
    if retries > 0:
        wrapped = with_retries(wrapped, attempts=retries, wait_max=wait_max)
    if fallback is not None:
        wrapped = with_fallback(wrapped, fallback)
    if pbar is not None:
        wrapped = with_progress(
            wrapped,
            pbar=pbar,
            progress_callback=progress_callback,
            min_iters=min_iters,
        )

    return wrapped


def all_with_policies(
    func: AsyncFunc,
    kwds: list[dict] | None = None,
    policies: dict[str, Any] | None = None,
    labels: str | list | None = None,
) -> list[Coroutine]:
    """
    Create a list of wrapped coroutines for many parameter sets.

    Args:
        func: The async function to wrap.
        kwds: Iterable of keyword-argument dicts.
        policies: Same keyword arguments as `with_policies`.

    Returns:
        List of coroutines, each wrapped with the given policies, ready to be awaited or gathered
    """
    kwds = kwds or [{}]
    policies = policies or {}

    if not isinstance(labels, list):
        labels = [labels] * len(kwds)

    if (sem := policies.get("semaphore")) and isinstance(sem, int | float):
        policies["semaphore"] = Semaphore(int(sem))
    elif "n_concurrent" in policies:
        policies["semaphore"] = Semaphore(int(policies.pop("n_concurrent")))

    factories = []
    for params, label in zip(kwds, labels, strict=True):  # type: ignore
        factory = lambda kwds=params: func(**kwds)  # noqa: E731
        wrapped = with_policies(factory, label=label, **policies)
        factories.append(wrapped)

    return [fac() for fac in factories]
