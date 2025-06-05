"""Helpers to convert various input formats to iterables of dictionaries."""

from collections.abc import Iterable
from warnings import warn

from pandas import DataFrame

AnyContext = dict | list[dict] | DataFrame


def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def iterrecords(df: DataFrame, index: bool = False):
    """Iterate over rows of a DataFrame as dictionaries."""
    for row in df.itertuples(index=index):
        yield row._asdict()  # type: ignore


def contexts_from_dataframe(
    df: DataFrame,
    required: list[str] | None,
) -> tuple[Iterable[dict] | None, int]:
    """Convert a DataFrame to an interable of dictionaries with variables needed by prompt only."""
    if not required:
        warn(
            "Prompt doesn't require context, but it was provided. Ignoring context.",
            stacklevel=2,
        )
        return None, 0

    missing = [k for k in required if k not in df]
    if missing:
        raise ValueError(f"Missing required columns in context DataFrame: {', '.join(missing)}")

    return iterrecords(df[required]), len(df)


def contexts_from_dict(
    context: dict,
    required: list[str] | None,
) -> tuple[Iterable[dict] | None, int]:
    """Convert a dict of iterables to an iterable of dicts with keys needed by prompt only."""
    if not required:
        warn(
            "Prompt doesn't require context, but it was provided. Ignoring context.",
            stacklevel=2,
        )
        return None, 0

    missing = [k for k in required if k not in context]
    if missing:
        raise ValueError(f"Missing required keys in context dictionary: {', '.join(missing)}")

    context = {k: v for k, v in context.items() if k in required}
    keys = context.keys()
    values = context.values()
    lengths = [len(v) for v in values]
    if len(set(lengths)) != 1:
        raise ValueError("All lists must have the same length.")

    context = ({k: v[i] for k, v in zip(keys, values, strict=True)} for i in range(lengths[0]))  # type: ignore
    return context, lengths[0]


def contexts_from_iterable(
    context: Iterable[dict],
    required: list[str] | None,
) -> Iterable[dict] | None:
    """Convert an iterable of dicts to an iterable of dicts with keys needed by prompt only."""
    if not required:
        warn(
            "Prompt doesn't require context, but it was provided. Ignoring context.",
            stacklevel=2,
        )
        return None

    for i, item in enumerate(context):
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError(
                f"Missing required keys in context dictionary {i}: {', '.join(missing)}"
            )

        yield {k: v for k, v in item.items() if k in required}


def iter_context(
    context: AnyContext | None,
    required: list[str] | None,
) -> tuple[Iterable[dict] | None, int]:
    """Ensure context is an iterable of dicts."""
    if required and context is None:
        raise ValueError("Context is required for prompt but wasn't provided!")

    if isinstance(context, DataFrame):
        return contexts_from_dataframe(context, required)

    if isinstance(context, dict):
        return contexts_from_dict(context, required)

    if isinstance(context, list) and isinstance(context[0], dict):
        return contexts_from_iterable(context, required), len(context)

    raise ValueError(
        "Context must be a DataFrame, a dictionary of iterables, or a list of dictionaries. "
        f"Got:\n {context}"
    )


def context_is_iterable(context: AnyContext | None) -> bool:
    """Check if context is iterable."""
    if context is None:
        return False
    if isinstance(context, DataFrame):
        return True
    if isinstance(context, dict):
        return all(is_iterable(v) for v in context.values())

    return isinstance(context, list) and all(isinstance(d, dict) for d in context)
