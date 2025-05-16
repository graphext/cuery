from collections.abc import Iterable
from warnings import warn

from pandas import DataFrame

from .utils import iterrecords


def contexts_from_dataframe(df: DataFrame, required: list[str] | None) -> Iterable[dict] | None:
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


def contexts_from_dict(context: dict, required: list[str] | None) -> Iterable[dict] | None:
    """Convert a dict of iterables to an iterable of dicts with keys needed by prompt only."""
    if not required:
        warn(
            "Prompt doesn't require context, but it was provided. Ignoring context.",
            stacklevel=2,
        )
        return None

    missing = [k for k in required if k not in context]
    if missing:
        raise ValueError(f"Missing required keys in context dictionary: {', '.join(missing)}")

    context = {k: v for k, v in context.items() if k in required}
    keys = context.keys()
    values = context.values()
    lengths = [len(v) for v in values]
    if len(set(lengths)) != 1:
        raise ValueError("All lists must have the same length.")

    context = ({k: v[i] for k, v in zip(keys, values, strict=True)} for i in range(lengths[0]))
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


def check_context_iterable(
    context: dict | list[dict] | DataFrame | None,
    required: list[str] | None,
) -> tuple[Iterable[dict], int | None]:
    """Ensure context is an iterable of dicts."""
    if required and context is None:
        raise ValueError("Context is required for prompt but wasn't provided!")

    if isinstance(context, DataFrame):
        return contexts_from_dataframe(context, required)

    if isinstance(context, dict):
        return contexts_from_dict(context, required)

    if isinstance(context, list) and isinstance(context[0], dict):
        return contexts_from_iterable, len(context)

    raise ValueError(
        "Context must be a DataFrame, a dictionary of iterables, or a list of dictionaries. "
        f"Got:\n {context}"
    )
