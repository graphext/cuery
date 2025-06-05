"""Base classes for LLM responses and response sets.

Faciliates conversion of responses to simpler Python objects and DataFrames,
as well as caching raw API responses for token usage calculation etc.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args, get_origin

import pandas as pd
import pydantic
from instructor.cli.usage import calculate_cost
from pandas import DataFrame, Series
from pydantic import BaseModel, Field

from .context import AnyContext, iter_context
from .pretty import Console, ConsoleOptions, Group, Padding, Panel, RenderResult, Text
from .utils import LOG, get_config, pretty_field_info

TYPES = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
}


class Response(BaseModel):
    """Base class for all response models.

    Adds functionality to cache the raw response from the API call, calculate token usage,
    and to create a fallback instance, which by default is an empty model with all fields
    set to None.

    Also implements rich's console protocol for pretty printing of the model's fields,
    and allows inspection of the model's fields to determine if it has a single
    multivalued field (a list) or not (which can be used to automatically "explode"
    items into DataFrame rows e.g.).
    """

    _raw_response: Any | None = None

    def token_usage(self) -> dict | None:
        """Get the token usage from the raw response."""
        if self._raw_response is None:
            return None

        return {
            "prompt": self._raw_response.usage.prompt_tokens,
            "completion": self._raw_response.usage.completion_tokens,
        }

    def to_dict(self) -> dict:
        """Convert the model to a dictionary."""
        return json.loads(self.model_dump_json())

    @classmethod
    def fallback(cls) -> "Response":
        return cls.model_construct(**dict.fromkeys(cls.model_fields, None))

    @classmethod
    def iterfield(cls) -> str | None:
        """Check if a pydantic model has a single field that is a list."""
        fields = cls.model_fields
        if len(fields) != 1:
            return None

        name = next(iter(fields.keys()))
        field = fields[name]
        if get_origin(field.annotation) is list:
            return name

        return None

    @classmethod
    def is_multivalued(cls) -> bool:
        """Check if a pydantic model has a single field that is a list."""
        return cls.iterfield() is not None

    @staticmethod
    def from_dict(name: str, fields: dict) -> "ResponseClass":
        """Create an instance of the model from a dictionary."""
        fields = fields.copy()
        for field_name, field_params in fields.items():
            field_type = TYPES[field_params.pop("type")]
            fields[field_name] = (field_type, Field(..., **field_params))

        return pydantic.create_model(name, __base__=Response, **fields)

    @classmethod
    def from_config(cls, source: str | Path | dict, *keys: list) -> "ResponseClass":
        """Create an instance of the model from a configuration dictionary."""
        config = get_config(source, *keys)
        return Response.from_dict(keys[-1], config)  # type: ignore

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        cls = self.__class__
        title = Text(f"RESPONSE: {cls.__name__}", style="bold")

        field_panels = []
        nested_models = []

        for name, field in cls.model_fields.items():
            field_panels.append(pretty_field_info(name, field))
            typ = field.annotation
            if typ is not None and issubclass(typ, Response):
                nested_models.append(typ.fallback())
            elif typ_args := get_args(typ):
                for typ_arg in typ_args:
                    if issubclass(typ_arg, Response):
                        nested_models.append(typ_arg.fallback())

        group = Group(*field_panels)

        if nested_models:
            models = Group(*nested_models)
            group = Group(group, Padding(models, 1))

        yield Panel(group, title=title, padding=(1, 1), expand=False)


def token_usage(responses: Iterable[Response]) -> DataFrame:
    return DataFrame([r.token_usage() for r in responses])


def with_cost(usage: DataFrame, model: str) -> DataFrame:
    cost = Series(
        [
            calculate_cost(model, prompt, compl)  # type: ignore
            for prompt, compl in zip(usage.prompt, usage.completion, strict=True)
        ]
    )
    return pd.concat([usage, cost.rename("cost")], axis=1)


ResponseClass = type[Response]


class ResponseSet:
    """A collection of responses

    This class is used to manage multiple responses, allowing iteration over them,
    conversion to records or DataFrame, and calculating token usage across all responses.
    """

    def __init__(
        self,
        responses: Response | list[Response],
        context: AnyContext | None,
        required: list[str] | None,
    ):
        self.responses = [responses] if isinstance(responses, Response) else responses
        self.context = [context] if isinstance(context, dict) else context
        self.required = required
        self.iterfield = self.responses[0].iterfield()

    def __iter__(self):
        return iter(self.responses)

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, index: int) -> Response:
        return self.responses[index]

    def to_records(self, explode: bool = True) -> list[dict] | DataFrame:
        """Convert to list of dicts, optionally with original context merged in."""
        context, responses = self.context, self.responses

        if context is not None:
            contexts, _ = iter_context(context, self.required)
        else:
            contexts = ({} for _ in responses)

        records = []
        if explode and self.iterfield is not None:
            for ctx, response in zip(contexts, responses, strict=True):  # type: ignore
                for item in getattr(response, self.iterfield):
                    if isinstance(item, Response):
                        records.append(ctx | dict(item))
                    else:
                        records.append(ctx | {self.iterfield: item})
        else:
            for ctx, response in zip(contexts, responses, strict=True):  # type: ignore
                records.append(ctx | dict(response))

        return records

    def to_pandas(self, explode: bool = True) -> DataFrame:
        """Convert list of responses to DataFrame."""
        return DataFrame.from_records(self.to_records(explode=explode))

    def usage(self) -> DataFrame:
        """Get the token usage for all responses."""
        usage = token_usage(self.responses)
        try:
            usage = with_cost(usage, self.responses[0]._raw_response.model)  # type: ignore
        except Exception as exc:
            LOG.error(f"Failed to calculate cost: {exc}")

        return usage

    def __str__(self) -> str:
        return self.responses.__str__()

    def __repr__(self) -> str:
        return self.responses.__repr__()
