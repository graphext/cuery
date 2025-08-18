"""Base classes for LLM responses and response sets.

Faciliates conversion of responses to simpler Python objects and DataFrames,
as well as caching raw API responses for token usage calculation etc.
"""

import contextlib
import inspect
import json
import uuid
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, get_args, get_origin

import pandas as pd
import pydantic
from datamodel_code_generator import DataModelType, InputFileType, PythonVersion, generate
from instructor.cli.usage import calculate_cost
from pandas import DataFrame, Series
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import to_jsonable_python

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

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

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
        return self.model_dump(mode="json")

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
            field_type = field_params.pop("type")
            if field_type in TYPES:
                field_type = TYPES[field_type]
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


def transpose(dicts: list[dict]) -> dict[str, list]:
    """Transpose a list of dictionaries into a dictionary of lists."""
    return {k: [dic[k] for dic in dicts] for k in dicts[0]}


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

    @staticmethod
    def to_dict(item: Any, fallback_name: str | None = None) -> Any:
        """Convert an item to a dictionary.

        If the item is not a dict-like object, return a fallback dict if a fallback name is
        provided, otherwise return the item as is.
        """
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")

        try:
            item = to_jsonable_python(item)
        except Exception:
            with contextlib.suppress(Exception):
                item = dict(item)

        if isinstance(item, dict) or fallback_name is None:
            return item

        return {fallback_name: item}

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
                    item_dict = self.to_dict(item, fallback_name=self.iterfield)
                    records.append(ctx | item_dict)
        else:
            for ctx, response in zip(contexts, responses, strict=True):  # type: ignore
                if self.iterfield is not None:
                    items = [self.to_dict(item) for item in getattr(response, self.iterfield)]
                    records.append(ctx | {self.iterfield: items})
                else:
                    records.append(ctx | self.to_dict(response))

        return records

    def to_pandas(
        self,
        explode: bool = True,
        normalize: bool = True,
        prefix: str | None = None,
    ) -> DataFrame:
        """Convert list of responses to DataFrame."""
        df = DataFrame.from_records(self.to_records(explode=explode))

        if not explode and normalize and self.iterfield is not None:
            # Convert single column with list of dicts to one list column per key
            try:
                df[self.iterfield] = df[self.iterfield].apply(transpose)
            except:  # noqa: E722, S110
                pass
            else:
                response_df = pd.json_normalize(df.pop(self.iterfield))
                prefix = prefix if prefix is not None else self.iterfield
                response_df.columns = [f"{prefix}_{col}" for col in response_df]
                df = pd.concat([df, response_df], axis=1)

        return df

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


def is_response_subclass(obj):
    return issubclass(obj, Response) and obj is not Response


def get_module_responses(module: ModuleType):
    members = inspect.getmembers(module, lambda obj: is_response_subclass(obj))
    return [member[1] for member in members]


def models_from_jsonschema(schema: str | dict, log: bool = False) -> list[ResponseClass]:
    """Create response models dynamically from configuration files.

    Also see:
    - https://koxudaxi.github.io/datamodel-code-generator/using_as_module/
    - https://github.com/koxudaxi/datamodel-code-generator/issues/331
    - https://github.com/koxudaxi/datamodel-code-generator/issues/278#issuecomment-764498857
    - https://github.com/VRSEN/agency-swarm/blob/main/agency_swarm/tools/ToolFactory.py
    """
    if isinstance(schema, dict):
        schema = json.dumps(schema)

    module = "_dynamic_model_" + uuid.uuid4().hex
    output = Path(f"{module}.py")
    generate(
        input_=schema,
        input_file_type=InputFileType.JsonSchema,
        output=output,
        output_model_type=DataModelType.PydanticV2BaseModel,
        base_class="cuery.response.Response",
        target_python_version=PythonVersion.PY_312,
    )

    if log:
        LOG.info(output.read_text())

    module = import_module(module)
    output.unlink(missing_ok=True)
    return get_module_responses(module)
