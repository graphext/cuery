from pathlib import Path
from typing import get_args, get_origin

import pydantic
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from .utils import get_config, pretty_field_info

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


class ResponseModel(BaseModel):
    """Base class for all response models."""

    @classmethod
    def fallback(cls) -> "ResponseModel":
        return cls.model_construct(**dict.fromkeys(cls.model_fields, None))

    @classmethod
    def is_multi_output(cls) -> tuple[bool, str | None]:
        """Check if a pydantic model has a single field that is a list."""
        fields = cls.model_fields
        if len(fields) != 1:
            return False, None

        name = next(iter(fields.keys()))
        field = fields[name]
        if get_origin(field.annotation) is list:
            return True, name

        return False, None

    @staticmethod
    def from_dict(name: str, fields: dict) -> "ResponseModel":
        """Create an instance of the model from a dictionary."""
        fields = fields.copy()
        for field_name, field_params in fields.items():
            field_type = TYPES[field_params.pop("type")]
            fields[field_name] = (field_type, Field(..., **field_params))

        return pydantic.create_model(name, **fields)

    @classmethod
    def from_config(cls, source: str | Path | dict, *keys: list) -> "ResponseModel":
        """Create an instance of the model from a configuration dictionary."""
        config = get_config(source, *keys)
        return ResponseModel.from_dict(keys[-1], config)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        cls = self.__class__
        title = Text(f"RESPONSE: {cls.__name__}", style="bold")

        field_panels = []
        nested_models = []

        for name, field in cls.model_fields.items():
            field_panels.append(pretty_field_info(name, field))
            typ = field.annotation
            if issubclass(typ, ResponseModel):
                nested_models.append(typ.fallback())
            elif typ_args := get_args(typ):
                for typ_arg in typ_args:
                    if issubclass(typ_arg, ResponseModel):
                        nested_models.append(typ_arg.fallback())

        group = Group(*field_panels)

        if nested_models:
            models = Group(*nested_models)
            group = Group(group, Padding(models, 1))

        yield Panel(group, title=title, padding=(1, 1), expand=False)


ResponseClass = type[ResponseModel]
