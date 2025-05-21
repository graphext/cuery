import json
import os
from functools import partial
from importlib.resources import files
from inspect import cleandoc
from pathlib import Path
from typing import get_args

import yaml
from glom import glom
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType
from rich import box, panel
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

NO_BOTTOM_BOX = box.Box(
    "╭─┬╮\n"  # top
    "│ ││\n"  # head
    "├─┼┤\n"  # headrow
    "│ ││\n"  # mid
    "├─┼┤\n"  # row
    "├─┼┤\n"  # foot row
    "│ ││\n"  # foot
    # "╰─┴╯\n"  # bottom
    "    \n"  # bottom
)

DEFAULT_BOX = box.ROUNDED

panel.Panel = partial(Panel, box=DEFAULT_BOX)


BaseModelClass = type[BaseModel]

DEFAULT_PATH = Path().home() / "Development/config/ai-api-keys.json"


def load_api_keys(path: str | Path | None = DEFAULT_PATH) -> dict:
    if path is None:
        path = DEFAULT_PATH

    with open(path) as file:
        return json.load(file)


def set_api_keys(keys: dict | str | Path | None = None):
    if not isinstance(keys, dict):
        keys = load_api_keys(keys)

    for key, value in keys.items():
        name = key.upper() + "_API_KEY"
        os.environ[name] = value


def resource_path(relpath: str | Path) -> Path:
    relpath = Path(relpath)
    dp, fn = relpath.parent, relpath.name
    dp = Path("cuery") / dp
    dp = str(dp).replace("/", ".")
    return files(dp).joinpath(str(fn))


def load_yaml(relpath: str | Path) -> dict:
    """Load a YAML file from a local, relative resource path."""
    relpath = Path(relpath)
    if not relpath.suffix:
        relpath = relpath.with_suffix(".yaml")

    path = resource_path(relpath)
    with open(path) as f:
        return yaml.safe_load(f)


def dedent(text):
    """Dedent a string, removing leading whitespace like yaml blocks."""
    text = cleandoc(text)
    paragraphs = text.split("\n\n")
    paragraphs = [p.replace("\n", " ") for p in paragraphs]
    return "\n\n".join(paragraphs).strip()


def get(dct, *keys, on_error="raise"):
    """Safely access a nested obj with variable length path."""
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, TypeError, IndexError):
            if isinstance(key, str):
                try:
                    dct = getattr(dct, key)
                except AttributeError:
                    if on_error == "raise":
                        raise
                    return on_error
            else:
                if on_error == "raise":
                    raise
                return on_error
    return dct


def get_config(source: str | Path | dict):
    """Load a (subset) of configuration from a local file.

    Supports glom-style dot and bracket notation to access nested keys/objects.
    """
    if isinstance(source, str | Path):
        if ":" in source:
            source, spec = str(source).split(":")
        else:
            spec = None

        source = load_yaml(source)

    return glom(source, spec) if spec else source


def pretty_field_info(name: str, field: FieldInfo):
    group = []
    if desc := field.description:
        group.append(Padding(Text(desc), (0, 0, 1, 0)))

    info = {
        "required": field.is_required(),
    }
    for k in ("metadata", "examples", "json_schema_extra"):
        if v := getattr(field, k):
            info[k] = v

    if not isinstance((default := field.get_default()), PydanticUndefinedType):
        info["default"] = default

    group.append(Pretty(info))

    typ = field.annotation if get_args(field.annotation) else field.annotation.__name__
    title = Text(f"{name}: {typ}", style="bold")
    return Panel(Padding(Group(*group), 1), title=title, title_align="left", box=DEFAULT_BOX)
