"""Utility functions."""

import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import date, datetime
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from inspect import cleandoc
from math import inf as INF
from pathlib import Path
from textwrap import dedent as pydedent
from typing import Any, get_args

import numpy as np
import pandas as pd
import yaml
from glom import glom
from jinja2 import Environment, meta
from jinja2.sandbox import SandboxedEnvironment
from pandas import isna
from pandas.api.types import is_scalar
from pandas.api.typing import NAType
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType
from tiktoken import Encoding, encoding_for_model, get_encoding
from tqdm.auto import tqdm as auto_tqdm

from .cost import cost_per_token
from .pretty import DEFAULT_BOX, Group, Padding, Panel, Pretty, Text

LOG = logging.getLogger("cuery")

if LOG.hasHandlers():
    LOG.handlers.clear()

LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(
    "%(asctime)s [%(name)s|%(levelname)s] %(message)s",
    datefmt=time_format,
)
ch.setFormatter(formatter)
LOG.addHandler(ch)


DEFAULT_PATH = Path().home() / "Development/config/ai-api-keys.json"

BaseModelClass = type(BaseModel)

NpNa = float
Missing = None | NAType | NpNa
"""Type hint for missing values."""


class progress(auto_tqdm):
    """A tqdm progress bar that calls an external callback on each update."""

    def __init__(self, *args, **kwds):
        self.callback = kwds.pop("callback", None)
        super().__init__(*args, **kwds)

    def update(self, n=1):
        displayed = super().update(n)
        if displayed and self.callback is not None:
            self.callback(self.format_dict)
        return displayed


def on_apify():
    """Check if the code is running on Apify's platform."""
    return os.environ.get("APIFY_IS_AT_HOME") == "1"


def json_encode(obj: Any) -> Any:  # noqa: PLR0911
    """Convert a value to a JSON string."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if is_scalar(obj) and pd.isna(obj):
        return None
    if isinstance(obj, type):
        if hasattr(obj, "__name__"):
            return obj.__name__
        return str(obj)

    raise TypeError(f"Type {type(obj)} not serializable to JSON.")


def apply_template(text: str, context: dict[str, Any]) -> str:
    """Apply Jinja2 template to the given text."""
    env = SandboxedEnvironment()
    env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
    env.policies["json.dumps_kwargs"]["sort_keys"] = False
    env.policies["json.dumps_kwargs"]["default"] = json_encode
    return pydedent(env.from_string(text).render(**context))


def load_api_keys(path: str | Path | None = DEFAULT_PATH) -> dict:
    """Load API keys from a JSON configuration file."""
    if path is None:
        path = DEFAULT_PATH

    with open(path) as file:
        api_keys = json.load(file)

    env = {}
    for k, v in api_keys.items():
        if "_api_key" not in k.lower():
            k = k + "_api_key"  # noqa: PLW2901

        env[k.upper()] = v

    return env


def set_api_keys(keys: dict | str | Path | None = None):
    """Set API keys as environment variables from a dictionary or file."""
    if not isinstance(keys, dict):
        keys = load_api_keys(keys)

    for key, value in keys.items():
        os.environ[key.upper()] = value


def resource_path(relpath: str | Path) -> Traversable:
    """Get the absolute path to a resource file within the cuery package."""
    relpath = Path(relpath)
    dp, fn = relpath.parent, relpath.name
    dp = Path("cuery") / dp
    dp = str(dp).replace("/", ".")
    return files(dp).joinpath(str(fn))


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file from a local, relative resource path."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")

    try:
        with open(path) as fp:
            return yaml.safe_load(fp)
    except FileNotFoundError:
        trv = resource_path(path)
        with as_file(trv) as f, open(f) as fp:
            return yaml.safe_load(fp)


def dedent(text):
    """Dedent a string, removing leading whitespace like yaml blocks."""

    def is_markdown_list_item(line):
        """Check if a line is a markdown list item."""
        line = line.strip()

        # Unordered lists
        if line.startswith(("- ", "* ", "+ ")):
            return True

        # Ordered lists: Detect a number or single letter followed by a dot and space
        return bool(re.match(r"^\d+\. |^[a-zA-Z]\. ", line))

    text = cleandoc(text)
    paragraphs = text.split("\n\n")
    paragraphs = [p.replace("\n", " ") if not is_markdown_list_item(p) else p for p in paragraphs]
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
        source = str(source).strip()
        if ":" in source:
            source, spec = str(source).split(":")
        else:
            spec = None

        source = load_yaml(source)

    return glom(source, spec) if spec else source


def pretty_field_info(name: str, field: FieldInfo):
    """Create a pretty-printed panel displaying field information for Pydantic models."""
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


def jinja_vars(template: str) -> list[str]:
    """Find undeclared Jinja variables in a template file."""
    parsed = Environment(autoescape=True).parse(template)
    return list(meta.find_undeclared_variables(parsed))


def render_template(template: str, **context: dict) -> str:
    """Render a Jinja template with the given context."""
    env = Environment(autoescape=True)
    env.globals.update(context)
    return env.from_string(template).render(context)


def model_encoding(model: str) -> Encoding:
    """Get the encoding name for a given model."""
    if "/" in model:
        provider, model = model.split("/", 1)
    else:
        provider = ""

    try:
        return encoding_for_model(model)
    except LookupError:
        if "gpt-4.1" in model.lower():
            return encoding_for_model("gpt-4o")
        if model.lower().startswith("o4"):
            return encoding_for_model("o3")
        if "google" in provider.lower() or "gemini" in model.lower():
            LOG.warning(
                f"Model {model} is not supported by tiktoken. Using cl100k_base encoding as a "
                "fallback for google/gemini models."
            )
            return get_encoding("cl100k_base")

        raise


def concat_up_to(
    texts: Iterable[str | Missing],
    model: str,
    max_dollars: float | None = None,
    max_tokens: float | None = None,
    max_texts: float | None = None,
    separator: str = "\n",
) -> str:
    """Concatenate texts until the total token count reaches max_tokens."""
    if max_dollars is None:
        max_dollars = INF

    if max_tokens is None:
        max_tokens = INF

    if max_texts is None:
        max_texts = INF

    enc = model_encoding(model)

    try:
        token_cost = cost_per_token(model, "input")
    except ValueError as e:
        LOG.warning(f"Can't get cost per token for model {model}: {e}.\n\nWon't limit by cost!")
        token_cost = 0.0

    if all(limit == INF for limit in (max_tokens, max_dollars, max_texts)):
        raise ValueError(
            "Must have one of max_dollars, max_tokens, or max_texts to limit concatenation!"
        ) from None

    total_texts = 0
    total_tokens = 0
    total_cost = 0
    result = []

    linebreak = re.compile(r"((\r\n)|\r|\n|\t|\n\v)+")

    for text in texts:
        if isna(text) or not text:
            continue

        text = linebreak.sub("", text).strip()  # noqa: PLW2901

        try:
            tokens = enc.encode(text)
        except Exception:
            LOG.error(f"Error encoding text '{text}' with model {model}.")
            raise

        n_tokens = len(tokens)
        n_dollars = token_cost * n_tokens

        if (total_tokens + n_tokens) > max_tokens:
            break

        if (total_cost + n_dollars) > max_dollars:
            break

        result.append(text)
        total_texts += 1
        total_tokens += n_tokens
        total_cost += n_dollars

        if total_texts >= max_texts:
            break

    LOG.info(
        f"Concatenated {total_texts:,} texts with {total_tokens:,} tokens "
        f"and total cost of ${total_cost:.5f}"
    )

    return separator.join(result)


def customize_fields(model: BaseModelClass, class_name: str, **fields) -> BaseModelClass:
    """Create a subclass of pydantic model changing field parameters."""
    if not fields:
        return model

    field_args = {}
    for field_name, new_args in fields.items():
        args = model.model_fields[field_name]._attributes_set | new_args
        field_args[field_name] = (args.pop("annotation"), Field(**args))

    return create_model(class_name, **field_args, __base__=model)


class Configurable(BaseModel):
    """Base class for configurations. Hashable so we can cache API calls using them."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    def __hash__(self) -> int:
        return self.model_dump_json().__hash__()

    def __repr__(self) -> str:
        name = self.__class__.__name__
        params = self.model_dump_json(indent=2)
        return f"{name}\n{'—' * len(name)}\n{params}\n"

    def __str__(self) -> str:
        return self.__repr__()
