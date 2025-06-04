import json
import logging
import os
import re
from collections.abc import Iterable
from importlib.resources import files
from inspect import cleandoc
from math import inf as INF
from pathlib import Path
from typing import get_args

import yaml
from glom import glom
from jinja2 import Environment, meta
from pandas import isna
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType
from tiktoken import encoding_for_model

from .cost import cost_per_token
from .pretty import DEFAULT_BOX, Group, Padding, Panel, Pretty, RichHandler, Text

BaseModelClass = type[BaseModel]

if not logging.getLogger("cuery").hasHandlers():
    LOG = logging.getLogger("cuery")
    LOG.addHandler(RichHandler(markup=False))
    LOG.setLevel(logging.INFO)
else:
    LOG = logging.getLogger("cuery")


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


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file from a local, relative resource path."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")

    try:
        with open(path) as fp:
            return yaml.safe_load(fp)
    except FileNotFoundError:
        path = resource_path(path)
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


def jinja_vars(template: str) -> list[str]:
    """Find undeclared Jinja variables in a template file."""
    parsed = Environment(autoescape=True).parse(template)
    return list(meta.find_undeclared_variables(parsed))


def concat_up_to(
    texts: Iterable[str],
    model: str,
    max_dollars: float,
    max_tokens: float | None = None,
    separator: str = "\n",
) -> str:
    """Concatenate texts until the total token count reaches max_tokens."""
    if max_dollars is None:
        raise ValueError("max_dollars must be specified!")

    if max_tokens is None:
        max_tokens = INF
        LOG.warning(
            f"The max_tokens param was not provided. Total length will be limited only by "
            f"a maximum total cost of ${max_dollars:.2f}."
        )

    try:
        enc = encoding_for_model(model)
    except LookupError:
        # Known models here: https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
        if "gpt-4.1" in model.lower():
            enc = encoding_for_model("gpt-4o")
        elif model.lower().startswith("o4"):
            enc = encoding_for_model("o3")

    try:
        token_cost = cost_per_token(model, "input")
    except ValueError as e:
        LOG.error(f"Error getting cost per token for model {model}: {e}")
        raise

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
        except Exception as e:
            LOG.error(f"Error encoding text '{text}' with model {model}.")
            raise

        n_tokens = len(tokens)
        n_dollars = token_cost * n_tokens

        if total_tokens + n_tokens > max_tokens:
            break

        if total_cost + n_dollars > max_dollars:
            break

        result.append(text)
        total_texts += 1
        total_tokens += n_tokens
        total_cost += n_dollars

    LOG.info(f"Concatenated {total_texts:,} texts with {total_tokens:,} tokens and total cost of ${total_cost:.5f}")

    return separator.join(result)
