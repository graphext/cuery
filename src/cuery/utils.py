import json
import os
from importlib.resources import files
from inspect import cleandoc
from pathlib import Path

import yaml
from pydantic import BaseModel

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


def get_config(source: str | Path | dict, *keys: list, on_error="raise"):
    if isinstance(source, str | Path):
        source = load_yaml(source)

    return get(source, *keys, on_error=on_error)
