import json
import os
from importlib.resources import files
from pathlib import Path
from typing import get_origin

import yaml
from pandas import DataFrame
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


def iterrecords(df: DataFrame, index: bool = False):
    """Iterate over rows of a DataFrame as dictionaries."""
    for row in df.itertuples(index=index):
        yield row._asdict()


def is_multi_output(model: BaseModel | BaseModelClass) -> tuple[bool, str | None]:
    """Check if a pydantic model has a single field that is a list."""
    if isinstance(model, BaseModel):
        model = model.__class__

    fields = model.model_fields
    if len(fields) != 1:
        return False

    name = next(iter(fields.keys()))
    field = fields[name]
    if get_origin(field.annotation) is list:
        return True, name

    return False, None
