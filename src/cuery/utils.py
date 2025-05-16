import json
import os
from importlib.resources import files
from pathlib import Path

import yaml
from pandas import DataFrame

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
    relpath = Path(relpath)
    if not relpath.suffix:
        relpath = relpath.with_suffix(".yaml")

    path = resource_path(relpath)
    with open(path) as f:
        return yaml.safe_load(f)


def iterrecords(df: DataFrame, index: bool = False):
    for row in df.itertuples(index=index):
        yield row._asdict()
