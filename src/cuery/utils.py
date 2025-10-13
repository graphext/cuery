"""Utility functions."""

import asyncio
import base64
import functools
import inspect
import json
import logging
import os
import re
from collections.abc import Coroutine, Iterable
from contextlib import contextmanager
from datetime import date, datetime
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from inspect import cleandoc
from math import inf as INF
from pathlib import Path
from textwrap import dedent as pydedent
from typing import Any, get_args
from urllib.parse import ParseResult, parse_qs, urlparse

import dotenv
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
from tldextract import TLDExtract
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

THIS_DIR = Path(__file__).parent
PKG_DIR = THIS_DIR.parent.parent

BaseModelClass = type(BaseModel)

NpNa = float
Missing = None | NAType | NpNa
"""Type hint for missing values."""


def with_log_level(logger: logging.Logger):
    """Decorator factory that adds a `log_level` parameter to the wrapped function.

    Temporarily sets the given logger's level during the call, then restores it.
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, log_level=None, **kwargs):
                old_level = logger.level
                if log_level is not None:
                    logger.setLevel(log_level)
                try:
                    return await func(*args, **kwargs)
                finally:
                    logger.setLevel(old_level)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, log_level=None, **kwargs):
            old_level = logger.level
            if log_level is not None:
                logger.setLevel(log_level)
            try:
                return func(*args, **kwargs)
            finally:
                logger.setLevel(old_level)

        return sync_wrapper

    return decorator


@contextmanager
def set_log_level(logger: logging.Logger, level: int | str):
    """Context manager to temporarily set the log level of a logger."""
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


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


def encode_json_b64(value):
    """Encode value in base64 JSON string."""
    b64 = base64.b64encode(json.dumps(value).encode("utf-8"))
    return b64.decode("ascii")


def decode_json_b64(value):
    """Decode a base64-encoded JSON string."""
    str_val = value.encode("ascii")
    return json.loads(base64.b64decode(str_val).decode("utf-8"))


class Secret(str):
    """A string that hides its content when printed."""

    def __repr__(self) -> str:
        return "****"

    def __str__(self) -> str:
        return "****"

    def reveal(self) -> str:
        """Get the actual string value."""
        return super().__str__()


def load_env(path: str | Path = PKG_DIR / ".env") -> dict[str, Secret]:
    """Load environment variables from a .env file into a dict masking their values."""
    secrets = dotenv.dotenv_values(dotenv_path=path, verbose=True)
    return {k: Secret(v) for k, v in secrets.items() if v}


def set_env(
    path: str | Path = PKG_DIR / ".env",
    apify_secrets: bool = False,
    return_vars=False,
) -> dict[str, Secret] | None:
    """Set environment variables from a .env file and optionally set local Apify environment."""
    dotenv.load_dotenv(dotenv_path=path, override=True, verbose=True)
    if apify_secrets:
        secrets = load_env(path)
        for key, secret in secrets.items():
            os.system(f"apify secrets rm {key} >/dev/null 2>&1")  # noqa: S605
            os.system(f"apify secrets add {key} '{secret.reveal()}'")  # noqa: S605

    if return_vars:
        return load_env(path)

    return None


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


async def gather_with_progress(
    coros: list[Coroutine],
    min_iters: int | None = None,
    progress_callback: Coroutine | None = None,
) -> list:
    """Gather a list of awaitables with a progress bar and optioncal callback."""
    tqdm_position = -1 if on_apify() else None
    total = len(coros)

    pbar = auto_tqdm(
        desc="Gathering searches",
        total=total,
        position=tqdm_position,
        miniters=min_iters,
    )

    async def with_progress(coro: Coroutine):
        result = await coro
        pbar.update()
        if progress_callback is not None:  # noqa: SIM102
            if (pbar.n % min_iters == 0) or (pbar.n == total):
                await progress_callback(pbar.format_dict)  # type: ignore
        return result

    coros = [with_progress(c) for c in coros]
    return await asyncio.gather(*coros)


def parse_url(url: str) -> ParseResult:
    """Parse a URL, adding scheme if missing."""
    if not url.startswith(("http://", "https://", "//")):
        url = "http://" + url

    return urlparse(url)


def is_google_translate_url(url: str | ParseResult) -> bool:
    """Check if a URL is a Google Translate URL."""
    if not isinstance(url, ParseResult):
        url = parse_url(url)

    return (
        url.netloc == "translate.google.com"
        and url.path.startswith("/translate")
        and "u=" in url.query
    )


TLD_EXTRACTOR = TLDExtract()


def extract_domain(
    url: str | None,
    with_subdomain: bool = False,
    resolve_google_translate: bool = True,
) -> str | None:
    """ "Extract the domain from a URL."""
    if not url:
        return None

    parsed = parse_url(url)
    if resolve_google_translate and is_google_translate_url(parsed):
        original_url = parse_qs(parsed.query)["u"][0]
        parsed = urlparse(original_url)

    tld = TLD_EXTRACTOR.extract_urllib(parsed)
    if with_subdomain:
        return tld.fqdn.replace("www.", "")

    return tld.top_domain_under_public_suffix


def clean_column_name(name: str) -> str:
    """Clean a string to be used as a pandas DataFrame column name."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)
