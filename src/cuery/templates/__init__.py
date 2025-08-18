"""Jinja templates for common tasks in cuery."""

from importlib.resources import as_file
from pathlib import Path
from string import Template

from ..utils import resource_path


def load_template(name: str | Path) -> str:
    """Load a template from a local, relative resource path."""
    relpath = Path("templates") / name
    if not relpath.suffix:
        relpath = relpath.with_suffix(".jinja")

    trv = resource_path(relpath)
    with as_file(trv) as f, open(f) as fp:
        return fp.read()


def prompt_with_template(
    prompt: str,
    template: str,
    separator: str = "\n",
    params: dict | None = None,
) -> str:
    """Get a prompt with the chosen template appended."""
    template = load_template(template)
    result = prompt + separator + template

    if params:
        result = Template(result).substitute(**params)

    return result
