"""Scaffold a new Apify actor from a FlexTool subclass.

This script inspects a given subclass of ``cuery.tools.flex.FlexTool`` and creates a new
actor directory in ``actors/`` with the common structure and files used in this repo:

actors/
  <actor_dir>/
    .actor/
      actor.json
      input_schema.json
      dataset_schema.json
      example_input.json
      README.md
      Dockerfile
    src/
      __main__.py
    storage/

It also creates a new module stub in ``src/cuery/actors/<module_name>.py`` that wires the
FlexTool into the standard runner using ``run_flex_tool``.

Usage (from project root):

  python -m cuery.actors.scaffold cuery.tools.flex.classify.Classifier --actor-name classifier

You can override names and descriptions with CLI options. By default, names are derived
from the tool class name.
"""

from __future__ import annotations
# isort: skip_file

import importlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import stubs
from ..utils import LOG


def snake_case(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.replace("-", "_")
    return name.lower()


def kebab_case(name: str) -> str:
    return snake_case(name).replace("_", "-")


def title_case(name: str) -> str:
    words = re.split(r"[_\-\s]+", snake_case(name))
    return " ".join(w.capitalize() for w in words if w)


def load_class(path: str):
    """Load a class given a full import path ``module.submodule:Class`` or ``module.Class``."""
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def repo_root_from_here() -> Path:
    """Return the repository root based on this file location.

    This file lives at: <root>/src/cuery/actors/scaffold/make.py → parents[4] is the repo root.
    """
    return Path(__file__).resolve().parents[4]


def read_project_version(root: Path) -> str:
    pyproject = root / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
        return data.get("project", {}).get("version")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def write_text(path: Path, data: str) -> None:
    path.write_text(data, encoding="utf-8")


def derive_actor_names(
    tool_class_name: str,
    actor_name: str | None,
    module_name: str | None,
    title: str | None,
) -> tuple[str, str, str]:
    """Return (actor_dir_name, module_name, title)."""
    default_mod = snake_case(tool_class_name)
    default_actor = default_mod
    default_title = title_case(tool_class_name)

    return (
        actor_name or default_actor,
        module_name or default_mod,
        title or default_title,
    )


def _apify_editor_for_field(name: str, schema: dict[str, Any]) -> str | None:
    t = schema.get("type")
    if name == "attrs" and t == "array":
        return "stringList"
    if name in {"instructions", "response_schema"}:
        return "textarea"
    if name in {"categories", "topics", "entities", "schema", "response_model"}:
        return "json"
    if t == "string" and "enum" in schema:
        return "select"
    if t in {"string", "number", "integer"}:
        return "textfield"
    return None


def build_input_schema(
    tool_cls,
    title: str,
    description: str | None,
) -> dict[str, Any]:
    """Convert a Pydantic model JSON schema into Apify actor input_schema.json.

    - Adds dataset_id
    - Drops 'records'
    - Attempts to set Apify UI editors heuristically
    """
    try:
        schema = tool_cls.model_json_schema()
    except Exception as exc:  # pragma: no cover - safety
        raise RuntimeError(f"Failed to get model_json_schema(): {exc}") from exc

    props: dict[str, Any] = {}
    required: set[str] = set(schema.get("required", []))

    # Always require dataset_id
    props["dataset"] = {
        "title": "Dataset",
        "type": "string",
        "description": (
            "Dataset containing the data records to process. "
            "Either an Apify dataset ID or the URL of a Parquet file."
        ),
        "editor": "textfield",
    }
    required.add("dataset")

    for name, field_schema in (schema.get("properties") or {}).items():
        if name == "records":
            continue  # handled via dataset_id

        # Shallow copy to avoid mutating original
        fs = dict(field_schema)

        # Prefer a clean, concise title
        fs.setdefault("title", title_case(name))

        # Map JSON Schema validation keywords as-is
        mapped: dict[str, Any] = {
            k: v
            for k, v in fs.items()
            if k
            in {
                "type",
                "title",
                "description",
                "default",
                "enum",
                "pattern",
                "minimum",
                "maximum",
                "minItems",
                "maxItems",
                "items",
                "anyOf",
                "allOf",
                "oneOf",
                "format",
            }
        }

        # Heuristics for Apify UI editor
        editor = _apify_editor_for_field(name, fs)
        if editor:
            mapped["editor"] = editor

        props[name] = mapped

    # Build input schema root
    return {
        "title": title,
        "description": description or f"Actor for {title}",
        "type": "object",
        "schemaVersion": 1,
        "properties": props,
        "required": sorted([r for r in required if r != "records"]),
    }


def default_dataset_schema(title: str, description: str | None) -> dict[str, Any]:
    return {
        "actorSpecification": 1,
        "title": title,
        "description": description or f"Processed records from {title}",
        "views": {
            "output": {
                "title": "Output",
                "description": "Input data with processed fields",
                "transformation": {"fields": ["*"]},
                "display": {"component": "table", "properties": {}},
            }
        },
    }


def make_actor_json(
    name: str,
    title: str,
    description: str | None,
    module_name: str,
) -> dict[str, Any]:
    return stubs.ACTOR_JSON | {
        "name": kebab_case(name),
        "title": title,
        "description": description or f"Actor for {title}",
        "main": f"python -m cuery.actors.{module_name}",
    }


def make_dockerfile(cuery_version: str | None, module_name: str) -> str:
    pin = f'"cuery[seo]=={cuery_version}"' if cuery_version else '"cuery[seo]"'
    return stubs.DOCKERFILE.format(pin=pin, module_name=module_name)


def make_readme(title: str, description: str | None) -> str:
    return stubs.README.format(title=title, description=description or "")


def make_actor_main(module_name: str) -> str:
    return stubs.ACTOR_MAIN.format(module_name=module_name)


def make_module_stub(tool_path: str) -> str:
    module_name, class_name = tool_path.rsplit(".", 1)
    return stubs.MODULE.format(module_name=module_name, class_name=class_name)


@dataclass
class ScaffoldSpec:
    tool_class_path: str
    actor_dir_name: str
    module_name: str
    title: str
    description: str | None
    overwrite: bool = False


def scaffold(spec: ScaffoldSpec) -> Path:  # noqa: PLR0912, PLR0915
    # Resolve file system locations
    root = repo_root_from_here()
    actors_root = root / "actors"
    actor_dir = actors_root / spec.actor_dir_name
    actor_actor_dir = actor_dir / ".actor"
    actor_src_dir = actor_dir / "src"
    actor_storage_dir = actor_dir / "storage"

    # Validate existence
    if actor_dir.exists() and not spec.overwrite:
        raise FileExistsError(
            f"Actor directory already exists: {actor_dir}. Use --force to overwrite."
        )

    # Introspect the tool
    tool_cls = load_class(spec.tool_class_path)

    # Verify it's a FlexTool subclass without importing here to avoid circulars
    try:
        from cuery.tools.flex.base import FlexTool as _FlexTool
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not import FlexTool base: {exc}") from exc
    if not issubclass(tool_cls, _FlexTool):
        raise TypeError(f"{tool_cls} is not a subclass of cuery.tools.flex.FlexTool")

    # Build artifacts
    input_schema = build_input_schema(tool_cls, spec.title, spec.description)
    dataset_schema = default_dataset_schema(spec.title, spec.description)
    actor_json = make_actor_json(
        spec.actor_dir_name, spec.title, spec.description, spec.module_name
    )

    # Example input
    example: dict[str, Any] = {"dataset": "YOUR_DATASET_ID"}
    try:
        schema = tool_cls.model_json_schema()
        for name, fs in (schema.get("properties") or {}).items():
            if name == "records":
                continue
            if "default" in fs:
                example[name] = fs["default"]
            elif fs.get("type") == "string" and "enum" in fs:
                example[name] = (fs["enum"] or [""])[0]
            elif fs.get("type") == "object":
                example[name] = {}
            elif fs.get("type") == "array":
                example[name] = []
            elif fs.get("type") in {"number", "integer"}:
                example[name] = fs.get("minimum", 0)
            else:
                # reasonable default for unknowns
                example[name] = None
    except Exception as exc:
        LOG.exception("Failed to build example input from schema: %s", exc)

    # Write files
    ensure_dir(actor_actor_dir)
    ensure_dir(actor_src_dir)
    ensure_dir(actor_storage_dir)

    write_json(actor_actor_dir / "input_schema.json", input_schema)
    write_json(actor_actor_dir / "dataset_schema.json", dataset_schema)
    write_json(actor_actor_dir / "actor.json", actor_json)
    write_json(actor_actor_dir / "example_input.json", example)

    write_text(actor_actor_dir / "README.md", make_readme(spec.title, spec.description))

    # Dockerfile pin from pyproject
    cuery_version = read_project_version(root)
    write_text(actor_actor_dir / "Dockerfile", make_dockerfile(cuery_version, spec.module_name))

    # __main__.py for Apify runner
    write_text(actor_src_dir / "__main__.py", make_actor_main(spec.module_name))

    # Module stub in src/cuery/actors/
    package_mod_path = root / "src" / "cuery" / "actors" / f"{spec.module_name}.py"
    if package_mod_path.exists() and not spec.overwrite:
        raise FileExistsError(
            "Module already exists: "
            f"{package_mod_path}. Use --force to overwrite or choose --module-name."
        )
    write_text(package_mod_path, make_module_stub(spec.tool_class_path))

    return actor_dir


def make_scaffold(
    tool: str,
    actor_name: str | None = None,
    module_name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    force: bool = False,
) -> None:
    """Create a new actor directory and module for a given FlexTool subclass."""
    tool_cls = load_class(tool)

    doc = (tool_cls.__doc__ or "").strip() or None
    actor_dir_name, mod_name, ttl = derive_actor_names(
        tool_cls.__name__,
        actor_name,
        module_name,
        title,
    )

    spec = ScaffoldSpec(
        tool_class_path=tool,
        actor_dir_name=actor_dir_name,
        module_name=mod_name,
        title=ttl,
        description=description or doc,
        overwrite=force,
    )

    out = scaffold(spec)
    LOG.info(f"Created actor at: {out}")
