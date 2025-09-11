import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .actors.scaffold.flex import make_scaffold
from .builder.ui import launch
from .seo import SeoConfig
from .task import Task
from .utils import set_env

app = typer.Typer()

PROJ_DIR = Path(__file__).resolve().parent.parent.parent


@app.command("tasks")
def list_tasks():
    """List all registered Task instances (pretty print)."""
    console = Console()
    table = Table(title="Registered Tasks")
    table.add_column("Task", style="bold cyan")
    table.add_column("Response Type", style="bold green")
    if not Task.registry:
        console.print("[red]No Task instances registered.[/red]")
        return
    for task in Task.registry.values():
        response_name = getattr(task.response, "__name__", str(task.response))
        table.add_row(task.name, response_name)
    console.print(table)


@app.command("run")
def run_task(task_name: str, csv: Path, output: Path):
    """Execute a Task instance by id with a CSV file as input."""
    task = Task.registry.get(task_name)  # type: ignore
    if not task:
        typer.echo(f"No Task found with name {task_name}")
        raise typer.Exit(1)

    df = pd.read_csv(csv)  # noqa: PD901
    result = asyncio.run(task(df))
    result = result.to_pandas()
    result.to_csv(output, index=False)


@app.command("builder")
def launch_builder():
    """Launch the interactive schema builder interface."""
    launch()


@app.command("seo-schema")
def generate_seo_schema(output: Path = Path("input_schema.json")):
    """Generate the SEO schema JSON file."""
    schema = SeoConfig.model_json_schema()
    with open(output, "w") as fp:
        json.dump(schema, fp, indent=2)
    typer.echo(f"SEO schema written to {output}")


@app.command("set-env")
def set_env_(apify_secrets: bool = True):
    """Set environment variables from configuration files."""
    set_env(apify_secrets=apify_secrets)


@app.command("actor")
def actor(name: str, apify_secrets: bool = True):
    os.chdir(PROJ_DIR)
    vars = set_env(apify_secrets=apify_secrets, return_vars=True)
    os.chdir(PROJ_DIR / "actors" / name)
    os.system(f"apify login --token {vars['APIFY_TOKEN'].reveal()}")  # noqa: S605
    cmd = f"uv run --env-file {PROJ_DIR / '.env'} apify run --purge --input-file=.actor/example_input.json"
    print(f"Running actor {name} with command: {cmd}")
    os.system(cmd)  # noqa: S605, S607


@app.command("scaffold")
def _make_scaffold(
    tool: str = typer.Argument(
        ..., help="Import path to FlexTool subclass, e.g. cuery.tools.flex.classify.Classifier"
    ),
    actor_name: str | None = typer.Option(
        None, help="Directory name for the new actor (default from tool class)"
    ),
    module_name: str | None = typer.Option(
        None, help="Module name to create in src/cuery/actors (default from tool class)"
    ),
    title: str | None = typer.Option(None, help="Actor title (default from tool class name)"),
    description: str | None = typer.Option(
        None, help="Actor description (default from tool class docstring)"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
) -> None:
    """Create a new actor directory and module for a given FlexTool subclass."""
    make_scaffold(
        tool=tool,
        actor_name=actor_name,
        module_name=module_name,
        title=title,
        description=description,
        force=force,
    )


if __name__ == "__main__":
    app()
