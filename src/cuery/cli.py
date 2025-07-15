import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from .builder.ui import launch
from .seo import SeoConfig
from .task import Task
from .utils import load_api_keys

app = typer.Typer()


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


DEFAULT_CONFIG_DIR = "~/Development/config"


@app.command("set-vars")
def set_env_vars(cfg_dir: Path = Path(DEFAULT_CONFIG_DIR)):
    config_dir = cfg_dir.expanduser().resolve()
    vars = {}

    # Set Apify token
    apify_token_path = config_dir / "apify_api_token.txt"
    with open(apify_token_path) as f:
        vars["APIFY_TOKEN"] = f.read().strip()

    # Set Google Ads credentials
    google_ads_yaml_path = config_dir / "google-ads.yaml"
    with open(google_ads_yaml_path) as f:
        google_ads_config = yaml.load(f, Loader=yaml.SafeLoader)
        for key, value in google_ads_config.items():
            vars[f"GOOGLE_ADS_{key.upper()}"] = str(value)

    if key_path := vars.get("GOOGLE_ADS_JSON_KEY_FILE_PATH"):  # noqa: SIM102
        with open(key_path) as f:
            vars["GOOGLE_ADS_JSON_KEY"] = json.dumps(json.load(f))
            vars.pop("GOOGLE_ADS_JSON_KEY_FILE_PATH")

    vars |= load_api_keys(config_dir / "ai-api-keys.json")

    for key, value in vars.items():
        os.environ[key] = value
        # Update apify local secrets via the command line
        os.system(f"apify secrets rm {key} >/dev/null 2>&1")  # noqa: S605
        os.system(f"apify secrets add {key} '{value}'")  # noqa: S605


if __name__ == "__main__":
    app()
