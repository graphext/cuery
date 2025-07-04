import asyncio
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from cuery.builder.ui import launch
from cuery.task import Task

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


if __name__ == "__main__":
    app()
