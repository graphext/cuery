from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from cuery.task import Chain, Task

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
def run_task(task: str, csv: Path, output: Path):
    """Execute a Task instance by id with a CSV file as input."""
    task = Task.registry.get(task)  # type: ignore
    if not task:
        typer.echo(f"No Task found with name {task}")
        raise typer.Exit(1)

    df = pd.read_csv(csv)
    result = task(df).to_pandas()
    result.to_csv(output, index=False)


# @app.command()
# def chain(task_ids: str, csv: Path, output: Path):
#     """Execute a chain of Task instances by id (comma-separated) with a CSV file as input."""
#     ids = [int(i) for i in task_ids.split(",")]
#     tasks = []
#     for i in ids:
#         task = Task.registry.get(i)
#         if not task:
#             typer.echo(f"No Task found with id {i}")
#             raise typer.Exit(1)
#         tasks.append(task)
#     chain = Chain(*tasks)
#     df = pd.read_csv(csv)
#     result = typer.run(lambda: chain(df))
#     if hasattr(result, "to_csv"):
#         if output:
#             result.to_csv(output, index=False)
#             typer.echo(f"Output written to {output}")
#         else:
#             typer.echo(result)
#     else:
#         typer.echo(result)


# if __name__ == "__main__":
#     app()
