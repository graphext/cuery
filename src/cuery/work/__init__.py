from ..task import Task
from .models import Jobs, JobTasks, Sectors

SpanishSectors = Task(
    name="SpanishSectors", prompt="work/prompts.yaml:spanish_sectors", response=Sectors
)
CountrySectors = Task(
    name="CountrySectors", prompt="work/prompts.yaml:country_sectors", response=Sectors
)
DirceJobs = Task(name="DirceJobs", prompt="work/prompts.yaml:dirce_jobs", response=Jobs)
DirceTasks = Task(name="DirceTasks", prompt="work/prompts.yaml:dirce_tasks", response=JobTasks)

__all__ = [
    "DirceJobs",
    "DirceTasks",
]
