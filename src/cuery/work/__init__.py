from ..task import Task
from .models import Jobs, JobTasks, Sectors

SpanishSectors = Task(prompt="work/prompts.yaml:spanish_sectors", response=Sectors)
CountrySectors = Task(prompt="work/prompts.yaml:country_sectors", response=Sectors)
DirceJobs = Task(prompt="work/prompts.yaml:dirce_jobs", response=Jobs)
DirceTasks = Task(prompt="work/prompts.yaml:dirce_tasks", response=JobTasks)

__all__ = [
    "DirceJobs",
    "DirceTasks",
]
