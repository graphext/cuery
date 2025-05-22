from ..task import Task
from .models import Jobs, JobTasks

DirceJobs = Task(prompt="work/prompts.yaml:dirce_jobs", response=Jobs)
DirceTasks = Task(prompt="work/prompts.yaml:dirce_tasks", response=JobTasks)

__all__ = [
    "DirceJobs",
    "DirceTasks",
]
