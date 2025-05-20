from pydantic import Field

from ..prompt import Prompt
from ..response import ResponseModel
from ..task import Task


class Job(ResponseModel):
    name: str = Field(
        description="Name of the job role (job title, less than 50 characters)",
        min_length=5,
        max_length=50,
    )
    description: str = Field(
        description="A short description of the job role (less than 200 characters)",
        min_length=20,
        max_length=200,
    )
    automation_potential: int = Field(
        description="A score from 1 to 10 indicating the job's potential for automation",
        ge=0,
        le=10,
    )
    reason: str = Field(
        description=(
            "A short explanation of no more than 10 words, of why the job "
            "is likely to be automatable with the given potential score."
        ),
        min_length=30,
        max_length=300,
    )


class Jobs(ResponseModel):
    jobs: list[Job] = Field(
        description=(
            "A list of jobs with their AI automation potential and reasons for that potential"
        ),
        min_items=3,
    )


class JobTask(ResponseModel):
    name: str = Field(
        description="Name of the task automatable with AI (less than 50 characters)",
        min_length=5,
        max_length=50,
    )
    description: str = Field(
        description="A short description of the task (less than 200 characters)",
        min_length=20,
        max_length=200,
    )
    automation_potential: int = Field(
        description="A score from 1 to 10 indicating the task's potential for automation",
        ge=0,
        le=10,
    )
    intelligence: int = Field(
        description=(
            "A score from 1 to 10 indicating the intelligence level needed to perform the task"
        ),
        ge=0,
        le=10,
    )
    sexyness: int = Field(
        description=(
            "A score from 1 to 10 indicating the sexiness level of the task"
            " (how appealing it is to do the task)"
        ),
        ge=0,
        le=10,
    )
    scalability: int = Field(
        description=(
            "A score from 1 to 10 indicating the scalability level of the task"
            " (how easy it is to scale the task)"
        ),
        ge=0,
        le=10,
    )
    data_needs: str = Field(
        description=(
            "A short explanation of no more than 10 words, of the data needs for the task"
        ),
        min_length=30,
        max_length=300,
    )
    products: list[str] = Field(
        description=(
            "A list of existing AI startups/companies or AI software solutions and providers "
            "(products or services) to automate the task, less than 10 words each, less than 50 "
            "characters each), ideally just the name and url."
        ),
        min_items=1,
        max_items=5,
    )


class JobTasks(ResponseModel):
    tasks: list[JobTask] = Field(
        description="A list of tasks automatable with AI software.",
        min_items=3,
        max_items=10,
    )


PROMPTS = "work/prompts.yaml"

DirceJobs = Task(prompt=Prompt.from_config(PROMPTS, "dirce_jobs"), response=Jobs)
DirceTasks = Task(prompt=Prompt.from_config(PROMPTS, "dirce_tasks"), response=JobTasks)
