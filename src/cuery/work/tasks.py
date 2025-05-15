from functools import partial

from pydantic import BaseModel, Field

from .. import prompt
from ..task import Task


class Job(BaseModel):
    name: str = Field(
        description="Name of the job role (less than 50 characters)",
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
        description="""
        A short explanation of no more than 10 words, of why the job is likely to be automatable with the given potential score.
        """,
        min_length=30,
        max_length=300,
    )


class Jobs(BaseModel):
    jobs: list[Job] = Field(
        description="A list of jobs with their AI automation potential and reasons for that potential",
        min_items=3,
    )


class JobTask(BaseModel):
    name: str = Field(description="The user's full name")
    description: str
    intelligence: int
    sexyness: int
    scalability: int
    data_needs: str
    solutions: list[str]


class JobTasks(BaseModel):
    tasks: list[JobTask]


PROMPTS = prompt.load("work/prompts.yaml")

DirceJobs = Task(prompt=PROMPTS["dirce_jobs"], response=Jobs)
