"""Base classes for tools with well-defined input and output interfaces.

Tools are essentially wrappers of Tasks. But they use pydantic models to also define the
*input* interface. The inputs are then used to customize a task's prompt, response
model and input context. They're only really useful if a task's prompt or response are
configurable, e.g. a classification task with configurable number of classes.
"""

from abc import ABC, abstractmethod
from functools import cached_property

from pydantic import ConfigDict, Field

from .context import AnyContext
from .prompt import Prompt
from .response import Response, ResponseClass, ResponseSet
from .task import Task
from .utils import Configurable


class Tool(Configurable, ABC):
    """Base class for all tools.

    Subclasses need to implement prompt and response models; either statically as ClassVars,
    or dynamically as (executable) instance properties.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field("openai/gpt-3.5-turbo", pattern=r"^[\w\.-]+/[\w\.-]+$")  # type: ignore
    """The LLM provider and model to use."""

    @property
    @abstractmethod
    def response_model(self) -> ResponseClass:
        """Defines the response model for this tool (ClassVar or property)."""

    @property
    @abstractmethod
    def prompt(self) -> Prompt:
        """Defines the prompt for this tool (ClassVar or property)."""

    @cached_property
    def task(self) -> Task:
        """Create a Task instance for this tool."""
        return Task(prompt=self.prompt, response=self.response_model, model=self.model)

    @cached_property
    def context(self) -> AnyContext | None:
        return None

    async def __call__(self, **kwargs) -> Response | ResponseSet:
        return await self.task(context=self.context, **kwargs)
