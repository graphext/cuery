"""Prompt base class.

Encapsulates lists of messages with functionality for loading from configuration files,
rendering to rich text, handling Jinja templating for dynamic content, and validating
required variables
"""

from pathlib import Path

from pydantic import BaseModel, Field

from .pretty import (
    Console,
    ConsoleOptions,
    Group,
    Padding,
    Panel,
    Pretty,
    RenderResult,
    Syntax,
    Text,
)
from .utils import get_config, jinja_vars

ROLE_STYLES = {
    "system": "bold cyan",
    "user": "bold green",
    "assistant": "bold yellow",
    "function": "bold magenta",
}


class Message(BaseModel):
    """Message class for chat completions."""

    content: str
    role: str = "user"

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        style = ROLE_STYLES.get(self.role, "bold")
        text = Syntax(
            self.content,
            "django",
            code_width=None,
            word_wrap=True,
            theme="friendly",
            padding=0,
        )
        title = f"[{style}]{self.role.upper()}"
        yield Panel(text, title=title, expand=True)


class Prompt(BaseModel):
    """Prompt class for chat completions.

    This class represents a chat prompt consisting of multiple messages.
    Each message can have a role (e.g., user, assistant) and content.
    It can be constructed manually or from a configuration file or a string. In the
    latter case, automatically detects the required variables used by
    the Jinja template, if any.
    """

    messages: list[Message] = Field(min_length=1)
    required: list[str] = Field(default_factory=list)

    def __iter__(self):
        yield from (dict(message) for message in self.messages)

    @classmethod
    def from_config(cls, source: str | Path | dict) -> "Prompt":
        config = get_config(source)
        return cls(**config)

    @classmethod
    def from_string(cls, p: str) -> "Prompt":
        """Create a Prompt from a string."""
        messages = [Message(content=p)]
        required = jinja_vars(p)
        return cls(messages=messages, required=required)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        group = []
        if self.required:
            group.append(
                Padding(
                    Group(
                        Text("Required: ", end=""),
                        Pretty(self.required),
                    ),
                    1,
                )
            )

        for message in self.messages:
            group.append(message)

        yield Panel(Group(*group), title=Text("PROMPT", style="bold"), expand=False)
