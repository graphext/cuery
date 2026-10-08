from typing import Any

import instructor.templating
from rich import print as pprint

from .clients import DEFAULT_MODEL, create_client, prepare_parameters
from .context import AnyContext
from .prompt import Message, Prompt
from .response import Field, Response, ResponseClass, ResponseSet
from .task import Chain, Task
from .tool import Tool
from .utils import apply_template, load_env, set_env

# Patch the instructor templating to use our custom apply_template
instructor.templating.apply_template = apply_template


async def ask(prompt: str, model: str | None = None, response_model: Any = str, **kwds) -> Any:
    """Simple text chat without structured output."""
    client = create_client(model or DEFAULT_MODEL)
    return await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
        **prepare_parameters(client, kwds),
    )


__all__ = [
    "AnyContext",
    "load_env",
    "pprint",
    "set_env",
    "Chain",
    "Field",
    "Message",
    "Prompt",
    "Response",
    "ResponseClass",
    "ResponseSet",
    "Task",
    "Tool",
]
