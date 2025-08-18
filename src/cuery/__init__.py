import instructor.templating
from rich import print as pprint

from .context import AnyContext
from .prompt import Message, Prompt
from .response import Field, Response, ResponseClass, ResponseSet
from .task import Chain, Task
from .tool import Tool
from .utils import apply_template, set_api_keys

# Patch the instructor templating to use our custom apply_template
instructor.templating.apply_template = apply_template


async def ask(prompt: str, model: str | None = None, **kwds) -> str:
    """Simple text chat without structured output."""
    if model is None:
        model = "openai/gpt-3.5-turbo"

    client = instructor.from_provider(model, async_client=True)
    return await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_model=str,
        **kwds,
    )


__all__ = [
    "AnyContext",
    "pprint",
    "set_api_keys",
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
