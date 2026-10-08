"""Client and request settings for the GPT-6 text models.

These models need Responses for structured tool output with reasoning. Legacy model
IDs are intentionally not rewritten here: saved configurations must be migrated.
"""

import re

import instructor
from instructor import Mode

DEFAULT_MODEL = "openai/gpt-6-luna"


def gpt6_model(model: str) -> str | None:
    """Identify supported OpenAI model aliases and dated snapshots."""
    name = model.removeprefix("openai/")
    match = re.fullmatch(r"(gpt-6-luna|gpt-6\.1-sol)(?:-\d{4}-\d{2}-\d{2})?", name)
    return match[1] if match else None


def create_client(model: str = DEFAULT_MODEL):
    """Create an async Instructor client with the appropriate API transport."""
    options = {"mode": Mode.RESPONSES_TOOLS} if gpt6_model(model) else {}
    return instructor.from_provider(model, async_client=True, **options)


def responses_parameters(model: str, parameters: dict) -> dict:
    """Copy and normalize GPT-6 parameters without modifying caller configuration."""
    params = parameters.copy()
    family = gpt6_model(model)
    if not family:
        return params

    reasoning = dict(params.get("reasoning") or {})
    effort = params.pop("reasoning_effort", reasoning.get("effort"))
    if effort is None:
        effort = "none" if family == "gpt-6-luna" else "medium"
    if effort == "minimal" or (family == "gpt-6.1-sol" and effort == "none"):
        effort = "low"
    reasoning["effort"] = effort
    if effort == "none":
        reasoning.pop("summary", None)
    else:
        for name in ("temperature", "top_p", "top_logprobs", "logprobs"):
            params.pop(name, None)
        if "include" in params:
            params["include"] = [
                item for item in params["include"] if item != "message.output_text.logprobs"
            ]
    params["reasoning"] = reasoning

    # Responses has no Chat Completions logprobs flag, even without reasoning.
    params.pop("logprobs", None)
    max_tokens = params.pop("max_tokens", None)
    max_completion_tokens = params.pop("max_completion_tokens", max_tokens)
    if "max_output_tokens" not in params and max_completion_tokens is not None:
        params["max_output_tokens"] = max_completion_tokens
    return params


def prepare_parameters(client, parameters: dict) -> dict:
    """Normalize parameters for the model selected by an Instructor client."""
    model = parameters.get("model", client.kwargs.get("model", ""))
    return responses_parameters(model, parameters)
