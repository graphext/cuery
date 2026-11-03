"""Client and request settings for supported text models.

These models need Responses for structured tool output with reasoning. Legacy model
IDs are intentionally not rewritten here: saved configurations must be migrated.
"""

import re

import instructor
from instructor import Mode

from .gemini import Gemini35Hooks

DEFAULT_MODEL = "openai/gpt-6-luna"


def gpt6_model(model: str) -> str | None:
    """Identify supported OpenAI model aliases and dated snapshots."""
    name = model.removeprefix("openai/")
    match = re.fullmatch(r"(gpt-6-luna|gpt-6\.1-sol)(?:-\d{4}-\d{2}-\d{2})?", name)
    return match[1] if match else None


def create_client(model: str = DEFAULT_MODEL):
    """Create an async Instructor client with the appropriate API transport."""
    # Instructor 1.11's basic Responses parser assumes output[0] is a tool call.
    # The mixed-output parser locates the named function after reasoning items.
    options = {"mode": Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS} if gpt6_model(model) else {}
    if model.removeprefix("google/") == "gemini-3.5-flash-lite":
        options["hooks"] = Gemini35Hooks()
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


def gemini_parameters(model: str, parameters: dict) -> dict:
    """Normalize Gemini 3.5 settings while preserving explicit legacy model calls."""
    if model.removeprefix("google/") != "gemini-3.5-flash-lite":
        return parameters.copy()

    from google.genai.types import ThinkingConfig

    params = parameters.copy()
    generation = dict(params.get("generation_config") or {})
    for settings in (params, generation):
        # Gemini 3.5 ignores sampling controls; do not forward stale settings.
        for key in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty"):
            settings.pop(key, None)
        for key in ("n", "candidate_count"):
            count = settings.pop(key, None)
            if count is not None and count != 1:
                raise ValueError("Gemini 3.5 supports only one candidate per request")
    if "generation_config" in params:
        params["generation_config"] = generation
    config = params.get("config")
    nested_thinking = (
        config.get("thinking_config")
        if isinstance(config, dict)
        else getattr(config, "thinking_config", None)
    )
    thinking = {}
    # Explicit top-level settings win over config, then generation_config.
    for setting in (
        generation.pop("thinking_config", None),
        nested_thinking,
        params.pop("thinking_config", None),
    ):
        if isinstance(setting, ThinkingConfig):
            setting = setting.model_dump(exclude_none=True)
        thinking.update(
            {key: value for key, value in (setting or {}).items() if value is not None}
        )
    thinking.pop("thinking_budget", None)
    params.pop("thinking_budget", None)
    level = params.pop("thinking_level", thinking.get("thinking_level", "minimal"))
    thinking["thinking_level"] = level
    params["thinking_config"] = ThinkingConfig(**thinking)
    return params


def prepare_parameters(client, parameters: dict) -> dict:
    """Normalize parameters for the model selected by an Instructor client."""
    model = parameters.get("model", client.kwargs.get("model", ""))
    return gemini_parameters(model, responses_parameters(model, parameters))
