"""Pricing information extracted with chatgpt passing it a screenshot of the table here: https://platform.openai.com/docs/pricing."""

from typing import Literal

COST = [
    {
        "model": "gpt-4.1",
        "version": "gpt-4.1-2025-04-14",
        "input": "2.00",
        "cached_input": "0.50",
        "output": "8.00",
    },
    {
        "model": "gpt-4.1-mini",
        "version": "gpt-4.1-mini-2025-04-14",
        "input": "0.40",
        "cached_input": "0.10",
        "output": "1.60",
    },
    {
        "model": "gpt-4.1-nano",
        "version": "gpt-4.1-nano-2025-04-14",
        "input": "0.10",
        "cached_input": "0.025",
        "output": "0.40",
    },
    {
        "model": "gpt-4.5-preview",
        "version": "gpt-4.5-preview-2025-02-27",
        "input": "75.00",
        "cached_input": "37.50",
        "output": "150.00",
    },
    {
        "model": "gpt-4o",
        "version": "gpt-4o-2024-08-06",
        "input": "2.50",
        "cached_input": "1.25",
        "output": "10.00",
    },
    {
        "model": "gpt-4o-audio-preview",
        "version": "gpt-4o-audio-preview-2024-12-17",
        "input": "2.50",
        "output": "10.00",
    },
    {
        "model": "gpt-4o-realtime-preview",
        "version": "gpt-4o-realtime-preview-2024-12-17",
        "input": "5.00",
        "cached_input": "2.50",
        "output": "20.00",
    },
    {
        "model": "gpt-4o-mini",
        "version": "gpt-4o-mini-2024-07-18",
        "input": "0.15",
        "cached_input": "0.075",
        "output": "0.60",
    },
    {
        "model": "gpt-4o-mini-audio-preview",
        "version": "gpt-4o-mini-audio-preview-2024-12-17",
        "input": "0.15",
        "output": "0.60",
    },
    {
        "model": "gpt-4o-mini-realtime-preview",
        "version": "gpt-4o-mini-realtime-preview-2024-12-17",
        "input": "0.60",
        "cached_input": "0.30",
        "output": "2.40",
    },
    {
        "model": "o1",
        "version": "o1-2024-12-17",
        "input": "15.00",
        "cached_input": "7.50",
        "output": "60.00",
    },
    {
        "model": "o1-pro",
        "version": "o1-pro-2025-03-19",
        "input": "150.00",
        "output": "600.00",
    },
    {
        "model": "o3",
        "version": "o3-2025-04-16",
        "input": "10.00",
        "cached_input": "2.50",
        "output": "40.00",
    },
    {
        "model": "o4-mini",
        "version": "o4-mini-2025-04-16",
        "input": "1.10",
        "cached_input": "0.275",
        "output": "4.40",
    },
    {
        "model": "o3-mini",
        "version": "o3-mini-2025-01-31",
        "input": "1.10",
        "cached_input": "0.55",
        "output": "4.40",
    },
    {
        "model": "o1-mini",
        "version": "o1-mini-2024-09-12",
        "input": "1.10",
        "cached_input": "0.55",
        "output": "4.40",
    },
    {
        "model": "codex-mini-latest",
        "version": "codex-mini-latest",
        "input": "1.50",
        "cached_input": "0.375",
        "output": "6.00",
    },
    {
        "model": "gpt-4o-mini-search-preview",
        "version": "gpt-4o-mini-search-preview-2025-03-11",
        "input": "0.15",
        "output": "0.60",
    },
    {
        "model": "gpt-4o-search-preview",
        "version": "gpt-4o-search-preview-2025-03-11",
        "input": "2.50",
        "output": "10.00",
    },
    {
        "model": "computer-use-preview",
        "version": "computer-use-preview-2025-03-11",
        "input": "3.00",
        "output": "12.00",
    },
    {
        "model": "gpt-image-1",
        "version": "gpt-image-1",
        "input": "5.00",
        "cached_input": "1.25",
    },
    {
        "model": "o3",
        "version": "o3-2025-04-16",
        "input": "5.00",
        "cached_input": "1.25",
        "output": "20.00",
    },
    {
        "model": "o4-mini",
        "version": "o4-mini-2025-04-16",
        "input": "0.55",
        "cached_input": "0.138",
        "output": "2.20",
    },
]


def cost_per_token(model_name: str, kind: Literal["input", "output", "cached_input"]) -> float:
    """Get the cost per 1 million tokens for a given model."""
    if kind not in ("input", "output", "cached_input"):
        raise ValueError("kind must be one of 'input', 'output', or 'cached_input'")

    if "/" in model_name:
        provider, model_name = model_name.split("/", 1)
        if provider.lower() != "openai":
            raise ValueError(f"Provider '{provider}' not yet supported for cost estimation!")
    else:
        provider = ""

    model = next((m for m in COST if m["model"] == model_name), None)

    if not model:
        models = [m["model"] for m in COST]
        raise ValueError(
            f"Model {model_name} not found in cost data. Known models: {', '.join(models)}"
        )

    c = model.get(kind)
    if c is None:
        raise ValueError(f"Cost for {model} with kind '{kind}' not found! Have info: {model}")

    return float(c) / 1_000_000
