"""Offline requests through the real OpenAI SDK and Instructor parser."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import openai
import pytest

from cuery import Prompt, Response, Task, ask
from cuery.clients import responses_parameters


class Answer(Response):
    result: int


@pytest.fixture
def transport(monkeypatch):
    requests, clients = [], []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/responses":
            payload = {
                "id": "resp_offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [
                    {"type": "reasoning", "id": "rs_offline", "summary": []},
                    {
                        "type": "function_call",
                        "id": "fc_offline",
                        "call_id": "call_offline",
                        "name": body["tools"][0]["name"],
                        "arguments": '{"result":4}',
                        "status": "completed",
                    },
                ],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                    "input_tokens_details": {"cached_tokens": 10},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        else:
            payload = {
                "id": "chat_offline",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_offline",
                                    "type": "function",
                                    "function": {
                                        "name": body["tools"][0]["function"]["name"],
                                        "arguments": '{"result":4}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            }
        return httpx.Response(200, json=payload)

    class OfflineOpenAI(openai.AsyncOpenAI):
        def __init__(self, **kwargs):
            super().__init__(
                api_key="offline-test",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            )
            clients.append(self)

    monkeypatch.setattr(openai, "AsyncOpenAI", OfflineOpenAI)
    return requests, clients


@pytest.mark.parametrize("entry", ["task", "ask", "override"])
@pytest.mark.parametrize(
    "model,effort,expected",
    [
        (None, None, "none"),
        ("openai/gpt-6-luna", "high", "high"),
        ("openai/gpt-6-luna", "medium", "medium"),
        ("openai/gpt-6.1-sol", None, "medium"),
        ("openai/gpt-6.1-sol", "none", "low"),
        ("openai/gpt-6.1-sol", "minimal", "low"),
        ("openai/gpt-6.1-sol", "max", "max"),
    ],
)
def test_real_responses_transport(transport, entry, model, effort, expected):
    requests, clients = transport
    original = {"temperature": 0.2, "max_tokens": 60}
    if effort is not None:
        original["reasoning_effort"] = effort
    before = deepcopy(original)

    async def run():
        try:
            if entry == "ask":
                return await ask("What is 2 + 2?", model=model, response_model=Answer, **original)
            task = Task(
                prompt=Prompt(messages="What is 2 + 2?"),
                response=Answer,
                model="openai/gpt-4.1" if entry == "override" else model,
            )
            result = await task(
                model=(model or "openai/gpt-6-luna") if entry == "override" else None,
                fallback=False,
                **original,
            )
            return result.responses[0]
        finally:
            for client in clients:
                await client.close()

    result = asyncio.run(run())
    assert result.result == 4
    assert result.token_usage() == {"prompt": 20, "completion": 5}
    assert len(requests) == 1
    path, body = requests[0]
    assert path == "/v1/responses"
    assert body["model"] == (model or "openai/gpt-6-luna").removeprefix("openai/")
    assert body["reasoning"] == {"effort": expected}
    assert body["max_output_tokens"] == 60
    assert ("temperature" in body) == (expected == "none")
    assert "max_tokens" not in body and "reasoning_effort" not in body
    assert original == before


def test_legacy_model_uses_chat_without_alias(transport):
    requests, clients = transport

    async def run():
        try:
            return await ask(
                "What is 2 + 2?",
                model="openai/gpt-4.1-mini",
                response_model=Answer,
                temperature=0.2,
                max_tokens=60,
            )
        finally:
            for client in clients:
                await client.close()

    result = asyncio.run(run())
    assert result.result == 4
    assert result.token_usage() == {"prompt": 20, "completion": 5}
    assert requests[0][0] == "/v1/chat/completions"
    assert requests[0][1]["model"] == "gpt-4.1-mini"
    assert requests[0][1]["max_tokens"] == 60


def test_nested_reasoning_priority_and_parameter_copy():
    original = {
        "reasoning": {"effort": "high", "summary": "auto"},
        "reasoning_effort": "medium",
        "temperature": 0.2,
        "top_p": 0.3,
        "logprobs": True,
        "top_logprobs": 4,
        "include": ["message.output_text.logprobs", "reasoning.encrypted_content"],
        "max_tokens": 10,
        "max_completion_tokens": 20,
        "max_output_tokens": 30,
    }
    before = deepcopy(original)
    params = responses_parameters("gpt-6.1-sol", original)
    assert params == {
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 30,
    }
    assert original == before
    assert responses_parameters("gpt-6-luna", {"reasoning": {"effort": "high"}}) == {
        "reasoning": {"effort": "high"}
    }
    assert responses_parameters("gpt-6-luna", {"max_completion_tokens": 20}) == {
        "reasoning": {"effort": "none"},
        "max_output_tokens": 20,
    }


def test_missing_usage():
    result = Answer(result=4)
    assert result.token_usage() is None
    result._raw_response = SimpleNamespace(usage=None)
    assert result.token_usage() is None
