import asyncio
import json

import httpx
import openai
import pytest

from cuery import Prompt, Response, Task, ask, search


class Answer(Response):
    result: int


@pytest.fixture
def sdk(monkeypatch):
    state = {"bodies": [], "mode": "ok", "clients": []}

    def respond(request):
        body = json.loads(request.content)
        state["bodies"].append(body)
        mode = state["mode"]
        if isinstance(mode, int):
            return httpx.Response(
                mode,
                json={"error": {"message": "Synthetic error", "type": "invalid_request_error"}},
            )
        tool = body.get("tools", [{}])[0]
        if tool.get("type") == "function":
            properties = tool["parameters"]["properties"]
            arguments = {
                key: 4 if schema.get("type") == "integer" else "Four"
                for key, schema in properties.items()
            }
            if mode == "retry" and len(state["bodies"]) == 1:
                arguments = {"result": "invalid integer"}
            output = [
                {"type": "reasoning", "id": "rs_test", "summary": []},
                {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": tool["name"],
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                },
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Four", "annotations": []}],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": body["model"],
                "status": "completed",
                "output": output,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "total_tokens": 16,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    original = openai.AsyncOpenAI

    class Offline(original):
        def __init__(self, **kwargs):
            super().__init__(
                api_key="offline",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            )
            state["clients"].append(self)

    monkeypatch.setattr(openai, "AsyncOpenAI", Offline)
    monkeypatch.setattr(search, "AsyncOpenAI", Offline)
    return state


def run(sdk, func):
    async def execute():
        try:
            return await func()
        finally:
            for client in sdk["clients"]:
                await client.close()

    return asyncio.run(execute())


@pytest.mark.parametrize(
    "model,effort", [("openai/gpt-6-luna", "none"), ("openai/gpt-6.1-sol", "low")]
)
def test_plain_string_ask(sdk, model, effort):
    result = run(
        sdk, lambda: ask("Return Four", model=model, reasoning_effort=effort, max_tokens=128)
    )
    assert result == "Four"
    assert sdk["bodies"][0]["reasoning"]["effort"] == effort


@pytest.mark.parametrize("model", ["openai/gpt-6-luna", "openai/gpt-6.1-sol"])
def test_structured_validation_retry(sdk, model):
    sdk["mode"] = "retry"
    task = Task(prompt=Prompt(messages="Return integer 4"), response=Answer, model=model)
    result = run(sdk, lambda: task(fallback=False, max_retries=2))
    assert result.responses[0].result == 4
    assert len(sdk["bodies"]) == 2
    assert sdk["bodies"][1]["reasoning"] == sdk["bodies"][0]["reasoning"]
    assert result.responses[0].token_usage()["prompt"] >= 12


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_not_hidden_by_fallback(sdk, status):
    sdk["mode"] = status
    task = Task(prompt=Prompt(messages="Return 4"), response=Answer)
    with pytest.raises(Exception):
        run(sdk, lambda: task(fallback=True, max_retries=1))


def test_transient_failure_can_fallback(sdk):
    sdk["mode"] = 500
    task = Task(prompt=Prompt(messages="Return 4"), response=Answer)
    result = run(sdk, lambda: task(fallback=True, max_retries=1))
    assert result.responses[0].result is None


def test_search_cache_and_model_key(sdk):
    search.query_openai.cache_clear()

    async def query():
        first = await search.query_openai("Return Four", use_search=False, model="gpt-6-luna")
        again = await search.query_openai("Return Four", use_search=False, model="gpt-6-luna")
        sol = await search.query_openai("Return Four", use_search=False, model="gpt-6.1-sol")
        return first, again, sol

    results = run(sdk, query)
    assert all(result.answer == "Four" for result in results)
    assert len(sdk["bodies"]) == 2
    assert sdk["bodies"][0]["reasoning"]["effort"] == "none"
    assert sdk["bodies"][1]["reasoning"]["effort"] == "medium"


@pytest.mark.parametrize("model", ["openai/gpt-6-luna", "openai/gpt-6.1-sol"])
def test_topic_context_preparation_with_unmapped_tokenizer(monkeypatch, model):
    from cuery import utils
    from cuery.tools.topics import TopicExtractor

    def unsupported_model(name):
        raise KeyError(name)

    monkeypatch.setattr(utils, "encoding_for_model", unsupported_model)
    extractor = TopicExtractor(
        model=model, texts=["Cats are pets.", "Football is a sport."], max_texts=2
    )
    assert "Cats are pets." in extractor.context["texts"]
    assert "Football is a sport." in extractor.context["texts"]
