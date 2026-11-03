"""Offline Gemini request regression tests through Cuery and Instructor."""

import asyncio
from copy import deepcopy

import pytest
from google import genai
from google.genai import types
from google.genai.models import AsyncModels

from cuery import Response, ask, search
from cuery.clients import gemini_parameters


class Answer(Response):
    result: int


@pytest.fixture
def requests(monkeypatch):
    captured = []
    original_client = genai.Client

    def client(**kwargs):
        return original_client(api_key="offline-test")

    async def generate(self, **kwargs):
        captured.append(kwargs)
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(name="Answer", args={"result": 4})
                            )
                        ],
                    )
                )
            ]
        )

    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setattr(genai.client, "Client", client)
    monkeypatch.setattr(AsyncModels, "generate_content", generate)
    return captured


@pytest.mark.parametrize("level", ["minimal", "medium", "high"])
def test_ask_instructor_request(requests, level):
    result = asyncio.run(
        ask(
            "What is 2+2?",
            model="google/gemini-3.5-flash-lite",
            response_model=Answer,
            temperature=0.2,
            top_p=0.8,
            top_k=30,
            thinking_level=level,
            generation_config={"temperature": 0.1, "max_tokens": 50},
        )
    )
    assert result.result == 4
    request = requests[0]
    assert request["model"] == "gemini-3.5-flash-lite"
    config = request["config"]
    assert config.thinking_config.thinking_level.value == level.upper()
    assert config.thinking_config.thinking_budget is None
    assert config.temperature is None and config.top_p is None and config.top_k is None
    assert config.max_output_tokens == 50
    assert config.tools[0].function_declarations[0].name == "Answer"


@pytest.mark.parametrize(
    "model,level,budget",
    [
        ("gemini-3.5-flash-lite", "MINIMAL", None),
        ("gemini-2.5-flash", None, 256),
        ("gemini-2.5-flash-lite", None, 256),
        ("gemini-2.0-flash", None, None),
    ],
)
def test_grounded_search_request(requests, model, level, budget):
    asyncio.run(search.query_gemini("Test", model=model, thinking_budget=256, validate=False))
    config = requests[0]["config"]
    assert config.tools[0].google_search is not None
    if level or budget:
        assert config.thinking_config.thinking_budget == budget
        actual_level = config.thinking_config.thinking_level
        assert (actual_level.value if actual_level else None) == level
    else:
        assert config.thinking_config is None


def test_parameter_copy_and_legacy_compatibility():
    original = {
        "temperature": 0.2,
        "thinking_config": {"thinking_budget": -1},
        "generation_config": {"top_p": 0.5},
    }
    before = deepcopy(original)
    params = gemini_parameters("google/gemini-3.5-flash-lite", original)
    assert params["thinking_config"].thinking_level.value == "MINIMAL"
    assert original == before
    assert gemini_parameters("google/gemini-2.5-flash", original) == original
    with pytest.raises(ValueError, match="one candidate"):
        gemini_parameters("google/gemini-3.5-flash-lite", {"n": 2})


def test_validation_retry_preserves_native_id_and_signature(requests, monkeypatch):
    from instructor.core.hooks import Hooks

    native_part = types.Part(
        function_call=types.FunctionCall(name="Answer", args={"result": "invalid"}, id="native-1"),
        thought_signature=b"native-signature",
    )
    captured = []

    async def generate(self, **kwargs):
        captured.append(kwargs)
        if len(captured) == 1:
            part = native_part
        else:
            contents = kwargs["contents"]
            assert contents[-2].parts[0] is native_part
            assert contents[-2].parts[0].thought_signature == b"native-signature"
            assert contents[-1].role == "user"
            response = contents[-1].parts[0].function_response
            assert response.name == "Answer" and response.id == "native-1"
            part = types.Part(function_call=types.FunctionCall(name="Answer", args={"result": 4}))
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
        )

    monkeypatch.setattr(AsyncModels, "generate_content", generate)
    result = asyncio.run(
        ask(
            "What is 2+2?",
            model="google/gemini-3.5-flash-lite",
            response_model=Answer,
            max_retries=2,
            hooks=Hooks(),
        )
    )
    assert result.result == 4
    assert len(captured) == 2


def test_parallel_response_id_matching_and_ambiguity():
    from cuery.gemini import Gemini35Hooks

    def call(name, identifier):
        return types.Part(function_call=types.FunctionCall(name=name, id=identifier, args={}))

    def response(name, identifier=None):
        return types.Part(
            function_response=types.FunctionResponse(
                name=name,
                id=identifier,
                response={"error": "invalid"},
            )
        )

    hook = Gemini35Hooks()
    first, second = response("Answer", "b"), response("Answer")
    contents = [
        types.Content(role="model", parts=[call("Answer", "a"), call("Answer", "b")]),
        types.Content(role="tool", parts=[first, second]),
    ]
    hook.emit_completion_arguments(contents=contents)
    assert first.function_response.id == "b"
    assert second.function_response.id == "a"
    ambiguous = [contents[0], types.Content(role="tool", parts=[response("Answer")])]
    with pytest.raises(ValueError, match="unambiguously"):
        hook.emit_completion_arguments(contents=ambiguous)
    wrong = [contents[0], types.Content(role="tool", parts=[response("Other", "a")])]
    with pytest.raises(ValueError, match="unambiguously"):
        hook.emit_completion_arguments(contents=wrong)


def test_legacy_client_has_no_gemini35_guard(requests):
    from cuery.clients import create_client
    from cuery.gemini import Gemini35Hooks

    assert isinstance(create_client("google/gemini-3.5-flash-lite").hooks, Gemini35Hooks)
    assert not isinstance(create_client("google/gemini-2.5-flash").hooks, Gemini35Hooks)


@pytest.mark.parametrize(
    "settings,expected",
    [
        (
            {"config": {"thinking_config": {"thinking_level": "high", "thinking_budget": -1}}},
            "HIGH",
        ),
        (
            {
                "config": types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="medium", thinking_budget=-1
                    )
                )
            },
            "MEDIUM",
        ),
        (
            {
                "generation_config": {
                    "thinking_config": {"thinking_level": "high", "thinking_budget": -1}
                }
            },
            "HIGH",
        ),
        (
            {
                "thinking_level": "minimal",
                "thinking_config": {"thinking_level": "medium"},
                "config": {"thinking_config": {"thinking_level": "high"}},
            },
            "MINIMAL",
        ),
        (
            {
                "thinking_config": {"thinking_level": "medium"},
                "config": {"thinking_config": {"thinking_level": "high"}},
            },
            "MEDIUM",
        ),
    ],
)
def test_nested_thinking_serialized_native_request(monkeypatch, settings, expected):
    import json

    from google.genai._api_client import SdkHttpResponse

    native = genai.Client(api_key="offline-test")
    sent = []
    original = deepcopy(settings)

    async def transport(method, path, payload, http_options=None):
        request = native._api_client._build_request(method, path, payload, http_options)
        sent.append(json.loads(json.dumps(request.data)))
        return SdkHttpResponse(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call-1",
                                            "name": "Answer",
                                            "args": {"result": 4},
                                        }
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            )
        )

    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    monkeypatch.setattr(native._api_client, "async_request", transport)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: native)
    result = asyncio.run(
        ask("Return 4", model="google/gemini-3.5-flash-lite", response_model=Answer, **settings)
    )
    assert result.result == 4
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinking_level": expected}
    assert settings == original


def test_retry_wire_uses_user_role(monkeypatch):
    import base64
    import json

    from google.genai._api_client import SdkHttpResponse

    native = genai.Client(api_key="offline-test")
    sent = []
    signature = base64.b64encode(b"signed-part").decode()

    async def transport(method, path, payload, http_options=None):
        request = native._api_client._build_request(method, path, payload, http_options)
        sent.append(json.loads(json.dumps(request.data)))
        return SdkHttpResponse(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "thoughtSignature": signature,
                                        "functionCall": {
                                            "id": "native-1",
                                            "name": "Answer",
                                            "args": {"result": "invalid" if len(sent) == 1 else 4},
                                        },
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            )
        )

    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    monkeypatch.setattr(native._api_client, "async_request", transport)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: native)
    result = asyncio.run(
        ask("Return 4", model="google/gemini-3.5-flash-lite", response_model=Answer, max_retries=2)
    )
    assert result.result == 4
    contents = sent[1]["contents"]
    assert all(content["role"] in ("user", "model") for content in contents)
    assert contents[-1]["role"] == "user"
    assert contents[-1]["parts"][0]["functionResponse"]["id"] == "native-1"
    assert base64.urlsafe_b64decode(contents[-2]["parts"][0]["thoughtSignature"]) == b"signed-part"
