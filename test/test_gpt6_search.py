"""Exercise web-search requests and parsing with the actual Responses SDK."""

import asyncio
import json

import httpx
import openai
import pytest

from cuery import Response, search


class Answer(Response):
    answer: str


@pytest.mark.parametrize("model,expected", [("gpt-6-luna", "none"), ("gpt-6.1-sol", "medium")])
@pytest.mark.parametrize("structured", [False, True])
def test_search_responses_transport(monkeypatch, model, expected, structured):
    requests, clients = [], []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/responses"
        return httpx.Response(
            200,
            json={
                "id": "resp_offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [
                    {"type": "reasoning", "id": "rs_offline", "summary": []},
                    {"type": "web_search_call", "id": "ws_offline", "status": "completed"},
                    {
                        "type": "message",
                        "id": "msg_offline",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"answer":"Four"}' if structured else "Four",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "start_index": 0,
                                        "end_index": 4,
                                        "url": "https://example.com",
                                        "title": "Source",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        )

    def make_client():
        client = openai.AsyncOpenAI(
            api_key="offline-test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(search, "AsyncOpenAI", make_client)
    search.query_openai.cache_clear()

    async def run():
        try:
            return await search.query_openai(
                "What is 2 + 2?", model=model, response_format=Answer if structured else None
            )
        finally:
            for client in clients:
                await client.close()

    result = asyncio.run(run())
    assert result.answer == "Four"
    assert requests[0]["reasoning"] == {"effort": expected}
    assert requests[0]["tools"][0]["type"] == "web_search"
    if structured:
        assert isinstance(result, Answer)
        assert requests[0]["text"]["format"]["type"] == "json_schema"
    else:
        assert result.sources[0].url == "https://example.com"
