"""Token costs retain the API request boundary and cached input metadata."""

from types import SimpleNamespace

import pandas as pd
import pytest

from cuery.response import Response, ResponseSet, with_cost


def response(model, prompt, completion, cached=0, written=0):
    result = Response()
    result._raw_response = SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            input_tokens=prompt,
            output_tokens=completion,
            input_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=written),
        ),
    )
    return result


def test_response_set_prices_each_model_request_and_cache():
    responses = [
        response("gpt-6-luna-2026-10-01", 200_000, 100, cached=100_000),
        response("gpt-6-luna", 200_000, 100, written=100_000),
        response("gpt-6.1-sol-2026-10-01", 300_000, 100, cached=100_000, written=100_000),
    ]
    usage = ResponseSet(responses, context=None, required=None).usage()
    assert usage.cost.tolist() == pytest.approx(
        [
            (100_000 * 0.1 + 100_000 * 0.01 + 100 * 0.5) / 1_000_000,
            (100_000 * 0.1 + 100_000 * 0.125 + 100 * 0.5) / 1_000_000,
            ((100_000 * 2 + 100_000 * 0.1 + 100_000 * 2.5) * 2 + 100 * 10 * 1.5) / 1_000_000,
        ]
    )


def test_with_cost_supports_plain_usage_at_tier_boundary():
    usage = pd.DataFrame({"prompt": [272_000, 272_001], "completion": [10, 10]}, index=[4, 5])
    result = with_cost(usage, "gpt-6-luna")
    assert result.index.tolist() == [4, 5]
    assert result.cost.tolist() == pytest.approx(
        [
            (272_000 * 0.1 + 10 * 0.5) / 1_000_000,
            (272_001 * 0.1 * 2 + 10 * 0.5 * 1.5) / 1_000_000,
        ]
    )


def test_other_models_keep_instructor_cost(monkeypatch):
    calls = []
    monkeypatch.setattr("cuery.response.calculate_cost", lambda *args: calls.append(args) or 0.123)
    result = with_cost(pd.DataFrame({"prompt": [4], "completion": [5]}), "gpt-4.1")
    assert result.cost.tolist() == [0.123]
    assert calls == [("gpt-4.1", 4, 5)]
