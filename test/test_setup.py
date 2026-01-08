import os

import pytest

from cuery import Response, ask, set_env


@pytest.mark.asyncio
async def test_ask():
    set_env(apify_secrets=False, path="/Users/thomas/code/cuery/.env")

    class IntResponse(Response):
        result: int

    response = await ask("What is 2 + 2?", response_model=IntResponse)
    print(f"What is 2 + 2? => {response.result}")
    assert response.result == 4
