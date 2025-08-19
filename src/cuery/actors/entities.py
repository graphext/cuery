import asyncio

from apify import Actor

from ..tools.flex.entities import EntityExtractor
from .utils import run_flex_tool

MAX_RETRIES = 6
N_CONCURRENT = 100


async def main():
    async with Actor:
        await run_flex_tool(
            Actor,
            EntityExtractor,
            max_retries=MAX_RETRIES,
            n_concurrent=N_CONCURRENT,
        )


if __name__ == "__main__":
    asyncio.run(main())
