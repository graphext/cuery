import asyncio

from apify import Actor

from ..tools.flex.entities import EntityExtractor
from .utils import run_flex_tool


async def main():
    async with Actor:
        await run_flex_tool(Actor, EntityExtractor)


if __name__ == "__main__":
    asyncio.run(main())
