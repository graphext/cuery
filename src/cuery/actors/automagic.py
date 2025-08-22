import asyncio

from apify import Actor

from ..tools.flex.generic import Auto
from .utils import run_flex_tool


async def main():
    async with Actor:
        await run_flex_tool(Actor, Auto)


if __name__ == "__main__":
    asyncio.run(main())
