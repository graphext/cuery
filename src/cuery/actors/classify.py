import asyncio

from apify import Actor

from ..tools.flex.classify import Classifier
from .utils import run_flex_tool


async def main():
    async with Actor:
        await run_flex_tool(Actor, Classifier)


if __name__ == "__main__":
    asyncio.run(main())
