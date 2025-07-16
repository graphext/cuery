import asyncio
import json

from apify import Actor

from ..seo.tasks import SerpTopicExtractor
from ..utils import LOG
from .utils import fetch_dataset

MAX_RETRIES = 6


async def main():
    async with Actor:
        config = await Actor.get_input()

        dataset_id = config.pop("dataset_id")
        df = await fetch_dataset(Actor, id=dataset_id)

        extractor = SerpTopicExtractor(**config)
        topics = await extractor(df, max_retries=MAX_RETRIES)

        LOG.info("Extracted topic hierarchy")
        LOG.info(json.dumps(topics.to_dict(), indent=2))
        await Actor.set_value("topics", topics.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
