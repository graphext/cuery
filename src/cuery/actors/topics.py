import asyncio
import json

from apify import Actor

from ..tools.flex import TopicExtractor
from ..utils import LOG
from .utils import fetch_dataset

MAX_RETRIES = 6


async def main():
    async with Actor:
        config = await Actor.get_input()

        dataset_id = config.pop("dataset_id")
        df = await fetch_dataset(Actor, id=dataset_id)

        extractor = TopicExtractor(records=df, **config)
        topics = await extractor(max_retries=8)

        LOG.info("Extracted topic hierarchy")
        LOG.info(json.dumps(topics.to_dict(), indent=2))
        await Actor.set_value(
            f"topics-{dataset_id}",
            topics.to_dict(),
            content_type="application/json",
        )


if __name__ == "__main__":
    asyncio.run(main())
