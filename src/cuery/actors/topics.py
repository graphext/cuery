import asyncio
import json

from apify import Actor

from ..tools.flex import TopicExtractor
from ..utils import LOG
from .utils import fetch_apify_dataset, fetch_parquet_dataset

MAX_RETRIES = 8


async def main():
    async with Actor:
        config = await Actor.get_input()

        dataset_ref = config.pop("dataset")
        if dataset_ref.startswith("http"):
            columns = config.pop("attrs", None)
            df = fetch_parquet_dataset(dataset_ref, columns=columns)
        else:
            df = await fetch_apify_dataset(source=Actor, id=dataset_ref)

        extractor = TopicExtractor(records=df, **config)
        topics = await extractor(max_retries=MAX_RETRIES)

        LOG.info("Extracted topic hierarchy")
        LOG.info(json.dumps(topics.to_dict(), indent=2))
        await Actor.set_value(
            "topics",
            topics.to_dict(),
            content_type="application/json",
        )


if __name__ == "__main__":
    asyncio.run(main())
