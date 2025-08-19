import asyncio
import json

from apify import Actor

from ..tools.flex.topics import TopicAssigner
from ..utils import LOG
from .utils import fetch_dataset

MAX_RETRIES = 6
N_CONCURRENT = 100


async def main():
    async with Actor:
        config = await Actor.get_input()

        dataset_id = config.pop("dataset_id")
        df = await fetch_dataset(Actor, id=dataset_id)

        assigner = TopicAssigner(records=df, **config)
        result = await assigner(max_retries=MAX_RETRIES, n_concurrent=N_CONCURRENT)
        LOG.info(f"Assigned topics:\n{result}")

        records = json.loads(result.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
