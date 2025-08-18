import asyncio
import json

from apify import Actor

from cuery.seo.tools import SerpIntentAssigner
from cuery.utils import LOG
from cuery.actors.utils import fetch_dataset

MAX_RETRIES = 6


async def main():
    async with Actor:
        config = await Actor.get_input()

        dataset_id = config.pop("dataset_id")
        df = await fetch_dataset(Actor, id=dataset_id)

        assigner = SerpIntentAssigner(**config)
        result = await assigner(df, max_retries=MAX_RETRIES)
        LOG.info(f"Assigned intents:\n{result}")

        records = json.loads(result.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
