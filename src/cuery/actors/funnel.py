import asyncio
import json

from apify import Actor
from pandas import DataFrame

from ..seo.funnels import Funnel
from ..utils import LOG


async def main():
    async with Actor:
        input = await Actor.get_input()
        funnel = Funnel(**input)

        await funnel.seed()
        seed_df = funnel.to_pandas()[["stage", "category", "seed"]]
        LOG.info(f"Generated seed keywords:\n{seed_df}")

        result = funnel.keywords()
        if result is None or len(result) == 0:
            raise ValueError("No funnel keyword results were generated!")

        df: DataFrame = result
        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
