import asyncio
import json

from apify import Actor
from pandas import DataFrame

from ..seo.brands import BrandSearchConfig, search_brands


async def main():
    async with Actor:
        input = await Actor.get_input()
        config = BrandSearchConfig(**input)
        result = await search_brands(config, to_pandas=True)

        if result is None or len(result) == 0:
            raise ValueError("No brand results were generated!")

        # Type assertion: when to_pandas=True, result is a DataFrame
        df: DataFrame = result  # type: ignore
        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
