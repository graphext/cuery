import asyncio
import json

from apify import Actor

from ..seo.serps import SerpConfig, serps


async def main():
    async with Actor:
        input = await Actor.get_input()
        config = SerpConfig(**input)
        df = await serps(config, keywords=None)  # Passed via input config

        if df is None or len(df) == 0:
            raise ValueError("No SERP results were fetched!")

        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
