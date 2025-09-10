import asyncio
import json

from apify import Actor

from ..seo.geo import GeoConfig, analyse


async def main():
    async with Actor:
        input = await Actor.get_input()
        config = GeoConfig(**input)
        df = await analyse(config)

        if df is None or len(df) == 0:
            raise ValueError("No LLM results were generated!")

        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
