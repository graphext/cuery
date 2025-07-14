import asyncio
import json

from apify import Actor

from ..seo.keywords import GoogleKwdConfig, keywords


async def main():
    async with Actor:
        input = await Actor.get_input()
        config = GoogleKwdConfig(**input)
        df = keywords(config)
        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
