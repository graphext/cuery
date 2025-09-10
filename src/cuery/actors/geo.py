import asyncio
import json

from apify import Actor

from ..seo.geo import GeoConfig, analyse


async def main():
    async with Actor:

        async def set_progress_status(progress: dict):
            n = progress.get("n", 0)
            total = progress.get("total", "?")
            msg = f"Processed: {n}/{total} rows."
            await Actor.set_status_message(msg)

        input = await Actor.get_input()
        config = GeoConfig(**input)
        df = await analyse(config, progress_callback=set_progress_status)

        if df is None or len(df) == 0:
            raise ValueError("No LLM results were generated!")

        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
