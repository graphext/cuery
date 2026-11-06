"""Historical combined Keyword Planner actor, backed by current Cuery."""

import asyncio
import json

from apify import Actor

from .. import resources
from ..seo.keywords import GoogleKwdConfig, keywords


def make_config(actor_input):
    # Preserve historical validation for explicit locale values, including empty strings.
    if "language" in actor_input:
        resources.google_lang_id(actor_input["language"])
    if "country" in actor_input:
        resources.google_country_id(actor_input["country"])
    # Preserve historical locale defaults without overwriting explicit user values.
    return GoogleKwdConfig(**({"language": "en", "country": "us"} | actor_input))


async def main():
    async with Actor:
        config = make_config(await Actor.get_input())
        df = keywords(config)
        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
