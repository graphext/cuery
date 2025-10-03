import asyncio
import json

from apify import Actor

from ..seo.keywords import GoogleKwdConfig, keywords
from .utils import fetch_apify_dataset, fetch_parquet_dataset


async def input_keywords(params: dict):
    """Extract keywords from input parameters."""
    kwds = []

    # Keywords from input dataset
    dataset = params.pop("keyword_dataset", None)
    column = params.pop("keyword_column", None)
    if dataset and column:
        if dataset.startswith("http"):
            df = fetch_parquet_dataset(dataset, columns=[column])
        else:
            df = await fetch_apify_dataset(source=Actor, id=dataset)

        kwds.extend(df[column].dropna().astype(str).tolist())

    # Explicit keywords directly passed via parameter
    if kwd_lst := params.pop("keywords"):
        kwds.extend(kwd_lst)

    return kwds


async def main():
    async with Actor:
        params = await Actor.get_input()
        params |= {"ideas": False}
        params["keywords"] = await input_keywords(params)
        config = GoogleKwdConfig(**params)
        df = keywords(config)
        records = json.loads(df.to_json(orient="records", date_format="iso", index=False))
        await Actor.push_data(records)


if __name__ == "__main__":
    asyncio.run(main())
