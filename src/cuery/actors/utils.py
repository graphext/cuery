from apify import Actor
from apify_client import ApifyClientAsync
from pandas import DataFrame

ActorClass = type(Actor)


async def fetch_dataset(
    source: ActorClass | ApifyClientAsync,
    id: str,
    force_cloud: bool = True,
) -> DataFrame:
    """Fetch a dataset from Apify and return it as a DataFrame."""
    if isinstance(source, ApifyClientAsync):
        dataset = source.dataset(dataset_id=id)
    else:
        dataset = await source.open_dataset(id=id, force_cloud=force_cloud)

    records = [record async for record in dataset.iterate_items()]
    return DataFrame(records)
