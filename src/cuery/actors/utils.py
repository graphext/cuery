import json

from apify import Actor
from apify_client import ApifyClientAsync
from pandas import DataFrame

from ..tools.flex.base import FlexTool
from ..utils import LOG

ActorClass = type(Actor)
FlexToolClass = type(FlexTool)


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


async def run_flex_tool(Actor: ActorClass, Tool: FlexToolClass, **kwargs):
    """Run a flex tool with the given arguments."""
    config = await Actor.get_input()

    dataset_id = config.pop("dataset_id")
    df = await fetch_dataset(Actor, id=dataset_id)

    tool = Tool(records=df, **config)
    result = await tool(**kwargs)
    LOG.info(f"Result:\n{result}")

    records = json.loads(result.to_json(orient="records", date_format="iso", index=False))
    await Actor.push_data(records)
