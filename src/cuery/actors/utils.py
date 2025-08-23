import json
from time import time

import pandas as pd
from apify import Actor
from apify_client import ApifyClientAsync
from pandas import DataFrame

from ..tools.flex.base import FlexTool
from ..utils import LOG, json_encode

ActorClass = type(Actor)
FlexToolClass = type(FlexTool)

DEFAULT_FLEX_CONFIG = {
    "max_retries": 6,
    "n_concurrent": 100,
    "timeout": 10,
}


async def fetch_apify_dataset(
    source: ActorClass | ApifyClientAsync,
    id: str,
    force_cloud: bool = True,
) -> DataFrame:
    """Fetch a dataset from Apify and return it as a DataFrame."""
    LOG.info(f"Fetching Apify dataset with id '{id}'")
    if isinstance(source, ApifyClientAsync):
        dataset = source.dataset(dataset_id=id)
    else:
        dataset = await source.open_dataset(id=id, force_cloud=force_cloud)

    records = [record async for record in dataset.iterate_items()]
    return DataFrame(records)


def fetch_parquet_dataset(url: str, columns: list[str] | None = None) -> DataFrame:
    """Fetch a Parquet dataset from a URL and return it as a DataFrame.

    Since we can't validate column names before fetching, we try to read the dataset
    with and without specifying them.
    """
    LOG.info(f"Fetching Parquet dataset from '{url}' with columns {columns}")
    try:
        return pd.read_parquet(url, columns=columns)
    except Exception:
        if columns:
            # May have failed because columns are not present in the dataset
            LOG.warning(
                f"Failed to read specified columns {columns} from {url}."
                " Reading the full dataset instead."
            )
            return pd.read_parquet(url)

        raise


async def run_flex_tool(
    Actor: ActorClass,
    Tool: FlexToolClass,
    **kwargs,
):
    """Run a flex tool with the given arguments."""

    async def set_progress_status(progress: dict):
        n = progress.get("n", 0)
        total = progress.get("total", "?")
        msg = f"Processed: {n}/{total} rows."
        await Actor.set_status_message(msg)

    t0 = time()

    exec_config = DEFAULT_FLEX_CONFIG.copy() | (kwargs or {})
    config = await Actor.get_input()

    dataset_ref = config.pop("dataset")
    if dataset_ref.startswith("http"):
        columns = config.pop("attrs", None)
        df = fetch_parquet_dataset(dataset_ref, columns=columns)
    else:
        df = await fetch_apify_dataset(source=Actor, id=dataset_ref)

    LOG.info(f"Got dataset\n{df}\n")
    LOG.info(f"Executing tool '{Tool.__name__}'")
    LOG.info(f"Tool config:\n {json.dumps(config, indent=2, default=json_encode)}")
    LOG.info(f"Execution config:\n {json.dumps(exec_config, indent=2, default=json_encode)}")

    tool = Tool(records=df, **config)
    result = await tool(progress_callback=set_progress_status, **exec_config)
    LOG.info(f"Result:\n{result}")

    records = json.loads(result.to_json(orient="records", date_format="iso", index=False))
    await Actor.push_data(records)

    t1 = time()
    LOG.info(f"Done in {t1 - t0:.1f}s")
