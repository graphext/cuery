from functools import cached_property
from typing import Literal

import pandas as pd
from pandas import DataFrame
from pydantic import ConfigDict

from ... import Tool, templates


def preprocess_records(
    records: DataFrame | list[dict],
    attrs: list[str] | None = None,
    max_samples: int | None = None,
) -> DataFrame:
    """Sample and filter attributes/columns in records."""
    if not isinstance(records, DataFrame):
        records = DataFrame(records)

    if attrs is not None:
        records = records[attrs]

    if max_samples is not None and (len(records) > max_samples):
        records = records.sample(max_samples, random_state=42)

    return records


class FlexTool(Tool):
    """Base class for tools iterating over data records with arbitrary attributes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Needed to allow DataFrame and other non-Pydantic types."""

    records: DataFrame | list[dict]
    """Data records to iterate over."""
    attrs: list[str] | None = None
    """List of record attributes to use. If None, all attributes are used."""
    record_format: Literal["json", "md", "text"] = "text"
    """Format of the record in the prompt."""

    @cached_property
    def template(self) -> str:
        """Get the name of the record template for the prompt.

        Each record/row will be formatted into the prompt using this template.
        """
        return templates.load_template(f"record_to_{self.record_format}")

    @cached_property
    def context(self) -> list[dict]:
        """For each input record, returns an object with a single "record" key.

        Jinja templates can then iterate over the "record" object's attributes
        """
        df = preprocess_records(self.records, attrs=self.attrs)
        return [{"record": record} for record in df.to_dict(orient="records")]

    async def __call__(self, **kwargs) -> DataFrame:
        """Normalize the nested input records back into individual columns in output."""
        response = await self.task(context=self.context, **kwargs)
        result = response.to_pandas(explode=False)
        return pd.concat(
            [
                result.drop(columns="record"),
                pd.json_normalize(result["record"]),  # type: ignore
            ],
            axis=1,
        )
