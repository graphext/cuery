from functools import cached_property
from typing import Literal

import pandas as pd
from pandas import DataFrame

from ... import Prompt, Response, ResponseClass
from .base import FlexTool

PROMPT = """
# Instructions

From the Data Record section below extract entities in the following categories:

${definitions}

For each entity, provide the entity name/text as it appears, and the type/category of entity.
Ensure to report the names of entities always in lowercase and singular form, even if
they appear in plural or uppercase in the source titles, to avoid inconsistencies in the output.

Expected output format:

[{"name": "<entity name>", "type": "<entity type>"}, ...]

For example, if the data record contains "Apple iPhone 15 Pro Max Review", and entity
definitions include a "brand" category and a "product" category, the expected output would be:

[{"name": "apple", "type": "brand"}, {"name": "iphone 15", "type": "product"}]

# Data Record

${record_template}
"""


def prompt_definitions(entities: dict[str, str]) -> str:
    """Generate the definitions section for the entity extraction prompt."""
    return "\n".join(f"""- "{name}": {description}""" for name, description in entities.items())


def make_entity_model(entities: dict[str, str]) -> ResponseClass:
    """Dynamically create a response model for a score based on provided parameters."""

    class Entity(Response):
        """Entity extracted from a data record."""

        name: str
        """The entity name or text in singular lowercase form"""
        type: Literal[*entities.keys()]
        """The category/type of the entity"""

    return Entity


def make_entities_model(entities: dict[str, str]) -> ResponseClass:
    """Create a response model for a list of entities."""
    Entity = make_entity_model(entities)

    class Entities(Response):
        """Result containing all extracted entities from a data record."""

        entities: list[Entity]
        """List of extracted SEO-relevant entities"""

    return Entities


class EntityExtractor(FlexTool):
    """ "Extract SEO-relevant entities from Google SERP AI Overview data."""

    entities: dict[str, str]
    """Dictionary of entity names/categories and their descriptions."""

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt.from_string(PROMPT)
        prompt_args = {
            "definitions": prompt_definitions(self.entities),
            "record_template": self.template,
        }
        return prompt.substitute(**prompt_args)

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_entities_model(self.entities)

    async def __call__(self, **kwargs) -> DataFrame:
        responses = await self.task(context=self.context, **kwargs)
        df = responses.to_pandas(explode=False, normalize=False)
        df = pd.concat(
            [
                df.drop(columns="record"),
                pd.json_normalize(df["record"]),  # type: ignore
            ],
            axis=1,
        )

        for kind in self.entities:
            df[f"entities_{kind}"] = df.entities.apply(
                lambda ents, kind=kind: [e["name"] for e in ents if e["type"] == kind]
                if ents is not None
                else None
            )

        return df.drop(columns="entities")
