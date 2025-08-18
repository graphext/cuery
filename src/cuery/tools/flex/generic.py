"""Tool(s) to iterate over data with a response model defined by json schemas.

The idea is to first generate a schema with an LLM using the `SchemaGenerator` tool,
then use that schema process records with the `Generic` tool.
"""

import json
from functools import cached_property
from string import Template

import pandas as pd
from pandas import DataFrame

from ... import Field, Prompt, ResponseClass, ResponseSet, Task, ask, pprint
from ...response import models_from_jsonschema
from ...utils import LOG, dedent
from ..schema import SchemaGenerator
from .base import FlexTool

GENERIC_PROMPT = dedent("""
# Instructions

${instructions}

# Data Record

${record_template}
""")

SCHEMA_CONVERSION_PROMPT = dedent("""
A user has provided instructions (in below section) for a task that involves extracting information
from data records. Your task is to translate the user's instructions into new instructions for a
tool that will first generate a JSON schema that describes the information the user wants to
extract.

## Example 1

- User instruction: "Extract any emails and URLs from the text."
- Your instruction: "Create a JSON schema that defines fields for 'emails' and 'urls' allowing for multiple values."

## Example 2

- User instruction: "Extract the name and age of the person from his bio."
- Your instruction: "Create a JSON schema that defines fields for a person's 'name' and 'age'."

Make sure to capture all relevant details from the user's instructions, e.g. whether the fields
should be scalar or arrays, whether they should be required or optional, and any specific formats
(e.g. email, URL) that should be used.

# User instruction

${instructions}
""")


class Generic(FlexTool):
    """Tools that iterates over records with a JSON-schema response model."""

    response_schema: dict
    """JSON schema used as response model."""
    instructions: str
    """Instructions for the tool, describing its purpose and how to use it."""

    @cached_property
    def prompt(self) -> Prompt:
        """Generate a prompt string based on the instructions and current schema."""
        prompt = Prompt.from_string(GENERIC_PROMPT)
        return prompt.substitute(
            instructions=self.instructions,
            record_template=self.template,
        )

    @cached_property
    def response_model(self) -> ResponseClass:
        models = models_from_jsonschema(self.response_schema)
        return models[0]


class Auto(Generic):
    """Fully automatic, general-purpose tool for processing data records.

    First auto-generates a response model from the response model instructions,
    then iterates over the records using that model and the provided tools
    instructions.
    """

    response_schema: str | dict | None = None
    """Instructions to generate a JSON schema used as response model."""
    schema_model: str = Field("openai/gpt-4.1", pattern=r"^[\w\.-]+/[\w\.-]+$")
    """Specific model to use to generate the JSON schema."""

    _response: ResponseSet | None = None

    @cached_property
    def prompt(self) -> Prompt:
        """Generate a prompt string based on the instructions and current schema."""
        prompt = Prompt.from_string(GENERIC_PROMPT)
        return prompt.substitute(
            instructions=self.instructions,
            record_template=self.template,
        )

    async def response_model(self) -> ResponseClass:
        # Auto-generate the response model from the schema instructions
        schema = self.response_schema

        if schema is None:
            LOG.info("Generating instructions for response schema from general instructions...")
            schema_prompt = Template(SCHEMA_CONVERSION_PROMPT).substitute(
                instructions=self.instructions,
            )
            schema = await ask(
                prompt=schema_prompt,
                model=self.schema_model,
                max_retries=10,
            )
            LOG.info(f"Generated schema instructions:\n{schema}")

        if isinstance(schema, str):
            LOG.info("Auto-generating response JSON schema from instructions...")
            sgen = SchemaGenerator(
                instructions=schema,
                model=self.schema_model,
            )
            response = await sgen(max_retries=10)
            schema = response.json_schema
            LOG.info(response.reasoning)
            LOG.info(f"Generated JSON schema:\n{json.dumps(schema, indent=2)}")

        model = models_from_jsonschema(schema)[0]
        LOG.info("Auto-generated response model:")
        pprint(model.fallback())
        return model

    async def task(self) -> Task:
        """Create a Task instance for this tool."""
        response_model = await self.response_model()
        return Task(prompt=self.prompt, response=response_model, model=self.model)

    async def __call__(self, **kwargs) -> DataFrame:
        """Normalize the nested input records back into individual columns in output."""
        task = await self.task()
        self._response = await task(context=self.context, **kwargs)
        result = self._response.to_pandas(explode=False)
        return pd.concat(
            [
                result.drop(columns="record"),
                pd.json_normalize(result["record"]),  # type: ignore
            ],
            axis=1,
        )
