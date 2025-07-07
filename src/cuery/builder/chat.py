"""Implements graio-compatible chat functionality.

I.e. the logic of interaction with OpenAI etc.
"""

import json
import os
import random
from typing import Any

import instructor
import jsonschema as jss
from openai import OpenAI
from pandas import DataFrame
from pydantic import BaseModel, Field, field_validator

SYSTEM_PROMPT = """
You are an expert at helping users design JSON schemas for structured data extraction from LLMs.

Your role is to:
1. Understand what kind of data the user wants to extract/structure
2. Help them define appropriate field names, types, and descriptions
3. Suggest improvements and best practices
4. Return valid JSON schema specifications
5. Build schemas incrementally - adding, modifying, or removing fields as requested
6. Generate realistic example data that conforms to the schema

When working with schemas:
- If a CURRENT SCHEMA is provided, use it as the base and modify it according to the user's request
- When adding fields: merge new fields with existing ones
- When removing fields: remove specified fields from the existing schema
- When modifying fields: update types, descriptions, or constraints as requested
- Always preserve existing fields unless explicitly asked to remove or modify them

Schema design considerations:
- Use appropriate data types (string, number, integer, boolean, array, object)
- Provide clear, helpful descriptions for each field
- Consider whether fields should be required or optional
- Use consistent naming conventions (camelCase or snake_case)

Example data requirements:
- ALWAYS provide 2-3 realistic example records in the example_data field
- Examples should demonstrate different scenarios and edge cases
- All examples must conform exactly to the schema
- Use realistic, diverse data that shows the schema's intended use

You should ALWAYS provide a complete schema (not just the changes) when creating or modifying
schemas. The "answer" field should contain your conversational response to the user, the "structure"
field should contain the full, updated JSON schema as a dictionary, the example_data
field should contain a list of objects with realistic examples, and the "reasoning" field
should contain a brief explanation of your schema design choices.

Be helpful, concise, and focus on practical schema design.
""".strip()

SCHEMA_CONTEXT = """
Please modify, extend, or replace the schema in the following section based on the user's request.
If the user wants to add fields, merge them with the existing schema.
If they want to remove fields, remove them from the existing schema.
If they want to modify field types or descriptions, update the existing schema accordingly.

## Current Schema
{schema}

"""


class SchemaResponse(BaseModel):
    """Response from the AI that includes both conversation and schema update."""

    answer: str = Field(description="Conversational response to the user")
    reasoning: str = Field(description="Brief explanation of schema design choices")
    structure: dict[str, Any] = Field(
        description="Valid JSON schema as a dictionary defining a structured output"
    )
    example_data: list[dict[str, Any]] = Field(
        description="2-5 example records that conform to the schema",
        min_length=2,
        max_length=5,
    )

    @field_validator("structure")
    @classmethod
    def validate_json_schema(cls, structure: dict[str, Any]) -> dict[str, Any]:
        """Validate that the schema is a proper JSON schema."""
        try:
            jss.Draft7Validator.check_schema(structure)
            return structure
        except jss.SchemaError as exc:
            raise ValueError(f"Invalid JSON schema: {exc.message}") from exc

    @field_validator("example_data")
    @classmethod
    def validate_example_data_against_schema(
        cls,
        example_data: list[dict[str, Any]],
        info,
    ) -> list[dict[str, Any]]:
        """Validate that example data conforms to the schema."""
        structure = info.data.get("structure")

        try:
            validator = jss.Draft7Validator(structure)
            for i, example in enumerate(example_data):  # noqa: B007
                validator.validate(example)
            return example_data
        except jss.ValidationError as exc:
            raise ValueError(
                f"Example data item {i + 1} does not conform to schema: {exc.message}"
            ) from exc


CURRENT_SCHEMA = None
CLIENT = instructor.from_openai(OpenAI())


def ai_chat(message: str, history: list) -> tuple[str, str, DataFrame | None]:
    """Send message to OpenAI and get response with structured output."""
    global CURRENT_SCHEMA  # noqa: PLW0603

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for msg in history:
        if isinstance(msg, tuple) and len(msg) == 2:
            user_msg, assistant_msg = msg
            messages.append({"role": "user", "content": str(user_msg)})
            messages.append({"role": "assistant", "content": str(assistant_msg)})

    # Add current schema context if available
    if CURRENT_SCHEMA is not None:
        schema_str = json.dumps(CURRENT_SCHEMA, indent=2)
        schema_msg = SCHEMA_CONTEXT.format(schema=schema_str)
        messages.append({"role": "assistant", "content": schema_msg})

    # Add current message
    messages.append({"role": "user", "content": message})

    print(json.dumps(messages, indent=2))

    response = CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,  # type: ignore
        response_model=SchemaResponse,
        max_retries=5,
    )  # type: ignore

    print("Executed OpenAI API call successfully.", flush=True)

    # Extract response and schema
    response_text = response.answer
    if response.reasoning:
        response_text += f"\n\n Schema Design Notes:\n{response.reasoning}"

    CURRENT_SCHEMA = response.structure
    schema_str = json.dumps(response.structure, indent=2)
    example_df = DataFrame.from_records(response.example_data)
    print(schema_str)
    print(example_df)
    return response_text, schema_str, example_df


def chat(message: str, history: list[dict]) -> tuple[str, str, str]:
    """Call LLM API with message and history and return response"""
    response = random.choice(["Yes", "No"])  # noqa: S311
    schema = {"a": 1, "b": 2, "c": 3}
    examples = [
        {"field1": "value1", "field2": 123},
        {"field1": "value2", "field2": 456},
    ]
    return response, json.dumps(schema, indent=2), json.dumps(examples, indent=2)
