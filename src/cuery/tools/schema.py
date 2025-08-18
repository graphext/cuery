import json
from functools import cached_property
from typing import Any, ClassVar

import jsonschema as jss
from pydantic import Field, field_validator

from .. import Message, Prompt, Response, ResponseClass, Tool

SYSTEM_PROMPT = """
You are an expert at helping users design JSON schemas for structured data extraction from LLMs.
The JSON schema you return will be converted to a Pydantic model, so make sure to include
field attributes that may trigger inclusion of pydantic specific validation, e.g. email
or URL formats.

Your role is to:
1. Understand what kind of data the user wants to extract
2. Define appropriate field names, types, constraints and descriptions
4. Return valid a JSON schema specification

Schema design considerations:
- Use appropriate data types (string, number, integer, boolean, array, object)
- Include JSON-schema built-in constraints if appropriate (number of array items and type, minimum, maximum etc.)
- Also include "format" and other field attribute if these would translate to corresponding Pydantic validators
- Provide clear, helpful descriptions for each field
- Consider whether fields should be required or optional
- Use consistent naming conventions (camelCase or snake_case)

The "reasoning" field should contain your conversational response to the user, and the "json_schema"
field should contain the full JSON schema as a dictionary.

Example output:

{
  "reasoning": "<your thought-process behind generating the json schema ...>"
  "json_schema": <your generated json-schema ...>
}
""".strip()

CONVERSATIONAL_SYSTEM_PROMPT = """
You are an expert at helping users design JSON schemas for structured data extraction from LLMs.

Your role is to:
1. Understand what kind of data the user wants to extract
2. Help them define appropriate field names, types, and descriptions
3. Suggest improvements and best practices
4. Return valid JSON schema specifications
5. Build schemas incrementally - adding, modifying, or removing fields as requested
6. Generate a list of realistic examples that conform to the schema

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
- ALWAYS provide 2-3 realistic example records in the examples field
- Examples should demonstrate different scenarios and edge cases
- All examples must conform exactly to the schema
- Use realistic, diverse data that shows the schema's intended use

You should ALWAYS provide a complete schema (not just the changes) when creating or modifying
schemas. The "answer" field should contain your conversational response to the user, the "json_schema"
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


class SchemaResponse(Response):
    """Response from the AI that includes both conversation and schema update."""

    reasoning: str
    """Brief explanation of schema design choices"""
    json_schema: dict[str, Any]
    """Valid JSON schema as a dictionary defining a structured output"""

    @field_validator("json_schema")
    @classmethod
    def validate_json_schema(cls, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Validate that the schema is a proper JSON schema."""
        try:
            jss.Draft7Validator.check_schema(json_schema)
            return json_schema
        except jss.SchemaError as exc:
            raise ValueError(f"Invalid JSON schema: {exc.message}") from exc


class ConversationalSchemaResponse(SchemaResponse):
    """Response from the AI that includes both conversation and schema update."""

    answer: str
    """Conversational response to the user"""
    examples: list[dict[str, Any]] = Field(min_length=2, max_length=5)
    """Some example records that conform to the schema"""

    @field_validator("examples")
    @classmethod
    def validate_examples(
        cls,
        examples: list[dict[str, Any]],
        info,
    ) -> list[dict[str, Any]]:
        """Validate that example data conforms to the schema."""
        json_schema = info.data.get("json_schema")

        try:
            validator = jss.Draft7Validator(json_schema)
            for i, example in enumerate(examples):  # noqa: B007
                validator.validate(example)
            return examples
        except jss.ValidationError as exc:
            raise ValueError(f"Example {i + 1} does not conform to schema: {exc.message}") from exc


class SchemaGenerator(Tool):
    """Create or modify a JSON schema given a prompt and optionally an existing schema."""

    instructions: str
    """Prompt instructions with details of the schema to generate."""
    current_schema: dict | None = None
    """Optional existing schema to modify or extend."""

    response_model: ClassVar[ResponseClass] = SchemaResponse
    """All instances of this tool will use the SchemaResponse model."""

    @cached_property
    def prompt(self) -> Prompt:
        """Add system and assistant messages to user's prompt."""
        messages = [Message(role="system", content=SYSTEM_PROMPT)]

        if self.current_schema:
            schema_str = json.dumps(self.current_schema, indent=2)
            schema_msg = SCHEMA_CONTEXT.format(schema=schema_str)
            messages.append(Message(role="assistant", content=schema_msg))

        messages.append(Message(role="user", content=self.instructions))
        return Prompt(messages=messages)

    async def __call__(self, **kwds) -> SchemaResponse:
        """Extracts a two-level topic hierarchy from a list of texts."""
        responses = await super().__call__(**kwds)
        return responses[0]  # type: ignore
