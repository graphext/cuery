"""Flexible scorer tool for assigning scores to data records."""

from functools import cached_property
from typing import Literal

from pydantic import field_validator

from ... import Prompt, Response, ResponseClass
from ...utils import dedent
from .base import FlexTool

SCORER_PROMPT = dedent("""
# Instructions

Assign a ${name} (${type} score between ${min} and ${max}) to the data record below.

Score description: ${description}

Make sure to consider all the attributes of the data record, and assign the score
based on the content of the attributes.

# Data Record

${record_template}
""")

SCORER_PARAM_NAMES = ("name", "type", "min", "max", "description")

PURCHASE_PROB_PROMPT = """
Estimate the probability of a purchase action based on the keyword and its associated
attributes. The score should be between 0 and 10, where 0 means no purchase probability and
10 means very high purchase probability. Consider the context of the keyword, its intent,
and any other relevant attributes that might indicate a user's likelihood to purchase.
"""
"""Example description for a purchase probability score."""

PURCHASE_PROB_PARAMS = {
    "name": "purchase probability",
    "type": "integer",
    "min": 0,
    "max": 10,
    "description": PURCHASE_PROB_PROMPT,
}
"""Example parameters for a purchase probability score."""


def make_score_model(
    name: str,
    type: str,
    min: float,
    max: float,
    description: str,
) -> ResponseClass:
    """Dynamically create a response model for a score based on provided parameters."""
    name = name.strip().replace(" ", "_").lower()
    fields = {
        name: {
            "type": type,
            "description": description,
            "ge": min,
            "le": max,
        }
    }

    return Response.from_dict(name="Score", fields=fields)


class Scorer(FlexTool):
    """Classify intent for keywords based on their SERP results."""

    name: str
    """Name of the score to assign."""
    type: Literal["integer", "float"] = "float"
    """Whether to return the score as integer or float."""
    min: float
    """Minimum value of the score."""
    max: float
    """Maximum value of the score."""
    description: str
    """Description of the score to assign."""

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """Ensure the name is a valid Python identifier."""
        name = name.strip().replace(" ", "_").lower()
        if not name.isidentifier():
            raise ValueError(f"Invalid score name: {name}. Must be a valid identifier.")
        return name

    @cached_property
    def scorer_params(self) -> dict:
        """Get the parameters for the score model."""
        return {k: getattr(self, k) for k in SCORER_PARAM_NAMES}

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt.from_string(SCORER_PROMPT)
        prompt_args = self.scorer_params | {"record_template": self.template}
        return prompt.substitute(**prompt_args)

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_score_model(**self.scorer_params)
