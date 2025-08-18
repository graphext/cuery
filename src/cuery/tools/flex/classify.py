from functools import cached_property
from typing import Literal

from ... import Prompt, Response, ResponseClass
from ...utils import dedent
from .base import FlexTool

PROMPT = dedent("""
# Instructions

You're task is to classify the data record below into one of the following categories:

${categories}

Make sure to consider all the attributes of the data record, and assign the class
based on the content of the attributes.

${instructions}

# Data Record

${record_template}
""")


def prompt_definitions(categories: dict[str, str]) -> str:
    """Generate the definitions section for the entity extraction prompt."""
    return "\n".join(
        f"""- "{label}": {description}""" for label, description in categories.items()
    )


def make_category_model(categories: dict[str, str]) -> ResponseClass:
    """Dynamically create a response model for a score based on provided parameters."""

    class Category(Response):
        """Entity extracted from a data record."""

        label: Literal[*categories.keys()]
        """A category label."""

    return Category


class Classifier(FlexTool):
    """Zero-shot classify a data record with arbitrary attributes."""

    categories: dict[str, str]
    """Dictionary of category labels and their descriptions."""
    instructions: str = ""
    """Additional instructions (context) for the classification task."""

    @cached_property
    def prompt(self) -> Prompt:
        prompt = Prompt.from_string(PROMPT)
        prompt_args = {
            "categories": prompt_definitions(self.categories),
            "instructions": self.instructions,
            "record_template": self.template,
        }
        return prompt.substitute(**prompt_args)

    @cached_property
    def response_model(self) -> ResponseClass:
        return make_category_model(self.categories)
