"""Tools to apply SEO to LLM responses."""

from functools import lru_cache
from io import StringIO
from typing import Literal

from markdown import Markdown
from openai import OpenAI

from .. import Field, Prompt, Response, Task
from ..utils import dedent


@lru_cache(maxsize=10)
def openai_web_search(
    prompt: str,
    country: str = "ES",
    city: str = "Madrid",
    context_size: str = "high",
    temperature: float = 0.3,
):
    client = OpenAI()
    return client.responses.create(
        model="gpt-4o",
        input=prompt,
        tools=[
            {
                "type": "web_search_preview",
                "search_context_size": context_size,
                "user_location": {
                    "type": "approximate",
                    "country": country,
                    "city": city,
                },
            }  # type: ignore
        ],
        tool_choice="required",
        temperature=temperature,
    )


# https://stackoverflow.com/a/54923798
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
Markdown.output_formats["plain"] = unmark_element  # type: ignore
__md = Markdown(output_format="plain")  # type: ignore
__md.stripTopLevelTags = False


def unmark(text):
    """Convert Markdown text to plain text, stripping all formatting."""
    return __md.convert(text)


def process_search_response(response, plain: bool = False):
    output = response.output
    if not len(output) >= 2:  # noqa: PLR2004
        raise ValueError("Response must contain at least two elements.")

    if not output[0].type == "web_search_call":
        raise ValueError("First output must be of type 'web_search_call'.")

    annotations = []
    text = ""
    for content in output[1].content:
        if content.type == "output_text":
            if hasattr(content, "annotations"):
                for ann in content.annotations:
                    annotations.append(f"{ann.title} ({ann.url})")
            if hasattr(content, "text"):
                text += content.text + "\n"

    annotations = list(set(annotations))
    content = text + "\nReferences:\n" + "\n".join(annotations)

    if plain:
        content = unmark(content)

    return content


class Entity(Response):
    """Individual entity extracted from AI Overview content."""

    name: str = Field(..., description="The entity name or text as it appears")
    brand: str = Field(..., description="The brand or company name associated with the entity")
    type: Literal["brand_or_product", "other"] = Field(
        ..., description="The category/type of the entity"
    )


class Entities(Response):
    """Result containing all extracted entities from AI Overview data."""

    entities: list[Entity] = Field(default=[], description="List of extracted entities")


ENTITY_EXTRACTION_PROMPT = dedent("""
From the text below (in the "Text to analyze" markdown section), extract entities that are relevant
to SEO analysis, specifically brand, company and product names, including names of services or
technologies (e.g., "Google Cloud Platform", "Microsoft Azure", ChatGPT), shops (e.g. "Best Buy",
"Walmart") etc. Categorize these as "brand_or_product", and any other entities as "other".

For each entity, provide the clean entity name/text, its canonical brand name, and the
type/category of the entity. Pay special attention to URLs in the text, which may refer to brands,
companies or products. Ensure to report the names of entities always in lowercase and singular
form, even if they appear in plural or uppercase. The canonical brand name should be the most
general or widely recognized brand name for the entity, e.g., "google" for "Google Cloud Platform",
"microsoft" for "Microsoft Azure", "hp" for "HP Pavilion 360", "sony" for "Sony WH-1000XM4",
"patagonia" for "Patagonia Spain", etc (always lower case). But if the entity is not known, or
not affiliated with a known brand, either use the entity name itself (e.g. "le grand mazarin" for
"Le Grand Mazarin") or make a best guess (e.g. "cmf" for "cmf phone 2 pro").

Also, importanttly, ensure to report the entities in the order they appear in the text, as this is
important for SEO ranking analysis.

# Text to analyze

Only consider entities in the below text:

{{ text }}
""")


@lru_cache(maxsize=10)
async def extract_entities(plain_content: str):
    prompt = Prompt.from_string(ENTITY_EXTRACTION_PROMPT)
    task = Task(prompt=prompt, response=Entities)
    response = await task.call(context={"text": plain_content}, model="openai/gpt-3.5-turbo")
    ents = response[0].to_dict()["entities"]
    ents = [
        {"name": e["name"].lower(), "brand": e["brand"].lower()}
        for e in ents
        if e["type"] == "brand_or_product"
    ]

    # Report each entity only once, even if it appears multiple times, preserving the order
    seen = set()
    unique_entities = []
    for entity in ents:
        if entity["name"] not in seen:
            seen.add(entity["name"])
            unique_entities.append(entity)

    return unique_entities
