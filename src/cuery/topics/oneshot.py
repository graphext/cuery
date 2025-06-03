from collections.abc import Iterable

from .. import utils
from ..prompt import Prompt
from ..response import Field, ResponseModel
from ..task import Task

PROMPT_TEXT = """
From the list of texts below (separated by line breaks), extract a two-level nested markdown
list of topics. The top-level should not contain more than 10 topics, and each top-level should
not contain more than 5 subtopics. The texts come from a dataset of {{meta_topic}}, so the topics
should be relevant to that domain. Make sure top-level topics are generalizable and not too
specific, so they can be used as a hierarchy for the subtopics. Make sure also that subtopics are
not redundant (no similar ones within the the same top-level topic). Finally, make sure the result
is valid Markdown: top-level topics should be prefixed with a single dash ("-"), and subtopics with
two spaces and a dash ("  -").

# Texts

{{texts}}
"""

PROMPT = Prompt.from_string(utils.dedent(PROMPT_TEXT))


class Topics(ResponseModel):
    markdown: str = Field(..., description="A two-level nested markdown list of topics.")


ExtractTopics = Task(prompt=PROMPT, response=Topics)


def topics_from_markdown(markdown: str) -> dict:
    """Converts a two-level nested markdown list of topics into a dictionary."""
    lines = markdown.strip().split("\n")
    topics = {}
    current_topic = None

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if stripped_line.startswith("- "):
            # This is a top-level topic
            current_topic = stripped_line[2:].strip()
            topics[current_topic] = []
        elif stripped_line.startswith("  - "):
            # This is a subtopic
            if current_topic is not None:
                subtopic = stripped_line[4:].strip()
                topics[current_topic].append(subtopic)

    return topics


async def extract_topics(
    texts: Iterable[str],
    domain: str,
    model: str,
    max_dollars: float,
    max_tokens: float | None = None,
) -> dict:
    """Extracts a two-level topic hierarchy from a list of texts."""
    text = utils.concat_up_to(texts, model=model, max_dollars=max_dollars, max_tokens=max_tokens)
    context = {"texts": text, "meta_topic": domain}
    response = await ExtractTopics.call(context=context, model=model)
    return topics_from_markdown(response.markdown)  # type: ignore
