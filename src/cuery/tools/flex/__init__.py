"""Flex tools (flexible tools) are tools with flexible inputs.

Instead of expecting a fixed set of attributes in input records (DataFrame rows),
which are validated on execution, and mapped to specific substitutions in prompt templates,
flexible tools allow data records with arbitrary attributes. Each record is essentially
an object/dict with a single "record" key and a nested object as value, which in turn
can have arbitrary data. Prompts are dynamically generated from Jinja templates based on
the attributes present in the input records.
"""

from .classify import Classifier
from .entities import EntityExtractor
from .score import Scorer
from .topics import MultiTopicAssigner, TopicAssigner, TopicExtractor

__all__ = [
    "Classifier",
    "EntityExtractor",
    "Scorer",
    "TopicExtractor",
    "TopicAssigner",
    "MultiTopicAssigner",
]
