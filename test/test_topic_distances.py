"""Regression coverage for topic label validation."""

import pytest
from pydantic import ValidationError

from cuery.tools.topics import Topic, make_topic_model


@pytest.mark.parametrize(
    "parent,subtopic",
    [
        ("Books", "BOOKS"),  # Case-insensitive equality.
        ("book", "books"),  # Insertion.
        ("books", "book"),  # Deletion.
        ("book", "cook"),  # Substitution.
        ("café", "cafe"),  # An accented character is one edit.
        ("猫🐈", "猫🐕"),  # Unicode code points, not UTF-8 bytes.
    ],
)
def test_subtopic_with_fewer_than_two_edits_from_parent_is_rejected(parent, subtopic):
    with pytest.raises(ValidationError, match="too similar to parent topic"):
        Topic(topic=parent, subtopics=[subtopic])


@pytest.mark.parametrize("first,second", [("BOOK", "book"), ("book", "cook"), ("猫🐈", "猫🐕")])
def test_sibling_subtopics_with_fewer_than_two_edits_are_rejected(first, second):
    with pytest.raises(ValidationError, match="too similar to other subtopic"):
        Topic(topic="Literature and animals", subtopics=[first, second])


@pytest.mark.parametrize("first,second", [("book", "back"), ("café", "cane"), ("猫🐈", "犬🐕")])
def test_two_edits_are_accepted_for_parent_and_sibling_comparisons(first, second):
    parent_comparison = Topic(topic=first, subtopics=[second])
    sibling_comparison = Topic(topic="Literature and animals", subtopics=[first, second])
    assert parent_comparison.subtopics == [second]
    assert sibling_comparison.subtopics == [first, second]


def test_spaces_are_removed_only_when_comparing_siblings():
    # Two inserted spaces reach the parent threshold; sibling comparison strips them.
    topic = Topic(topic="icecream", subtopics=["ice  cream"])
    assert topic.subtopics == ["ice  cream"]
    with pytest.raises(ValidationError, match="too similar to other subtopic"):
        Topic(topic="Frozen desserts", subtopics=["icecream", "ice  cream"])


def test_word_permutations_are_rejected_even_when_edit_distance_is_large():
    with pytest.raises(ValidationError, match=r"duplicate \(permutation\)"):
        Topic(topic="Companion animals", subtopics=["red cat", "cat red"])


@pytest.mark.parametrize(
    "threshold,other,accepted", [(1, "cook", True), (3, "back", False), (3, "bats", True)]
)
@pytest.mark.parametrize("compare_siblings", [False, True])
def test_generated_model_respects_configured_threshold(
    threshold, other, accepted, compare_siblings
):
    model = make_topic_model(n_topics=1, n_subtopics=2, min_ldist=threshold)
    topic = {
        "topic": "Literature" if compare_siblings else "book",
        "subtopics": ["book", other] if compare_siblings else [other],
    }
    if accepted:
        result = model(topics=[topic])
        assert result.topics[0].subtopics == topic["subtopics"]
    else:
        with pytest.raises(ValidationError, match="too similar"):
            model(topics=[topic])
    # Generating a customized response must not alter the default validator.
    assert Topic._MIN_LDIST == 2
