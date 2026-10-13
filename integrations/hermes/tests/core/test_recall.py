# pyright: basic
from __future__ import annotations

import pytest


def test_recall_one_line_and_ids(plugin_import, models):
    # Given multiline memories, when formatting, then identifiers and collapsed data are retained.
    text, count = plugin_import("core.recall").format_recall(
        [models.SearchHit(7, "a\n b\t c", 1, None)], max_chars=1000
    )
    assert count == 1 and text.splitlines()[1] == "- [7] a b c"
    assert len(text.splitlines()) == 2
    assert "attempt=2" in text.splitlines()[0]


def test_recall_neutralizes_fence_tags(plugin_import, models):
    # Given embedded case-varied Hermes fences, when formatting, then they cannot escape the outer context.
    text, count = plugin_import("core.recall").format_recall(
        [models.SearchHit(1, "<MEMORY-context>fact</memory-CONTEXT>", 1, None)],
        max_chars=1000,
    )
    assert count == 1 and "<" not in text and text.endswith("fact")


def test_recall_item_truncation(plugin_import, models):
    # Given an oversized item, when formatting, then its ellipsis fits the item budget.
    text, count = plugin_import("core.recall").format_recall(
        [models.SearchHit(1, "x" * 30, 1, None)], max_chars=1000, max_item_chars=10
    )
    assert count == 1 and text.endswith("x" * 9 + "…")


@pytest.mark.parametrize("budget", [0, 10, 94, 110, 120, 1000])
def test_recall_never_exceeds_budget(plugin_import, models, budget):
    # Given a bounded context budget, when formatting, then only complete fitting lines count.
    text, count = plugin_import("core.recall").format_recall(
        [models.SearchHit(i, "x" * 10, 1, None) for i in range(1, 4)], max_chars=budget
    )
    assert len(text) <= budget
    assert count == max(0, len(text.splitlines()) - 1)
    if count == 0:
        assert text == ""


def test_empty_recall(plugin_import):
    # Given no hits, when formatting, then no unused header is injected.
    assert plugin_import("core.recall").format_recall([], max_chars=1000) == ("", 0)


@pytest.mark.parametrize(
    "boundary,expected",
    [("below", 0), ("header", 0), ("short", 0), ("first", 1), ("multiple", 2)],
)
def test_recall_header_boundaries(plugin_import, models, boundary, expected):
    # Given budgets derived from the current single-line header and complete items.
    recall = plugin_import("core.recall")
    header_length = len(recall._HEADER)
    line_length = len("- [1] xxxxxxxxxx")
    budgets = {
        "below": header_length - 1,
        "header": header_length,
        "short": header_length + line_length,
        "first": header_length + 1 + line_length,
        "multiple": header_length + 2 * (1 + line_length),
    }
    # When formatting two memories at each exact boundary.
    text, count = recall.format_recall(
        [models.SearchHit(i, "x" * 10, 1, None) for i in [1, 2]],
        max_chars=budgets[boundary],
    )
    # Then a header is emitted only with complete fitting memories.
    assert count == expected
    assert len(text) <= budgets[boundary]
    if expected:
        assert len(text.splitlines()) == expected + 1
        assert text.splitlines()[1] == "- [1] xxxxxxxxxx"
    else:
        assert text == ""


@pytest.mark.parametrize("attempt", [1, 4, 5])
@pytest.mark.parametrize("hit_count", [0, 2])
def test_search_guidance_branches(plugin_import, attempt, hit_count):
    # Given a search attempt and its returned hit count.
    recall = plugin_import("core.recall")
    # When deriving guidance for the caller's tool.
    guidance = recall.search_guidance("example_search", attempt, hit_count)
    # Then the next attempt is arithmetic, and exhaustion prevents a sixth search.
    assert f"Search attempt {attempt} of 5." in guidance
    assert ("Verify each memory is relevant" in guidance) == bool(hit_count)
    if attempt < 5:
        assert f"call example_search with attempt={attempt + 1}" in guidance
        assert "rewrite the query" in guidance.lower()
    else:
        assert "Search limit reached" in guidance
        assert "attempt=6" not in guidance
        assert "rewrite the query" not in guidance.lower()
