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
