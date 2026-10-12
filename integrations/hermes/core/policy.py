from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from collections.abc import Sequence

_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.ASCII)
    for pattern in (
        r"\b(ignore|disregard|forget|override)\b.{0,60}\b(all|any|previous|above|prior|earlier|system|developer)\b.{0,30}\binstructions?\b",
        r"do not follow (the )?(system|developer)",
        r"system prompt",
        r"developer message",
        r"<\s*(system|assistant|developer|tool|function|relevant-memories)\b",
        r"\b(run|execute|call|invoke)\b.{0,40}\b(tool|command)\b",
    )
)


class PolicyError(ValueError):
    def __init__(
        self,
        message: str,
        indices: tuple[int, ...] = (),
        *,
        code: str = "invalid_input",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.indices = indices


def normalize(text: str) -> str:
    return unicodedata.normalize(
        "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def looks_like_prompt_injection(text: str) -> bool:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return any(pattern.search(collapsed) for pattern in _INJECTION_PATTERNS)


def validate_store_batch(items: Sequence[object]) -> tuple[str, ...]:
    if isinstance(items, (str, bytes)) or not 1 <= len(items) <= 20:
        raise PolicyError("memories: expected 1 to 20 items")
    normalized: list[str] = []
    invalid: list[int] = []
    injections: list[int] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, str):
            invalid.append(index)
            continue
        text = normalize(item)
        normalized.append(text)
        if not text or len(text) > 30720:
            invalid.append(index)
        if looks_like_prompt_injection(text):
            injections.append(index)
    if invalid:
        raise PolicyError("memories: invalid items", tuple(invalid))
    if injections:
        raise PolicyError("memories: prompt injection rejected", tuple(injections))
    if sum(map(len, normalized)) > 500000:
        raise PolicyError("memories: aggregate character limit exceeded")
    return tuple(normalized)


def check_auto_write(text: str) -> str | None:
    try:
        return validate_store_batch([text])[0]
    except PolicyError:
        return None


def recall_query(text: str) -> str | None:
    return normalize(text)[:1000] or None


def validate_search_query(text: str) -> str:
    result = normalize(text)
    if not result or len(result) > 1000:
        raise PolicyError("query: expected 1 to 1000 characters")
    return result


def build_turn_record(
    user: str, assistant: str, *, platform: str, now: datetime
) -> str | None:
    user_text = normalize(user)
    if not user_text:
        return None
    assistant_text = normalize(assistant)
    user_text = user_text[:1999] + "…" if len(user_text) > 2000 else user_text
    assistant_text = (
        assistant_text[:1999] + "…" if len(assistant_text) > 2000 else assistant_text
    )
    instant = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    return (
        f"Conversation turn (Hermes {platform}, {instant.isoformat()}):\n"
        f"User: {user_text}\nAssistant: {assistant_text}"
    )
