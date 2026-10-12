"""Immutable contracts parsed from ReflectLog structured tool responses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Literal, TypeGuard

ErrorCode = Literal[
    "unavailable",
    "timeout",
    "outcome_unknown",
    "tool_error",
    "unauthorized",
    "incompatible",
    "malformed",
]
ERROR_CODES: Final[frozenset[ErrorCode]] = frozenset(
    {
        "unavailable",
        "timeout",
        "outcome_unknown",
        "tool_error",
        "unauthorized",
        "incompatible",
        "malformed",
    }
)


class ReflectLogError(Exception):
    """A safe client message with optional, separately held server detail."""

    def __init__(
        self, code: ErrorCode, message: str, *, server_detail: str | None = None
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError("Unknown ReflectLog error code")
        super().__init__(message)
        self.code = code
        self.server_detail = server_detail


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    memory: str
    created_at: str | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: int
    memory: str
    score: float
    created_at: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    score_kind: str
    limit: int


@dataclass(frozen=True, slots=True)
class MemoryPage:
    records: tuple[MemoryRecord, ...]
    total: int
    offset: int
    limit: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class Replacement:
    old_memory: str
    new_memory: str
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class AddResult:
    stored_count: int
    skipped_count: int
    replaced_count: int
    replacements: tuple[Replacement, ...]
    dry_run: bool


@dataclass(frozen=True, slots=True)
class DeleteResult:
    removed: tuple[int, ...]
    missing: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WriteJob:
    workspace_id: str
    memories: tuple[str, ...]
    source: str


def _is_dict(value: object) -> TypeGuard[dict[str, object]]:
    # The exact key check in _fields completes the string-key validation.
    return isinstance(value, dict)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _fields(value: object, keys: tuple[str, ...]) -> dict[str, object]:
    # Extra fields are ignored so a newer server can add fields without breaking us.
    if not _is_dict(value) or not value.keys() >= set(keys):
        raise ReflectLogError("malformed", "Missing structured response fields")
    return value


def _array(value: object, field: str) -> list[object]:
    if not _is_list(value):
        raise ReflectLogError("malformed", f"Expected an array for {field}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ReflectLogError("malformed", f"Expected a string for {field}")
    return value


def _timestamp(value: object) -> str | None:
    return None if value is None else _string(value, "created_at")


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReflectLogError("malformed", f"Expected a valid integer for {field}")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReflectLogError("malformed", f"Expected a number for {field}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ReflectLogError("malformed", f"Expected a finite {field}") from error
    if not isfinite(number):
        raise ReflectLogError("malformed", f"Expected a finite {field}")
    return number


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ReflectLogError("malformed", f"Expected a boolean for {field}")
    return value


def parse_memory_record(value: object) -> MemoryRecord:
    fields = _fields(value, ("id", "memory", "created_at"))
    return MemoryRecord(
        _integer(fields["id"], "id", 1),
        _string(fields["memory"], "memory"),
        _timestamp(fields["created_at"]),
    )


def parse_search_hit(value: object) -> SearchHit:
    fields = _fields(value, ("id", "memory", "score", "created_at"))
    return SearchHit(
        _integer(fields["id"], "id", 1),
        _string(fields["memory"], "memory"),
        _number(fields["score"], "score"),
        _timestamp(fields["created_at"]),
    )


def parse_search_result(value: object) -> SearchResult:
    fields = _fields(value, ("results", "score_kind", "limit"))
    return SearchResult(
        tuple(parse_search_hit(item) for item in _array(fields["results"], "results")),
        _string(fields["score_kind"], "score_kind"),
        _integer(fields["limit"], "limit", 1),
    )


def parse_memory_page(value: object) -> MemoryPage:
    fields = _fields(value, ("records", "total", "offset", "limit", "truncated"))
    return MemoryPage(
        tuple(
            parse_memory_record(item) for item in _array(fields["records"], "records")
        ),
        _integer(fields["total"], "total"),
        _integer(fields["offset"], "offset"),
        _integer(fields["limit"], "limit"),
        _boolean(fields["truncated"], "truncated"),
    )


def parse_get_result(value: object) -> MemoryRecord | None:
    fields = _fields(value, ("record",))
    record = fields["record"]
    return None if record is None else parse_memory_record(record)


def parse_replacement(value: object) -> Replacement:
    fields = _fields(value, ("old_memory", "new_memory", "confidence", "reason"))
    return Replacement(
        _string(fields["old_memory"], "old_memory"),
        _string(fields["new_memory"], "new_memory"),
        _number(fields["confidence"], "confidence"),
        _string(fields["reason"], "reason"),
    )


def parse_add_result(value: object) -> AddResult:
    fields = _fields(
        value,
        ("stored_count", "skipped_count", "replaced_count", "replacements", "dry_run"),
    )
    return AddResult(
        _integer(fields["stored_count"], "stored_count"),
        _integer(fields["skipped_count"], "skipped_count"),
        _integer(fields["replaced_count"], "replaced_count"),
        tuple(
            parse_replacement(item)
            for item in _array(fields["replacements"], "replacements")
        ),
        _boolean(fields["dry_run"], "dry_run"),
    )


def parse_delete_result(value: object) -> DeleteResult:
    fields = _fields(value, ("removed", "missing"))
    return DeleteResult(
        tuple(
            _integer(item, "removed id", 1)
            for item in _array(fields["removed"], "removed")
        ),
        tuple(
            _integer(item, "missing id", 1)
            for item in _array(fields["missing"], "missing")
        ),
    )
