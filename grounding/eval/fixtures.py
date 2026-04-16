"""Fixture schema and loader for the retrieval evaluation harness (Story 16.1).

Parses a YAML fixture file into typed dataclasses and validates structure with
actionable errors. Performs no corpus or FAISS I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml

SCHEMA_VERSION = 1
_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+/ch_\d{4}$")


class FixtureValidationError(Exception):
    """Raised when a fixture file fails schema validation."""

    def __init__(
        self,
        path: Path,
        reason: str,
        *,
        item_id: str | None = None,
        field: str | None = None,
    ) -> None:
        self.path = path
        self.item_id = item_id
        self.field = field
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        parts = []
        if self.item_id is not None:
            parts.append(f"item={self.item_id}")
        if self.field is not None:
            parts.append(f"field={self.field}")
        context = f" [{' '.join(parts)}]" if parts else ""
        return f"{self.path}{context}: {self.reason}"


class UnknownAgentError(FixtureValidationError):
    """Raised when the agent named in a fixture has no matching agents/<name>.yaml."""


@dataclass(frozen=True)
class Expected:
    doc_ids: Tuple[str, ...]
    chunk_ids: Tuple[str, ...] = ()
    page: int | Tuple[int, int] | None = None
    section: str | None = None


@dataclass(frozen=True)
class FixtureItem:
    id: str
    query: str
    expected: Expected
    tags: Tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class FixtureSet:
    agent: str
    version: int
    items: Tuple[FixtureItem, ...]
    source_path: Path


def load_fixtures(path: Path, *, agents_dir: Path) -> FixtureSet:
    """Load and validate a fixture YAML file.

    Args:
        path: Path to the fixture YAML file.
        agents_dir: Directory holding agent YAML files. The fixture's `agent`
            field must resolve to `{agents_dir}/{agent}.yaml`.

    Returns:
        A frozen FixtureSet.

    Raises:
        FixtureValidationError: Schema or structural problems, with `.path`,
            `.item_id`, `.field`, and `.reason` attributes.
        UnknownAgentError: Fixture's agent has no matching YAML in agents_dir.
    """
    path = Path(path)
    if not path.exists():
        raise FixtureValidationError(path, "fixture file not found")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FixtureValidationError(path, f"malformed YAML: {exc}") from exc

    if raw is None:
        raise FixtureValidationError(path, "fixture file is empty")
    if not isinstance(raw, dict):
        raise FixtureValidationError(path, "top-level YAML must be a mapping")

    agent = _require_str(raw, "agent", path)
    version = _require_int(raw, "version", path)
    if version != SCHEMA_VERSION:
        raise FixtureValidationError(
            path,
            f"unsupported version {version}; expected {SCHEMA_VERSION}",
            field="version",
        )

    agent_yaml = Path(agents_dir) / f"{agent}.yaml"
    if not agent_yaml.exists():
        raise UnknownAgentError(
            path,
            f"agent '{agent}' not found at {agent_yaml}",
            field="agent",
        )

    raw_items = raw.get("items")
    if raw_items is None:
        raise FixtureValidationError(path, "missing required field", field="items")
    if not isinstance(raw_items, list):
        raise FixtureValidationError(path, "must be a list", field="items")
    if not raw_items:
        raise FixtureValidationError(path, "must be non-empty", field="items")

    items: list[FixtureItem] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        item = _parse_item(raw_item, index, path)
        if item.id in seen_ids:
            raise FixtureValidationError(
                path,
                "duplicate item id",
                item_id=item.id,
                field="id",
            )
        seen_ids.add(item.id)
        items.append(item)

    return FixtureSet(
        agent=agent,
        version=version,
        items=tuple(items),
        source_path=path,
    )


def _parse_item(raw: object, index: int, path: Path) -> FixtureItem:
    fallback_id = f"<index {index}>"
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path, "item must be a mapping", item_id=fallback_id
        )

    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise FixtureValidationError(
            path,
            "missing or empty required field",
            item_id=fallback_id,
            field="id",
        )

    query = raw.get("query")
    if not isinstance(query, str) or not query.strip():
        raise FixtureValidationError(
            path,
            "missing or empty required field",
            item_id=item_id,
            field="query",
        )

    expected_raw = raw.get("expected")
    if not isinstance(expected_raw, dict):
        raise FixtureValidationError(
            path,
            "missing required field",
            item_id=item_id,
            field="expected",
        )
    expected = _parse_expected(expected_raw, item_id, path)

    tags = _parse_str_list(raw.get("tags", []), item_id, "tags", path)
    notes_raw = raw.get("notes", "")
    if not isinstance(notes_raw, str):
        raise FixtureValidationError(
            path, "must be a string", item_id=item_id, field="notes"
        )

    return FixtureItem(
        id=item_id,
        query=query,
        expected=expected,
        tags=tuple(tags),
        notes=notes_raw,
    )


def _parse_expected(raw: dict, item_id: str, path: Path) -> Expected:
    doc_ids = _parse_str_list(
        raw.get("doc_ids"), item_id, "expected.doc_ids", path, required=True
    )
    if not doc_ids:
        raise FixtureValidationError(
            path, "must be non-empty", item_id=item_id, field="expected.doc_ids"
        )

    chunk_ids_raw = raw.get("chunk_ids", [])
    chunk_ids = _parse_str_list(
        chunk_ids_raw, item_id, "expected.chunk_ids", path
    )
    for chunk_id in chunk_ids:
        if not _CHUNK_ID_RE.match(chunk_id):
            raise FixtureValidationError(
                path,
                f"chunk id '{chunk_id}' does not match <doc_id>/ch_NNNN",
                item_id=item_id,
                field="expected.chunk_ids",
            )

    page = _parse_expected_page(raw.get("page"), item_id, path)
    section = _parse_expected_section(raw.get("section"), item_id, path)

    return Expected(
        doc_ids=tuple(doc_ids),
        chunk_ids=tuple(chunk_ids),
        page=page,
        section=section,
    )


def _parse_expected_page(
    raw: object, item_id: str, path: Path
) -> int | Tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise FixtureValidationError(
            path,
            "must be a positive int or a [start, end] pair of positive ints",
            item_id=item_id,
            field="expected.page",
        )
    if isinstance(raw, int):
        if raw < 1:
            raise FixtureValidationError(
                path,
                f"must be a positive int, got {raw}",
                item_id=item_id,
                field="expected.page",
            )
        return raw
    if isinstance(raw, list):
        if len(raw) != 2:
            raise FixtureValidationError(
                path,
                f"range must have exactly 2 elements, got {len(raw)}",
                item_id=item_id,
                field="expected.page",
            )
        start, end = raw
        for value in (start, end):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise FixtureValidationError(
                    path,
                    "range entries must be positive ints",
                    item_id=item_id,
                    field="expected.page",
                )
        if start > end:
            raise FixtureValidationError(
                path,
                f"range start {start} must be <= end {end}",
                item_id=item_id,
                field="expected.page",
            )
        return (int(start), int(end))
    raise FixtureValidationError(
        path,
        "must be a positive int or a [start, end] pair of positive ints",
        item_id=item_id,
        field="expected.page",
    )


def _parse_expected_section(
    raw: object, item_id: str, path: Path
) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise FixtureValidationError(
            path,
            "must be a non-empty string",
            item_id=item_id,
            field="expected.section",
        )
    return raw


def _parse_str_list(
    raw: object,
    item_id: str,
    field_name: str,
    path: Path,
    *,
    required: bool = False,
) -> list[str]:
    if raw is None:
        if required:
            raise FixtureValidationError(
                path,
                "missing required field",
                item_id=item_id,
                field=field_name,
            )
        return []
    if not isinstance(raw, list):
        raise FixtureValidationError(
            path, "must be a list", item_id=item_id, field=field_name
        )
    result: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise FixtureValidationError(
                path,
                "entries must be non-empty strings",
                item_id=item_id,
                field=field_name,
            )
        result.append(entry)
    return result


def _require_str(raw: dict, key: str, path: Path) -> str:
    if key not in raw:
        raise FixtureValidationError(path, "missing required field", field=key)
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise FixtureValidationError(
            path, "must be a non-empty string", field=key
        )
    return value


def _require_int(raw: dict, key: str, path: Path) -> int:
    if key not in raw:
        raise FixtureValidationError(path, "missing required field", field=key)
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixtureValidationError(path, "must be an integer", field=key)
    return value
