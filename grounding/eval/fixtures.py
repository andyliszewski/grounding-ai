"""Fixture schema and loader for the retrieval evaluation harness (Story 16.1).

Parses a YAML fixture file into typed dataclasses and validates structure with
actionable errors. Performs no corpus or FAISS I/O.

Story 25.1 adds two optional pieces used by the grounded-answer benchmark
(``grounding eval-answers``) without changing what the retrieval harness reads:

* ``items[].answer``: category, gold answer, optional numeric value with a
  tolerance, optional ``must_include`` facts, and ``answerable`` (default true).
  Items with ``answerable: false`` may leave ``expected.doc_ids`` empty and
  carry no page; the retrieval runner skips them. They must say why they are
  unanswerable in ``unanswerable_kind``: ``no_source`` (no source answers the
  question as asked: a false premise, missing information, or unpublished
  data) or ``not_in_corpus`` (answered elsewhere, but not by the corpus).
* top-level ``page_offsets``: ``doc_id`` to integer, used to map a printed page
  number to a PDF page index (``pdf_page = printed_page + offset``), or to the
  string ``section`` for a section-paged work (a handbook paged "5-20"), whose
  printed pages cannot be mapped at all.
* top-level ``editions``: ``doc_id`` to a positive integer, declaring a
  document's edition when its name does not carry one ("10th Edition" or
  "10e"). A citation that states an edition is only checked against a
  document whose edition is known; otherwise it is unresolvable.
* top-level ``revisions``: ``doc_id`` to a revision letter ("B", "Rev. C"),
  a numbered revision ("Rev. 1") or a year (2016), for documents whose names
  do not carry one ("NASA-STD-5001B", "ISO 13485:2016"). Checked like
  editions.
* top-level ``identifiers``: ``doc_id`` to a document identifier such as
  "DOE-HDBK-1018-93" or "NASA-STD-5001B", for documents whose names do not
  carry it. A citation that names an identifier only matches a document with
  that identifier.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple

import yaml

SCHEMA_VERSION = 1
_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+/ch_\d{4}$")

#: Allowed values for ``items[].answer.category`` (Epic 25, Story 25.1).
ANSWER_CATEGORIES: Tuple[str, ...] = (
    "table",
    "formula",
    "standard",
    "guidance",
    "judgment",
    "unanswerable",
)
#: Allowed values for ``items[].answer.unanswerable_kind``, required when
#: ``answerable: false``. ``no_source`` items are scored in every condition;
#: ``not_in_corpus`` items only in grounded ones, because a model without the
#: corpus may know the answer from elsewhere.
UNANSWERABLE_KINDS: Tuple[str, ...] = ("no_source", "not_in_corpus")
#: ``page_offsets`` value marking a document as section-paged ("5-20").
SECTION_PAGED = "section"
#: Allowed ``source_license`` values. ``public_domain`` (US government work)
#: is the opt-in that lets ``--publishable --include-source-text`` keep
#: passages, transcripts and blind CSVs; the default strips them (D7).
PUBLIC_DOMAIN = "public_domain"
SOURCE_LICENSES: Tuple[str, ...] = (PUBLIC_DOMAIN, "restricted")
_ANSWER_KEYS = frozenset(
    {"category", "gold", "numeric", "must_include", "answerable", "unanswerable_kind"}
)
_NUMERIC_KEYS = frozenset({"value", "unit", "rel_tol", "abs_tol"})
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_REVISION_VALUE_RE = re.compile(
    r"^\s*(?:rev(?:ision)?\.?\s*)?"
    # A month before a year is how a guidance dates its revision ("September 2023").
    r"(?:(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(?:\d{1,2},\s*)?)?"
    r"(?P<value>(?:19|20)\d{2}|[A-Za-z]{1,2}|\d{1,3})\s*$", re.I
)


def parse_revision_value(value: object) -> Tuple[str, str | int] | None:
    """Read a declared or cited revision: ("revision", "C") or ("year", 2016).

    Accepts a revision letter ("B", "Rev. C", "Revision AA"), a numbered
    revision ("Rev. 1") or a four-digit year (2016 or "2016"). Returns None
    for anything else.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("year", value) if _YEAR_RE.fullmatch(str(value)) else None
    if not isinstance(value, str):
        return None
    match = _REVISION_VALUE_RE.match(value)
    if not match:
        return None
    raw = match.group("value")
    if _YEAR_RE.fullmatch(raw):
        return ("year", int(raw))
    return ("revision", raw.upper())


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
class NumericAnswer:
    """A numeric gold value with exactly one tolerance (Story 25.1).

    ``rel_tol`` is a fraction of ``value`` (0.02 means within 2 percent);
    ``abs_tol`` is in the same unit as ``value``.
    """

    value: float
    unit: str | None = None
    rel_tol: float | None = None
    abs_tol: float | None = None

    def tolerance(self) -> float:
        """Absolute half-width of the accepted band around ``value``."""
        if self.rel_tol is not None:
            return abs(self.value) * self.rel_tol
        return float(self.abs_tol or 0.0)

    def accepts(self, candidate: float) -> bool:
        """True when ``candidate`` lies inside the tolerance band (inclusive)."""
        # A tiny epsilon keeps exact-boundary values (and abs_tol: 0 integer
        # answers) from failing on binary floating point noise.
        return abs(candidate - self.value) <= self.tolerance() + 1e-9 * max(
            1.0, abs(self.value)
        )


@dataclass(frozen=True)
class AnswerSpec:
    """Gold answer block for the grounded-answer benchmark (Story 25.1)."""

    category: str
    gold: str = ""
    numeric: NumericAnswer | None = None
    must_include: Tuple[str, ...] = ()
    answerable: bool = True
    # One of UNANSWERABLE_KINDS when answerable is false, else None.
    unanswerable_kind: str | None = None


@dataclass(frozen=True)
class FixtureItem:
    id: str
    query: str
    expected: Expected
    tags: Tuple[str, ...] = ()
    notes: str = ""
    answer: AnswerSpec | None = None


@dataclass(frozen=True)
class FixtureSet:
    agent: str
    version: int
    items: Tuple[FixtureItem, ...]
    source_path: Path
    # doc_id -> offset, pdf_page = printed_page + offset (Story 25.1, D4), or
    # SECTION_PAGED for a work whose printed pages cannot be mapped.
    # Excluded from hashing so the frozen dataclass stays hashable.
    page_offsets: Mapping[str, int | str] = field(default_factory=dict, hash=False)
    # doc_id -> edition, for documents whose names do not state one.
    editions: Mapping[str, int] = field(default_factory=dict, hash=False)
    # doc_id -> revision letter or year, as written in the fixture.
    revisions: Mapping[str, str | int] = field(default_factory=dict, hash=False)
    # doc_id -> document identifier ("DOE-HDBK-1018-93"), for names without one.
    identifiers: Mapping[str, str] = field(default_factory=dict, hash=False)
    # Who the answer model answers for ("a practicing mechanical engineer");
    # None leaves the benchmark's default wording.
    persona: str | None = None
    # "public_domain" when the corpus behind this fixture may be published
    # (US government documents); None or "restricted" keeps source text local.
    source_license: str | None = None


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

    page_offsets = _parse_page_offsets(raw.get("page_offsets"), path)
    editions = _parse_editions(raw.get("editions"), path)
    revisions = _parse_revisions(raw.get("revisions"), path)
    identifiers = _parse_identifiers(raw.get("identifiers"), path)

    return FixtureSet(
        agent=agent,
        version=version,
        items=tuple(items),
        source_path=path,
        page_offsets=page_offsets,
        editions=editions,
        revisions=revisions,
        identifiers=identifiers,
        persona=_parse_persona(raw.get("persona"), path),
        source_license=_parse_source_license(raw.get("source_license"), path),
    )


def _parse_source_license(raw: object, path: Path) -> str | None:
    """Parse the optional top-level ``source_license`` (Story 25.4, D7).

    ``public_domain`` is the opt-in that lets a publishable report carry
    source text; anything else keeps the default, which strips it.
    """
    if raw is None:
        return None
    if raw not in SOURCE_LICENSES:
        raise FixtureValidationError(
            path,
            f"unknown source_license {raw!r}; expected one of {list(SOURCE_LICENSES)}",
            field="source_license",
        )
    return str(raw)


def _parse_persona(raw: object, path: Path) -> str | None:
    """Parse the optional top-level ``persona`` ("a NASA structures engineer")."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip() or "\n" in raw or len(raw) > 200:
        raise FixtureValidationError(
            path,
            "must be a short one-line noun phrase naming who the answers are for, "
            'such as "a practicing mechanical engineer"',
            field="persona",
        )
    return raw.strip()


def _parse_revisions(raw: object, path: Path) -> dict[str, str | int]:
    """Parse the optional top-level ``revisions`` map (doc_id -> letter or year)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path,
            "must be a mapping of doc_id to a revision letter, 'Rev. C' or a year",
            field="revisions",
        )
    out: dict[str, str | int] = {}
    for doc_id, value in raw.items():
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise FixtureValidationError(
                path, "keys must be non-empty doc_id strings", field="revisions"
            )
        if parse_revision_value(value) is None:
            raise FixtureValidationError(
                path,
                f"revision for '{doc_id}' must be a revision letter ('B', 'Rev. C'), a "
                f"numbered revision ('Rev. 1') or a four-digit year, got {value!r}; use "
                "editions: for a numbered edition",
                field="revisions",
            )
        out[doc_id] = value
    return out


def _parse_identifiers(raw: object, path: Path) -> dict[str, str]:
    """Parse the optional top-level ``identifiers`` map.

    A value is one identifier ("DOE-HDBK-1018-93") or a list of them, for a
    document known by several ("89 FR 7496", "FDA-2021-N-0507"). A list is
    joined with "; ", which is how the citation parser separates designations.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path,
            "must be a mapping of doc_id to a document identifier such as 'NASA-STD-5001B', "
            "or to a list of them",
            field="identifiers",
        )
    out: dict[str, str] = {}
    for doc_id, value in raw.items():
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise FixtureValidationError(
                path, "keys must be non-empty doc_id strings", field="identifiers"
            )
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise FixtureValidationError(
                path,
                f"identifier for '{doc_id}' must be a non-empty string, or a list of them, "
                f"got {value!r}",
                field="identifiers",
            )
        out[doc_id] = "; ".join(v.strip() for v in values)
    return out


def _parse_editions(raw: object, path: Path) -> dict[str, int]:
    """Parse the optional top-level ``editions`` map (doc_id -> positive int)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path, "must be a mapping of doc_id to a positive integer", field="editions"
        )
    editions: dict[str, int] = {}
    for doc_id, edition in raw.items():
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise FixtureValidationError(
                path, "keys must be non-empty doc_id strings", field="editions"
            )
        if isinstance(edition, bool) or not isinstance(edition, int) or edition < 1:
            raise FixtureValidationError(
                path,
                f"edition for '{doc_id}' must be a positive integer, got {edition!r}",
                field="editions",
            )
        editions[doc_id] = edition
    return editions


def _parse_page_offsets(raw: object, path: Path) -> dict[str, int | str]:
    """Parse the optional top-level ``page_offsets`` map.

    Each value is an integer offset (``pdf_page = printed_page + offset``) or
    the string ``section``, which marks a section-paged work: its printed
    pages ("5-20") cannot be mapped to PDF pages, so ungrounded citations to
    it by page are scored unresolvable instead of mapped.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path,
            f"must be a mapping of doc_id to an integer or '{SECTION_PAGED}'",
            field="page_offsets",
        )
    offsets: dict[str, int | str] = {}
    for doc_id, offset in raw.items():
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise FixtureValidationError(
                path, "keys must be non-empty doc_id strings", field="page_offsets"
            )
        if offset == SECTION_PAGED:
            offsets[doc_id] = SECTION_PAGED
            continue
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise FixtureValidationError(
                path,
                f"offset for '{doc_id}' must be an integer, or '{SECTION_PAGED}' "
                f"for a section-paged work, got {offset!r}",
                field="page_offsets",
            )
        offsets[doc_id] = offset
    return offsets


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

    # The answer block is parsed first because it decides how strictly
    # ``expected`` is validated: unanswerable items need no doc_ids or page.
    answer = None
    if "answer" in raw:
        answer = _parse_answer(raw.get("answer"), item_id, path)
    unanswerable = answer is not None and not answer.answerable

    expected_raw = raw.get("expected")
    if expected_raw is None and unanswerable:
        expected_raw = {}
    if not isinstance(expected_raw, dict):
        raise FixtureValidationError(
            path,
            "missing required field",
            item_id=item_id,
            field="expected",
        )
    expected = _parse_expected(
        expected_raw, item_id, path, allow_empty_doc_ids=unanswerable
    )
    if answer is not None and answer.answerable and expected.page is None:
        raise FixtureValidationError(
            path,
            "answerable items with an answer block need the gold PDF page",
            item_id=item_id,
            field="expected.page",
        )

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
        answer=answer,
    )


def _parse_answer(raw: object, item_id: str, path: Path) -> AnswerSpec:
    """Validate an ``items[].answer`` block (Story 25.1)."""
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path, "must be a mapping", item_id=item_id, field="answer"
        )
    unknown = sorted(set(raw) - _ANSWER_KEYS)
    if unknown:
        raise FixtureValidationError(
            path,
            f"unknown key(s) {unknown}; allowed: {sorted(_ANSWER_KEYS)}",
            item_id=item_id,
            field="answer",
        )

    category = raw.get("category")
    if not isinstance(category, str) or not category.strip():
        raise FixtureValidationError(
            path,
            f"missing or empty required field; expected one of {list(ANSWER_CATEGORIES)}",
            item_id=item_id,
            field="answer.category",
        )
    if category not in ANSWER_CATEGORIES:
        raise FixtureValidationError(
            path,
            f"unknown category '{category}'; expected one of {list(ANSWER_CATEGORIES)}",
            item_id=item_id,
            field="answer.category",
        )

    answerable = raw.get("answerable", True)
    if not isinstance(answerable, bool):
        raise FixtureValidationError(
            path, "must be true or false", item_id=item_id, field="answer.answerable"
        )
    if (category == "unanswerable") == answerable:
        raise FixtureValidationError(
            path,
            "category 'unanswerable' and 'answerable: false' must go together",
            item_id=item_id,
            field="answer.answerable",
        )

    kind = raw.get("unanswerable_kind")
    if answerable and kind is not None:
        raise FixtureValidationError(
            path,
            "only unanswerable items (answerable: false) take unanswerable_kind",
            item_id=item_id,
            field="answer.unanswerable_kind",
        )
    if not answerable and kind is None:
        raise FixtureValidationError(
            path,
            "unanswerable items need unanswerable_kind: no_source (no source answers it "
            "as asked) or not_in_corpus (answered elsewhere, not by the corpus)",
            item_id=item_id,
            field="answer.unanswerable_kind",
        )
    if kind is not None and kind not in UNANSWERABLE_KINDS:
        raise FixtureValidationError(
            path,
            f"unknown unanswerable_kind {kind!r}; expected one of {list(UNANSWERABLE_KINDS)}",
            item_id=item_id,
            field="answer.unanswerable_kind",
        )

    gold = raw.get("gold", "")
    if gold is None:
        gold = ""
    if not isinstance(gold, str):
        raise FixtureValidationError(
            path, "must be a string", item_id=item_id, field="answer.gold"
        )
    if answerable and not gold.strip():
        raise FixtureValidationError(
            path,
            "answerable items need a non-empty gold answer",
            item_id=item_id,
            field="answer.gold",
        )

    numeric = None
    if raw.get("numeric") is not None:
        if not answerable:
            raise FixtureValidationError(
                path,
                "unanswerable items cannot carry a numeric gold value",
                item_id=item_id,
                field="answer.numeric",
            )
        numeric = _parse_numeric(raw["numeric"], item_id, path)

    must_include = _parse_str_list(
        raw.get("must_include", []), item_id, "answer.must_include", path
    )
    if must_include and not answerable:
        raise FixtureValidationError(
            path,
            "unanswerable items cannot carry must_include facts",
            item_id=item_id,
            field="answer.must_include",
        )

    return AnswerSpec(
        category=category,
        gold=gold,
        numeric=numeric,
        must_include=tuple(must_include),
        answerable=answerable,
        unanswerable_kind=kind,
    )


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _parse_numeric(raw: object, item_id: str, path: Path) -> NumericAnswer:
    if not isinstance(raw, dict):
        raise FixtureValidationError(
            path, "must be a mapping", item_id=item_id, field="answer.numeric"
        )
    unknown = sorted(set(raw) - _NUMERIC_KEYS)
    if unknown:
        raise FixtureValidationError(
            path,
            f"unknown key(s) {unknown}; allowed: {sorted(_NUMERIC_KEYS)}",
            item_id=item_id,
            field="answer.numeric",
        )
    value = raw.get("value")
    if not _is_number(value):
        raise FixtureValidationError(
            path,
            "numeric answers need a finite number 'value'",
            item_id=item_id,
            field="answer.numeric.value",
        )

    unit = raw.get("unit")
    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise FixtureValidationError(
            path,
            "unit must be a non-empty string when given",
            item_id=item_id,
            field="answer.numeric.unit",
        )

    tolerances = {k: raw.get(k) for k in ("rel_tol", "abs_tol") if raw.get(k) is not None}
    if len(tolerances) != 1:
        raise FixtureValidationError(
            path,
            f"numeric answers need exactly one of rel_tol or abs_tol, got {sorted(tolerances) or 'none'}",
            item_id=item_id,
            field="answer.numeric",
        )
    (tol_name, tol_value), = tolerances.items()
    if not _is_number(tol_value) or float(tol_value) < 0:
        raise FixtureValidationError(
            path,
            f"{tol_name} must be a non-negative number",
            item_id=item_id,
            field=f"answer.numeric.{tol_name}",
        )

    return NumericAnswer(
        value=float(value),
        unit=unit.strip() if isinstance(unit, str) else None,
        rel_tol=float(tol_value) if tol_name == "rel_tol" else None,
        abs_tol=float(tol_value) if tol_name == "abs_tol" else None,
    )


def _parse_expected(
    raw: dict, item_id: str, path: Path, *, allow_empty_doc_ids: bool = False
) -> Expected:
    doc_ids = _parse_str_list(
        raw.get("doc_ids"),
        item_id,
        "expected.doc_ids",
        path,
        required=not allow_empty_doc_ids,
    )
    if not doc_ids and not allow_empty_doc_ids:
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
