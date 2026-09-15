"""Citation extraction and resolution for the answer benchmark (Story 25.3).

Extraction finds two kinds of citation in an answer:

* ``prefix``: a bracketed citation in the ``grounding/citations.py`` formats,
  ``[slug]``, ``[slug, p.N]``, ``[slug, p.N-M]`` (en dash or hyphen),
  ``[slug, §Section]`` or ``[slug, p.N, §Section]``. Several may share one
  bracket, separated by semicolons.
* ``free_text``: a bracketed or parenthesized group that names a work and a
  location, such as ``(Shigley's Mechanical Engineering Design, 10th ed.,
  p. 250)``. It needs a location marker (page, section sign, clause, table,
  figure, chapter or edition) and a title with at least one real word.

Resolution puts every citation in exactly one bucket (spec, metric 2):

* grounded conditions: the citation must match a chunk that a tool call in
  the same transcript returned, or it is ``invented``. A match goes on to the
  support judge (``verified`` or ``unsupported``). Before a citation is
  declared ``invented``: a slug that names no returned document is
  fuzzy-matched against the documents this transcript returned (the model
  dropped a sort prefix or an edition token), and a citation whose page
  matches a returned chunk but whose section does not is ``located`` with a
  ``section_mismatch`` note.
* both conditions: a citation that names a table, figure, equation or
  clause label ("Table A-20", "Eq. (5-19)", "Clause 7.1", "para. 3.2") and
  whose document resolves is located by searching for the label (match type
  ``label``), independent of page mapping. A caption (the label opening a
  line or a section heading) locates it; a single clause number needs its
  keyword ("Section 5"), a dotted one does not ("7.1 Planning"). In grounded
  conditions only the chunks a tool call returned are searched: a label that
  is only in the rest of the document is ``invented``
  (``label_outside_returned_chunks``), and a passing mention inside a
  returned chunk resolves with a ``label_mention_only`` note. In the
  ungrounded condition the whole cited document is searched, and a label
  found only in passing does not locate text (``label_mention_only``).
* ungrounded condition: the title is fuzzy-matched to a document of the
  agent's index on ``slug`` and ``orig_name``; no match is ``unresolvable``.
  A cited version (edition, revision letter or year) must equal the
  document's (``edition_mismatch``), and a document whose version is unknown
  cannot confirm one (``edition_unknown``); both are ``unresolvable``, so a
  wrong edition's page is never mapped into another edition. A citation that
  names a document identifier (``DOE-HDBK-1018-93``, ``NASA-STD-5001B``,
  ``21 CFR 820``) only matches a document carrying that identifier, because a
  different number is a different document; a document whose name carries no
  identifier cannot confirm one (``identifier_unknown``, fixable with the
  fixture's ``identifiers:`` map). A section-paged work
  (``page_offsets: {doc: section}``) is ``unresolvable`` by page, and its
  labels are the way its text is reached. The printed page maps to a PDF page through
  ``page_offsets``; no offset is ``unresolvable`` (never guessed, D4). A
  mapped page outside the work is ``invented``; a page inside the work with
  no extracted text is ``unresolvable`` (``no_text_at_page``). Otherwise the
  text at that page goes to the support judge.

Title matching (``match_title``) scores the whole cited title and each of its
comma, semicolon or colon separated segments against every document, and a
candidate passes one of three rules:

* ``eponym``: the whole title, at least ``TITLE_MATCH_THRESHOLD`` of its words
  are in the document's names, and its first word is the first word of the
  document's title ("Roark", "Shigley's Mechanical Engineering Design");
* ``precision_recall``: at least ``TITLE_MATCH_THRESHOLD`` of the candidate's
  words are in the document's names and it covers at least
  ``TITLE_RECALL_THRESHOLD`` of the document's title words. This is what
  lets the "Shigley's Mechanical Engineering Design" segment of
  "Budynas & Nisbett, Shigley's Mechanical Engineering Design" match, while
  a short segment such as "ASME" cannot match a long standard's name;
* ``recall``: the candidate covers at least ``RECALL_PATH_RECALL`` of the
  document's title words, in order, and at least ``RECALL_PATH_PRECISION``
  of its own words are explained (an author prefix with no comma).

Document title words drop numbering noise (pure numbers, edition tokens such
as ``10e``), so numbers are matched as identifiers rather than as words. The
best-scoring document (F1 of precision and recall) wins; ties are ambiguous
unless the cited version separates them.

This module does no API calls; the judge step lives in ``scoring.py``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from grounding.citations import _derive_slug, format_citation_prefix
from grounding.eval.fixtures import SECTION_PAGED, parse_revision_value

# "partial": a grounded citation that named a returned document but gave no page or
# section, and whose text supports the claim. It is reported on its own and never
# counted as verified, so the headline verified rate only credits findable locations.
BUCKETS = ("verified", "partial", "unsupported", "invented", "unresolvable")

TITLE_MATCH_THRESHOLD = 0.75  # share of the cited title's words found in the doc's names
TITLE_RECALL_THRESHOLD = 0.6  # share of the doc's title words a segment must cover
RECALL_PATH_RECALL = 0.75  # recall rule: the candidate covers most of the doc's title...
RECALL_PATH_PRECISION = 0.6  # ...and most of its own words are explained
TOKEN_MATCH_RATIO = 0.85
MAX_PASSAGE_CHARS = 8000

_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2212"
# One level of nested parentheses is allowed inside a parenthesized citation,
# so "(Shigley's ..., Eq. (5-19), p. 250)" is one citation, not zero.
_GROUP_RE = re.compile(
    r"(?<!!)\[([^\[\]\n]{1,400})\](?!\()"
    r"|\(((?:[^()\n]|\([^()\n]{0,60}\)){1,400})\)"
)
_PREFIX_RE = re.compile(
    r"^(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)"
    rf"(?:\s*,\s*pp?\.\s*(?P<ps>\d+)(?:\s*[{_DASHES}]\s*(?P<pe>\d+))?)?"
    r"(?:\s*,\s*§\s*(?P<section>.+?))?\s*$"
)
_PAGE_RE = re.compile(
    rf"\b(?:pp?\.|pages?)\s*(?P<ps>\d+)(?:\s*[{_DASHES}]\s*(?P<pe>\d+))?", re.I
)
_SECTION_RE = re.compile(r"§\s*(?P<section>[^,;]+)")
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_REVISION_RE = re.compile(
    r"\brev(?:ision)?\.?\s*(?P<rev>(?:19|20)\d{2}\b|[A-Za-z]{1,2}\b|\d{1,3}\b)", re.I
)
# A segment that states only a year ("2015", "2016 edition", "April 2016").
_YEAR_SEGMENT_RE = re.compile(
    r"^(?:[A-Za-z]+\.?\s+)?(?:19|20)\d{2}\s*(?:ed\b\.?|edition|printing|revision|rev\b\.?)?$",
    re.I,
)
_EDITION_RE = re.compile(
    r"\b(?:(?P<num>\d{1,2})(?:st|nd|rd|th)|(?P<word>" + "|".join(_ORDINALS) + r"))"
    r"[\s\-_]*(?:ed\b\.?|edition\b)|\bed(?:ition)?\.?\s*(?P<num2>\d{1,2})\b",
    re.I,
)
# Document names only (never cited text): the "10e" / "27e" / "2e" convention, as
# in "machinerys-handbook-27e". A whole token, and not a mantissa: "1e-3" is
# scientific notation (a 1-3 digit exponent follows), while "27e-2004" is an
# edition followed by a year.
_DOC_EDITION_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?P<num>\d{1,2})e(?![A-Za-z0-9])(?![\s_+\-]*[+\-]?\d{1,3}(?!\d))"
)
_EDITION_TOKEN_RE = re.compile(r"\d{1,2}e")
_LOCATION_RE = re.compile(
    r"§|\b(?:pp?\.\s*\d|pages?\s+\d|clause\s+\d|section\s+\d|para(?:graph)?\.?\s*\d|"
    r"table\s+[\dA-Z]|fig(?:ure)?\.?\s*\d|eq(?:uation)?\.?\s*\(?\d|ch(?:apter)?\.?\s*\d)",
    re.I,
)
_LABEL_SEP = r"[.\-\u2010-\u2015\u2212]"
_LABEL_ID = (
    r"\(\s*[A-Za-z]?\d+(?:\s*" + _LABEL_SEP + r"\s*\d+)*\s*\)"
    r"|[A-Za-z]?\d+(?:\s*" + _LABEL_SEP + r"\s*\d+)*[a-z]?"
    r"|[A-Za-z](?:\s*" + _LABEL_SEP + r"\s*\d+)+"
)
_LABEL_RE = re.compile(
    r"\b(?P<kind>tables?|figs?\.?|figures?|eqs?\.?|equations?|clauses?|sections?"
    r"|paragraphs?|paras?\.?)"
    r"\s*(?:no\.?\s*)?(?P<id>" + _LABEL_ID + r")",
    re.I,
)
_SECTION_NUMBER_RE = re.compile(r"§\s*(?P<id>\d+(?:\.\d+)*)")
_LABEL_HEADS = {
    "table": r"\btables?\s*(?:no\.?\s*)?",
    "figure": r"\bfig(?:ure)?s?\.?\s*",
    "equation": r"\beq(?:uation)?s?\.?\s*",
    "clause": r"(?:\bclauses?\s*|\bsections?\s*|\bparagraphs?\s*|\bparas?\.?\s*|§\s*)",
}
_LEADIN_RE = re.compile(r"^(?:see(?:\s+also)?|per|cf\.?|from|in|according\s+to)\s+", re.I)
_STOPWORDS = {
    "the", "of", "and", "a", "an", "for", "in", "on", "to", "with", "by", "at",
    "ed", "edition", "vol", "volume", "see", "also", "per", "cf", "pp", "p",
    "page", "pages", "pdf", "et", "al",
}
_ABBREVIATIONS = {
    "p.", "pp.", "e.g.", "i.e.", "etc.", "fig.", "eq.", "no.", "vs.", "approx.",
    "ed.", "vol.", "cf.", "al.", "ch.", "sec.", "min.", "max.", "dia.", "ref.",
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Document identifiers and versions
# ---------------------------------------------------------------------------

# Series codes of document identifiers, normalized (lowercase, hyphen-joined),
# mapped to the canonical series an identifier is compared under. Government
# and standards documents are known by these designations, and a different
# number is a different document, so they are identity, not title words:
# DOE-HDBK-1019 is not DOE-HDBK-1018. Add a series here when the corpus gains
# one; a number outside a recognized series is not an identity token.
_ID_SERIES: Dict[str, str] = {
    "doe-hdbk": "doe-hdbk", "doe-std": "doe-std", "doe-spec": "doe-spec",
    "nasa-std": "nasa-std", "nasa-hdbk": "nasa-hdbk", "nasa-spec": "nasa-spec",
    "nasa-sp": "sp", "nasa-rp": "rp", "nasa-tm": "tm", "nasa-tp": "tp", "nasa-cr": "cr",
    "rp": "rp",
    "npr": "npr", "npd": "npd",
    "mil-hdbk": "mil-hdbk", "mil-std": "mil-std", "mil-prf": "mil-prf",
    "mil-dtl": "mil-dtl", "mil-spec": "mil-spec",
    "dod-hdbk": "dod-hdbk", "dod-std": "dod-std", "fed-std": "fed-std",
    "asme": "asme", "iso": "iso", "iec": "iec", "ansi": "ansi", "nfpa": "nfpa",
}
_ID_SERIES_ALT = "|".join(sorted((re.escape(s) for s in _ID_SERIES), key=len, reverse=True))
_ID_RE = re.compile(
    rf"(?<![a-z0-9])(?P<series>{_ID_SERIES_ALT})-(?P<groups>[a-z]?\d+[a-z]?(?:-\d+[a-z]?)*)"
    rf"(?:-rev(?:ision)?-(?P<rev>[a-z]{{1,2}}|\d{{1,3}}))?(?![a-z0-9])"
)
# Designations that do not fit "series then number", each with its own shape:
#
# * CFR parts, either way round: "21 CFR 820", "21 CFR Part 888.3080", "cfr-21-part-888".
#   The title and the part identify it; a section after the part is a location.
# * FDA dockets: "FDA-2013-D-1530", "FDA-2021-N-0507", and the pre-2014 form
#   "2006D-0020". A guidance keeps its docket across revisions, so the docket is
#   identity and the issue date (``revisions:``) is the version.
# * Federal Register: a cite "89 FR 7496" (volume and first page), optionally a
#   range "89 FR 7496 to 7525", and the FR document number "FR Doc. 2024-01709".
# * DOT report numbers: "DOT/FAA/CT-93/69.I", "DOT-VNTSC-FAA-93-13.I".
_CFR_RE = re.compile(
    r"(?<![a-z0-9])(?:(?P<title>\d{1,2})-cfr-(?:part-)?(?P<part>\d{1,4})"
    r"|cfr-(?P<title2>\d{1,2})-(?:part-)?(?P<part2>\d{1,4}))(?!\d)"
)
_FDA_DOCKET_RE = re.compile(
    r"(?<![a-z0-9])(?:fda-(?P<year>\d{4})-(?P<kind>[a-z])-(?P<serial>\d{3,5})"
    r"|(?P<year2>\d{4})(?P<kind2>[a-z])-(?P<serial2>\d{3,5}))(?![a-z0-9])"
)
_FR_CITE_RE = re.compile(
    r"(?<![a-z0-9])(?P<vol>\d{1,3})-fr-(?P<page>\d{2,6})"
    r"(?:-(?:to-|through-)?(?P<end>\d{2,6}))?(?!\d)"
)
_FR_DOC_RE = re.compile(
    r"(?<![a-z0-9])fr-(?:doc-)?(?P<year>(?:19|20)\d{2})-(?P<serial>\d{4,6})(?![a-z0-9])"
)
_FR_DOC_BARE_RE = re.compile(r"^(?P<year>(?:19|20)\d{2})-(?P<serial>\d{5})$")
_DOT_MODES = "faa|vntsc|fhwa|nhtsa|fra|fta|phmsa|marad|tsa"
_DOT_RE = re.compile(
    rf"(?<![a-z0-9])dot-(?P<agency>(?:{_DOT_MODES})(?:-[a-z]{{2,6}}){{0,2}})-"
    r"(?P<num>\d{1,4}(?:-\d{1,4})*)(?:-(?P<part>[ivx]{1,4}|[a-z]))?(?![a-z0-9])"
)
_VERSION_KINDS = ("edition", "revision", "year")


def _normalize(text: str) -> str:
    """Lowercase, with every run of other characters turned into one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


@dataclass(frozen=True)
class Version:
    """Which version of a work a citation or a document name states.

    ``edition`` is a numbered edition ("10th ed.", "10e"), ``revision`` a
    revision letter or number ("Rev. C", the B of NASA-STD-5001B), ``year`` a
    revision year ("ISO 13485:2016", the -93 of DOE-HDBK-1018-93). A work may
    state more than one (a 27th edition printed in 2004).
    """

    edition: int | None = None
    revision: str | None = None
    year: int | None = None

    @property
    def stated(self) -> bool:
        return any(getattr(self, kind) is not None for kind in _VERSION_KINDS)

    def merge(self, other: "Version") -> "Version":
        """``self`` wins per kind; ``other`` fills the kinds it leaves open."""
        return Version(*[getattr(self, k) if getattr(self, k) is not None else getattr(other, k)
                         for k in _VERSION_KINDS])

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in _VERSION_KINDS if getattr(self, k) is not None}


def compare_versions(cited: Version | None, doc: Version) -> str:
    """"unchecked" (nothing stated), "matched", "mismatch" or "unknown".

    Only the kinds the citation states and the document knows are compared, so
    a citation that gives an edition and a year matches a document that knows
    only its edition, and one that gives only a year cannot be checked against
    a document that knows only its edition ("unknown"): a page is never mapped
    into a version the metadata cannot confirm (D9).
    """
    if cited is None or not cited.stated:
        return "unchecked"
    comparable = [k for k in _VERSION_KINDS
                  if getattr(cited, k) is not None and getattr(doc, k) is not None]
    if any(getattr(cited, k) != getattr(doc, k) for k in comparable):
        return "mismatch"
    return "matched" if comparable else "unknown"


@dataclass(frozen=True)
class DocIdentifier:
    """A document designation: DOE-HDBK-1018-93, NASA-STD-5001B, 21 CFR 820,
    FDA-2013-D-1530, 89 FR 7496, DOT/FAA/CT-93/69.I."""

    series: str
    number: Tuple[str, ...]
    version: Version = Version()
    # A Federal Register rule runs over a page range, and citations name a page
    # inside it; a document declared "89 FR 7496 to 7525" covers "89 FR 7520".
    span: Tuple[int, int] | None = None

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        """What makes it the same document; the revision and year are the version."""
        return (self.series, self.number)

    @property
    def text(self) -> str:
        return "-".join((self.series, *self.number)).upper()

    def covers(self, cited: "DocIdentifier") -> bool:
        """Is ``cited`` this same document? Exact, or a page inside an FR range."""
        if self.key == cited.key:
            return True
        if (self.span and self.series == cited.series == "fr"
                and self.number[:1] == cited.number[:1]):
            return self.span[0] <= int(cited.number[1]) <= self.span[1]
        return False


def _identifier_from(series: str, groups: List[str], rev: str | None) -> DocIdentifier | None:
    canonical = _ID_SERIES[series]
    if canonical == "rp" and not re.fullmatch(r"\d{4}", groups[0]):
        return None  # "RP-1228" is a NASA report number; "RP 2A" is something else
    year = revision = None
    if len(groups) > 1 and re.fullmatch(r"(?:19|20)\d{2}", groups[-1]):
        year = int(groups.pop())
    elif len(groups) > 1 and series.startswith("doe-") and re.fullmatch(r"\d{2}", groups[-1]):
        # The DOE convention: DOE-HDBK-1018-93 is the 1993 revision of 1018.
        two = int(groups.pop())
        year = 1900 + two if two >= 50 else 2000 + two
    if not groups:
        return None
    tail = re.fullmatch(r"([a-z]?\d+)([a-z])", groups[-1])
    if tail:
        groups[-1] = tail.group(1)
        revision = tail.group(2).upper()
    if rev:
        revision = rev.upper()
    return DocIdentifier(canonical, tuple(groups), Version(revision=revision, year=year))


def parse_identifiers(text: str) -> List[DocIdentifier]:
    """Document identifiers named in ``text``, in order, without duplicates.

    Each comma or semicolon separated segment is read on its own, so a page or
    a table number after the identifier is never swallowed into it.
    """
    found: List[DocIdentifier] = []
    for segment in re.split(r"[,;]", text or ""):
        normalized = _normalize(segment)
        for match in _CFR_RE.finditer(normalized):
            title = match.group("title") or match.group("title2")
            part = match.group("part") or match.group("part2")
            found.append(DocIdentifier("cfr", (title, part)))
        for match in _FDA_DOCKET_RE.finditer(normalized):
            found.append(DocIdentifier("fda-docket", (
                match.group("year") or match.group("year2"),
                (match.group("kind") or match.group("kind2")).upper(),
                match.group("serial") or match.group("serial2"),
            )))
        for match in _FR_CITE_RE.finditer(normalized):
            page = match.group("page")
            end = match.group("end")
            found.append(DocIdentifier(
                "fr", (match.group("vol"), page),
                span=(int(page), int(end)) if end else None,
            ))
        bare = _FR_DOC_BARE_RE.match(normalized)
        for match in list(_FR_DOC_RE.finditer(normalized)) + ([bare] if bare else []):
            found.append(DocIdentifier("fr-doc", (match.group("year"), match.group("serial"))))
        for match in _DOT_RE.finditer(normalized):
            parts = (*match.group("agency").split("-"), *match.group("num").split("-"))
            if match.group("part"):
                parts += (match.group("part"),)
            found.append(DocIdentifier("dot", parts))
        for match in _ID_RE.finditer(normalized):
            identifier = _identifier_from(
                match.group("series"), match.group("groups").split("-"), match.group("rev")
            )
            if identifier is not None:
                found.append(identifier)
    out: List[DocIdentifier] = []
    for identifier in found:
        if identifier not in out:
            out.append(identifier)
    return out


def parse_cited_version(text: str) -> Version:
    """The edition, revision or year a piece of cited text states, if any."""
    identifiers = parse_identifiers(text)
    stated = Version(_parse_edition(text), _parse_revision(text),
                     _cited_year(text, text, identifiers))
    return stated.merge(_identifier_version(identifiers))


def _year_outside_identifiers(text: str) -> int | None:
    """A year in ``text`` that is not part of a designation.

    The 2013 of FDA-2013-D-1530 is the docket's year, not the guidance's
    revision: a guidance keeps its docket when it is reissued.
    """
    normalized = _normalize(text)
    for pattern in (_ID_RE, _CFR_RE, _FDA_DOCKET_RE, _FR_CITE_RE, _FR_DOC_RE, _DOT_RE):
        normalized = pattern.sub(" ", normalized)
    match = _YEAR_RE.search(normalized)
    return int(match.group()) if match else None


def _identifier_version(identifiers: Sequence[DocIdentifier]) -> Version:
    version = Version()
    for identifier in identifiers:
        version = version.merge(identifier.version)
    return version


@dataclass(frozen=True)
class Label:
    """A table, figure, equation or clause label named in a citation."""

    kind: str  # "table" | "figure" | "equation" | "clause"
    parts: Tuple[str, ...]  # ("A", "20") for "Table A-20"

    @property
    def text(self) -> str:
        joined = ("." if self.kind == "clause" else "-").join(self.parts)
        return f"{self.kind} {joined}"


def _label_kind(raw: str) -> str:
    word = raw.lower()
    if word.startswith("tab"):
        return "table"
    if word.startswith("fig"):
        return "figure"
    if word.startswith("eq"):
        return "equation"
    return "clause"


def extract_labels(text: str) -> List[Label]:
    """Table, figure, equation and clause labels in a citation, in order."""
    found: List[Label] = []
    for match in _LABEL_RE.finditer(text or ""):
        parts = tuple(p.upper() for p in re.findall(r"[A-Za-z]+|\d+", match.group("id")))
        label = Label(_label_kind(match.group("kind")), parts)
        if parts and label not in found:
            found.append(label)
    for match in _SECTION_NUMBER_RE.finditer(text or ""):
        label = Label("clause", tuple(match.group("id").split(".")))
        if label not in found:
            found.append(label)
    return found


def _label_regexes(label: Label) -> Tuple[re.Pattern, re.Pattern, re.Pattern]:
    """(anywhere in a chunk, as a caption or heading, at the start of a section heading).

    A clause caption needs its keyword ("Section 5", "Clause 7.1", "§ 4.2",
    "para. 3.1") or a dotted number ("7.1 Planning", "4.2.3 General"). A
    bare single number never opens a clause on its own: "5 mm bolts" and a
    table row starting "5 " are not section 5.
    """
    id_re = re.escape(label.parts[0])
    for prev, part in zip(label.parts, label.parts[1:]):
        optional = "" if prev.isdigit() and part.isdigit() else "?"
        id_re += r"\s*" + _LABEL_SEP + optional + r"\s*" + re.escape(part)
    tail = r"(?![A-Za-z0-9])(?!\s*" + _LABEL_SEP + r"\s*\d)"
    head = _LABEL_HEADS[label.kind]
    ident = r"\(?\s*" + id_re + r"\s*\)?" if label.kind == "equation" else id_re
    anywhere = [head + ident + tail]
    caption = [r"^[\s#>*|_]*" + head + ident + tail]
    if label.kind == "equation":
        anywhere.append(r"\(\s*" + id_re + r"\s*\)")  # the equation's own tag
        caption.append(r"\(\s*" + id_re + r"\s*\)\s*$")
    bare_number = label.kind == "clause" and len(label.parts) == 1
    if label.kind == "clause" and not bare_number:
        # "7.1 Planning ...": a dotted number followed by a capitalized word
        # opens a clause without its keyword ("4.2 mm" does not).
        caption.append(r"^[\s#>*]*" + id_re + tail + r"[ \t]+(?-i:[A-Z])")
    # A section heading may drop the label word ("A-20 ...", "7.1 ..."), except
    # for a bare clause number, which needs its keyword there too.
    heading = r"^\s*(?:" + head + r")" + ("" if bare_number else "?") + ident + tail
    return (re.compile("|".join(anywhere), re.I),
            re.compile("|".join(caption), re.I | re.M),
            re.compile(heading, re.I))


@dataclass
class Citation:
    cite_id: str
    kind: str  # "prefix" | "free_text"
    text: str
    span: Tuple[int, int]
    claim: str = ""
    slug: str | None = None
    title: str | None = None
    edition: int | None = None
    revision: str | None = None
    year: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    labels: List[Label] = field(default_factory=list)
    identifiers: List[DocIdentifier] = field(default_factory=list)

    @property
    def version(self) -> Version:
        """The edition, revision or year this citation states."""
        return Version(self.edition, self.revision, self.year).merge(
            _identifier_version(self.identifiers)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cite_id": self.cite_id,
            "kind": self.kind,
            "text": self.text,
            "span": list(self.span),
            "claim": self.claim,
            # Claim pairing: how much answer text one citation has to carry.
            "claim_chars": len(self.claim),
            "claim_words": len(re.findall(r"\w+", self.claim)),
            "slug": self.slug,
            "title": self.title,
            "edition": self.edition,
            "revision": self.revision,
            "year": self.year,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "labels": [label.text for label in self.labels],
            "identifiers": [identifier.text for identifier in self.identifiers],
        }


def _content_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("'", "").replace("’", ""))
    return [
        t for t in tokens
        if t not in _STOPWORDS and not re.fullmatch(r"\d+(?:st|nd|rd|th)", t)
    ]


def _parse_edition(text: str) -> int | None:
    """Edition stated in cited text: '10th ed.', 'tenth edition', 'ed. 10'."""
    match = _EDITION_RE.search(text)
    if not match:
        return None
    if match.group("num"):
        return int(match.group("num"))
    if match.group("num2"):
        return int(match.group("num2"))
    return _ORDINALS.get(match.group("word").lower())


def parse_doc_edition(name: str) -> int | None:
    """Edition of a document from its name: '10th Edition' or the '10e' convention."""
    edition = _parse_edition(name.replace("-", " ").replace("_", " "))
    if edition:
        return edition
    match = _DOC_EDITION_RE.search(name)
    return int(match.group("num")) if match else None


def _parse_revision(text: str) -> str | None:
    """Revision stated in text: 'Rev. C', 'Revision 1'. A year is not a revision."""
    match = _REVISION_RE.search(text or "")
    if not match:
        return None
    value = match.group("rev")
    return None if _YEAR_RE.fullmatch(value) else value.upper()


def parse_doc_version(name: str) -> Version:
    """Edition, revision and year of a document from its name.

    '...-10e' or '10th Edition' is the edition, 'NASA-STD-5001B' or 'Rev. C'
    the revision, and a four-digit year in the name (or the DOE '-93'
    convention) the year.
    """
    spaced = name.replace("-", " ").replace("_", " ")
    identifiers = parse_identifiers(name)
    named = Version(
        edition=parse_doc_edition(name),
        revision=_parse_revision(spaced),
        year=_year_outside_identifiers(name),
    )
    # An identifier's own revision and year are the most specific reading of a name.
    return _identifier_version(identifiers).merge(named)


def _title_tokens(name: str) -> List[str]:
    """A document name's title words, without numbering noise (000, 10e, 2018)."""
    return [
        t for t in _content_tokens(name.replace("-", " ").replace("_", " "))
        if len(t) > 1 and not t.isdigit() and not _EDITION_TOKEN_RE.fullmatch(t)
    ]


def _pages(ps: str | None, pe: str | None) -> Tuple[int | None, int | None]:
    if ps is None:
        return None, None
    start = int(ps)
    end = int(pe) if pe else start
    return (start, end) if end >= start else (start, start)


def _parse_prefix(part: str, known_slugs: Iterable[str]) -> Dict[str, Any] | None:
    match = _PREFIX_RE.match(part)
    if not match:
        return None
    slug = match.group("slug")
    if not re.search(r"[a-z]", slug):
        return None  # [1], [23]: footnote numbers, not citations
    has_location = match.group("ps") or match.group("section")
    if not has_location and "-" not in slug and slug not in set(known_slugs):
        return None  # a lone bracketed word such as [sic] or [note]
    ps, pe = _pages(match.group("ps"), match.group("pe"))
    section = match.group("section")
    return {
        "kind": "prefix",
        "slug": slug,
        "page_start": ps,
        "page_end": pe,
        "section": section.strip() if section else None,
        "labels": extract_labels(section or ""),
    }


def _cited_year(part: str, title: str, identifiers: Sequence[DocIdentifier]) -> int | None:
    """The year a citation states: "ISO 13485:2016", ", 2015,", a year in the title.

    Never a page or a label number: only an identifier's own year, a segment
    that states nothing but a year, or a year inside the work's name counts.
    """
    for identifier in identifiers:
        if identifier.version.year is not None:
            return identifier.version.year
    for segment in re.split(r"[,;]", part):
        stripped = segment.strip(" ()[]")
        if _YEAR_SEGMENT_RE.match(stripped):
            return int(_YEAR_RE.search(stripped).group())
    return _year_outside_identifiers(title)


def _parse_free_text(part: str) -> Dict[str, Any] | None:
    location = _LOCATION_RE.search(part)
    edition_match = _EDITION_RE.search(part)
    revision_match = _REVISION_RE.search(part)
    marks = [m for m in (location, edition_match, revision_match) if m is not None]
    if not marks:
        return None
    cut = min(m.start() for m in marks)
    title = _LEADIN_RE.sub("", part[:cut].strip(" ,;:*_\"'“”‘’"))
    title = title.strip(" ,;:*_\"'“”‘’")
    if not any(len(t) >= 3 and re.search(r"[a-z]", t) for t in _content_tokens(title)):
        return None  # no work named, e.g. "(see Table 3)" or "(p. 12)"
    page = _PAGE_RE.search(part)
    section = _SECTION_RE.search(part)
    ps, pe = _pages(page.group("ps"), page.group("pe")) if page else (None, None)
    identifiers = parse_identifiers(part)
    return {
        "kind": "free_text",
        "title": title,
        "edition": _parse_edition(part),
        "revision": _parse_revision(part),
        "year": _cited_year(part, title, identifiers),
        "page_start": ps,
        "page_end": pe,
        "section": section.group("section").strip() if section else None,
        "labels": extract_labels(part),
        "identifiers": identifiers,
    }


def _sentence_spans(text: str, masked: str) -> List[Tuple[int, int]]:
    """Sentence boundaries within lines, computed on text with citations masked."""
    spans: List[Tuple[int, int]] = []
    line_start = 0
    for line in masked.split("\n"):
        line_end = line_start + len(line)
        start = line_start
        for match in re.finditer(r"[.!?](?=\s+\S)", line):
            end = line_start + match.end()
            word = re.findall(r"\S+$", masked[start:end])
            if word and word[0].lower() in _ABBREVIATIONS:
                continue
            if re.search(r"\d\.$", masked[max(start, end - 3):end]) and re.match(
                r"\s*\d", masked[end:]
            ):
                continue
            spans.append((start, end))
            start = end
        spans.append((start, line_end))
        line_start = line_end + 1
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _clean_claim(text: str, spans: Sequence[Tuple[int, int]], lo: int, hi: int) -> str:
    pieces = []
    cursor = lo
    for s, e in sorted(spans):
        if e <= lo or s >= hi:
            continue
        pieces.append(text[cursor:max(cursor, s)])
        cursor = max(cursor, e)
    pieces.append(text[cursor:hi])
    claim = re.sub(r"\s+", " ", "".join(pieces)).strip()
    claim = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", claim)
    claim = re.sub(r"\s+([.,;:!?])", r"\1", claim)
    # A citation introduced by "see", "cf." or "per" leaves that word dangling.
    claim = re.sub(
        r"[,;:]?\s*\b(?:for details,?\s+)?(?:see(?:\s+also)?|cf\.?|per)\s*[.;,:]?$",
        "",
        claim,
        flags=re.I,
    ).rstrip(" ,;:")
    return claim[:600]


def extract_citations(text: str, known_slugs: Iterable[str] = ()) -> List[Citation]:
    """All citations in ``text``, in order of appearance, each with its claim."""
    known = set(known_slugs)
    found: List[Tuple[Tuple[int, int], str, Dict[str, Any]]] = []
    for group in _GROUP_RE.finditer(text):
        bracket = group.group(1) is not None
        content = group.group(1) if bracket else group.group(2)
        for raw_part in content.split(";"):
            part = raw_part.strip()
            if not part:
                continue
            parsed = _parse_prefix(part, known) if bracket else None
            if parsed is None:
                parsed = _parse_free_text(part)
            if parsed is None:
                continue
            shown = f"[{part}]" if bracket else f"({part})"
            found.append((group.span(), shown, parsed))

    group_spans = sorted({span for span, _, _ in found})
    masked = list(text)
    for s, e in group_spans:
        masked[s:e] = ["█"] * (e - s)
    masked_text = "".join(masked)
    sentences = _sentence_spans(text, masked_text)

    citations: List[Citation] = []
    for index, (span, shown, parsed) in enumerate(found, start=1):
        claim = ""
        for pos, (s, e) in enumerate(sentences):
            if s <= span[0] < e:
                claim = _clean_claim(text, group_spans, s, e)
                if len(re.findall(r"\w+", claim)) < 3 and pos > 0:
                    ps_, pe_ = sentences[pos - 1]
                    claim = _clean_claim(text, group_spans, ps_, pe_)
                break
        citations.append(Citation(cite_id=f"c{index}", text=shown, span=span, claim=claim, **parsed))
    return citations


def strip_citations(text: str, citations: Sequence[Citation]) -> str:
    """``text`` with every citation group removed (what the correctness judge reads)."""
    out = text
    for s, e in sorted({c.span for c in citations}, reverse=True):
        out = out[:s].rstrip(" ") + out[e:]
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


# ---------------------------------------------------------------------------
# Corpus lookup
# ---------------------------------------------------------------------------

@dataclass
class ChunkText:
    doc_id: str
    page_start: int | None
    page_end: int | None
    section: str | None
    prefix: str
    body: str
    chunk_id: Any = None


@dataclass
class DocInfo:
    doc_id: str
    slug: str
    orig_name: str
    prefix_slug: str
    chunk_paths: List[str] = field(default_factory=list)
    # From the fixture's top-level ``editions:``; overrides the name.
    declared_edition: int | None = None
    # From ``revisions:`` (a letter or a year) and ``identifiers:``.
    declared_revision: str | None = None
    declared_year: int | None = None
    declared_identifier: str | None = None

    @property
    def names(self) -> List[str]:
        return [n for n in (self.slug, Path(self.orig_name).stem, self.prefix_slug) if n]

    @cached_property
    def tokens(self) -> set:
        """Every content word of every name (what cited words are looked up in)."""
        out: set = set()
        for name in self.names:
            out.update(_content_tokens(name.replace("-", " ").replace("_", " ")))
        return out

    @cached_property
    def title_token_lists(self) -> List[List[str]]:
        """Each name's title words in order, duplicates removed."""
        out: List[List[str]] = []
        for name in self.names:
            words = _title_tokens(name)
            if words and words not in out:
                out.append(words)
        return out

    @cached_property
    def version(self) -> Version:
        """What the fixture declares about this document's version, then its names."""
        declared = Version(self.declared_edition, self.declared_revision, self.declared_year)
        if self.declared_identifier:
            declared = declared.merge(parse_doc_version(self.declared_identifier))
        for name in self.names:
            declared = declared.merge(parse_doc_version(name))
        return declared

    @property
    def edition(self) -> int | None:
        return self.version.edition

    @cached_property
    def identifiers(self) -> List[DocIdentifier]:
        """Identifiers the fixture declares for this document, then its names."""
        found = parse_identifiers(self.declared_identifier or "")
        for name in self.names:
            for identifier in parse_identifiers(name):
                if identifier not in found:
                    found.append(identifier)
        return found

    @cached_property
    def identifier_keys(self) -> set:
        return {i.key for i in self.identifiers}


@dataclass
class TitleMatch:
    doc: DocInfo | None
    score: float
    detail: str
    # The document the title matched before the edition check (set for
    # "matched", "edition_unknown", and a single-document "edition_mismatch").
    candidate: DocInfo | None = None
    rule: str | None = None
    precision: float | None = None
    recall: float | None = None


@dataclass
class _TitleFit:
    score: float  # F1 of precision and recall; ranks documents
    precision: float
    recall: float
    rule: str | None  # None: no rule passed


def _tokens_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    return len(a) >= 4 and len(b) >= 4 and SequenceMatcher(None, a, b).ratio() >= TOKEN_MATCH_RATIO


def _in_order(covered: Sequence[str], cited: Sequence[str]) -> bool:
    position = -1
    for word in covered:
        nxt = next((j for j in range(position + 1, len(cited)) if _tokens_equal(word, cited[j])), None)
        if nxt is None:
            return False
        position = nxt
    return True


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _title_candidates(title: str) -> List[Tuple[List[str], bool]]:
    """The whole cited title plus its comma/semicolon/colon segments, as word lists."""
    whole = _content_tokens(title)
    out = [(whole, True)] if whole else []
    parts = [p for p in re.split(r"[,;:]", title) if p.strip()]
    if len(parts) > 1:
        for part in parts:
            words = _content_tokens(part)
            if words and words != whole and (words, False) not in out:
                out.append((words, False))
    return out


def _fit_candidate(cited: List[str], whole: bool, doc: DocInfo) -> _TitleFit:
    precision = sum(1 for t in cited if _token_in(t, doc.tokens)) / len(cited)
    cited_set = set(cited)
    best_pass: _TitleFit | None = None
    best_any = _TitleFit(0.0, precision, 0.0, None)
    for title in doc.title_token_lists:
        covered = [w for w in title if _token_in(w, cited_set)]
        recall = len(covered) / len(title)
        score = _f1(precision, recall)
        rule = None
        if precision >= TITLE_MATCH_THRESHOLD and recall >= TITLE_RECALL_THRESHOLD:
            rule = "precision_recall"
        elif (recall >= RECALL_PATH_RECALL and precision >= RECALL_PATH_PRECISION
              and len(covered) >= 2 and _in_order(covered, cited)):
            rule = "recall"
        elif whole and precision >= TITLE_MATCH_THRESHOLD and _tokens_equal(cited[0], title[0]):
            rule = "eponym"
        fit = _TitleFit(score, precision, recall, rule)
        if rule and (best_pass is None or score > best_pass.score):
            best_pass = fit
        if score > best_any.score:
            best_any = fit
    return best_pass or best_any


def _word_matches(docs: Sequence[DocInfo], title: str) -> List[Tuple[_TitleFit, DocInfo]]:
    """Documents whose names the cited title matches by one of the word rules, best first."""
    candidates = _title_candidates(title)
    passing: List[Tuple[_TitleFit, DocInfo]] = []
    for doc in docs:
        fits = [_fit_candidate(words, whole, doc) for words, whole in candidates]
        passed = [f for f in fits if f.rule]
        if passed:
            passing.append((max(passed, key=lambda f: f.score), doc))
    passing.sort(key=lambda pair: pair[0].score, reverse=True)
    return passing


def _best_word_score(docs: Sequence[DocInfo], title: str) -> float:
    candidates = _title_candidates(title)
    return max([0.0] + [_fit_candidate(words, whole, doc).score
                        for doc in docs for words, whole in candidates])


def match_title_among(
    docs: Iterable[DocInfo],
    title: str,
    version: Version | int | None = None,
    *,
    identifiers: Sequence[DocIdentifier] | None = None,
) -> TitleMatch:
    """Match a cited title to one of ``docs`` (see the module docstring).

    ``version`` is what the citation states about the edition, revision or
    year (an integer is read as an edition). ``identifiers`` are the document
    designations the citation names; by default they are read from ``title``.
    When a citation names an identifier, only a document carrying that exact
    identifier can match it: DOE-HDBK-1019 is not DOE-HDBK-1018, and
    NASA-STD-5005 Rev. C is not 5005D.
    """
    if isinstance(version, int):
        version = Version(edition=version)
    docs = list(docs)
    cited_ids = parse_identifiers(title) if identifiers is None else list(identifiers)
    version = (version or Version()).merge(_identifier_version(cited_ids))
    if not _title_candidates(title):
        return TitleMatch(None, 0.0, "no_title")
    if cited_ids:
        named = [d for d in docs
                 if any(mine.covers(cited) for mine in d.identifiers for cited in cited_ids)]
        if not named:
            # A document whose name carries no identifier cannot confirm the
            # cited one; declare it under the fixture's identifiers: map.
            unnamed = [d for d in docs if not d.identifier_keys]
            by_words = _word_matches(unnamed, title)
            score = round(by_words[0][0].score, 4) if by_words else 0.0
            if len(by_words) == 1:
                return TitleMatch(None, score, "identifier_unknown", candidate=by_words[0][1],
                                  rule=by_words[0][0].rule)
            detail = "ambiguous_title" if by_words else "work_not_in_corpus"
            return TitleMatch(None, score or round(_best_word_score(docs, title), 4), detail)
        # The identifier decides which documents are candidates; words only rank them.
        by_words = _word_matches(named, title)
        top = by_words[0][0].score if by_words else 0.0
        best = [d for fit, d in by_words if abs(fit.score - top) < 1e-9] or named
        info = dict(score=round(_best_word_score(named, title), 4), rule="identifier")
        return _pick_version(best, version, info)
    passing = _word_matches(docs, title)
    if not passing:
        return TitleMatch(None, round(_best_word_score(docs, title), 4), "work_not_in_corpus")
    top_fit = passing[0][0]
    best = [doc for fit, doc in passing if abs(fit.score - top_fit.score) < 1e-9]
    info = dict(score=round(top_fit.score, 4), rule=top_fit.rule,
                precision=round(top_fit.precision, 4), recall=round(top_fit.recall, 4))
    return _pick_version(best, version, info)


def _pick_version(best: Sequence[DocInfo], version: Version, info: Dict[str, Any]) -> TitleMatch:
    """One document of ``best``, or why the cited version leaves it unresolved."""
    if version.stated:
        verdicts = [(compare_versions(version, d.version), d) for d in best]
        same = [d for verdict, d in verdicts if verdict == "matched"]
        if same:
            best = same
        else:
            unknown = [d for verdict, d in verdicts if verdict == "unknown"]
            if not unknown:
                # Never map a wrong edition's or revision's page into another one.
                return TitleMatch(None, detail="edition_mismatch",
                                  candidate=best[0] if len(best) == 1 else None, **info)
            if len(unknown) == 1:
                return TitleMatch(None, detail="edition_unknown", candidate=unknown[0], **info)
            return TitleMatch(None, detail="ambiguous_title", **info)
    if len(best) > 1:
        return TitleMatch(None, detail="ambiguous_title", **info)
    return TitleMatch(best[0], detail="matched", candidate=best[0], **info)


class CorpusIndex:
    """Documents in the agent's index, with lazy access to their chunk text.

    Only documents present in the index's chunk map count as "in the corpus",
    so ungrounded citations are resolved against exactly what the grounded
    conditions could have retrieved.
    """

    def __init__(
        self,
        corpus_dir: Path,
        embeddings_dir: Path,
        *,
        editions: Mapping[str, int] | None = None,
        revisions: Mapping[str, Any] | None = None,
        identifiers: Mapping[str, str] | None = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        editions = dict(editions or {})
        declared_versions: Dict[str, Tuple[str | None, int | None]] = {}
        for doc_id, raw in dict(revisions or {}).items():
            parsed = parse_revision_value(raw)
            if parsed is None:
                continue  # the fixture loader rejects these; a caller may not have
            kind, value = parsed
            declared_versions[doc_id] = (
                (str(value), None) if kind == "revision" else (None, int(value))
            )
        identifiers = dict(identifiers or {})
        manifest = json.loads((self.corpus_dir / "_index.json").read_text(encoding="utf-8"))
        chunk_map = json.loads((Path(embeddings_dir) / "_chunk_map.json").read_text(encoding="utf-8"))
        entries = chunk_map if isinstance(chunk_map, list) else chunk_map.get("chunks", [])
        paths_by_doc: Dict[str, List[str]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("deleted_utc"):
                continue
            doc_id, path = entry.get("doc_id"), entry.get("file_path")
            if doc_id and path:
                paths_by_doc.setdefault(doc_id, []).append(path)
        self.docs: Dict[str, DocInfo] = {}
        for doc in manifest.get("docs", []):
            doc_id = doc.get("doc_id")
            if doc_id not in paths_by_doc:
                continue
            orig = doc.get("orig_name", "")
            revision, year = declared_versions.get(doc_id, (None, None))
            self.docs[doc_id] = DocInfo(
                doc_id=doc_id,
                slug=doc.get("slug", ""),
                orig_name=orig,
                prefix_slug=_derive_slug(orig) if orig else doc.get("slug", ""),
                chunk_paths=sorted(paths_by_doc[doc_id]),
                declared_edition=editions.get(doc_id),
                declared_revision=revision,
                declared_year=year,
                declared_identifier=identifiers.get(doc_id),
            )
        self._chunks: Dict[str, List[ChunkText]] = {}

    @property
    def prefix_slugs(self) -> set:
        return {d.prefix_slug for d in self.docs.values()} | {d.slug for d in self.docs.values()}

    def chunks(self, doc_id: str) -> List[ChunkText]:
        if doc_id not in self._chunks:
            out = []
            for rel in self.docs[doc_id].chunk_paths:
                try:
                    raw = (self.corpus_dir / rel).read_text(encoding="utf-8")
                except OSError:
                    continue
                meta: Dict[str, Any] = {}
                body = raw
                if raw.startswith("---"):
                    parts = raw.split("---", 2)
                    if len(parts) >= 3:
                        meta = yaml.safe_load(parts[1]) or {}
                        body = parts[2].strip()
                ps, pe = meta.get("page_start"), meta.get("page_end")
                ps = ps if isinstance(ps, int) and not isinstance(ps, bool) else None
                pe = pe if isinstance(pe, int) and not isinstance(pe, bool) else ps
                section = meta.get("section_heading") or None
                source = meta.get("source") or self.docs[doc_id].orig_name
                out.append(
                    ChunkText(
                        doc_id=doc_id,
                        page_start=ps,
                        page_end=pe,
                        section=section,
                        prefix=format_citation_prefix(source, ps, pe, section),
                        body=body,
                        chunk_id=meta.get("chunk_id"),
                    )
                )
            self._chunks[doc_id] = out
        return self._chunks[doc_id]

    def max_page(self, doc_id: str) -> int | None:
        pages = [c.page_end for c in self.chunks(doc_id) if c.page_end is not None]
        return max(pages) if pages else None

    def match_title(
        self,
        title: str,
        version: Version | int | None = None,
        *,
        identifiers: Sequence[DocIdentifier] | None = None,
        restrict_to: Iterable[str] | None = None,
    ) -> TitleMatch:
        """Match a cited title to one document (slug and orig_name).

        ``restrict_to`` limits the search to those doc_ids (for example the
        documents a grounded transcript's tool calls returned).
        """
        docs = self.docs.values()
        if restrict_to is not None:
            allowed = set(restrict_to)
            docs = [d for d in docs if d.doc_id in allowed]
        return match_title_among(docs, title, version, identifiers=identifiers)


def _token_in(token: str, doc_tokens: Iterable[str]) -> bool:
    if token in doc_tokens:
        return True
    if len(token) < 4:
        return False
    return any(_tokens_equal(token, d) for d in doc_tokens)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    bucket: str | None  # "invented" / "unresolvable" now; None -> needs the judge
    detail: str
    match: str | None = None  # "exact" | "located" | "partial" for grounded matches
    doc_id: str | None = None
    pdf_pages: Tuple[int, int] | None = None
    passage_prefixes: List[str] = field(default_factory=list)
    passage: str = ""
    title_score: float | None = None
    # The document the title pointed at when it was refused (edition checks),
    # so a human audit can see what the scorer considered.
    candidate_doc_id: str | None = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "detail": self.detail,
            "match": self.match,
            "doc_id": self.doc_id,
            "pdf_pages": list(self.pdf_pages) if self.pdf_pages else None,
            "passage_prefixes": self.passage_prefixes,
            "passage_chars": len(self.passage),
            "title_score": self.title_score,
            "candidate_doc_id": self.candidate_doc_id,
            "notes": list(self.notes),
        }


def _norm_section(section: str | None) -> str:
    return re.sub(r"\s+", " ", (section or "")).strip().casefold()


def _overlaps(a0: int, a1: int, b0: int | None, b1: int | None) -> bool:
    if b0 is None:
        return False
    return a0 <= (b1 if b1 is not None else b0) and b0 <= a1


def _passage(chunks: Sequence[Mapping[str, Any]]) -> Tuple[List[str], str]:
    prefixes, parts, total = [], [], 0
    for chunk in chunks:
        block = f"{chunk['prefix']}\n{chunk['content']}"
        if total and total + len(block) > MAX_PASSAGE_CHARS:
            break
        prefixes.append(chunk["prefix"])
        parts.append(block[:MAX_PASSAGE_CHARS])
        total += len(block)
    return prefixes, "\n\n".join(parts)


def returned_chunks(tool_calls: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Unique chunks returned by the transcript's tool calls, in first-seen order."""
    seen, out = set(), []
    for call in tool_calls:
        for result in call.get("results") or []:
            key = (result.get("doc_id"), result.get("chunk_id"), result.get("prefix"))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(result))
    return out


def _corpus_chunk_dicts(corpus: CorpusIndex | None, doc_id: str) -> List[Dict[str, Any]]:
    """Every chunk of a document in the index, as passage dicts."""
    if corpus is None or doc_id not in corpus.docs:
        return []
    return [
        {"prefix": c.prefix, "content": c.body, "page_start": c.page_start,
         "page_end": c.page_end, "section": c.section, "doc_id": doc_id}
        for c in corpus.chunks(doc_id)
    ]


def _returned_chunk_dicts(returned: Sequence[Mapping[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
    """The chunks of one document that a tool call returned: all the model saw of it."""
    return [{**dict(r), "section": r.get("section_heading")}
            for r in returned if r.get("doc_id") == doc_id]


def _label_hits(
    chunks: Sequence[Mapping[str, Any]], labels: Sequence[Label],
    pages: Tuple[int, int] | None = None,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """(caption hits, mention-only hits) for ``labels`` among ``chunks``.

    A caption hit is a chunk where the label opens a line or the chunk's
    section heading (the table itself, the clause's heading): that is where
    the label points. A mention-only hit names the label in passing ("see
    Table A-20") and holds no caption. Within each list, chunks on the cited
    page come first, then document order.
    """
    compiled = [_label_regexes(label) for label in labels]
    captions, mentions = [], []
    for order, chunk in enumerate(chunks):
        text, section = str(chunk.get("content") or ""), str(chunk.get("section") or "")
        is_caption = any(c.search(text) or h.search(section) for _, c, h in compiled)
        is_mention = not is_caption and any(a.search(text) for a, _, _ in compiled)
        if not (is_caption or is_mention):
            continue
        on_page = pages is not None and _overlaps(pages[0], pages[1],
                                                  chunk.get("page_start"), chunk.get("page_end"))
        (captions if is_caption else mentions).append(((0 if on_page else 1, order), chunk))
    return ([c for _, c in sorted(captions, key=lambda p: p[0])],
            [c for _, c in sorted(mentions, key=lambda p: p[0])])


def _label_found(citation: Citation, doc_id: str, hits, *, pages, notes: List[str],
                 title_score=None) -> Resolution:
    prefixes, passage = _passage(hits)
    return Resolution(None, "label_found", "label", doc_id, pages, prefixes, passage,
                      title_score=title_score, notes=notes + [f"label:{citation.labels[0].text}"])


def _fuzzy_returned_doc(slug: str, returned: Sequence[Mapping[str, Any]],
                        corpus: CorpusIndex | None) -> str | None:
    """The one returned document an unrecognized slug most plausibly names.

    Only documents this transcript's tool calls returned are candidates, so a
    fuzzy match can never credit a location the model was not shown. A slug
    that exactly names another corpus document is not unrecognized: that
    document simply was not returned. An edition in the slug ("-9e") must
    agree with the document's.
    """
    if corpus is not None and slug in corpus.prefix_slugs:
        return None
    docs: Dict[str, DocInfo] = {}
    for r in returned:
        doc_id = r.get("doc_id")
        if not doc_id or doc_id in docs:
            continue
        known = corpus.docs.get(doc_id) if corpus is not None else None
        docs[doc_id] = known or DocInfo(doc_id=doc_id, slug=r.get("slug") or "",
                                        orig_name=r.get("source") or "",
                                        prefix_slug=r.get("slug") or "")
    if not docs:
        return None
    match = match_title_among(docs.values(), slug, parse_doc_version(slug))
    return match.doc.doc_id if match.doc is not None else None


def _located(citation: Citation, candidates: Sequence[Mapping[str, Any]],
             notes: List[str]) -> List[Mapping[str, Any]]:
    """Returned chunks at the cited page and section; a page match survives a section mismatch."""
    hits = list(candidates)
    if citation.page_start is not None:
        hits = [r for r in hits
                if _overlaps(citation.page_start, citation.page_end or citation.page_start,
                             r.get("page_start"), r.get("page_end"))]
    if citation.section:
        same = [r for r in hits
                if _norm_section(r.get("section_heading")) == _norm_section(citation.section)]
        if same:
            return same
        if citation.page_start is not None and hits:
            # The page takes a reader to the text; the section string is off.
            notes.append("section_mismatch")
            return hits
        return []
    return hits


def resolve_grounded(
    citation: Citation, returned: Sequence[Mapping[str, Any]], corpus: CorpusIndex | None
) -> Resolution:
    """A grounded citation must name a chunk some tool call returned."""
    notes: List[str] = []
    title_score = None
    if citation.kind == "prefix":
        canonical = format_citation_prefix(
            citation.slug or "", citation.page_start, citation.page_end, citation.section
        )
        exact = [r for r in returned if r.get("prefix") == canonical]
        if exact:
            prefixes, passage = _passage(exact)
            return Resolution(None, "matched_returned_chunk", "exact",
                              exact[0].get("doc_id"), None, prefixes, passage)
        candidates = [r for r in returned if r.get("slug") == citation.slug]
        if not candidates:
            fuzzy = _fuzzy_returned_doc(citation.slug or "", returned, corpus)
            if fuzzy is not None:
                candidates = [r for r in returned if r.get("doc_id") == fuzzy]
                notes.append(f"slug_fuzzy_match:{candidates[0].get('slug')}")
    else:
        returned_ids = {r.get("doc_id") for r in returned if r.get("doc_id")}
        match = corpus.match_title(citation.title or "", citation.version,
                                   identifiers=citation.identifiers) if corpus else None
        if match is not None and match.detail == "ambiguous_title":
            # Two corpus documents share the title; only one may have been returned.
            match = corpus.match_title(citation.title or "", citation.version,
                                       identifiers=citation.identifiers,
                                       restrict_to=returned_ids)
        doc = match.doc if match else None
        if match is not None and doc is None and match.detail in ("edition_unknown",
                                                                  "identifier_unknown"):
            # The model read the returned text and may state the version or
            # the designation printed in it; our metadata cannot confirm it,
            # which is not the model's error.
            doc = match.candidate
            notes.append("edition_unchecked" if match.detail == "edition_unknown"
                         else "identifier_unchecked")
        if doc is None:
            detail = ("edition_mismatch" if match and match.detail == "edition_mismatch"
                      else "title_not_a_returned_document")
            return Resolution("invented", detail, title_score=match.score if match else None,
                              candidate_doc_id=match.candidate.doc_id
                              if match and match.candidate else None)
        title_score = match.score
        candidates = [r for r in returned if r.get("doc_id") == doc.doc_id]

    if not candidates:
        return Resolution("invented", "no_returned_chunk_at_location", notes=notes)
    doc_id = candidates[0].get("doc_id")
    has_location = citation.page_start is not None or bool(citation.section)
    if has_location:
        hits = _located(citation, candidates, notes)
        if hits:
            prefixes, passage = _passage(hits)
            # A page or section is enough for a reader to find the text, even
            # without the exact prefix string ("located").
            return Resolution(None, "matched_returned_chunk", "located", doc_id, None,
                              prefixes, passage, title_score=title_score, notes=notes)
    if citation.labels:
        pages = ((citation.page_start, citation.page_end or citation.page_start)
                 if citation.page_start is not None else None)
        # Only what a tool call returned counts: a label is located in the
        # returned chunks of the document, never in the rest of it.
        captions, mentions = _label_hits(_returned_chunk_dicts(returned, doc_id),
                                         citation.labels, pages)
        if captions or mentions:
            if not captions:
                # A passing mention the model was shown: the judge reads that chunk.
                notes.append("label_mention_only")
            return _label_found(citation, doc_id, captions or mentions, pages=pages,
                                notes=notes, title_score=title_score)
        elsewhere = _label_hits(_corpus_chunk_dicts(corpus, doc_id), citation.labels)
        if any(elsewhere):
            # The label is in the document, but in text no tool call returned.
            return Resolution("invented", "label_outside_returned_chunks", doc_id=doc_id,
                              title_score=title_score,
                              notes=notes + [f"label:{citation.labels[0].text}"])
        notes.append("label_not_found")
    if not has_location:
        # Naming only the document is "partial": never counted as verified.
        prefixes, passage = _passage(candidates)
        return Resolution(None, "matched_returned_chunk", "partial", doc_id, None,
                          prefixes, passage, title_score=title_score, notes=notes)
    return Resolution("invented", "no_returned_chunk_at_location", notes=notes)


def resolve_ungrounded(
    citation: Citation, corpus: CorpusIndex, page_offsets: Mapping[str, int | str]
) -> Resolution:
    """Title to document, then a label or the printed page, then the text there."""
    title = citation.title if citation.kind == "free_text" else citation.slug
    # A bracketed slug states its version the way a document name does ("...-10e").
    version = (citation.version if citation.kind == "free_text"
               else parse_doc_version(title or ""))
    match = corpus.match_title(title or "", version,
                               identifiers=citation.identifiers or None)
    if match.doc is None:
        # No match, an ambiguous title, or an edition the corpus cannot confirm:
        # the citation cannot be checked, which is not the same as wrong.
        return Resolution("unresolvable", match.detail, title_score=match.score,
                          candidate_doc_id=match.candidate.doc_id if match.candidate else None)
    doc = match.doc
    offset = page_offsets.get(doc.doc_id)
    notes: List[str] = []
    mapped = None
    if isinstance(offset, int) and citation.page_start is not None:
        mapped = (citation.page_start + offset, (citation.page_end or citation.page_start) + offset)
    if citation.labels:
        # A label locates text without any page mapping, so it also works for
        # section-paged works and documents with no page offset. Only a caption
        # (the table, the clause heading) locates it; a passing mention does not.
        captions, mentions = _label_hits(_corpus_chunk_dicts(corpus, doc.doc_id),
                                         citation.labels, mapped)
        if captions:
            return _label_found(citation, doc.doc_id, captions, pages=mapped, notes=notes,
                                title_score=match.score)
        notes.append("label_mention_only" if mentions else "label_not_found")
    # Without a page to fall back on, a label found only in passing is the
    # reason the citation cannot be checked.
    no_page_detail = "label_mention_only" if "label_mention_only" in notes else None
    if offset == SECTION_PAGED:
        # Printed pages such as "5-20" (section 5, page 20) cannot be mapped.
        return Resolution("unresolvable", no_page_detail or "section_paged", doc_id=doc.doc_id,
                          title_score=match.score, notes=notes)
    if citation.page_start is None:
        return Resolution("unresolvable", no_page_detail or "no_page_cited", doc_id=doc.doc_id,
                          title_score=match.score, notes=notes)
    if offset is None:
        return Resolution("unresolvable", "no_page_offset", doc_id=doc.doc_id,
                          title_score=match.score, notes=notes)
    max_page = corpus.max_page(doc.doc_id)
    if max_page is None:
        return Resolution("unresolvable", "work_has_no_page_index", doc_id=doc.doc_id,
                          title_score=match.score, notes=notes)
    lo, hi = mapped
    if hi < 1 or lo > max_page:
        return Resolution("invented", "page_not_in_work", doc_id=doc.doc_id,
                          pdf_pages=mapped, title_score=match.score, notes=notes)
    chunks = [
        {"prefix": c.prefix, "content": c.body}
        for c in corpus.chunks(doc.doc_id)
        if _overlaps(max(lo, 1), min(hi, max_page), c.page_start, c.page_end)
    ]
    if not chunks:
        # The page exists but extraction left no text on it (a figure page, a
        # table the parser dropped): nothing to check, which is not "wrong".
        return Resolution("unresolvable", "no_text_at_page", doc_id=doc.doc_id,
                          pdf_pages=mapped, title_score=match.score, notes=notes)
    prefixes, passage = _passage(chunks)
    return Resolution(None, "mapped_printed_page", doc_id=doc.doc_id, pdf_pages=mapped,
                      passage_prefixes=prefixes, passage=passage, title_score=match.score,
                      notes=notes)
