"""Scoring tests for the grounded-answer benchmark (Epic 25, Story 25.3).

Scripted answers drive every citation bucket through the real extraction,
resolution and scoring code; a fake judge client returns scripted verdicts.
Assertions check the buckets, scores and files the code produced.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from grounding.citations import _derive_slug, format_citation_prefix
from grounding.eval.answers import prompts
from grounding.eval.answers.blind import (
    ANSWERS_CSV,
    BLIND_DIR,
    CITATIONS_CSV,
    KEY_FILE,
    cohens_kappa,
    draw_sample,
    export_blind,
    import_blind,
)
from grounding.eval.answers.citations import (
    CorpusIndex,
    Version,
    compare_versions,
    extract_citations,
    parse_doc_edition,
    parse_doc_version,
    parse_identifiers,
    resolve_grounded,
    resolve_ungrounded,
)
from grounding.eval.answers.runner import Budget, edition_warnings
from grounding.eval.answers.scoring import (
    Judge,
    abstained_as_desired,
    score_answer,
    score_run,
)
from grounding.eval.fixtures import (
    AnswerSpec,
    Expected,
    FixtureItem,
    FixtureSet,
    NumericAnswer,
    load_fixtures,
)
from tests.answers_fakes import (
    FakeJudge,
    _tag,
    MINI_AGENTS_DIR,
    MINI_ANSWERS_YAML,
    MINI_CORPUS,
    FakeClient,
    make_response,
    mini_chunks,
    read_jsonl,
    text_block,
)

FIXTURE = load_fixtures(MINI_ANSWERS_YAML, agents_dir=MINI_AGENTS_DIR)
ITEMS = {it.id: it for it in FIXTURE.items}
OFFSETS = FIXTURE.page_offsets
CHUNKS = {(c["doc_id"], c["meta"]["chunk_id"]): c for c in mini_chunks()}
GAMMA = CHUNKS[("doc-gamma", 1)]  # p.12, section "Falsifiability and Demarcation"
BETA_1 = CHUNKS[("doc-beta", 1)]  # p.247
BETA_2 = CHUNKS[("doc-beta", 2)]  # p.248-249


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_chunk_map(directory: Path) -> Path:
    """The chunk map CorpusIndex reads, for the real mini corpus files."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = [
        {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "file_path": c["file_path"],
         "embedding_index": i}
        for i, c in enumerate(mini_chunks())
    ]
    (directory / "_chunk_map.json").write_text(
        json.dumps({"format_version": "1.1", "chunks": entries}), encoding="utf-8"
    )
    return directory


@pytest.fixture
def corpus(tmp_path) -> CorpusIndex:
    return CorpusIndex(MINI_CORPUS, write_chunk_map(tmp_path / "idx"), editions=FIXTURE.editions)


def returned(chunk) -> dict:
    meta = chunk["meta"]
    return {
        "rank": 1,
        "source": meta["source"],
        "doc_id": meta["doc_id"],
        "chunk_id": meta["chunk_id"],
        "page_start": meta["page_start"],
        "page_end": meta["page_end"],
        "section_heading": meta.get("section_heading"),
        "content": chunk["body"],
        "slug": _derive_slug(meta["source"]),
        "prefix": format_citation_prefix(
            meta["source"], meta["page_start"], meta["page_end"], meta.get("section_heading")
        ),
    }


def transcript(item_id: str, condition: str, text: str, results=(), status="ok") -> dict:
    return {
        "run_id": "run-test",
        "item_id": item_id,
        "condition": condition,
        "status": status,
        "final_text": text,
        "tool_calls": [{"results": [returned(c) for c in results]}] if results else [],
    }


def judge_for(fake: FakeJudge, tmp_path: Path | None = None, budget: Budget | None = None) -> Judge:
    errors = tmp_path / "errors.jsonl" if tmp_path else None
    return Judge(FakeClient(fake), "claude-opus-5", budget or Budget(None), errors_path=errors)


def buckets(scored: dict) -> list[str]:
    return [c["bucket"] for c in scored["citations"]]


def lexical_support(claim: str, passage: str) -> bool:
    """Supported when every long word of the claim appears in the passage."""
    words = [w for w in re.findall(r"[a-z]+", claim.lower()) if len(w) >= 7]
    return bool(words) and all(w in passage.lower() for w in words)


# ---------------------------------------------------------------------------
# Grounded buckets
# ---------------------------------------------------------------------------

def test_grounded_exact_prefix_supported_by_passage_is_verified(corpus):
    text = f"Popper's demarcation criterion is falsifiability {returned(GAMMA)['prefix']}."
    fake = FakeJudge(supported=lexical_support)
    scored = score_answer(
        transcript("ans-001", "dense", text, [GAMMA, BETA_1]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    (cit,) = scored["citations"]
    assert cit["bucket"] == "verified"
    assert cit["resolution"]["match"] == "exact"
    assert cit["claim"] == "Popper's demarcation criterion is falsifiability."
    assert "falsifiability" in cit["passage"]
    assert scored["bucket_counts"] == {
        "verified": 1, "partial": 0, "unsupported": 0, "invented": 0, "unresolvable": 0,
    }


def test_grounded_citation_the_passage_does_not_support_is_unsupported(corpus):
    text = f"Popper insisted every theory be verified by induction {returned(GAMMA)['prefix']}."
    fake = FakeJudge(supported=lexical_support)
    scored = score_answer(
        transcript("ans-001", "hybrid", text, [GAMMA]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert buckets(scored) == ["unsupported"]
    assert scored["citations"][0]["judge"] == {"supported": False, "reason": "scripted", "cached": False}


@pytest.mark.parametrize(
    "citation, detail",
    [
        ("[shigley-design, p.40]", "no_returned_chunk_at_location"),  # slug never returned
        ("[gamma, p.13]", "no_returned_chunk_at_location"),  # real doc, page not returned
        ("[beta, §Results]", "no_returned_chunk_at_location"),  # a section that was never returned
    ],
)
def test_grounded_citation_no_tool_call_returned_is_invented(corpus, citation, detail):
    fake = FakeJudge()
    scored = score_answer(
        transcript("ans-001", "dense", f"The criterion is falsifiability {citation}.", [GAMMA, BETA_1]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert buckets(scored) == ["invented"]
    assert scored["citations"][0]["resolution"]["detail"] == detail
    assert [k for k, _ in fake.requests] == ["correctness"]  # no support call for invented


def test_grounded_citation_needs_the_tool_result_in_the_same_transcript(corpus):
    """The same prefix is invented when this transcript's tool calls never returned it."""
    text = f"The criterion is falsifiability {returned(GAMMA)['prefix']}."
    scored = score_answer(
        transcript("ans-001", "dense", text, [BETA_1]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    assert buckets(scored) == ["invented"]


def test_grounded_partial_prefix_and_free_text_resolve_to_returned_chunks(corpus):
    text = (
        "The percentile interval reads the quantiles of the bootstrap replicates [beta, p.248-249]. "
        "Resampling draws with replacement (Beta Study, p. 247). "
        "Lattices discretize spacetime (Alpha Paper, p. 5)."
    )
    scored = score_answer(
        transcript("ans-002", "hybrid-rerank", text, [BETA_1, BETA_2]),
        ITEMS["ans-002"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    first, second, third = scored["citations"]
    # A page without the exact prefix string is still a findable location.
    assert (first["bucket"], first["resolution"]["match"]) == ("verified", "located")
    assert first["resolution"]["passage_prefixes"] == [returned(BETA_2)["prefix"]]
    assert (second["kind"], second["bucket"]) == ("free_text", "verified")
    assert second["resolution"]["passage_prefixes"] == [returned(BETA_1)["prefix"]]
    # Alpha was never returned in this transcript.
    assert third["bucket"] == "invented"


def test_grounded_citation_naming_only_the_document_is_partial_not_verified(corpus):
    text = "The percentile interval reads the quantiles of the bootstrap replicates [beta]."
    scored = score_answer(
        transcript("ans-002", "hybrid-rerank", text, [BETA_1, BETA_2]),
        ITEMS["ans-002"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    (cit,) = scored["citations"]
    assert (cit["bucket"], cit["resolution"]["match"]) == ("partial", "partial")
    assert scored["bucket_counts"]["verified"] == 0
    assert scored["bucket_counts"]["partial"] == 1


# ---------------------------------------------------------------------------
# Ungrounded buckets
# ---------------------------------------------------------------------------

def test_ungrounded_printed_page_maps_through_offset_and_is_verified(corpus):
    text = "A theory is scientific only if it is open to falsifiability (Gamma Notes, 1st ed., p. 10)."
    scored = score_answer(
        transcript("ans-001", "ungrounded", text), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge(supported=lexical_support)),
    )
    (cit,) = scored["citations"]
    assert cit["bucket"] == "verified"
    assert cit["resolution"]["doc_id"] == "doc-gamma"
    assert cit["resolution"]["pdf_pages"] == [12, 12]  # printed 10 + offset 2


def test_ungrounded_mapped_page_that_does_not_support_is_unsupported(corpus):
    text = "Popper endorsed verificationism (Gamma Notes, p. 10)."
    scored = score_answer(
        transcript("ans-001", "ungrounded", text), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge(supported=lexical_support)),
    )
    assert buckets(scored) == ["unsupported"]


def test_ungrounded_page_outside_the_work_is_invented(corpus):
    text = "Falsifiability is central (Gamma Notes, p. 40)."
    fake = FakeJudge()
    scored = score_answer(
        transcript("ans-001", "ungrounded", text), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    (cit,) = scored["citations"]
    assert cit["bucket"] == "invented"
    assert cit["resolution"]["detail"] == "page_not_in_work"
    assert cit["resolution"]["pdf_pages"] == [42, 42]
    assert "support" not in [k for k, _ in fake.requests]


@pytest.mark.parametrize(
    "citation, detail",
    [
        ("(Karl Popper, The Logic of Scientific Discovery, 2nd ed., p. 40)", "work_not_in_corpus"),
        ("(Alpha Paper, p. 3)", "no_page_offset"),  # in corpus, no page_offsets entry
        ("(Gamma Notes, 1st ed.)", "no_page_cited"),
    ],
)
def test_ungrounded_unmappable_citation_is_unresolvable(corpus, citation, detail):
    fake = FakeJudge()
    scored = score_answer(
        transcript("ans-001", "ungrounded", f"Falsifiability separates science {citation}."),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert buckets(scored) == ["unresolvable"]
    assert scored["citations"][0]["resolution"]["detail"] == detail
    assert "support" not in [k for k, _ in fake.requests]


def test_every_citation_lands_in_exactly_one_bucket(corpus):
    text = (
        "Falsifiability is the criterion (Gamma Notes, p. 10). "
        "Popper endorsed verificationism (Gamma Notes, p. 10). "
        "It appears late in the notes (Gamma Notes, p. 40). "
        "Kuhn disagreed (Thomas Kuhn, The Structure of Scientific Revolutions, 3rd ed., p. 5)."
    )
    scored = score_answer(
        transcript("ans-001", "ungrounded", text), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge(supported=lexical_support)),
    )
    assert buckets(scored) == ["verified", "unsupported", "invented", "unresolvable"]
    assert sum(scored["bucket_counts"].values()) == scored["n_citations"] == 4


# ---------------------------------------------------------------------------
# Title matching (fuzzy slug / orig_name match, editions)
# ---------------------------------------------------------------------------

def _corpus_with(tmp_path: Path, names: list[str]) -> CorpusIndex:
    root = tmp_path / "corpus"
    docs, entries = [], []
    for i, name in enumerate(names):
        slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")
        (root / slug / "chunks").mkdir(parents=True)
        (root / slug / "chunks" / "ch_0001.md").write_text(
            f"---\ndoc_id: d{i}\nsource: {name}\nchunk_id: 1\npage_start: 1\npage_end: 1\n---\nbody\n"
        )
        docs.append({"doc_id": f"d{i}", "slug": slug, "orig_name": name})
        entries.append({"chunk_id": f"d{i}_ch_0001", "doc_id": f"d{i}",
                        "file_path": f"{slug}/chunks/ch_0001.md"})
    (root / "_index.json").write_text(json.dumps({"docs": docs}))
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "_chunk_map.json").write_text(json.dumps({"chunks": entries}))
    return CorpusIndex(root, idx)


def test_title_matching_handles_short_names_editions_and_near_misses(tmp_path):
    corpus = _corpus_with(
        tmp_path,
        [
            "Shigleys Mechanical Engineering Design 10th Edition.pdf",
            "Shigleys Mechanical Engineering Design 9th Edition.pdf",
            "Mechanical Engineers Handbook.pdf",
            "Roarks Formulas for Stress and Strain.pdf",
        ],
    )
    assert corpus.match_title("Roark", None).doc.doc_id == "d3"
    assert corpus.match_title("Shigley's Mechanical Engineering Design", 10).doc.doc_id == "d0"
    assert corpus.match_title("Shigley's Mechanical Engineering Design", 9).doc.doc_id == "d1"
    assert corpus.match_title("Shigley", None).detail == "ambiguous_title"
    assert corpus.match_title("Shigley's Mechanical Engineering Design", 11).detail == "edition_mismatch"
    # Shares three words with the Kutz handbook but is a different work.
    marks = corpus.match_title("Marks' Standard Handbook for Mechanical Engineers", None)
    assert marks.doc is None and marks.detail == "work_not_in_corpus"


# Document names shaped like the real mechanical-engineer corpus: a sort prefix,
# the edition as "10e", or no edition at all.
REAL_SHAPED = [
    "000-a-a-shigleys-mechanical-engineering-design-10e",
    "machinerys-handbook-27e",
    "roarks-formulas-for-stress-and-strain",
    "mechanical-engineers-handbook",
    "asme-y14-5-2018",
]


def _paged_corpus(tmp_path: Path, slugs: list[str], pages: int = 3, editions=None,
                  revisions=None, identifiers=None) -> CorpusIndex:
    """One document per slug (orig_name = slug + .pdf), one chunk per PDF page."""
    root = tmp_path / "corpus"
    docs, entries = [], []
    for i, slug in enumerate(slugs):
        (root / slug / "chunks").mkdir(parents=True)
        for page in range(1, pages + 1):
            (root / slug / "chunks" / f"ch_{page:04d}.md").write_text(
                f"---\ndoc_id: d{i}\nsource: {slug}.pdf\nchunk_id: {page}\n"
                f"page_start: {page}\npage_end: {page}\n---\n{slug} text on page {page}.\n"
            )
            entries.append({"chunk_id": f"d{i}_ch_{page:04d}", "doc_id": f"d{i}",
                            "file_path": f"{slug}/chunks/ch_{page:04d}.md"})
        docs.append({"doc_id": f"d{i}", "slug": slug, "orig_name": f"{slug}.pdf"})
    (root / "_index.json").write_text(json.dumps({"docs": docs}))
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "_chunk_map.json").write_text(json.dumps({"chunks": entries}))
    return CorpusIndex(root, idx, editions=editions, revisions=revisions,
                       identifiers=identifiers)


def _free_text(citation: str):
    (cit,) = extract_citations(f"The claim holds {citation}.")
    return cit


@pytest.mark.parametrize(
    "citation, slug, rule",
    [
        # Author-prefixed citations: matched tokens / cited tokens was 4/6 = 0.67,
        # under the 0.75 threshold, so these used to be unresolvable.
        ("(Budynas & Nisbett, Shigley's Mechanical Engineering Design, 10th ed., p. 250)",
         "000-a-a-shigleys-mechanical-engineering-design-10e", "precision_recall"),
        ("(Oberg et al., Machinery's Handbook, 27th ed., p. 1520)",
         "machinerys-handbook-27e", "precision_recall"),
        ("(Young and Budynas, Roark's Formulas for Stress and Strain, Table 9.2)",
         "roarks-formulas-for-stress-and-strain", "precision_recall"),
        # No comma between author and title: the document's title words appear
        # in order and cover the whole title (the recall rule).
        ("(Budynas and Nisbett Shigley's Mechanical Engineering Design, 10th ed., p. 250)",
         "000-a-a-shigleys-mechanical-engineering-design-10e", "recall"),
    ],
)
def test_author_prefixed_titles_match_real_shaped_document_names(tmp_path, citation, slug, rule):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)
    cit = _free_text(citation)
    match = corpus.match_title(cit.title, cit.edition)
    assert match.detail == "matched"
    assert match.doc.slug == slug
    assert match.rule == rule


@pytest.mark.parametrize(
    "citation",
    [
        # Three shared words with mechanical-engineers-handbook, out of order.
        "(Marks' Standard Handbook for Mechanical Engineers, 11th ed., p. 5)",
        # The "ASME" segment alone must not claim a different ASME document.
        "(ASME, Boiler and Pressure Vessel Code, Section VIII, p. 12)",
        # A sub-phrase of Shigley's title is a different book (Dieter).
        "(Dieter, Engineering Design, 5th ed., p. 40)",
        # Shares "stress" with Roark and an author with Shigley's.
        "(Budynas, Advanced Strength and Applied Stress Analysis, 2nd ed., p. 40)",
    ],
)
def test_titles_of_other_works_do_not_match_real_shaped_names(tmp_path, citation):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)
    cit = _free_text(citation)
    match = corpus.match_title(cit.title, cit.edition)
    assert match.doc is None
    assert match.candidate is None
    assert match.detail == "work_not_in_corpus"


@pytest.mark.parametrize(
    "name, edition",
    [
        ("000-a-a-shigleys-mechanical-engineering-design-10e", 10),
        ("machinerys-handbook-27e", 27),
        ("statics-2e", 2),
        ("machinerys-handbook-27e-2004", 27),  # an edition followed by a year
        ("Shigleys Mechanical Engineering Design 10th Edition", 10),
        ("roarks-formulas-for-stress-and-strain", None),
        ("tolerance-table-1e-3", None),  # scientific notation, not an edition
        ("creep-at-1e-6-strain", None),
        ("asme-y14-5-2018", None),
    ],
)
def test_document_editions_parse_the_10e_convention_but_not_exponents(name, edition):
    assert parse_doc_edition(name) == edition


def test_cited_text_never_uses_the_document_edition_convention():
    assert _free_text("(Machinery's Handbook 27e, p. 5)").edition is None
    assert _free_text("(Roark's Formulas, 1e-3 strain, p. 5)").edition is None
    assert _free_text("(Machinery's Handbook, 27th ed., p. 5)").edition == 27


def test_a_wrong_edition_is_unresolvable_and_its_page_is_never_mapped(tmp_path):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)
    offsets = {"d1": 0}  # machinerys-handbook-27e, page 2 exists
    cit = _free_text("(Oberg et al., Machinery's Handbook, 26th ed., p. 2)")
    res = resolve_ungrounded(cit, corpus, offsets)
    assert (res.bucket, res.detail) == ("unresolvable", "edition_mismatch")
    assert res.pdf_pages is None and res.passage == ""
    assert res.candidate_doc_id == "d1"
    # The right edition maps the same printed page.
    right = resolve_ungrounded(_free_text("(Machinery's Handbook, 27th ed., p. 2)"), corpus, offsets)
    assert (right.bucket, right.detail, right.pdf_pages) == (None, "mapped_printed_page", (2, 2))


def test_a_cited_edition_on_a_document_of_unknown_edition_is_unresolvable(tmp_path):
    offsets = {"d2": 0}  # roarks-formulas-for-stress-and-strain: no edition in its name
    cit = _free_text("(Roark's Formulas for Stress and Strain, 8th ed., p. 2)")
    unknown = resolve_ungrounded(cit, _paged_corpus(tmp_path / "a", REAL_SHAPED), offsets)
    assert (unknown.bucket, unknown.detail) == ("unresolvable", "edition_unknown")
    assert unknown.pdf_pages is None and unknown.candidate_doc_id == "d2"
    # Declaring the document's edition in the fixture makes the citation checkable.
    declared = _paged_corpus(tmp_path / "b", REAL_SHAPED, editions={"d2": 8})
    mapped = resolve_ungrounded(cit, declared, offsets)
    assert (mapped.bucket, mapped.detail, mapped.pdf_pages) == (None, "mapped_printed_page", (2, 2))
    # No edition cited: nothing to check, the page maps as before.
    plain = resolve_ungrounded(_free_text("(Roark's Formulas for Stress and Strain, p. 2)"),
                               _paged_corpus(tmp_path / "c", REAL_SHAPED), offsets)
    assert plain.detail == "mapped_printed_page"


# ---------------------------------------------------------------------------
# Government and standards document identifiers, revisions and years (Epic 25,
# public-domain corpus readiness)
# ---------------------------------------------------------------------------

# Names shaped like a public-domain US government corpus.
GOV_SHAPED = [
    "doe-hdbk-1018-93-mechanical-science-volume-1",
    "doe-hdbk-1019-93-nuclear-physics-and-reactor-theory-volume-1",
    "nasa-std-5001b-structural-design-and-test-factors-of-safety",
    "mil-hdbk-5j-metallic-materials-and-elements-for-aerospace-vehicle-structures",
    "nasa-rp-1228-fastener-design-manual",
    "21-cfr-part-820-quality-system-regulation",
    "fda-guidance-cybersecurity-in-medical-devices",
]


@pytest.mark.parametrize(
    "text, key, version",
    [
        ("DOE-HDBK-1018-93", ("doe-hdbk", ("1018",)), {"year": 1993}),
        ("DOE-HDBK-1012/1-92", ("doe-hdbk", ("1012", "1")), {"year": 1992}),
        ("NASA-STD-5001B", ("nasa-std", ("5001",)), {"revision": "B"}),
        ("NASA-STD-5005 Rev. C", ("nasa-std", ("5005",)), {"revision": "C"}),
        ("MIL-HDBK-5J", ("mil-hdbk", ("5",)), {"revision": "J"}),
        ("NASA/RP-1228", ("rp", ("1228",)), {}),
        ("21 CFR 820.30", ("cfr", ("21", "820")), {}),
        ("21 CFR Part 820", ("cfr", ("21", "820")), {}),
        ("ASME Y14.5-2018", ("asme", ("y14", "5")), {"year": 2018}),
        ("ISO 13485:2016", ("iso", ("13485",)), {"year": 2016}),
    ],
)
def test_document_identifiers_parse_into_an_identity_and_a_version(text, key, version):
    (identifier,) = parse_identifiers(text)
    assert identifier.key == key
    assert identifier.version.as_dict() == version


@pytest.mark.parametrize(
    "text",
    [
        "Shigley's Mechanical Engineering Design, 10th ed., p. 250",
        "Marks' Standard Handbook for Mechanical Engineers, Section 5",
        "ASME, Boiler and Pressure Vessel Code, Section VIII",  # no number after the body
        "RP 2A",  # not a four-digit NASA report number
    ],
)
def test_text_without_a_document_designation_names_no_identifier(text):
    assert parse_identifiers(text) == []


@pytest.mark.parametrize(
    "name, version",
    [
        ("nasa-std-5001b-structural-design-and-test-factors-of-safety", {"revision": "B"}),
        ("mil-hdbk-5j-metallic-materials", {"revision": "J"}),
        ("doe-hdbk-1018-93-mechanical-science-volume-1", {"year": 1993}),
        ("asme-y14-5-2018", {"year": 2018}),
        ("machinerys-handbook-27e", {"edition": 27}),
        ("machinerys-handbook-27e-2004", {"edition": 27, "year": 2004}),
        ("fda-guidance-cybersecurity-in-medical-devices", {}),
        ("roarks-formulas-for-stress-and-strain", {}),
    ],
)
def test_document_names_give_their_edition_revision_or_year(name, version):
    assert parse_doc_version(name).as_dict() == version


@pytest.mark.parametrize(
    "citation, slug",
    [
        ("(DOE-HDBK-1018-93, Mechanical Science, Volume 1, p. 12)",
         "doe-hdbk-1018-93-mechanical-science-volume-1"),
        # The identifier alone is enough: the document's name carries it.
        ("(DOE-HDBK-1019, p. 12)", "doe-hdbk-1019-93-nuclear-physics-and-reactor-theory-volume-1"),
        ("(NASA-STD-5001B, Structural Design and Test Factors of Safety, p. 12)",
         "nasa-std-5001b-structural-design-and-test-factors-of-safety"),
        ("(MIL-HDBK-5J, Metallic Materials and Elements for Aerospace Vehicle Structures, p. 12)",
         "mil-hdbk-5j-metallic-materials-and-elements-for-aerospace-vehicle-structures"),
        ("(NASA RP-1228, Fastener Design Manual, p. 12)", "nasa-rp-1228-fastener-design-manual"),
        ("(21 CFR 820.30, Design controls, p. 12)", "21-cfr-part-820-quality-system-regulation"),
    ],
)
def test_a_cited_identifier_matches_the_document_that_carries_it(tmp_path, citation, slug):
    corpus = _paged_corpus(tmp_path, GOV_SHAPED)
    cit = _free_text(citation)
    match = corpus.match_title(cit.title, cit.version, identifiers=cit.identifiers)
    assert match.detail == "matched"
    assert match.doc.slug == slug
    assert match.rule == "identifier"


@pytest.mark.parametrize(
    "citation, detail",
    [
        # A different number is a different document, however close the title is.
        ("(DOE-HDBK-1020, Mechanical Science, Volume 1, p. 12)", "work_not_in_corpus"),
        ("(NASA-STD-5002, Structural Design and Test Factors of Safety, p. 12)",
         "work_not_in_corpus"),
        ("(21 CFR 211.100, p. 12)", "work_not_in_corpus"),
        ("(NASA RP-1229, Fastener Design Manual, p. 12)", "work_not_in_corpus"),
        # A different revision of the same document is not that document either.
        ("(MIL-HDBK-5H, Metallic Materials, p. 12)", "edition_mismatch"),
        ("(NASA-STD-5001A, p. 12)", "edition_mismatch"),
        ("(NASA-STD-5001 Rev. C, p. 12)", "edition_mismatch"),
        # Same handbook number, another year: a revision mismatch, not another work.
        ("(DOE-HDBK-1018-2009, Mechanical Science, p. 12)", "edition_mismatch"),
    ],
)
def test_a_near_miss_identifier_or_revision_never_matches(tmp_path, citation, detail):
    corpus = _paged_corpus(tmp_path, GOV_SHAPED)
    cit = _free_text(citation)
    match = corpus.match_title(cit.title, cit.version, identifiers=cit.identifiers)
    assert match.doc is None
    assert match.detail == detail
    # The page is never mapped into the near miss.
    res = resolve_ungrounded(cit, corpus, {f"d{i}": 0 for i in range(len(GOV_SHAPED))})
    assert res.bucket == "unresolvable" and res.pdf_pages is None and res.passage == ""


# Names and declared designations shaped like the public-domain benchmark corpus
# (benchmarks/public-corpus): FDA guidances known by their docket, a Federal
# Register rule, a CFR part, an FAA report and a NASA standard.
PUBLIC_SHAPED = [
    "fda-reporting-computational-modeling-2016",
    "fda-cms-credibility-2023",
    "cfr-21-part-888-2025",
    "fr-qmsr-final-rule-2024",
    "faa-damage-tolerance-handbook-vol1",
    "nasa-std-5001b-chg3",
]
PUBLIC_IDENTIFIERS = {
    "d0": "FDA-2013-D-1530",
    "d1": "FDA-2021-D-0980",
    # The rule is known by its Federal Register cite and by its docket.
    "d3": "89 FR 7496 to 7525; FDA-2021-N-0507",
    "d4": "DOT/FAA/CT-93/69.I",
}


def _public_corpus(tmp_path: Path) -> CorpusIndex:
    return _paged_corpus(tmp_path, PUBLIC_SHAPED, identifiers=PUBLIC_IDENTIFIERS,
                         revisions={"d1": "November 2023"})


@pytest.mark.parametrize(
    "citation, slug",
    [
        ("(Reporting of Computational Modeling Studies, FDA-2013-D-1530, p. 12)",
         "fda-reporting-computational-modeling-2016"),
        ("(Assessing the Credibility of Computational Modeling, docket FDA-2021-D-0980, p. 12)",
         "fda-cms-credibility-2023"),
        ("(21 CFR 888.3080, p. 2)", "cfr-21-part-888-2025"),
        ("(21 CFR Part 888, p. 2)", "cfr-21-part-888-2025"),
        # A Federal Register rule is cited at its first page or at a page inside it.
        ("(Quality Management System Regulation, 89 FR 7496, p. 2)", "fr-qmsr-final-rule-2024"),
        ("(Quality Management System Regulation, 89 FR 7520, p. 2)", "fr-qmsr-final-rule-2024"),
        ("(QMSR final rule, docket FDA-2021-N-0507, p. 2)", "fr-qmsr-final-rule-2024"),
        ("(Damage Tolerance Assessment Handbook, DOT/FAA/CT-93/69.I, p. 12)",
         "faa-damage-tolerance-handbook-vol1"),
        ("(NASA-STD-5001B, p. 12)", "nasa-std-5001b-chg3"),
    ],
)
def test_public_corpus_designations_match_the_document_that_declares_them(tmp_path, citation, slug):
    corpus = _public_corpus(tmp_path)
    cit = _free_text(citation)
    match = corpus.match_title(cit.title, cit.version, identifiers=cit.identifiers)
    assert match.detail == "matched"
    assert match.doc.slug == slug


@pytest.mark.parametrize(
    "citation",
    [
        # One digit out is a different guidance, however close the title is.
        "(Reporting of Computational Modeling Studies, FDA-2013-D-1531, p. 12)",
        "(Assessing the Credibility of Computational Modeling, FDA-2021-D-0981, p. 12)",
        # 21 CFR 820 is the quality system regulation, not the orthopedic part.
        "(21 CFR 820.30, p. 2)",
        # A Federal Register page outside the rule's range.
        "(Quality Management System Regulation, 89 FR 9000, p. 2)",
        # Volume II of the FAA handbook is a different report.
        "(Damage Tolerance Assessment Handbook, DOT/FAA/CT-93/69.II, p. 12)",
        "(NASA-STD-5002, p. 12)",
    ],
)
def test_public_corpus_near_miss_designations_never_match(tmp_path, citation):
    corpus = _public_corpus(tmp_path)
    cit = _free_text(citation)
    match = corpus.match_title(cit.title, cit.version, identifiers=cit.identifiers)
    assert match.doc is None
    assert match.detail in ("work_not_in_corpus", "identifier_unknown")
    res = resolve_ungrounded(cit, corpus, {f"d{i}": 0 for i in range(len(PUBLIC_SHAPED))})
    assert res.bucket == "unresolvable" and res.pdf_pages is None


def test_a_guidance_revision_is_the_date_the_fixture_declares(tmp_path):
    corpus = _public_corpus(tmp_path)  # d1 declares "November 2023"
    current = _free_text("(Assessing the Credibility of Computational Modeling, "
                         "FDA-2021-D-0980, 2023, p. 2)")
    assert resolve_ungrounded(current, corpus, {"d1": 0}).detail == "mapped_printed_page"
    draft = _free_text("(Assessing the Credibility of Computational Modeling, "
                       "FDA-2021-D-0980, 2021, p. 2)")
    assert resolve_ungrounded(draft, corpus, {"d1": 0}).detail == "edition_mismatch"


WARNING_SHAPED = [
    "nasa-std-5001b-structural-design-and-test-factors-of-safety",  # revision B in the name
    "nasa-std-5005-welding-and-brazing",                            # a designation, no revision
    "fda-guidance-cybersecurity-in-medical-devices",                # no designation, no year
    "machinerys-handbook-27e",                                      # edition in the name
    "shigleys-mechanical-engineering-design",                       # a sibling edition below
    "shigleys-mechanical-engineering-design-9e",
]


def _fixture_set(items=(), **kwargs) -> FixtureSet:
    return FixtureSet(agent="mini", version=1, items=tuple(items),
                      source_path=Path("fixture.yaml"), **kwargs)


def _item(item_id: str, query: str, doc_ids: tuple[str, ...]) -> FixtureItem:
    return FixtureItem(id=item_id, query=query, expected=Expected(doc_ids=doc_ids, page=1),
                       answer=AnswerSpec(category="standard", gold="gold"))


def test_edition_warnings_fire_only_where_a_version_is_likely_to_matter(tmp_path):
    corpus = _paged_corpus(tmp_path, WARNING_SHAPED)
    offsets = {f"d{i}": 0 for i in range(len(WARNING_SHAPED))}
    warned = edition_warnings(_fixture_set(page_offsets=offsets), corpus)
    flagged = {w.split()[1] for w in warned}
    # d1 carries a designation but no revision; d4 has a sibling edition.
    assert flagged == {"d1", "d4"}
    assert "NASA-STD-5005" in " ".join(warned)
    assert "same title words" in " ".join(warned)
    # A government document with neither a designation nor a sibling stays quiet
    # until a fixture item states a version of it.
    cited = edition_warnings(
        _fixture_set([_item("g-1", "What did the 2023 cybersecurity guidance require?", ("d2",))],
                     page_offsets=offsets),
        corpus,
    )
    assert {w.split()[1] for w in cited} == {"d1", "d2", "d4"}
    assert "fixture item g-1" in " ".join(cited)
    # Nothing to warn about when no document has a page offset.
    assert edition_warnings(_fixture_set(), corpus) == []


def test_a_declared_identifier_that_no_series_recognizes_is_reported(tmp_path):
    corpus = _paged_corpus(tmp_path, WARNING_SHAPED, identifiers={"d2": "Fundamentals Handbook"})
    (warning,) = edition_warnings(
        _fixture_set(identifiers={"d2": "Fundamentals Handbook"}), corpus
    )
    assert "not in a recognized series" in warning and "d2" in warning


def test_a_cited_year_is_checked_like_an_edition_and_can_be_declared(tmp_path):
    cited = _free_text("(FDA, Cybersecurity in Medical Devices, 2023, p. 2)")
    assert cited.year == 2023
    plain = _paged_corpus(tmp_path / "a", GOV_SHAPED)
    unknown = resolve_ungrounded(cited, plain, {"d6": 0})
    assert (unknown.bucket, unknown.detail) == ("unresolvable", "edition_unknown")
    assert unknown.candidate_doc_id == "d6"
    # revisions: declares the year the name does not carry.
    declared = _paged_corpus(tmp_path / "b", GOV_SHAPED, revisions={"d6": 2023})
    mapped = resolve_ungrounded(cited, declared, {"d6": 0})
    assert (mapped.bucket, mapped.detail, mapped.pdf_pages) == (None, "mapped_printed_page", (2, 2))
    # A different year is a different revision of the guidance.
    older = _free_text("(FDA, Cybersecurity in Medical Devices, 2018, p. 2)")
    assert resolve_ungrounded(older, declared, {"d6": 0}).detail == "edition_mismatch"


def test_a_standards_year_revision_is_checked(tmp_path):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)  # holds asme-y14-5-2018
    same = _free_text("(ASME Y14.5-2018, Section 4.2, p. 2)")
    assert corpus.match_title(same.title, same.version, identifiers=same.identifiers).doc.slug \
        == "asme-y14-5-2018"
    older = _free_text("(ASME Y14.5-2009, Section 4.2, p. 2)")
    assert corpus.match_title(older.title, older.version,
                              identifiers=older.identifiers).detail == "edition_mismatch"


def test_a_document_whose_name_hides_its_identifier_can_declare_one(tmp_path):
    names = ["fastener-design-manual", "doe-hdbk-1018-93-mechanical-science-volume-1"]
    cit = _free_text("(NASA RP-1228, Fastener Design Manual, p. 2)")
    unknown = resolve_ungrounded(cit, _paged_corpus(tmp_path / "a", names), {"d0": 0})
    assert (unknown.bucket, unknown.detail) == ("unresolvable", "identifier_unknown")
    assert unknown.candidate_doc_id == "d0"
    declared = _paged_corpus(tmp_path / "b", names, identifiers={"d0": "NASA RP-1228"})
    mapped = resolve_ungrounded(cit, declared, {"d0": 0})
    assert (mapped.bucket, mapped.detail, mapped.pdf_pages) == (None, "mapped_printed_page", (2, 2))


@pytest.mark.parametrize(
    "cited, doc, verdict",
    [
        ({"edition": 10}, {"edition": 10}, "matched"),
        ({"edition": 10, "year": 2015}, {"edition": 10}, "matched"),
        ({"edition": 9}, {"edition": 10}, "mismatch"),
        ({"year": 2015}, {"edition": 10}, "unknown"),
        ({"revision": "C"}, {"revision": "B"}, "mismatch"),
        ({"revision": "B"}, {}, "unknown"),
        ({}, {"edition": 10}, "unchecked"),
    ],
)
def test_version_comparison_checks_only_what_both_sides_state(cited, doc, verdict):
    assert compare_versions(Version(**cited), Version(**doc)) == verdict


def test_section_paged_works_are_unresolvable_by_page(tmp_path):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)
    # Handbook pages "5-20" mean section 5, page 20, not PDF pages 5 to 20.
    cit = _free_text("(Machinery's Handbook, 27th ed., p. 5-20)")
    assert (cit.page_start, cit.page_end) == (5, 20)
    res = resolve_ungrounded(cit, corpus, {"d1": "section"})
    assert (res.bucket, res.detail) == ("unresolvable", "section_paged")
    assert res.pdf_pages is None and res.passage == ""


# ---------------------------------------------------------------------------
# Label location (tables, figures, equations, clauses)
# ---------------------------------------------------------------------------

SHIGLEY = "000-a-a-shigleys-mechanical-engineering-design-10e"
ISO = "iso-13485-2016"
MARKS = "marks-standard-handbook-for-mechanical-engineers"
LABEL_DOCS = {
    SHIGLEY: [
        (1, "Chapter 2 Materials", "Yield strengths vary by grade; see Table A-20 for values."),
        (2, "Chapter 2 Materials", "Table A-201 lists something else entirely."),
        (3, "Appendix A", "Table A\u201320 Deterministic ASTM Minimum Tensile and Yield Strengths\n"
                          "| 1018 | CD | 440 | 370 |"),
        (4, "5-4 Maximum Shear Stress", "The factor of safety is n = Sy / (2 tau_max)    (5\u201319)"),
    ],
    ISO: [
        (1, "7.1 Planning of product realization", "The organization shall plan product realization."),
        (2, "7.10 Something later", "Unrelated later clause."),
        (3, "Annex B", "4.2 mm is the smallest gap measured.\nThe gap tolerance is given in clause 4.2 below."),
    ],
    MARKS: [
        # Lines that start with a bare number are not section 5's caption.
        (40, "Bolted joints", "5 mm bolts are common in light assemblies.\n"
                              "5 | 8.8 | 640\nFastener strength is covered in section 5."),
        (41, "Section 6 Materials", "Section 6 Materials\nCast irons are brittle."),
    ],
}


def _label_corpus(tmp_path: Path) -> CorpusIndex:
    root = tmp_path / "corpus"
    docs, entries = [], []
    for i, (slug, chunks) in enumerate(LABEL_DOCS.items()):
        (root / slug / "chunks").mkdir(parents=True)
        for n, (page, section, body) in enumerate(chunks, start=1):
            (root / slug / "chunks" / f"ch_{n:04d}.md").write_text(
                f"---\ndoc_id: d{i}\nsource: {slug}.pdf\nchunk_id: {n}\npage_start: {page}\n"
                f"page_end: {page}\nsection_heading: \"{section}\"\n---\n{body}\n"
            )
            entries.append({"chunk_id": f"d{i}_ch_{n:04d}", "doc_id": f"d{i}",
                            "file_path": f"{slug}/chunks/ch_{n:04d}.md"})
        docs.append({"doc_id": f"d{i}", "slug": slug, "orig_name": f"{slug}.pdf"})
    (root / "_index.json").write_text(json.dumps({"docs": docs}))
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "_chunk_map.json").write_text(json.dumps({"chunks": entries}))
    return CorpusIndex(root, idx)


def _returned_from(corpus: CorpusIndex, doc_id: str, chunk_id: int) -> dict:
    chunk = next(c for c in corpus.chunks(doc_id) if c.chunk_id == chunk_id)
    source = f"{corpus.docs[doc_id].slug}.pdf"
    return {"doc_id": doc_id, "chunk_id": chunk_id, "source": source, "slug": _derive_slug(source),
            "page_start": chunk.page_start, "page_end": chunk.page_end,
            "section_heading": chunk.section, "content": chunk.body, "prefix": chunk.prefix}


def test_labels_are_extracted_from_citations():
    cit = _free_text("(Shigley's Mechanical Engineering Design, 10th ed., Table A-20, Eq. (5-19), p. 1040)")
    assert [label.text for label in cit.labels] == ["table A-20", "equation 5-19"]
    assert [label.text for label in _free_text("(ISO 13485:2016, Clause 7.1)").labels] == ["clause 7.1"]
    assert [label.text for label in _free_text("(Beta Study, §3.2)").labels] == ["clause 3.2"]


def test_ungrounded_table_label_is_located_without_any_page_mapping(tmp_path):
    corpus = _label_corpus(tmp_path)
    cit = _free_text("(Shigley's Mechanical Engineering Design, 10th ed., Table A-20)")
    res = resolve_ungrounded(cit, corpus, {})  # no page_offsets at all
    assert (res.bucket, res.detail, res.match) == (None, "label_found", "label")
    # Only the caption chunk (en dash in the book) is the location: the passing
    # reference on p.1 is not added to the passage, and "Table A-201" is not a match.
    assert [p.split(",")[1].strip() for p in res.passage_prefixes] == ["p.3"]
    assert "see Table A-20 for values" not in res.passage
    assert "A-201" not in res.passage
    assert "label:table A-20" in res.notes


def test_equation_and_clause_labels_find_their_chunks(tmp_path):
    corpus = _label_corpus(tmp_path)
    eq = resolve_ungrounded(
        _free_text("(Shigley's Mechanical Engineering Design, 10th ed., Eq. (5-19), p. 250)"), corpus, {}
    )
    assert eq.match == "label" and eq.passage_prefixes[0].split(",")[1].strip() == "p.4"
    clause = resolve_ungrounded(_free_text("(ISO 13485, Clause 7.1)"), corpus, {})
    assert clause.match == "label"
    assert clause.passage_prefixes == [c.prefix for c in corpus.chunks("d1") if c.chunk_id == 1]


def test_label_path_works_for_section_paged_works_and_falls_back_when_missing(tmp_path):
    corpus = _label_corpus(tmp_path)
    found = resolve_ungrounded(
        _free_text("(Shigley's Mechanical Engineering Design, 10th ed., Table A-20, p. 5-20)"),
        corpus, {"d0": "section"},
    )
    assert found.match == "label"
    missing = resolve_ungrounded(
        _free_text("(Shigley's Mechanical Engineering Design, 10th ed., Table A-99, p. 5-20)"),
        corpus, {"d0": "section"},
    )
    assert (missing.bucket, missing.detail) == ("unresolvable", "section_paged")
    assert "label_not_found" in missing.notes


def test_grounded_label_is_searched_only_in_returned_chunks(tmp_path):
    corpus = _label_corpus(tmp_path)
    cit = _free_text("(Shigley's Mechanical Engineering Design, Table A-20)")
    caption, mention = _returned_from(corpus, "d0", 3), _returned_from(corpus, "d0", 1)
    # The tool returned the table itself (and the passing reference): the caption is the location.
    both = resolve_grounded(cit, [mention, caption], corpus)
    assert (both.bucket, both.match, both.detail) == (None, "label", "label_found")
    assert both.passage_prefixes == [caption["prefix"]]
    assert "label_mention_only" not in both.notes


def test_grounded_label_in_a_non_returned_chunk_of_a_returned_document_is_invented(tmp_path):
    """The tool returned Shigley's p.4, but Table A-20 sits on p.3 and p.1, which it never returned."""
    corpus = _label_corpus(tmp_path)
    shown = _returned_from(corpus, "d0", 4)
    for citation in ("(Shigley's Mechanical Engineering Design, Table A-20)",
                     "(Shigley's Mechanical Engineering Design, Table A-20, p. 9)"):
        res = resolve_grounded(_free_text(citation), [shown], corpus)
        assert (res.bucket, res.detail) == ("invented", "label_outside_returned_chunks")
        assert res.passage == "" and res.passage_prefixes == []
        assert "label:table A-20" in res.notes
    # A label the document does not contain at all falls back as before: the
    # document alone is "partial", never located.
    missing = resolve_grounded(_free_text("(Shigley's Mechanical Engineering Design, Table A-99)"),
                               [shown], corpus)
    assert (missing.bucket, missing.match) == (None, "partial")
    assert "label_not_found" in missing.notes
    # A document no tool call returned cannot be located by label.
    other = resolve_grounded(_free_text("(ISO 13485, Clause 7.1)"), [shown], corpus)
    assert other.bucket == "invented"


def test_grounded_label_mentioned_in_a_returned_chunk_resolves_to_that_chunk_only(tmp_path):
    corpus = _label_corpus(tmp_path)
    mention = _returned_from(corpus, "d0", 1)  # "see Table A-20 for values", p.1
    res = resolve_grounded(_free_text("(Shigley's Mechanical Engineering Design, Table A-20)"),
                           [mention], corpus)
    assert (res.bucket, res.match) == (None, "label")
    assert "label_mention_only" in res.notes
    # The judge reads what the model was shown, not the table on p.3.
    assert res.passage_prefixes == [mention["prefix"]]
    assert "1018 | CD" not in res.passage


def test_scored_grounded_label_outside_returned_chunks_never_reaches_the_judge(tmp_path):
    corpus = _label_corpus(tmp_path)
    row = {"run_id": "r", "item_id": "ans-001", "condition": "hybrid-rerank", "status": "ok",
           "final_text": "1018 CD steel yields at 370 MPa (Shigley's Mechanical Engineering "
                         "Design, Table A-20).",
           "tool_calls": [{"results": [_returned_from(corpus, "d0", 4)]}]}
    fake = FakeJudge()
    scored = score_answer(row, ITEMS["ans-001"], corpus=corpus, page_offsets={},
                          judge=judge_for(fake))
    (cit,) = scored["citations"]
    assert cit["bucket"] == "invented"
    assert cit["resolution"]["detail"] == "label_outside_returned_chunks"
    assert "support" not in [k for k, _ in fake.requests]


def test_a_bare_clause_number_needs_its_keyword_to_be_a_caption(tmp_path):
    corpus = _label_corpus(tmp_path)
    marks = "(Marks' Standard Handbook for Mechanical Engineers, Section 5, p. 5-20)"
    # "5 mm bolts" and the table row "5 | 8.8 | 640" open lines with 5, but
    # neither is section 5; the text only mentions "section 5" in passing.
    res = resolve_ungrounded(_free_text(marks), corpus, {"d2": "section"})
    assert (res.bucket, res.detail) == ("unresolvable", "label_mention_only")
    assert res.passage == ""
    # With its keyword, a bare number is a caption: "Section 6 Materials".
    six = resolve_ungrounded(
        _free_text("(Marks' Standard Handbook for Mechanical Engineers, Section 6, p. 6-2)"),
        corpus, {"d2": "section"},
    )
    assert (six.match, six.passage_prefixes) == ("label", [c.prefix for c in corpus.chunks("d2")
                                                          if c.chunk_id == 2])
    # A dotted number opens a clause on its own only before a capitalized word:
    # "4.2 mm is the smallest gap" is not clause 4.2.
    gap = resolve_ungrounded(_free_text("(ISO 13485, Clause 4.2)"), corpus, {})
    assert (gap.bucket, gap.detail) == ("unresolvable", "label_mention_only")


def test_ungrounded_mention_only_label_falls_back_to_the_page(tmp_path):
    corpus = _label_corpus(tmp_path)
    cit = _free_text("(Marks' Standard Handbook for Mechanical Engineers, Section 5, p. 2)")
    # With an integer offset the printed page still maps; the note records why
    # the label did not locate text.
    res = resolve_ungrounded(cit, corpus, {"d2": 38})
    assert (res.bucket, res.detail, res.pdf_pages) == (None, "mapped_printed_page", (40, 40))
    assert "label_mention_only" in res.notes
    # No page cited: the mention is the reason the citation cannot be checked.
    no_page = resolve_ungrounded(
        _free_text("(Marks' Standard Handbook for Mechanical Engineers, Section 5)"), corpus, {}
    )
    assert (no_page.bucket, no_page.detail) == ("unresolvable", "label_mention_only")


# ---------------------------------------------------------------------------
# Grounded slugs and sections (fuzzy slug match, section mismatch)
# ---------------------------------------------------------------------------

def _grounded(citation: str):
    (cit,) = extract_citations(f"The claim holds {citation}.")
    return cit


@pytest.mark.parametrize(
    "citation",
    ["[shigleys-mechanical-engineering-design-10e, p.2]",  # dropped the sort prefix
     "[shigleys-mechanical-engineering-design, p.2]"],  # and the edition token
)
def test_unrecognized_slug_is_fuzzy_matched_to_a_returned_document(tmp_path, citation):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)
    shown = _returned_from(corpus, "d0", 2)
    res = resolve_grounded(_grounded(citation), [shown], corpus)
    assert (res.bucket, res.match) == (None, "located")
    assert res.passage_prefixes == [shown["prefix"]]
    assert res.notes == [f"slug_fuzzy_match:{shown['slug']}"]


@pytest.mark.parametrize(
    "citation",
    ["[shigleys-mechanical-engineering-design-9e, p.2]",  # another edition
     "[roarks-formulas-for-stress-and-strain, p.2]",  # a real document that was not returned
     "[machinery-design-handbook, p.2]"],  # nothing like the returned document
)
def test_fuzzy_slug_match_never_credits_another_document(tmp_path, citation):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED)
    res = resolve_grounded(_grounded(citation), [_returned_from(corpus, "d0", 2)], corpus)
    assert res.bucket == "invented"


def test_correct_page_with_a_mismatched_section_is_located_with_a_note(corpus):
    scored = score_answer(
        transcript("ans-002", "dense",
                   "Resampling draws with replacement from the observed sample [beta, p.247, §Results].",
                   [BETA_1]),
        ITEMS["ans-002"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    (cit,) = scored["citations"]
    assert (cit["bucket"], cit["resolution"]["match"]) == ("verified", "located")
    assert cit["resolution"]["notes"] == ["section_mismatch"]
    assert cit["resolution"]["passage_prefixes"] == [returned(BETA_1)["prefix"]]


# ---------------------------------------------------------------------------
# Correctness and abstention
# ---------------------------------------------------------------------------

def _numeric_item(**numeric) -> FixtureItem:
    return FixtureItem(
        id="num-1",
        query="What is the modulus of elasticity of carbon steel?",
        expected=Expected(doc_ids=("doc-beta",), page=247),
        answer=AnswerSpec(category="table", gold="About 200 GPa.", numeric=NumericAnswer(**numeric)),
    )


@pytest.mark.parametrize(
    "answer, score, method",
    [
        ("Carbon steel has E of about 205 GPa [beta, p.247].", 1.0, "numeric"),
        ("Carbon steel has E of 150 GPa.", 0.0, "numeric"),
        ("Somewhere between 190 and 210 GPa.", 0.5, "judge"),  # range: judge decides
        ("E is roughly 29,000 ksi.", 0.5, "judge"),  # different unit: judge decides
    ],
)
def test_numeric_items_auto_score_before_the_judge(corpus, answer, score, method):
    fake = FakeJudge(correctness=lambda cand: ("0.5", False))
    item = _numeric_item(value=200.0, unit="GPa", rel_tol=0.05)
    scored = score_answer(
        transcript(item.id, "dense", answer, [BETA_1]), item,
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert scored["correctness"]["score"] == score
    assert scored["correctness"]["method"] == method
    judged = [k for k, _ in fake.requests if k == "correctness"]
    assert len(judged) == (1 if method == "judge" else 0)


def test_correctness_judge_never_sees_retrieved_text_or_citations(corpus):
    text = f"You resample with replacement and read percentiles of the bootstrap distribution {returned(BETA_2)['prefix']}."
    fake = FakeJudge()
    score_answer(
        transcript("ans-002", "dense", text, [BETA_1, BETA_2]), ITEMS["ans-002"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    (correctness_params,) = [p for k, p in fake.requests if k == "correctness"]
    user = correctness_params["messages"][0]["content"]
    assert "read percentiles of the bootstrap distribution." in _tag(user, "candidate_answer")
    assert "[beta" not in user
    for chunk in (BETA_1, BETA_2):
        for sentence in chunk["body"].split(". "):
            assert sentence.strip()[:40] not in user
    assert _tag(user, "reference_answer") == ITEMS["ans-002"].answer.gold
    assert "- resample with replacement" in _tag(user, "required_facts")


def test_empty_and_refused_answers(corpus, tmp_path):
    fake = FakeJudge()
    empty = score_answer(
        transcript("ans-001", "dense", ""), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert empty["correctness"] == {"score": 0.0, "method": "empty_answer", "declined": False,
                                    "reason": "the answer has no text", "numeric_check": None}
    refused = score_answer(
        transcript("ans-001", "dense", "", status="refusal"), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert refused["scored"] is False and refused["correctness"] is None
    assert fake.requests == []


NO_SOURCE_ITEM = FixtureItem(
    id="ns-1",
    query="What yield strength does ASTM A36 specify for the quenched and tempered condition?",
    expected=Expected(doc_ids=()),
    answer=AnswerSpec(category="unanswerable", answerable=False, unanswerable_kind="no_source"),
)


def test_not_in_corpus_scores_acknowledging_the_gap_as_the_desired_behavior(corpus):
    """answer-v3 asks the model to flag the gap and to label any general knowledge."""
    fake = FakeJudge(
        gap_flagged=lambda cand: "library" in cand,
        gave_answer=lambda cand: "250 MPa" in cand,
        answer_labeled=lambda cand: "from memory" in cand,
    )
    declined = score_answer(
        transcript("ans-004", "dense", "The library cannot support an answer to this."),
        ITEMS["ans-004"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    flagged_then_answered = score_answer(
        transcript("ans-004", "hybrid", "The library lacks this; from memory it is 250 MPa."),
        ITEMS["ans-004"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    unlabeled = score_answer(
        transcript("ans-004", "hybrid-rerank", "It is 250 MPa."),
        ITEMS["ans-004"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    ungrounded = score_answer(
        transcript("ans-004", "ungrounded", "It is 250 MPa (ASTM A36 Standard, p. 2)."),
        ITEMS["ans-004"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert declined["abstention"] == {
        "method": "judge", "reason": "scripted", "gap_flagged": True, "gave_answer": False,
        "answer_labeled": False, "acknowledged_gap": True, "declined": True,
    }
    assert declined["correctness"] is None
    # Flagged the gap, then answered from labeled general knowledge: the
    # instruction was followed, so it is the desired behavior but not a decline.
    assert flagged_then_answered["abstention"]["declined"] is False
    assert flagged_then_answered["abstention"]["acknowledged_gap"] is True
    assert abstained_as_desired(flagged_then_answered["abstention"], "not_in_corpus") is True
    # Answered without saying the sources lack it: neither.
    assert unlabeled["abstention"]["acknowledged_gap"] is False
    assert abstained_as_desired(unlabeled["abstention"], "not_in_corpus") is False
    # The ungrounded model had no library: the item never counts against it.
    assert ungrounded["scored"] is False
    assert ungrounded["excluded"] == "not_in_corpus_ungrounded"
    assert ungrounded["abstention"] is None and ungrounded["citations"] == []
    # Three abstention calls, all with the not_in_corpus framing; none for ungrounded.
    assert fake.abstention_kinds == ["not_in_corpus"] * 3
    # A no_source question still scores plain declining.
    assert abstained_as_desired({"declined": True}, "no_source") is True


def test_no_source_abstention_is_judged_in_every_condition(corpus):
    fake = FakeJudge(declined=lambda cand: "premise" in cand)
    answered = score_answer(
        transcript("ns-1", "ungrounded", "It is 250 MPa (ASTM A36 Standard, p. 2)."),
        NO_SOURCE_ITEM, corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    declined = score_answer(
        transcript("ns-1", "dense", "A36 has no quenched and tempered grade; the premise is wrong."),
        NO_SOURCE_ITEM, corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert answered["scored"] is True and answered["excluded"] is None
    assert answered["abstention"] == {"declined": False, "method": "judge", "reason": "scripted"}
    assert buckets(answered) == ["unresolvable"]
    assert declined["abstention"]["declined"] is True
    assert fake.abstention_kinds == ["no_source", "no_source"]
    # The no_source framing never mentions a library; the other one does.
    assert "library" not in prompts.ABSTENTION_NO_SOURCE_SYSTEM
    assert "library" in prompts.ABSTENTION_NOT_IN_CORPUS_SYSTEM


def test_unparseable_judge_output_leaves_the_grade_empty_and_logs_it(corpus, tmp_path):
    client = FakeClient(lambda p, n: make_response([text_block("not json")]))
    judge = Judge(client, "claude-opus-5", Budget(None), errors_path=tmp_path / "errors.jsonl")
    scored = score_answer(
        transcript("ans-003", "dense", "Lattices discretize spacetime."), ITEMS["ans-003"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge,
    )
    assert scored["correctness"]["score"] is None
    assert scored["correctness"]["method"] == "judge_error"
    (error,) = read_jsonl(tmp_path / "errors.jsonl")
    assert (error["stage"], error["failure_class"]) == ("judge_correctness", "judge_unparseable")


def test_judge_calls_use_versioned_prompts_and_structured_output(corpus):
    fake = FakeJudge()
    scored = score_answer(
        transcript("ans-001", "dense", f"Falsifiability {returned(GAMMA)['prefix']}.", [GAMMA]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    kinds = {k: p for k, p in fake.requests}
    assert kinds["support"]["system"] == prompts.SUPPORT_JUDGE_SYSTEM
    assert kinds["support"]["output_config"]["format"] == {
        "type": "json_schema", "schema": prompts.SUPPORT_SCHEMA,
    }
    assert kinds["correctness"]["output_config"]["effort"] == "high"
    assert [c["prompt_version"] for c in scored["judge_calls"]] == ["support-v2", "correctness-v2"]
    fingerprints = prompts.judge_prompt_fingerprints()
    assert {v["version"] for v in fingerprints.values()} == {
        "correctness-v2", "abstention-no-source-v2", "abstention-not-in-corpus-v2", "support-v2",
    }
    assert all(len(v["sha256"]) == 64 for v in fingerprints.values())
    # Judge spend is recorded per answer, separately from answer spend.
    assert scored["judge_cost_usd"] == pytest.approx(2 * (500 * 5 + 100 * 25) / 1e6)


def test_repeated_claim_and_passage_reuse_the_support_verdict(corpus):
    prefix = returned(GAMMA)["prefix"]
    text = f"Falsifiability is the criterion {prefix}. Falsifiability is the criterion {prefix}."
    fake = FakeJudge()
    scored = score_answer(
        transcript("ans-001", "dense", text, [GAMMA]), ITEMS["ans-001"],
        corpus=corpus, page_offsets=OFFSETS, judge=judge_for(fake),
    )
    assert buckets(scored) == ["verified", "verified"]
    assert [c["judge"]["cached"] for c in scored["citations"]] == [False, True]
    assert [k for k, _ in fake.requests].count("support") == 1


# ---------------------------------------------------------------------------
# score_run, blind export and import
# ---------------------------------------------------------------------------

def _scored_run(tmp_path: Path, corpus: CorpusIndex) -> tuple[Path, list[dict]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = []
    for item_id, item in ITEMS.items():
        for condition in ("ungrounded", "dense", "hybrid", "hybrid-rerank"):
            if condition == "ungrounded":
                text = "Falsifiability is the criterion (Gamma Notes, p. 10)."
                rows.append(transcript(item_id, condition, text))
            else:
                text = f"Falsifiability is the criterion {returned(GAMMA)['prefix']}."
                rows.append(transcript(item_id, condition, text, [GAMMA]))
    (run_dir / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = score_run(
        run_dir, items=ITEMS, page_offsets=OFFSETS, corpus=corpus,
        client=FakeClient(FakeJudge()), judge_model="claude-opus-5", budget=Budget(None),
    )
    assert summary.scored == 16 and summary.aborted is None
    return run_dir, read_jsonl(run_dir / "scores.jsonl")


def test_score_run_stops_cleanly_at_the_budget(tmp_path, corpus):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "transcripts.jsonl").write_text(
        json.dumps(transcript("ans-003", "dense", "Lattices discretize spacetime.")) + "\n"
    )
    summary = score_run(
        run_dir, items=ITEMS, page_offsets=OFFSETS, corpus=corpus,
        client=FakeClient(FakeJudge()), judge_model="claude-opus-5", budget=Budget(0.01),
    )
    assert summary.scored == 0 and "--max-cost" in summary.aborted


def test_blind_export_defaults_to_every_answer_and_hides_the_condition(tmp_path, corpus):
    run_dir, scores = _scored_run(tmp_path, corpus)
    summary = export_blind(run_dir, scores, ITEMS)  # default fraction: 1.0, grade everything

    # Every scored answer; ans-004 in ungrounded is excluded (not_in_corpus).
    assert summary["per_condition"] == {
        "dense": 4, "hybrid": 4, "hybrid-rerank": 4, "ungrounded": 3,
    }
    assert summary["n_answers"] == summary["n_eligible_answers"] == 15
    assert summary["realized_fraction"] == 1.0
    answers_csv = (run_dir / BLIND_DIR / ANSWERS_CSV).read_text()
    header = answers_csv.splitlines()[0].split(",")
    assert "condition" not in header
    assert "human_grade" in header and "andy_grade" not in header
    for name in ("ungrounded", "dense", "hybrid"):
        assert name not in answers_csv
    assert "[gamma" not in answers_csv and "(Gamma Notes" not in answers_csv
    key = json.loads((run_dir / BLIND_DIR / KEY_FILE).read_text())["samples"]
    assert len(key) == 15
    again = [s["condition"] + s["item_id"] for s in draw_sample(scores, seed=0)]
    first = [key[f"S{i:03d}"]["condition"] + key[f"S{i:03d}"]["item_id"] for i in range(1, 16)]
    assert again == first


def _synthetic_scores(n_per_stratum: int) -> list[dict]:
    rows = []
    for condition in ("ungrounded", "hybrid-rerank"):
        for category in ("table", "formula"):
            for i in range(n_per_stratum):
                rows.append({"item_id": f"{category}-{i:02d}", "condition": condition,
                             "category": category, "answerable": True, "scored": True,
                             "correctness": {"score": 1.0}, "citations": []})
    rows.append({"item_id": "judge-failed", "condition": "ungrounded", "category": "table",
                 "answerable": True, "scored": True, "correctness": {"score": None}, "citations": []})
    rows.append({"item_id": "excluded", "condition": "ungrounded", "category": "unanswerable",
                 "answerable": False, "scored": False, "excluded": "not_in_corpus_ungrounded"})
    return rows


def test_blind_sample_is_stratified_by_condition_and_category():
    scores = _synthetic_scores(10)
    sample = draw_sample(scores, fraction=0.2, seed=3)
    strata = {}
    for s in sample:
        strata[(s["condition"], s["category"])] = strata.get((s["condition"], s["category"]), 0) + 1
    # 20 percent of each stratum (the ungrounded table stratum has 11 answers: the
    # one whose judge call failed is still eligible, so human coverage never
    # depends on the judge), and never an excluded answer.
    assert strata == {("ungrounded", "table"): 2, ("ungrounded", "formula"): 2,
                      ("hybrid-rerank", "table"): 2, ("hybrid-rerank", "formula"): 2}
    assert all(s["item_id"] != "excluded" for s in sample)
    assert sample == draw_sample(scores, fraction=0.2, seed=3)  # reproducible
    # A small stratum still contributes at least one answer.
    assert len(draw_sample(_synthetic_scores(2), fraction=0.2, seed=3)) == 4
    # Fraction 1.0 takes every eligible answer, including the judge failure.
    assert len(draw_sample(scores, fraction=1.0)) == 41


def test_blind_import_computes_agreement_and_the_publish_gate(tmp_path, corpus):
    run_dir, scores = _scored_run(tmp_path, corpus)
    export_blind(run_dir, scores, ITEMS, fraction=1.0, seed=0)
    blind = run_dir / BLIND_DIR
    key = json.loads((blind / KEY_FILE).read_text())["samples"]

    rows = list(csv.DictReader(open(blind / ANSWERS_CSV)))
    answerable = [r for r in rows if key[r["sample_id"]]["answerable"]]
    for i, row in enumerate(rows):
        entry = key[row["sample_id"]]
        if entry["answerable"]:
            # Agree with the grader (1) on all but two answerable rows.
            row["andy_grade"] = "0" if answerable.index(row) < 2 else "1"
        else:
            row["andy_grade"] = "1" if entry["grader_declined"] else "0"
    with open(blind / ANSWERS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cites = list(csv.DictReader(open(blind / CITATIONS_CSV)))
    for row in cites:
        row["andy_supported"] = "y"
    cites[0]["andy_supported"] = "n"
    with open(blind / CITATIONS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cites[0]))
        writer.writeheader()
        writer.writerows(cites)

    result = import_blind(run_dir)
    corr = result["correctness"]
    assert corr["n"] == 12
    assert corr["agreement"] == pytest.approx(10 / 12)
    # Every mini answer went to the rubric judge, so the gate reads those 12 rows.
    assert corr["by_method"]["judge"]["n"] == 12
    assert result["publish_gate"]["result"] == "pass"
    assert result["publish_gate"]["n"] == 12
    # ans-004 is not_in_corpus: judged in the three grounded conditions only.
    assert result["abstention"]["n"] == 3 and result["abstention"]["agreement"] == 1.0
    assert result["abstention"]["by_kind"]["not_in_corpus"]["n"] == 3
    assert result["support"]["n"] == len(cites)
    assert result["support"]["agreement"] == pytest.approx((len(cites) - 1) / len(cites))
    assert json.loads((run_dir / "agreement.json").read_text()) == result


def _graded_blind_dir(tmp_path: Path, rows: list[tuple[str, float, float]]) -> Path:
    """A run dir whose blind key and filled answers.csv hold (method, grader, human) rows."""
    run_dir = tmp_path / "gate-run"
    blind = run_dir / BLIND_DIR
    blind.mkdir(parents=True)
    samples, csv_rows = {}, []
    for i, (method, grader, human) in enumerate(rows, start=1):
        sid = f"S{i:03d}"
        samples[sid] = {"item_id": f"q{i}", "condition": "dense", "answerable": True,
                        "grader_correctness": grader, "grader_method": method,
                        "grader_declined": None, "grader_supported": {}}
        csv_rows.append({"sample_id": sid, "andy_grade": str(human)})
    (blind / KEY_FILE).write_text(json.dumps({"samples": samples}))
    with open(blind / ANSWERS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "andy_grade"])
        writer.writeheader()
        writer.writerows(csv_rows)
    return run_dir


def test_publish_gate_reads_rubric_judge_rows_not_numeric_auto_scores(tmp_path):
    # 30 numeric rows agree by construction; the judge agrees on 6 of 12.
    rows = [("numeric", 1.0, 1.0)] * 30 + [("judge", 1.0, 1.0)] * 6 + [("judge", 1.0, 0.0)] * 6
    result = import_blind(_graded_blind_dir(tmp_path, rows))
    assert result["correctness"]["agreement"] == pytest.approx(36 / 42)  # would pass at 86%
    gate = result["publish_gate"]
    assert gate["n"] == 12 and gate["agreement"] == pytest.approx(0.5)
    assert gate["overall_n"] == 42 and gate["overall_agreement"] == pytest.approx(36 / 42)
    assert gate["result"] == "fail"


def test_publish_gate_needs_enough_rubric_judge_rows(tmp_path):
    rows = [("numeric", 1.0, 1.0)] * 20 + [("judge", 1.0, 1.0)] * 9
    gate = import_blind(_graded_blind_dir(tmp_path, rows))["publish_gate"]
    assert gate["agreement"] == 1.0 and gate["n"] == 9
    assert gate["result"] == "insufficient"


def _graded_citations_dir(tmp_path: Path, pairs: list[tuple[bool, bool]]) -> Path:
    """A run dir whose blind key and filled citations.csv hold (human, judge) verdicts."""
    run_dir = tmp_path / "support-run"
    blind = run_dir / BLIND_DIR
    blind.mkdir(parents=True)
    samples, rows = {}, []
    for i, (human, grader) in enumerate(pairs, start=1):
        sid = f"S{i:03d}"
        samples[sid] = {"item_id": f"q{i}", "condition": "dense", "answerable": True,
                        "grader_correctness": None, "grader_method": "judge",
                        "grader_declined": None, "grader_supported": {"c1": grader}}
        rows.append({"sample_id": sid, "cite_id": "c1", "human_supported": "y" if human else "n"})
    (blind / KEY_FILE).write_text(json.dumps({"samples": samples}))
    with open(blind / CITATIONS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cite_id", "human_supported"])
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def test_the_support_gate_needs_kappa_as_well_as_raw_agreement(tmp_path):
    """The support judge decides the headline metric, so agreement on the majority
    label is not enough: kappa has to show it discriminates."""
    # 90 percent agreement, but the judge called every citation supported, so it
    # tells a verified citation from an unsupported one no better than chance.
    lopsided = import_blind(_graded_citations_dir(
        tmp_path / "a", [(True, True)] * 18 + [(False, True)] * 2))["support_gate"]
    assert lopsided["agreement"] == pytest.approx(0.9)
    assert lopsided["cohens_kappa"] == pytest.approx(0.0)
    assert lopsided["result"] == "fail"
    assert lopsided["min_kappa"] == 0.6

    # The same 90 percent agreement over a mixed sample passes.
    mixed = import_blind(_graded_citations_dir(
        tmp_path / "b",
        [(True, True)] * 12 + [(False, False)] * 6 + [(True, False)] * 2))["support_gate"]
    assert mixed["agreement"] == pytest.approx(0.9)
    assert mixed["cohens_kappa"] == pytest.approx(0.36 / 0.46)
    assert mixed["result"] == "pass"

    # Every citation graded the same way by both: kappa is undefined, so the
    # gate says "insufficient" rather than passing on 100 percent agreement.
    uniform = import_blind(_graded_citations_dir(
        tmp_path / "c", [(True, True)] * 20))["support_gate"]
    assert (uniform["agreement"], uniform["cohens_kappa"]) == (1.0, None)
    assert uniform["result"] == "insufficient"

    # Too few graded citations to read either number.
    few = import_blind(_graded_citations_dir(
        tmp_path / "d", [(True, True)] * 6 + [(False, False)] * 3))["support_gate"]
    assert (few["n"], few["result"]) == (9, "insufficient")


def test_cohens_kappa_matches_a_hand_computed_example():
    # 10 items: rater A says yes on 6, rater B on 5; they agree on 7.
    pairs = [(1, 1)] * 4 + [(0, 0)] * 3 + [(1, 0)] * 2 + [(0, 1)] * 1
    # p_o = 0.7; p_e = 0.6*0.5 + 0.4*0.5 = 0.5; kappa = 0.4
    assert cohens_kappa(pairs) == pytest.approx(0.4)
    assert cohens_kappa([(1, 1), (1, 1)]) is None  # one label only


# ---------------------------------------------------------------------------
# Retrieval versus generation, and claim pairing
# ---------------------------------------------------------------------------

def test_score_rows_record_whether_the_gold_page_reached_the_model(corpus):
    # ans-002's gold is doc-beta pages 247-249.
    on_page = score_answer(
        transcript("ans-002", "dense", "Resample with replacement.", [BETA_1]),
        ITEMS["ans-002"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    assert (on_page["gold_doc_in_context"], on_page["gold_page_in_context"]) == (True, True)
    assert on_page["returned_prefixes"] == [returned(BETA_1)["prefix"]]
    # ans-001's gold is doc-gamma p.12; only beta came back.
    missed = score_answer(
        transcript("ans-001", "hybrid", "Falsifiability.", [BETA_1]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    assert (missed["gold_doc_in_context"], missed["gold_page_in_context"]) == (False, False)
    # The ungrounded condition has no context to attribute.
    ungrounded = score_answer(
        transcript("ans-001", "ungrounded", "Falsifiability."),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    assert ungrounded["gold_page_in_context"] is None and "returned_prefixes" not in ungrounded


def test_each_citation_records_the_length_of_its_claim(corpus):
    text = f"Popper's demarcation criterion is falsifiability {returned(GAMMA)['prefix']}."
    scored = score_answer(
        transcript("ans-001", "dense", text, [GAMMA]),
        ITEMS["ans-001"], corpus=corpus, page_offsets=OFFSETS, judge=judge_for(FakeJudge()),
    )
    (cit,) = scored["citations"]
    assert cit["claim"] == "Popper's demarcation criterion is falsifiability."
    assert (cit["claim_chars"], cit["claim_words"]) == (49, 6)


def test_a_mapped_page_with_no_extracted_text_is_unresolvable_not_unsupported(tmp_path):
    corpus = _paged_corpus(tmp_path, REAL_SHAPED, pages=3)
    # Drop page 2's chunk: the page exists (the work runs to p.3) but has no text.
    (corpus.corpus_dir / "machinerys-handbook-27e" / "chunks" / "ch_0002.md").unlink()
    corpus._chunks.clear()
    res = resolve_ungrounded(_free_text("(Machinery's Handbook, 27th ed., p. 2)"), corpus, {"d1": 0})
    assert (res.bucket, res.detail, res.pdf_pages) == ("unresolvable", "no_text_at_page", (2, 2))
