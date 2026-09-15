"""Shared test doubles for the grounded-answer benchmark tests (Epic 25).

* ``FakeClient``: stands in for ``anthropic.Anthropic``; ``messages.create``
  records every request and returns whatever the scripted responder returns
  (or raises it, when the responder returns an exception). No network.
* ``ScriptedModel``: a deterministic responder that plays the answer model
  (searches once, then cites a returned prefix) and, from Story 25.3 on, the
  judge (JSON verdicts chosen by the test).
* ``build_stub_mini_index``: a FAISS + BM25 index of the real mini corpus
  chunks, embedded with a deterministic stub so no model download is needed.
  Chunk ``file_path`` values are real, so the MCP server's ``search_corpus``
  reads the real chunk files in-process.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import numpy as np
import yaml

FIXTURES_ROOT = Path(__file__).resolve().parent / "eval_fixtures"
MINI_CORPUS = FIXTURES_ROOT / "mini_corpus"
MINI_ANSWERS_YAML = FIXTURES_ROOT / "mini_answers.yaml"
MINI_AGENTS_DIR = FIXTURES_ROOT / "agents"
DIM = 384

PREFIX_LINE_RE = re.compile(r"^\[[^\]\n]+\]$", re.M)

# ---------------------------------------------------------------------------
# Stub embedder
# ---------------------------------------------------------------------------

_TOPIC_TOKENS = {
    0: {"quantum", "mechanics", "transformer", "field", "lattice", "renormalization"},
    1: {"bootstrap", "confidence", "interval", "percentile", "resampling", "resample"},
    2: {"falsifiability", "popper", "pseudoscience", "demarcation", "scientific"},
}


def stub_vector(text: str) -> np.ndarray:
    """Deterministic 384-d unit vector keyed on topic words."""
    vec = np.zeros(DIM, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        for axis, words in _TOPIC_TOKENS.items():
            if token in words:
                vec[axis] += 1.0
    vec[3 + (len(text) % (DIM - 3))] += 0.01  # break exact ties
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class StubEncoder:
    """Mimics SentenceTransformer.encode for the MCP server."""

    def encode(self, texts, normalize_embeddings: bool = True):
        return np.stack([stub_vector(t) for t in texts])


def lexical_rerank(query, chunks, *, config, text_key="content"):
    """Deterministic stand-in for the cross-encoder: query-word overlap."""
    words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for position, chunk in enumerate(chunks):
        body = set(re.findall(r"\w+", str(chunk.get(text_key, "")).lower()))
        score = float(len(words & body)) - position * 1e-6
        new = dict(chunk)
        new.update(rerank_score=score, faiss_distance=chunk.get("score"), score=score)
        scored.append(new)
    scored.sort(key=lambda d: d["rerank_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Stub index over the real mini corpus
# ---------------------------------------------------------------------------

def mini_chunks() -> List[Dict[str, Any]]:
    """Every mini-corpus chunk with its front matter and body."""
    out = []
    for doc_dir in sorted(p for p in MINI_CORPUS.iterdir() if p.is_dir()):
        for path in sorted((doc_dir / "chunks").glob("ch_*.md")):
            text = path.read_text(encoding="utf-8")
            _, front, body = text.split("---", 2)
            meta = yaml.safe_load(front)
            out.append(
                {
                    "doc_id": meta["doc_id"],
                    "chunk_id": f"{meta['doc_id']}_{path.stem}",
                    "file_path": str(path.relative_to(MINI_CORPUS)),
                    "meta": meta,
                    "body": body.strip(),
                }
            )
    return out


def build_stub_mini_index(parent: Path, name: str = "mini") -> Path:
    from grounding.bm25 import write_bm25_index
    from grounding.vector_store import write_vector_index

    out = Path(parent) / name
    out.mkdir(parents=True, exist_ok=True)
    chunks = mini_chunks()
    write_vector_index(
        {c["chunk_id"]: stub_vector(c["body"]) for c in chunks},
        out,
        chunk_metadata={
            c["chunk_id"]: {"doc_id": c["doc_id"], "file_path": c["file_path"], "is_music": False}
            for c in chunks
        },
    )
    write_bm25_index(
        [c["body"] for c in chunks],
        [c["chunk_id"] for c in chunks],
        out,
        chunk_doc_ids=[c["doc_id"] for c in chunks],
    )
    return out


class FakeSearchTool:
    """MCP-free stand-in for ``SearchCorpusTool``.

    Ranks the real mini-corpus chunks with the stub embedder and formats them
    the way the MCP server's ``format_results_for_context`` does, so tests
    that need real chunk text flowing through a run (the publishable-report
    proof) also run where the optional ``mcp`` package is not installed.
    """

    def __init__(self, condition=None) -> None:
        self.condition = condition

    def execute(self, name, tool_input):
        from grounding.citations import _derive_slug, format_citation_prefix
        from grounding.eval.answers.tool import ToolExecution

        query = (tool_input or {}).get("query", "")
        top_k = int((tool_input or {}).get("top_k", 5))
        qv = stub_vector(query)
        ranked = sorted(mini_chunks(), key=lambda c: -float(np.dot(qv, stub_vector(c["body"]))))
        results, lines = [], [
            "## Corpus Search Results", f"Query: {query}", f"Found {min(top_k, len(ranked))} relevant chunks:", "",
        ]
        for rank, chunk in enumerate(ranked[:top_k], start=1):
            meta = chunk["meta"]
            prefix = format_citation_prefix(
                meta["source"], meta["page_start"], meta["page_end"], meta.get("section_heading")
            )
            results.append({
                "rank": rank, "score": 1.0 / rank, "source": meta["source"],
                "doc_id": meta["doc_id"], "chunk_id": meta["chunk_id"],
                "page_start": meta["page_start"], "page_end": meta["page_end"],
                "section_heading": meta.get("section_heading"), "content": chunk["body"],
                "slug": _derive_slug(meta["source"]), "prefix": prefix,
            })
            lines += [f"### [{rank}] {meta['source']} (score: {1.0 / rank})", prefix,
                      f"*doc_id: {meta['doc_id']}, chunk: {meta['chunk_id']}*", "",
                      chunk["body"], "", "---", ""]
        return ToolExecution(query, top_k, "\n".join(lines), False, 0.0, results)


def stub_run_eval(fixture_set, agent, **kwargs):
    """The Epic 16 runner with the stub embedder in both retrieval paths."""
    from grounding.eval.runner import run_eval
    from grounding.hybrid import search_hybrid

    def hybrid_fn(query, embeddings_dir, *, top_k, pool_size, k_rrf):
        return search_hybrid(
            query, embeddings_dir, top_k=top_k, pool_size=pool_size, k_rrf=k_rrf,
            embed_fn=stub_vector,
        )

    return run_eval(fixture_set, agent, embed_fn=stub_vector, hybrid_fn=hybrid_fn, **kwargs)


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------

def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block(text: str = "Planning the answer.") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, signature="sig-fake")


def tool_use_block(block_id: str, query: str, top_k: int | None = None) -> SimpleNamespace:
    tool_input: Dict[str, Any] = {"query": query}
    if top_k is not None:
        tool_input["top_k"] = top_k
    return SimpleNamespace(type="tool_use", id=block_id, name="search_corpus", input=tool_input)


def make_response(
    blocks: List[Any],
    *,
    model: str = "claude-opus-5",
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_fake",
        type="message",
        role="assistant",
        model=model,
        stop_reason=stop_reason,
        content=blocks,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


class _FakeMessages:
    def __init__(self, responder: Callable[[Dict[str, Any], int], Any]) -> None:
        self._responder = responder
        self.calls: List[Dict[str, Any]] = []

    def create(self, **params):
        self.calls.append(copy.deepcopy(params))
        result = self._responder(params, len(self.calls))
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self, responder: Callable[[Dict[str, Any], int], Any]) -> None:
        self.messages = _FakeMessages(responder)


class FakeAPIError(Exception):
    def __init__(self, status_code: int, message: str = "fake API error") -> None:
        super().__init__(message)
        self.status_code = status_code


def system_text(params: Dict[str, Any]) -> str:
    system = params.get("system", "")
    if isinstance(system, list):
        return " ".join(b.get("text", "") for b in system)
    return system


def last_tool_result_text(params: Dict[str, Any]) -> str | None:
    last = params["messages"][-1]
    if last["role"] == "user" and isinstance(last["content"], list):
        return "\n".join(str(b.get("content", "")) for b in last["content"])
    return None


class ScriptedModel:
    """Deterministic responder for answer calls (and judge calls, see 25.3).

    Grounded conditions: search once with the question, then answer citing the
    first prefix the tool returned. Ungrounded: answer with the text given in
    ``ungrounded_answers[item query]`` or a default free-text citation.
    ``judge`` (optional) handles calls whose system prompt is a judge prompt.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        ungrounded_answers: Dict[str, str] | None = None,
        grounded_answer: Callable[[str, List[str]], str] | None = None,
        judge: Callable[[Dict[str, Any], int], Any] | None = None,
    ) -> None:
        self.model = model
        self.ungrounded_answers = ungrounded_answers or {}
        self.grounded_answer = grounded_answer or (
            lambda question, prefixes: (
                f"The library answers this directly {prefixes[0]}." if prefixes
                else "The library does not contain the answer."
            )
        )
        self.judge = judge

    def __call__(self, params: Dict[str, Any], n: int):
        # Judge calls are the ones that request structured JSON output.
        if self.judge is not None and (params.get("output_config") or {}).get("format"):
            return self.judge(params, n)
        question = params["messages"][0]["content"]
        if "tools" in params:
            tool_text = last_tool_result_text(params)
            if tool_text is None:
                return make_response(
                    [thinking_block(), tool_use_block(f"toolu_{n}", question)],
                    model=self.model,
                    stop_reason="tool_use",
                )
            prefixes = PREFIX_LINE_RE.findall(tool_text)
            return make_response(
                [text_block(self.grounded_answer(question, prefixes))], model=self.model
            )
        answer = self.ungrounded_answers.get(
            question, "I believe this is covered in the Gamma Notes (Gamma Notes, 1st ed., p. 10)."
        )
        return make_response([text_block(answer)], model=self.model)


def _tag(user: str, name: str) -> str:
    match = re.search(rf"<{name}>\n(.*?)\n</{name}>", user, re.S)
    return match.group(1) if match else ""


class FakeJudge:
    """Scripted judge: verdicts come from the rules the test passes in."""

    def __init__(self, supported=None, correctness=None, declined=None, model=None,
                 gap_flagged=None, gave_answer=None, answer_labeled=None):
        self.supported = supported or (lambda claim, passage: True)
        self.correctness = correctness or (lambda candidate: ("1", False))
        self.declined = declined or (lambda candidate: True)
        # The not_in_corpus judge (abstention-not-in-corpus-v2) reports three
        # facts; by default they follow ``declined``.
        self.gap_flagged = gap_flagged or (lambda candidate: self.declined(candidate))
        self.gave_answer = gave_answer or (lambda candidate: not self.declined(candidate))
        self.answer_labeled = answer_labeled or (lambda candidate: True)
        # None: report whichever model was requested, as the real API does.
        self.model = model
        self.requests: list[tuple[str, dict]] = []
        # Which abstention framing each abstention call used (no_source / not_in_corpus).
        self.abstention_kinds: list[str] = []

    def __call__(self, params, n):
        system = params["system"]
        user = params["messages"][0]["content"]
        from grounding.eval.answers import prompts

        if system == prompts.SUPPORT_JUDGE_SYSTEM:
            kind = "support"
            ok = self.supported(_tag(user, "claim"), _tag(user, "passage"))
            data = {"supported": ok, "reason": "scripted"}
        elif system == prompts.CORRECTNESS_JUDGE_SYSTEM:
            kind = "correctness"
            score, declined = self.correctness(_tag(user, "candidate_answer"))
            data = {"score": score, "declined": declined, "missing_facts": [], "reason": "scripted"}
        elif system == prompts.ABSTENTION_NO_SOURCE_SYSTEM:
            kind = "abstention"
            self.abstention_kinds.append("no_source")
            data = {"declined": self.declined(_tag(user, "candidate_answer")), "reason": "scripted"}
        elif system == prompts.ABSTENTION_NOT_IN_CORPUS_SYSTEM:
            kind = "abstention"
            self.abstention_kinds.append("not_in_corpus")
            candidate = _tag(user, "candidate_answer")
            data = {"gap_flagged": self.gap_flagged(candidate),
                    "gave_answer": self.gave_answer(candidate),
                    "answer_labeled": self.answer_labeled(candidate), "reason": "scripted"}
        else:  # pragma: no cover - a test bug
            raise AssertionError("unexpected judge prompt")
        self.requests.append((kind, params))
        return make_response([text_block(json.dumps(data))],
                             model=self.model or params.get("model", "claude-opus-5"),
                             input_tokens=500, output_tokens=100)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
