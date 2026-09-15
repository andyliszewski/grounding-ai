"""Run provenance for the answer benchmark (Epic 25, D8).

Each run records the git commit it ran from (and whether the tree was dirty),
a fingerprint of the retrieval index and corpus manifest, and hashes of the
fixture and prompts, so a result can be traced to exactly what produced it.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

INDEX_FILES = ("_embeddings.faiss", "_chunk_map.json", "_bm25.pkl", "_bm25_map.json")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: Path | None = None) -> Dict[str, Any]:
    """Commit SHA and dirty flag of the checkout holding this package."""
    root = repo_root or _REPO_ROOT

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
    }


def index_fingerprint(embeddings_dir: Path, corpus_dir: Path) -> Dict[str, Any]:
    """SHA-256 and size of each retrieval artifact plus the corpus manifest."""
    files: Dict[str, Any] = {}
    for name in INDEX_FILES:
        path = Path(embeddings_dir) / name
        if path.exists():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = Path(corpus_dir) / "_index.json"
    if manifest.exists():
        files["corpus/_index.json"] = {
            "sha256": sha256_file(manifest),
            "bytes": manifest.stat().st_size,
        }
    combined = hashlib.sha256(
        "".join(f"{k}:{v['sha256']};" for k, v in sorted(files.items())).encode("utf-8")
    ).hexdigest()
    return {"combined_sha256": combined, "files": files}


def _module_constant(path: Path, name: str) -> str | None:
    """A module-level string constant, read without importing the module.

    The embedding modules import sentence-transformers and torch at import
    time; reading the constant from source keeps manifests cheap and works
    where those packages are absent.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def embedding_provenance(repo_root: Path | None = None) -> Dict[str, Any]:
    """The embedding models behind retrieval and the library versions (D8).

    ``query_model`` is what the corpus-search MCP server encodes queries with
    (the grounded conditions); ``index_model`` is what ``grounding
    embeddings`` builds indexes with. They must agree for dense retrieval to
    mean anything, so a mismatch is worth seeing in every run.
    """
    root = repo_root or _REPO_ROOT
    return {
        "query_model": _module_constant(root / "mcp_servers" / "corpus_search" / "server.py",
                                        "EMBEDDING_MODEL"),
        "index_model": _module_constant(root / "grounding" / "embedder.py", "MODEL_NAME"),
        "sentence_transformers": _dist_version("sentence-transformers"),
        "torch": _dist_version("torch"),
        "faiss": _dist_version("faiss-cpu"),
    }
