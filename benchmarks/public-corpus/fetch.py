#!/usr/bin/env python3
"""Fetch and verify the public-domain benchmark corpus listed in manifest.yaml.

Every document in the manifest is downloaded to ``<dest>/<id>.pdf`` and checked
against the SHA-256 recorded in the manifest, so anyone can rebuild the same
corpus from this directory alone.

Modes
-----
default        Download anything missing, verify every file against the manifest
               hash, and fail loudly on a mismatch or on an entry with no hash yet.
--record       First-run mode: download, then write sha256, bytes, pages,
               last_modified, fetched_utc and fetched_from into the manifest for
               entries that have no hash yet. An entry that already has a hash is
               still verified, never silently overwritten (use --update-hashes for
               a deliberate re-pin after an upstream re-post).
--verify-only  Never touch the network: check the files already on disk.

Behaviour
---------
* Idempotent: a file whose hash already matches is left alone.
* Streams to ``<id>.pdf.part`` and renames on success, so a partial download
  never masquerades as a finished one.
* Sends a normal browser User-Agent (several .gov hosts reject the default
  urllib one) and retries each URL once on a transient failure.
* Falls back to ``fallback_url`` when the primary URL fails.
* Rejects any response that is not a PDF (``%PDF-`` magic), which catches HTML
  interstitials and bot checks that come back with HTTP 200.

Dependencies: Python 3.10+, PyYAML. Page counts use pypdf when installed and
``pdfinfo`` (poppler) otherwise; if neither is available ``pages`` stays null.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.yaml"
DEFAULT_DEST = Path(
    os.environ.get("CORPORA_PUBLIC", str(Path.home() / "Documents" / "Corpora-public"))
) / "originals"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "grounding-ai-public-corpus/1.0 (+https://github.com/andyliszewski/grounding-ai)"
)
CHUNK = 1 << 20  # 1 MiB read size for streaming
RECORDED_FIELDS = ("sha256", "bytes", "pages", "last_modified", "fetched_utc", "fetched_from")


# ----------------------------------------------------------------------------
# Manifest I/O (keeps the leading comment header and field order)
# ----------------------------------------------------------------------------

def load_manifest(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    header: List[str] = []
    for line in text.splitlines():
        if line.startswith("#") or line.strip() == "":
            header.append(line)
        else:
            break
    # Trim trailing blank lines from the header so the rewrite stays tidy.
    while header and header[-1].strip() == "":
        header.pop()
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        raise SystemExit(f"{path}: expected a top-level 'documents:' list")
    return header, data


def save_manifest(path: Path, header: List[str], data: Dict[str, Any]) -> None:
    body = yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, width=1000, default_flow_style=None
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(header) + "\n\n" + body, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def page_count(path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001 - fall through to pdfinfo
        pass
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    try:
        out = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def is_pdf(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(5) == b"%PDF-"


class FetchError(Exception):
    pass


def _download_once(url: str, part: Path, timeout: float) -> Dict[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as out:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
    expected = headers.get("content-length")
    if expected is not None and expected.isdigit() and part.stat().st_size != int(expected):
        raise FetchError(f"short read: got {part.stat().st_size} of {expected} bytes")
    if not is_pdf(part):
        ctype = headers.get("content-type", "?")
        raise FetchError(f"response is not a PDF (content-type {ctype}); likely an HTML interstitial")
    return headers


def download(url: str, part: Path, timeout: float, retries: int = 1) -> Dict[str, str]:
    """Download ``url`` to ``part``, retrying ``retries`` times on transient failure."""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return _download_once(url, part, timeout)
        except urllib.error.HTTPError as exc:
            last = exc
            # 4xx is not transient; do not hammer the host.
            if 400 <= exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError, FetchError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(3)
    if part.exists():
        part.unlink()
    raise FetchError(f"{url}: {last}")


def fmt_bytes(n: int) -> str:
    return f"{n / 1_048_576:.2f} MB"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="directory for <id>.pdf files")
    ap.add_argument("--record", action="store_true", help="record hash, size, pages for entries with no hash yet")
    ap.add_argument("--update-hashes", action="store_true",
                    help="with --record: overwrite an existing hash that no longer matches (deliberate re-pin)")
    ap.add_argument("--verify-only", action="store_true", help="no downloads; verify files on disk")
    ap.add_argument("--only", default="", help="comma-separated manifest ids to process")
    ap.add_argument("--timeout", type=float, default=120.0, help="socket timeout per request, seconds")
    args = ap.parse_args(argv)

    header, data = load_manifest(args.manifest)
    docs: List[Dict[str, Any]] = data["documents"]
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        unknown = only - {d["id"] for d in docs}
        if unknown:
            print(f"ERROR: unknown ids in --only: {sorted(unknown)}", file=sys.stderr)
            return 2
    args.dest.mkdir(parents=True, exist_ok=True)

    ok: List[str] = []
    failed: List[Tuple[str, str]] = []
    changed = False

    for doc in docs:
        doc_id = doc["id"]
        if only and doc_id not in only:
            continue
        dest = args.dest / f"{doc_id}.pdf"
        part = args.dest / f"{doc_id}.pdf.part"
        recorded = doc.get("sha256")

        # 1. Obtain the file (unless it is already there).
        fetched_from: Optional[str] = None
        headers: Dict[str, str] = {}
        if not dest.exists():
            if args.verify_only:
                failed.append((doc_id, "missing on disk"))
                print(f"MISSING  {doc_id}")
                continue
            candidates = [("url", doc.get("url")), ("fallback_url", doc.get("fallback_url"))]
            errors: List[str] = []
            for label, url in candidates:
                if not url:
                    continue
                print(f"GET      {doc_id}  <- {url}", flush=True)
                try:
                    headers = download(url, part, timeout=args.timeout)
                    fetched_from = label
                    break
                except FetchError as exc:
                    errors.append(str(exc))
                    print(f"  failed: {exc}", file=sys.stderr)
            if fetched_from is None:
                failed.append((doc_id, "; ".join(errors) or "no url"))
                continue
            os.replace(part, dest)

        # 2. Verify.
        if not is_pdf(dest):
            failed.append((doc_id, "file on disk is not a PDF"))
            print(f"BAD      {doc_id}  not a PDF: {dest}")
            continue
        digest = sha256_of(dest)
        size = dest.stat().st_size

        if recorded and digest == recorded:
            print(f"OK       {doc_id}  {fmt_bytes(size)}  sha256 matches")
            ok.append(doc_id)
            continue

        if recorded and digest != recorded:
            if args.record and args.update_hashes:
                print(f"REPIN    {doc_id}  hash changed, recording the new one (--update-hashes)")
            else:
                failed.append((doc_id, f"SHA-256 MISMATCH: manifest {recorded[:12]}..., file {digest[:12]}..."))
                print(f"MISMATCH {doc_id}  manifest {recorded}\n                  file     {digest}")
                continue
        elif not recorded and not args.record:
            failed.append((doc_id, "manifest has no sha256 for this entry; run with --record"))
            print(f"NOHASH   {doc_id}  file sha256 {digest} (not recorded; use --record)")
            continue

        # 3. Record (first run, or a deliberate re-pin).
        doc["sha256"] = digest
        doc["bytes"] = size
        doc["pages"] = page_count(dest)
        if fetched_from is not None:
            doc["fetched_from"] = fetched_from
            doc["last_modified"] = headers.get("last-modified")
            doc["fetched_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif not doc.get("fetched_from"):
            # File was already on disk from an earlier, unrecorded run.
            doc["fetched_from"] = "pre-existing file"
            doc["fetched_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        changed = True
        print(f"RECORDED {doc_id}  {fmt_bytes(size)}  pages={doc['pages']}  sha256={digest[:16]}...")
        ok.append(doc_id)

    if changed:
        save_manifest(args.manifest, header, data)
        print(f"manifest updated: {args.manifest}")

    print(f"\n{len(ok)} ok, {len(failed)} failed")
    for doc_id, why in failed:
        print(f"  FAIL {doc_id}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
