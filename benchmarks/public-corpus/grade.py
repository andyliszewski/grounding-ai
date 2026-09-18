#!/usr/bin/env python3
"""Local blind-grading UI for an answer-benchmark run (Epic 25, Story 25.6).

Serves one answer (or citation, or resolution row) at a time and writes every grade
straight back into the run's `blind/*.csv`, so `grounding eval-answers --run-dir <run>
--import-grades` picks them up unchanged. Nothing leaves the machine: no network calls,
no model calls, localhost only.

    ./venv/bin/python benchmarks/public-corpus/grade.py <run-dir> [--port 8765]

The CSVs carry no condition column, so grading stays blind. Citations are shown in a
seeded shuffle, so stopping early leaves a random sample rather than one answer's worth.
A one-time `.bak` copy of each CSV is made on first write.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TABLES = {
    "answers": ("sample_id", ("human_grade", "human_notes")),
    "citations": ("sample_id", ("human_supported", "human_notes")),
    "resolution_audit": ("audit_id", ("human_agrees", "human_bucket", "human_notes")),
}
LOCK = threading.Lock()


class Store:
    """The three CSVs, held in memory and rewritten in place on every save."""

    def __init__(self, blind_dir: Path) -> None:
        self.dir = blind_dir
        self.rows: dict[str, list[dict]] = {}
        self.fields: dict[str, list[str]] = {}
        for name in TABLES:
            path = self.dir / f"{name}.csv"
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self.fields[name] = list(reader.fieldnames or [])
                self.rows[name] = list(reader)
        # Citations get a seeded display order so a partial grading is a random sample.
        order = list(range(len(self.rows["citations"])))
        random.Random(0).shuffle(order)
        self.citation_order = order

    def key_of(self, name: str, row: dict) -> str:
        key_col = TABLES[name][0]
        return row[key_col] if name != "citations" else f"{row['sample_id']}/{row['cite_id']}"

    def save(self, name: str, key: str, values: dict) -> dict:
        editable = TABLES[name][1]
        with LOCK:
            target = next((r for r in self.rows[name] if self.key_of(name, r) == key), None)
            if target is None:
                raise KeyError(key)
            for field, value in values.items():
                if field not in editable:
                    raise KeyError(field)
                target[field] = value
            self._write(name)
        return target

    def _write(self, name: str) -> None:
        path = self.dir / f"{name}.csv"
        backup = path.with_suffix(".csv.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields[name])
            writer.writeheader()
            writer.writerows(self.rows[name])
        os.replace(tmp, path)

    def payload(self) -> dict:
        cites = [dict(self.rows["citations"][i], _key=self.key_of("citations", self.rows["citations"][i]))
                 for i in self.citation_order]
        answers = [dict(r, _key=self.key_of("answers", r)) for r in self.rows["answers"]]
        audits = [dict(r, _key=self.key_of("resolution_audit", r)) for r in self.rows["resolution_audit"]]
        return {"answers": answers, "citations": cites, "resolution_audit": audits}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Blind grading</title>
<style>
:root { --bg:#faf9f7; --fg:#1a1a18; --mut:#6b6b66; --line:#e3e1dc; --acc:#2d6a4f; }
* { box-sizing:border-box }
body { margin:0; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--fg) }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 20px; display:flex; gap:18px; align-items:center; flex-wrap:wrap }
nav button { font:inherit; border:1px solid var(--line); background:#fff; padding:6px 12px; border-radius:6px; cursor:pointer }
nav button.on { background:var(--acc); color:#fff; border-color:var(--acc) }
main { max-width:900px; margin:0 auto; padding:22px 20px 80px }
.card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:16px }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.07em; color:var(--mut); margin:0 0 8px }
pre { white-space:pre-wrap; word-wrap:break-word; font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; margin:0 }
.q { font-size:17px; font-weight:600 }
.grades { display:flex; gap:10px; flex-wrap:wrap; margin-top:6px }
.grades button { font:inherit; padding:9px 16px; border-radius:8px; border:1px solid var(--line); background:#fff; cursor:pointer }
.grades button:hover { border-color:var(--acc) }
.grades button.sel { background:var(--acc); color:#fff; border-color:var(--acc) }
input[type=text] { width:100%; font:inherit; padding:8px 10px; border:1px solid var(--line); border-radius:6px }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; flex:1; min-width:120px }
.bar div { height:100%; background:var(--acc) }
.meta { color:var(--mut); font-size:13px }
.nav2 { display:flex; gap:10px; align-items:center; margin-top:10px }
kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:1px 5px; font:12px ui-monospace,monospace; background:#fff }
</style></head><body>
<header>
  <nav>
    <button id="t-answers" class="on" onclick="tab('answers')">Answers</button>
    <button id="t-citations" onclick="tab('citations')">Citations</button>
    <button id="t-resolution_audit" onclick="tab('resolution_audit')">Resolution</button>
  </nav>
  <div class="bar"><div id="prog"></div></div>
  <span class="meta" id="count"></span>
  <span class="meta" id="saved"></span>
</header>
<main id="main"></main>
<script>
let DATA = {}, T = 'answers', I = 0;
const el = s => document.querySelector(s);
const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const graded = r => T==='answers' ? r.human_grade!=='' : T==='citations' ? r.human_supported!=='' : r.human_agrees!=='';

async function load() {
  DATA = await (await fetch('/data')).json();
  // A subset folder (such as a citation retest) leaves some tables empty: hide their tabs.
  const tables = ['answers', 'citations', 'resolution_audit'];
  tables.forEach(t => { if (!DATA[t].length) el('#t-'+t).style.display = 'none'; });
  const first = tables.find(t => DATA[t].length);
  if (first) tab(first); else render();
}
function tab(t) { T=t; I=0; document.querySelectorAll('nav button').forEach(b=>b.classList.remove('on')); el('#t-'+t).classList.add('on'); nextUngraded(); }
function nextUngraded() { const rows=DATA[T]; const j=rows.findIndex((r,k)=>k>=I && !graded(r)); I = j>=0 ? j : I; render(); }

async function save(vals) {
  const row = DATA[T][I];
  Object.assign(row, vals);
  el('#saved').textContent = 'saving…';
  await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({table:T, key:row._key, values:vals})});
  el('#saved').textContent = 'saved';
  render();
}
function grade(v) { save(T==='answers'?{human_grade:v}:T==='citations'?{human_supported:v}:{human_agrees:v}).then(()=>setTimeout(()=>{ if(I<DATA[T].length-1){I++; render();} },120)); }
function note(v) { save({human_notes:v}); }
function move(d) { I=Math.max(0,Math.min(DATA[T].length-1,I+d)); render(); }

function render() {
  const rows = DATA[T]||[], r = rows[I]; if(!r) return;
  const done = rows.filter(graded).length;
  el('#prog').style.width = (100*done/rows.length)+'%';
  el('#count').textContent = `${done} of ${rows.length} graded · showing ${I+1}`;
  let body = '';
  if (T==='answers') {
    const scale = r.grade_scale;
    const opts = scale.startsWith('1, 0.5') ? [['1','1 — correct'],['0.5','0.5 — partly'],['0','0 — wrong']]
                                            : [['1','1 — yes'],['0','0 — no']];
    body = `
    <div class="card"><h2>Question · ${esc(r.item_id)} · ${esc(r.category)}</h2><div class="q">${esc(r.question)}</div></div>
    <div class="card"><h2>Answer (condition hidden)</h2><pre>${esc(r.answer_without_citations)}</pre></div>
    <div class="card"><h2>Gold</h2><pre>${esc(r.gold)}</pre>
      ${r.must_include?`<h2 style="margin-top:12px">Must include</h2><pre>${esc(r.must_include)}</pre>`:''}</div>
    <div class="card"><h2>Grade</h2><div class="meta">${esc(scale)}</div>
      <div class="grades">${opts.map(([v,l])=>`<button class="${r.human_grade===v?'sel':''}" onclick="grade('${v}')">${l}</button>`).join('')}</div>
      <div class="nav2"><input type="text" placeholder="notes (optional)" value="${esc(r.human_notes)}" onchange="note(this.value)"></div></div>`;
  } else if (T==='citations') {
    body = `
    <div class="card"><h2>Question</h2><div class="q">${esc(r.question)}</div></div>
    <div class="card"><h2>Claim</h2><pre>${esc(r.claim)}</pre></div>
    <div class="card"><h2>Passage it cites</h2><pre>${esc(r.passage)}</pre></div>
    <div class="card"><h2>Does the passage support the claim?</h2>
      <div class="meta">Yes only when the passage states the claim or directly implies it, including any
        specific numbers, units and conditions. No when the passage is on topic but lacks the specific fact,
        contradicts it, or supports only part of a claim that has two or more parts. Judge against this
        passage only, not against what you know. Use the question only to work out what the claim refers to;
        whether the claim answers the question is not the test.</div>
      <div class="grades">
        <button class="${r.human_supported==='yes'?'sel':''}" onclick="grade('yes')">yes — supported</button>
        <button class="${r.human_supported==='no'?'sel':''}" onclick="grade('no')">no — not supported</button></div>
      <div class="nav2"><input type="text" placeholder="notes (optional)" value="${esc(r.human_notes)}" onchange="note(this.value)"></div></div>`;
  } else {
    body = `
    <div class="card"><h2>Question · ${esc(r.item_id)}</h2><div class="q">${esc(r.question)}</div></div>
    <div class="card"><h2>Citation</h2><pre>${esc(r.citation)}</pre><h2 style="margin-top:12px">Claim</h2><pre>${esc(r.claim)}</pre></div>
    <div class="card"><h2>What the scorer decided</h2><pre>${esc(r.auto_bucket)} · ${esc(r.auto_detail)}\n${esc(r.scorer_looked_at)}</pre></div>
    <div class="card"><h2>Do you agree with that call?</h2>
      <div class="grades">
        <button class="${r.human_agrees==='yes'?'sel':''}" onclick="grade('yes')">yes</button>
        <button class="${r.human_agrees==='no'?'sel':''}" onclick="grade('no')">no</button></div>
      <div class="nav2"><input type="text" placeholder="if no, the right bucket" value="${esc(r.human_bucket)}" onchange="save({human_bucket:this.value})">
      <input type="text" placeholder="notes" value="${esc(r.human_notes)}" onchange="note(this.value)"></div></div>`;
  }
  el('#main').innerHTML = body + `
    <div class="nav2"><button onclick="move(-1)">← previous</button><button onclick="move(1)">next →</button>
    <button onclick="nextUngraded()">next ungraded</button>
    <span class="meta">keys: <kbd>1</kbd> <kbd>h</kbd> (0.5) <kbd>0</kbd> · <kbd>y</kbd> <kbd>n</kbd> · <kbd>←</kbd> <kbd>→</kbd></span></div>`;
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key==='ArrowLeft') move(-1); else if (e.key==='ArrowRight') move(1);
  else if (e.key==='1') grade(T==='answers'?'1':'yes');
  else if (e.key==='h' && T==='answers') grade('0.5');
  else if (e.key==='0') grade(T==='answers'?'0':'no');
  else if (e.key==='y' && T!=='answers') grade('yes');
  else if (e.key==='n' && T!=='answers') grade('no');
});
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/data":
            self._send(200, json.dumps(self.store.payload()).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        try:
            self.store.save(req["table"], req["key"], req["values"])
        except KeyError as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            return
        self._send(200, b'{"ok":true}', "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind-grade an answer-benchmark run in the browser.")
    ap.add_argument("run_dir", type=Path, help="run directory (the one holding blind/)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args()

    blind = args.run_dir / "blind"
    if not (blind / "answers.csv").exists():
        raise SystemExit(f"no blind/answers.csv under {args.run_dir}")
    Handler.store = Store(blind)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    counts = {k: len(v) for k, v in Handler.store.rows.items()}
    print(f"grading {counts['answers']} answers, {counts['citations']} citations, "
          f"{counts['resolution_audit']} resolution rows")
    print(f"open {url}  (grades save to {blind}/*.csv as you go; ctrl-c to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; grades are saved")


if __name__ == "__main__":
    main()
