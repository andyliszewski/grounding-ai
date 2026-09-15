# Public corpus inventory (after ingestion, 2026-09-14)

One row per document as it landed in `~/Documents/Corpora-public/corpus/` after
`build.sh` (ingestion 25/25 succeeded, 0 failed, 0 skipped; total 9,117 chunks over
3,294 PDF pages; wall time 140 s including embeddings). Every document took the fast
pdftotext path (text yield above the 1,000 chars/MB threshold in `grounding/parser.py`),
so every chunk carries per-page metadata except the three noted below.

**page_start:** "yes" means every chunk of the document has an integer `page_start`.
Three documents have exactly one chunk without it: in each case it is `ch_0001`, a
437 to 459 byte chunk holding only the document's own YAML front-matter block (the
`doc.md` header, which the chunker treats as body text). In the other 22 documents that
same block is merged into the page-1 chunk, so this is a pipeline quirk in how the header
is chunked, not a property of the PDFs. It does not disqualify a document for
`eval-answers`, which refuses a gold document only when *no* chunk has `page_start`.

**Offset:** `pdf_page = printed_page + offset`, measured from one or two sampled pages'
running footers with `pdftotext -layout` (the PDF page shown and the printed number found
on it). Treat as provisional until re-checked in a viewer at the page you cite.
`section` marks a section- or module-paged work (printed pages like "3-98" or
"MS-01 Page 12") whose printed pages cannot be mapped to PDF pages.

| manifest id (= slug) | doc_id | pages | chunks | page_start | offset (sample) |
|---|---|---:|---:|---|---|
| fda-spinal-system-510k-2004 | 2407fde5 | 23 | 65 | yes | +3 (PDF p.12 shows "page 9") |
| fda-ibfd-special-controls-2007 | 28e33873 | 19 | 47 | yes | +3 (PDF p.10 shows "page 7") |
| fda-nonspinal-bone-screws-perf-criteria-2024 | f8c4da7f | 11 | 32 | yes | +2 (PDF p.6 shows 4) |
| fda-spinal-plating-perf-criteria-2020 | eccf8c84 | 10 | 27 | yes | +2 (PDF p.6 shows 4) |
| fda-modified-metallic-surfaces-1994 | e7aa1015 | 10 | 26 | yes | 0 (PDF p.5 shows 5) |
| fda-reporting-computational-modeling-2016 | 60302bab | 48 | 125 | yes | +3 (PDF p.12 shows 9) |
| fda-cms-credibility-2023 | 6715aa83 | 42 | 143 | yes | 0 (PDF p.12 shows 12) |
| fda-iso-10993-1-use-2023 | 199cba9b | 71 | 236 | 235 of 236 (ch_0001 is the front-matter block) | +5 (PDF p.12 shows 7) |
| fda-510k-program-substantial-equivalence-2014 | dff0cd46 | 42 | 172 | yes | +3 (PDF p.20 shows 17) |
| fr-qmsr-final-rule-2024 | fa8f8887 | 30 | 401 | yes | -7495 (PDF p.1 is 89 FR 7496; p.12 is 7507) |
| fda-bench-performance-testing-2019 | b5889d8d | 12 | 35 | yes | 0 (PDF p.8 shows 8) |
| fda-mr-safety-testing-labeling-2023 | 31fa320d | 32 | 93 | yes | +3 (PDF p.12 shows 9) |
| fda-510k-for-device-change-2017 | d944819e | 78 | 267 | 266 of 267 (ch_0001 is the front-matter block) | +1 (PDF p.12 shows 11) |
| cfr-21-part-888-2025 | 0ce2b093 | 34 | 174 | 173 of 174 (ch_0001 is the front-matter block) | -761 (PDF p.12 shows 773, p.30 shows 791) |
| nasa-std-5001b-chg3 | 5afd132e | 36 | 95 | yes | 0 (footer "12 of 36" on PDF p.12) |
| nasa-std-5019a-chg4 | edf7995e | 120 | 303 | yes | 0 (footer "12 of 120") |
| nasa-std-5020b | f327d38b | 114 | 288 | yes | 0 (footer "12 of 114") |
| nasa-std-6016c-chg1 | 70c700ff | 157 | 604 | yes | 0 (footer "12 of 157") |
| nasa-std-7009b | aa017943 | 88 | 230 | yes | to measure (no printed page number in the text layer on sampled pages) |
| nasa-rp-1228-fastener-design-manual | 987bcc1c | 100 | 205 | yes | +4 (PDF p.30 shows 26); 1990 scan with an OCR text layer, 2,035 chars/page |
| doe-hdbk-1017-1-material-science-v1 | 775f74b8 | 102 | 177 | yes | section (module-paged, "MS-01 Page 12") |
| doe-hdbk-1017-2-material-science-v2 | 41ec5758 | 112 | 210 | yes | section (module-paged, "MS-04 Page ii") |
| mil-hdbk-5j | 9f9fdea2 | 1,733 | 4,619 | yes | section (chapter-paged, "3-98"); front matter uses roman numerals |
| mil-std-882e-chg1 | c1afa9a7 | 106 | 258 | yes | +7 (PDF p.12 shows 5, p.30 shows 23) |
| faa-damage-tolerance-handbook-vol1 | 9a46f5c3 | 164 | 285 | yes | section (chapter-paged, "1-14"); 1993 scan with an OCR text layer, 1,819 chars/page |
| **total** | | **3,294** | **9,117** | | |

Per-collection chunk counts from the build log: public-fda-ortho 1,843; public-nasa-standards
1,725; public-doe 387; public-mil 4,877 (MIL-HDBK-5J alone is 4,619, 51% of the corpus);
public-faa 285.

## How doc_id is derived, and what a rebuild must match

`doc_id` is the first 8 hex characters of `hashes.doc_sha1` in each `meta.yaml`: the SHA-1
of the normalized Markdown, not of the PDF (`grounding/pipeline.py:297`,
`short_doc_id(doc_sha1)`). It is therefore stable only while the same PDF bytes (pinned by
the manifest's SHA-256) are parsed by a toolchain that produces byte-identical Markdown. A
rebuild on another machine should compare its `_index.json` doc_ids against this table; a
difference means a parser or chunker version changed, and any fixture `doc_ids` must be
remapped by slug.

Toolchain that produced this table (recorded in each `meta.yaml` under `tooling:`):

| component | version |
|---|---|
| grounding-ai | 0.3.0 at commit 33ce059 (branch `epic-25-grounded-answer-benchmark`, other files dirty) |
| parser (unstructured, but pdftotext fast path was used for all 25) | 0.22.18 |
| pdftotext (poppler) | 25.10.0 |
| chunker (langchain-text-splitters) | 1.1.1 |
| hashing module | 1.0.8 |
| sentence-transformers / faiss-cpu / rank-bm25 | 5.4.0 / 1.13.2 / 0.2.2 |
| Python | 3.13.5, macOS 26.6.2, arm64 |

Pinned parameters in every `meta.yaml`: `chunk_size: 1200`, `chunk_overlap: 150`,
`min_chunk_size: 200`, `parser: unstructured`, `ocr_mode: auto`.

## Embeddings

`~/Documents/Corpora-public/embeddings/implant-eng-public/`: `_embeddings.faiss` (14.0 MB),
`_chunk_map.json` (1.8 MB), `_bm25.pkl` (11.9 MB), `_bm25_map.json` (1.2 MB). All 25 documents
were selected by the agent's five collection tags. The BM25 sidecar is present, so the
`hybrid` and `hybrid-rerank` conditions of `eval-answers` are available.
