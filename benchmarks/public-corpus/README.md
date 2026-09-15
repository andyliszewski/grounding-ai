# Public-domain corpus for the Epic 25 grounded-answer benchmark

A small corpus of 25 US government engineering and regulatory documents (3,294 PDF pages,
96.4 MB, 9,117 chunks) that anyone can rebuild, byte for byte, from `manifest.yaml`. It
exists so the grounded-answer benchmark (`grounding eval-answers`, see
`docs/eval/README.md`, "Answer benchmark (Epic 25)") can be run and published without the
private mechanical-engineer corpus, whose books are copyrighted.

The subject is a practicing medical-device design engineer's working set: FDA orthopedic
guidances and the QMSR, NASA structural, fracture-control, fastener, materials and
modeling standards, DOE material-science handbooks, MIL-HDBK-5J allowables, MIL-STD-882E
system safety, and the FAA damage-tolerance handbook.

## Why every source is public-release

Every document is either a work of the United States Government or carries a
public-release distribution statement. The bases, with the key letters used in
`manifest.yaml` `public_basis`:

- **[G] 17 U.S.C. 105.** "Copyright protection under this title is not available for any
  work of the United States Government" (https://www.law.cornell.edu/uscode/text/17/105).
  Applies to the FDA guidances, the Federal Register and CFR editions on govinfo, the NASA
  standards and NASA RP-1228, the DOE handbooks, MIL-STD-882E, MIL-HDBK-5J and the FAA
  handbook.
- **[F] FDA website policy.** The contents of fda.gov "are not copyrighted. They are in the
  public domain and may be republished, reprinted and otherwise used freely by anyone
  without the need to obtain permission from FDA"
  (https://www.fda.gov/about-fda/about-website/website-policies).
- **[N] NASA Technical Standards System.** Each standard's landing page states "Export
  Control/Distribution Authorization: Internet Public" and "Internet Public -- Standard is
  cleared for public accessibility on the internet". NASA-STD-5019A and 6016C also carry
  "APPROVED FOR PUBLIC RELEASE - DISTRIBUTION IS UNLIMITED" in every page footer.
- **[T] NTRS.** The NASA RP-1228 record has `"distribution": "PUBLIC"` and copyright
  determination `GOV_PUBLIC_USE_PERMITTED`.
- **[A] Distribution Statement A** ("Approved for public release; distribution is
  unlimited") on the DOE handbook covers and on MIL-HDBK-5J.
- **[Q] ASSIST QuickSearch** shows Dist Stmt A for MIL-HDBK-5J and MIL-STD-882E.
- **[R] DTIC** public holdings are unclassified, unlimited reports (the FAA handbook's
  fallback mirror).

Two hosting caveats. MIL-HDBK-5J is fetched from an archive.org mirror (a third-party
upload of the DoD PDF) because DoD no longer serves the canceled handbook; the work is a
US Government work regardless of host. The FAA handbook comes from USDOT's ROSA P
repository, with a DTIC mirror on archive.org as fallback.

Deliberately excluded: MMPDS, CMH-17, ASTM, ISO and ISO 13485 (all sold), the withdrawn
1997 FDA Design Control guidance (only an Archive-It snapshot behind a bot check remains),
and the January 2024 draft coatings guidance (not final). The private corpus's copies of
any of these must never be added here.

## Rebuild in three commands

Prerequisites: Python 3.13, poppler (`brew install poppler` or `apt-get install
poppler-utils`; `pdftotext` is what gives every chunk its page number), and about 250 MB
of disk. Set `CORPORA_PUBLIC` to change the output root (default
`~/Documents/Corpora-public`).

```bash
python3.13 -m venv venv && ./venv/bin/pip install -e .          # 1. the grounding CLI
benchmarks/public-corpus/build.sh                                 # 2. fetch, verify SHA-256, ingest, embed
./venv/bin/grounding eval-answers --agent implant-eng-public \    # 3. prove it loads (no key, no spend)
    --agents-dir benchmarks/public-corpus/agents \
    --fixtures <your-questions.yaml> \
    --corpus ~/Documents/Corpora-public/corpus \
    --embeddings ~/Documents/Corpora-public/embeddings/implant-eng-public --dry-run
```

`build.sh` (a) runs `fetch.py`, which downloads every manifest entry to
`originals/<id>.pdf`, streams large files, retries once, falls back to `fallback_url`, and
fails loudly on any SHA-256 mismatch; (b) ingests one `grounding` run per collection group
with the pinned parameters `--chunk-size 1200 --chunk-overlap 150 --min-chunk-size 200
--parser unstructured --ocr auto`; (c) builds the agent's FAISS index and BM25 sidecar with
`grounding embeddings` and refuses to finish without all four index files. Flags:
`--record` (first run only, writes hashes into the manifest), `--clean`, `--skip-fetch`,
`--skip-embed`. `fetch.py --verify-only` checks the files on disk without the network.

What a rebuild reproduces exactly: the PDFs (SHA-256), the corpus slugs (they equal the
manifest ids), the collection tags, and, given the same parser toolchain, the Markdown,
chunk boundaries and `doc_id`s. `doc_id` is derived from the SHA-1 of the normalized
Markdown, not of the PDF, so it depends on the pdftotext and chunker versions recorded in
`inventory.md`; compare a rebuild's `_index.json` against that table before reusing
fixture `doc_ids`.

## Output layout

```
~/Documents/Corpora-public/
  originals/<id>.pdf                       the 25 PDFs, verified against manifest.yaml
  corpus/_index.json, <slug>/{doc.md,meta.yaml,chunks/ch_NNNN.md}
  embeddings/implant-eng-public/{_embeddings.faiss,_chunk_map.json,_bm25.pkl,_bm25_map.json}
  work/<collection-group>/<id>.pdf         symlinks, one directory per grounding run
```

## Files in this directory

| file | purpose |
|---|---|
| `manifest.yaml` | the contract: id, title, agency, number, URL, fallback, SHA-256, bytes, pages, Last-Modified, public basis, category hint, notes |
| `fetch.py` | download + verify (`--record`, `--verify-only`, `--only`, `--update-hashes`) |
| `build.sh` | fetch, ingest with pinned parameters, embed, check the BM25 sidecar |
| `agents/implant-eng-public.yaml` | the benchmark agent; its five collection tags select exactly these 25 documents |
| `inventory.md` | per-document doc_id, pages, chunk count, page_start coverage, printed-page offsets, toolchain versions |
| `questions-template.yaml` | public fixture template with `page_offsets`, `revisions` and `identifiers` stubbed for every document |

## Inventory

Chunk counts and doc_ids are from the 2026-09-14 build; see `inventory.md` for page_start
coverage and printed-page offsets.

| # | id | document | number | pages | chunks | doc_id |
|--:|---|---|---|--:|--:|---|
| 1 | fda-spinal-system-510k-2004 | Guidance for Industry and FDA Staff: Spinal System 510(k)s | FDA CDRH, May 3, 2004 | 23 | 65 | 2407fde5 |
| 2 | fda-ibfd-special-controls-2007 | Class II Special Controls Guidance: Intervertebral Body Fusion Device | FDA CDRH, June 12, 2007 | 19 | 47 | 28e33873 |
| 3 | fda-nonspinal-bone-screws-perf-criteria-2024 | Orthopedic Non-Spinal Metallic Bone Screws and Washers, Performance Criteria | FDA CDRH, Nov 2024 (first Dec 2020) | 11 | 32 | f8c4da7f |
| 4 | fda-spinal-plating-perf-criteria-2020 | Spinal Plating Systems, Performance Criteria | FDA CDRH, Dec 2020 | 10 | 27 | eccf8c84 |
| 5 | fda-modified-metallic-surfaces-1994 | Testing Orthopedic Implants with Modified Metallic Surfaces Apposing Bone or Bone Cement | FDA CDRH, April 1994 | 10 | 26 | e7aa1015 |
| 6 | fda-reporting-computational-modeling-2016 | Reporting of Computational Modeling Studies in Medical Device Submissions | FDA CDRH, Sept 2016 | 48 | 125 | 60302bab |
| 7 | fda-cms-credibility-2023 | Assessing the Credibility of Computational Modeling and Simulation in Medical Device Submissions | FDA CDRH, Nov 2023 | 42 | 143 | 6715aa83 |
| 8 | fda-iso-10993-1-use-2023 | Use of International Standard ISO 10993-1 | FDA CDRH, Sept 2023 | 71 | 236 | 199cba9b |
| 9 | fda-510k-program-substantial-equivalence-2014 | The 510(k) Program: Evaluating Substantial Equivalence | FDA CDRH, July 28, 2014 | 42 | 172 | dff0cd46 |
| 10 | fr-qmsr-final-rule-2024 | Medical Devices; Quality System Regulation Amendments (QMSR final rule) | 89 FR 7496, Feb 2, 2024 | 30 | 401 | fa8f8887 |
| 11 | fda-bench-performance-testing-2019 | Recommended Content and Format of Non-Clinical Bench Performance Testing Information | FDA CDRH, Dec 20, 2019 | 12 | 35 | b5889d8d |
| 12 | fda-mr-safety-testing-labeling-2023 | Testing and Labeling Medical Devices for Safety in the MR Environment | FDA CDRH, Oct 2023 | 32 | 93 | 31fa320d |
| 13 | fda-510k-for-device-change-2017 | Deciding When to Submit a 510(k) for a Change to an Existing Device | FDA CDRH, Oct 2017 | 78 | 267 | d944819e |
| 14 | cfr-21-part-888-2025 | 21 CFR Part 888, Orthopedic Devices | CFR annual edition, revised April 1, 2025 | 34 | 174 | 0ce2b093 |
| 15 | nasa-std-5001b-chg3 | Structural Design and Test Factors of Safety for Spaceflight Hardware | NASA-STD-5001B w/Change 3, 2022-10-24 | 36 | 95 | 5afd132e |
| 16 | nasa-std-5019a-chg4 | Fracture Control Requirements for Spaceflight Hardware | NASA-STD-5019A w/Change 4, revalidated 2025-09-05 | 120 | 303 | edf7995e |
| 17 | nasa-std-5020b | Requirements for Threaded Fastening Systems in Spaceflight Hardware | NASA-STD-5020B, 2021-08-06 | 114 | 288 | f327d38b |
| 18 | nasa-std-6016c-chg1 | Standard Materials and Processes Requirements for Spacecraft | NASA-STD-6016C w/Change 1, 2023-11-15 | 157 | 604 | 70c700ff |
| 19 | nasa-std-7009b | Standard for Models and Simulations | NASA-STD-7009B, 2024-03-05 | 88 | 230 | aa017943 |
| 20 | nasa-rp-1228-fastener-design-manual | Fastener Design Manual | NASA RP-1228, March 1990 | 100 | 205 | 987bcc1c |
| 21 | doe-hdbk-1017-1-material-science-v1 | DOE Fundamentals Handbook: Material Science, Vol 1 of 2 | DOE-HDBK-1017/1-93, Jan 1993 | 102 | 177 | 775f74b8 |
| 22 | doe-hdbk-1017-2-material-science-v2 | DOE Fundamentals Handbook: Material Science, Vol 2 of 2 | DOE-HDBK-1017/2-93, Jan 1993 | 112 | 210 | 41ec5758 |
| 23 | mil-hdbk-5j | Metallic Materials and Elements for Aerospace Vehicle Structures | MIL-HDBK-5J, 31 Jan 2003 (canceled 2006) | 1,733 | 4,619 | 9f9fdea2 |
| 24 | mil-std-882e-chg1 | System Safety | MIL-STD-882E w/Change 1, 27 Sept 2023 | 106 | 258 | c1afa9a7 |
| 25 | faa-damage-tolerance-handbook-vol1 | Damage Tolerance Assessment Handbook, Vol I | DOT/FAA/CT-93/69.I, Oct 1993 | 164 | 285 | 9a46f5c3 |

## Known quirks

- **MIL-HDBK-5J is 51% of the chunks** (4,619 of 9,117). It will dominate retrieval pools
  for materials questions; consider that when writing table items against the other
  sources, or cap it in a later revision.
- **Three documents have a page-less first chunk** (`fda-iso-10993-1-use-2023`,
  `fda-510k-for-device-change-2017`, `cfr-21-part-888-2025`): `ch_0001` holds only the
  document's own front-matter block. A pipeline quirk, harmless to `eval-answers`.
- **NASA-STD-7009B** shows no printed page number in its text layer, so its
  `page_offsets` entry is left to measure.
- **FDA re-posts PDFs at the same URL.** `manifest.yaml` records the `Last-Modified`
  header and SHA-256 seen at build time; a later re-post fails verification on purpose.
  Re-pin deliberately with `fetch.py --record --update-hashes` and re-ingest.
- **Fallback URLs are different files.** NASA-STD-5020B's fallback is a re-rendering and
  the FAA handbook's fallback is a different scan, so a fallback fetch cannot match the
  recorded hash; both primaries worked on 2026-09-14.
