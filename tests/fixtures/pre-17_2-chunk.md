---
doc_id: legacyab
source: legacy-paper.pdf
chunk_id: legacyab-0001
hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
created_utc: 2025-11-01T00:00:00+00:00
---

This fixture intentionally omits the page_start, page_end, and
section_heading fields introduced in Story 17.2 so Story 17.3 can
verify that retrieval output formatting degrades gracefully when
those fields are absent (pre-17.2 chunks and pdftotext fallback
paths).
