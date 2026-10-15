# Changelog

Changes describe the open-source **software artifact**, not edits to the submitted manuscript.

## Unreleased — preprint pending

- Removed the manuscript PDF and its version manifest from the current public checkout pending arXiv release.
- Updated English/Chinese README paper links to **arXiv coming soon**; retained the numerical results, seven figures and all demos.
- Updated documentation references and link tests to reflect the pending preprint.

## 0.1.0 — initial research-artifact preparation

### P0: verified core

- Preserved the user-supplied authoritative 25-page PDF with a SHA-256 provenance manifest; excluded incompatible 23-page TeX source from claims of PDF correspondence.
- Preserved original v0.6/v1.0 numerical scripts and frozen reference CSV/JSON.
- Added a standard-library scalar mathematical analysis module with the exact support horizon, velocity requirement, discrete corridor feasibility, bounded-BTL information calculations, and fixed-N certification helpers.
- Added three executable CPU demos: scalar horizon, feedback-extended certification, and local ordinary policy gradient.
- Added mathematical and numerical regression tests and one-command reproduction of all seven paper figures.

### P1: documentation and release engineering

- Added full English and Chinese READMEs, links to actual paper figures and demo outputs.
- Documented theoretical assumptions, API, paper results, figure provenance, numerical reproduction, and limits of applicability.
- Added an MIT software license with separate manuscript and third-party copyright clarification, `CITATION.cff` for the software artifact, package metadata, and contributor guidance.
- Added Python 3.10/3.12/3.13 test/build CI and a dedicated paper-reproduction GitHub Actions job.

### Publication note

The initial local Git history keeps P0 and P1 in separate sequential commits for traceability. A local archive or bundle is **not** by itself a public GitHub release. Confirm that a live repository, Actions run, and any eventual tag exist before marking an online release completed.
