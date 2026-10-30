# DEV-LPAIR diagnostic evidence — private backup

This repository is an **evidence backup**, created so the work is not lost if
it is forgotten. It is **not** a publication, **not** an official experiment
record, and **not** a claim that anything here is validated.

## What is in here

- `spec/` — the DEV-LPAIR diagnostic specification chain: the original
  specification plus Revisions 01–10 and the Revision-11 minimal erratum
  (12 files). Revision-10 is the unchanged historical source document;
  Revision-11 governs only the passages listed in its own change table.
- `pilot/` — the DEV-LPAIR-001 Option B pilot evidence: report, records,
  code, checks, run log, and trained weights (18 files).
- `MANIFEST.sha256` — SHA-256 hashes for every file in `spec/` and `pilot/`,
  generated from the copies committed here. Verify with
  `sha256sum -c MANIFEST.sha256` from the repository root.

## Status labels — read before using anything here

- **Specification:** Revision-11 was approved by the owner as a
  **specification-only** amendment to Revision-10. That approval does not
  authorise repository inspection, code execution, data inspection,
  diagnostic computation, or empirical validation. Code verification (open
  question OQ-1, independent reproduction of the carried-forward code audit)
  remains open. No diagnostic identity in the specification has been
  validated against data by this repository's existence.
- **Pilot:** DEV-LPAIR-001 Option B is **INCOMPLETE and unofficial**. It was
  stopped by a VM restart that killed the final training job (L-SVA W20
  seed 2), which was **not** rerun, resumed, or substituted. The missing job
  is **UNKNOWN, not zero**. Held-out results for completed jobs: L-SHARED
  0/120, L-SVA 0/100, matched 0/100 vs 0/100. There is **no winner, no causal
  diagnosis, and no established mechanism**. These records must never be
  pooled with official EXP1 / EXP1-COMB records.
- **Weights:** the `.npz` files are trained-model artefacts, not trajectory
  evidence. Generating trajectories from them would be a new execution, not
  a reading of existing evidence.
- **Deliberately excluded:** the two reading-split parts of Revision-10
  (they duplicate the single Revision-10 file, so only the single file is
  stored), and `__pycache__` directories (build artefacts, not evidence).

## Key hashes (also in MANIFEST.sha256)

- Revision-10: `81a1f9a8a6c423dd06f6f5d8138178953a107d1df32b4bce1aa6b1eb4b211969`
- Revision-11 erratum: `13134c06d4ec437eeabdf3ce4bab1534290c62076a2397ac42b4d21dbbaeb4ce`
- Pilot records (`pilot/records_dev_lpair_001.jsonl`, 55 records, 0 duplicates):
  `d39b512377bb4685971737739f6e9759240453feef2abb1e9f64cde35cc994dd`
- Pilot report (`pilot/REPORT-DEV-LPAIR-001-OPTION-B.md`):
  `5d51d0cb3de034010954735b86bae19ea3861a7543e4ce008d2aa7cd8c8b84f5`

## What this repository does not do

It does not complete the pilot, does not approve any collection,
computation, replay, or experiment, and does not change the status of the
workbench repository (frozen) or the sibling template repository
(announced). Any future step on this track requires its own explicit owner
authorisation in the owner's own words.
