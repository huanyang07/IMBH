# Repository Cleanup Results

- Date: 2026-07-11
- Branch: `codex/repository-cleanup-after-1e7438e`
- Functional cleanup commit: `0c311550bfeabd320f6e4ffe36d5261c1deeeb42`

## Safety and Recovery

The following annotated tags were created and pushed before any removal:

```text
legacy-steady-positive-flux-dae-2026-07-10
  -> 1e7438e167823500d6ffe5434a0f3c62cb2ba864

pre-cleanup-p0-2026-07-11
  -> 0a000767a915880c0710b8f4ec03eb0c64aa168a
```

External archive:

```text
/Users/huanyang/Documents/IMBH_QPE_legacy_archive_2026-07-11/
  IMBH_legacy_steady_dae_artifacts_pre-cleanup-p0-2026-07-11.tar.gz
```

Archive verification:

```text
files                         7,799
uncompressed bytes            558,320,131
archive bytes                 197,037,654
archive SHA-256               50ea39cadf7fef1b1fb5c8506a49966f9551238a2356ce9ca1ac54eec1a180a8
manifest SHA-256              2a5463c0af6e47af7dcc3a25f9f4f263cf29d59d2dd787842db8847b107bcb60
fresh extraction/hash audit   PASS
independent checksum command  PASS
```

Every extracted file matched its pre-cleanup size and SHA-256. The archive is
not stored in normal Git history; the remote tags provide a second recovery
path.

## Before and After

| Metric | Before | Functional cleanup | Reduction |
|---|---:|---:|---:|
| Tracked files | 7,995 | 282 | 96.47% |
| Tracked bytes | 562,953,617 | 10,092,690 | 98.21% |
| Raw `outputs/` files | 7,621 before P0 additions | 1 policy README | effectively all removed |
| Unstructured `Note/` reports | 160 | 0 | consolidated |
| Full-paper literature files | 12 | 0 | replaced by bibliography |
| Tests | 182 + 4 subtests | 186 + 4 subtests | coverage increased |

The checked-out source tree is approximately 11 MB. A fresh local clone is
approximately 187 MB because Git history was intentionally not rewritten; its
`.git` directory is approximately 176 MB.

## Canonical Evidence

Eight canonical cases replace the raw output ladder:

1. `no_wind_mdot5`
2. `stream_no_wind_mdot2_fs080`
3. `phase_dae_entry_N164`
4. `phase_endpoint_positive_N164`
5. `phase_endpoint_step_convergence`
6. `source_shape_comparison`
7. `global_composite_failure`
8. `p0_validity_ledger_outer_manifold`

The canonical set contains 52 checksummed files totaling 739,744 bytes. It
includes the previously hidden K12/K13/K14 phase regression dependencies, the
exit-refinement endpoint, and the model base anchor. No test depends on raw
`outputs/` paths.

## Exact Actions

The exact pre-cleanup disposition of every path is recorded in:

```text
docs/manifests/repository_inventory.csv
```

Counts from that manifest:

```text
KEEP       196 legacy paths
ARCHIVE  7,786 legacy paths
DELETE      13 legacy paths
```

The exact 7,799 paths included in the archive are listed in
`docs/manifests/archive_file_list.txt`. Exact duplicates and largest files are
recorded in `duplicate_files.csv` and `largest_tracked_files.csv`.

The `DELETE` set consists of the 12 local full-paper files, including the
extensionless 1988 PDF, plus the empty placeholder policy item recorded in the
inventory. Scientific full texts remain recoverable through the private archive
and immutable tag; the source branch now contains `references/references.bib`
and `references/REFERENCES.md`.

## Documentation and Reviewability

- `docs/PROJECT_STATUS.md` is the single canonical scientific handoff.
- `GPT_REPO_HANDOFF.md` is a short pointer, not a competing status document.
- Current reports live under `docs/reports/current/`.
- Decisive historical milestones and negative results are summarized in
  `docs/history/MILESTONES.md`.
- Canonical status and limitations are recorded per case in `provenance.json`.
- Two GitHub Actions workflows run tests and repository hygiene checks.

## Verification

Working tree and fresh-clone results:

```text
editable install                         PASS
canonical SHA-256 validation             PASS
repository hygiene                       PASS
pytest                                   186 passed, 4 subtests passed
raw-output-independent phase regression  PASS
```

The fresh clone used an isolated virtual environment with system scientific
packages and `pip install --no-build-isolation --no-deps -e .`. The default
isolated pip build attempted a network download of setuptools in the sandbox;
that network-only failure was not a repository failure.

## Script Decision

No legacy scientific runner was removed in this first pass. The dependency
inventory found extensive sibling-script imports and documentation references.
Eleven unreferenced archive candidates are identified, but removing them before
moving reusable helpers into `src/` would mix repository cleanup with a risky
numerical refactor. `scripts/README.md` identifies the current entry points.

## Remaining Risks

1. Git history is still large. No history rewrite or force push was performed.
2. The external archive currently has one durable local copy plus remote Git
   tags; an additional institutional or release-asset backup would improve
   resilience.
3. `run_mdot5_local_mdot_eta_continuation.py` remains a large legacy monolith
   because current phase workflows import its private helpers.
4. Historical current reports may mention archived raw paths; each report now
   points to canonical replacements or the recovery tag where relevant.

## Commit Sequence

```text
53f9dc6  Inventory repository cleanup candidates
3aa7ad7  Consolidate project status and documentation
ff85b28  Add compact canonical scientific results
4112f52  Record verified legacy artifact archive
7c31953  Include all literature files in legacy archive
e62e700  Migrate phase regression to canonical fixtures
da7c8e6  Remove archived generated artifacts from main tree
c9e999e  Replace literature PDFs with bibliography
0c31155  Add CI and repository hygiene gates
```

No scientific equation, closure, residual weight, or accepted numerical
threshold was changed by the cleanup.
