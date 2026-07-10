# Codex Repository Cleanup Specification

**Project:** IMBH/QPE stream-fed minidisk  
**Cleanup anchor:** `1e7438e167823500d6ffe5434a0f3c62cb2ba864`  
**Date:** 2026-07-10  
**Purpose:** Make the default branch small, reviewable, reproducible, and ready for the conservative time-dependent disk model without losing the evidence supporting the current steady-DAE results.

---

## 1. Executive decision

Clean the repository now, before the major equation/closure redesign.

However, use three different actions:

1. **KEEP IN MAIN** — source, tests, current documentation, compact regression fixtures, and a very small canonical result set.
2. **ARCHIVE OFF MAIN** — scientifically useful raw outputs, intermediate continuation states, exploratory figures/tables, and superseded development reports.
3. **DELETE PERMANENTLY** — caches, compiled Python files, duplicated/generated clutter, empty placeholder directories, and redistributable-paper copies that should not be hosted in the public source repository.

Do **not** treat “remove from the default branch” as “destroy the data.” Preserve the current state with an immutable tag and a verified artifact bundle first.

Do **not** rewrite Git history in the initial cleanup. History rewriting is a separate, later decision that requires explicit owner approval.

---

## 2. Non-negotiable safety gates

Codex must complete all of the following before deleting or untracking bulk data:

1. Create an annotated tag at the exact anchor:

   ```text
   legacy-steady-positive-flux-dae-2026-07-10
   -> 1e7438e167823500d6ffe5434a0f3c62cb2ba864
   ```

2. Create a complete inventory containing, at minimum:

   ```text
   path
   file type
   byte size
   SHA-256
   Git status/tracked status
   scientific category
   proposed action: KEEP / ARCHIVE / DELETE
   reason
   replacement or archive location
   ```

3. Create and verify a full artifact archive containing the raw generated scientific products that will leave the default branch.

   Suggested filenames:

   ```text
   IMBH_legacy_steady_dae_artifacts_1e7438e.tar.zst
   IMBH_legacy_steady_dae_artifacts_1e7438e_SHA256SUMS.txt
   IMBH_legacy_steady_dae_artifacts_1e7438e_MANIFEST.csv
   ```

4. Verify the archive by extracting it into a fresh temporary directory and checking every SHA-256.

5. Copy the small canonical result set described below into the new repository layout.

6. Run the complete existing test suite before cleanup and record the result.

7. Perform the cleanup in a dedicated branch:

   ```text
   cleanup/repository-layout-after-1e7438e
   ```

8. Run the same test suite after cleanup and show numerical parity for all retained regression fixtures.

9. Do not change scientific equations, closure logic, residual weights, solver tolerances, or accepted numerical thresholds in the cleanup commit.

10. Do not force-push or rewrite history in this work package.

---

## 3. KEEP IN MAIN

### 3.1 All scientific source code

Keep all tracked source files under:

```text
src/imri_qpe/**
```

This includes the existing steady positive-flux/transonic/phase-DAE implementation, even though a new conservative model is planned.

In particular, keep the current implementations of:

```text
layer1_hill_flow
layer2_scurve
layer3_minidisk_1d
transonic_collocation
transonic_local
transonic_thermo
transonic_potential
isolated_slim_solver
winds
entropy/advection/audit utilities
constants
parameters
scales
units
```

Rationale:

- The existing solver is a certified or partially certified scientific benchmark.
- The endpoint and global-composite failure must remain reproducible.
- It will be needed for regression comparisons against the future signed-flux and time-dependent conservative model.
- A change of physical formulation does not make the old code worthless; it changes its status to a legacy diagnostic/reference model.

Do not move these files into a new `legacy/` namespace during the first cleanup commit. Path moves create unnecessary diff noise and can break imports. Add status documentation first; refactor paths only after regression coverage is stronger.

---

### 3.2 All tests

Keep every existing file under:

```text
tests/**
```

Add, rather than remove, tests for:

```text
- the no-wind Mdot/Mdot_Edd = 5 anchor;
- the stream-fed no-wind fs = 0.80 anchor;
- the phase-DAE interface state;
- the accepted positive-p_R endpoint tail;
- the rejected step-sensitive signed crossing;
- the global-composite residual failure;
- source-shape invariance of R_* within the recorded tolerance;
- artifact manifest validation.
```

Regression tests may use small downsampled fixtures. They must not require thousands of continuation checkpoints.

---

### 3.3 Project and build configuration

Keep and improve:

```text
pyproject.toml
.gitignore
README.md
```

Add:

```text
LICENSE or an explicit private-research notice
.github/workflows/tests.yml
.github/workflows/repository-hygiene.yml
environment/ or lock file
CITATION.cff
```

The environment record should separate:

```text
runtime dependencies
solver dependencies
plotting/report dependencies
development/test dependencies
```

Every production output should record:

```text
Git SHA
Python version
NumPy version
SciPy version
BLAS/LAPACK information when practical
platform
command line
configuration file
random seed, if any
```

---

### 3.4 Current documentation

Replace the current fragmented/stale handoff structure with:

```text
docs/PROJECT_STATUS.md
docs/MODEL_EQUATIONS.md
docs/REPRODUCIBILITY.md
docs/ARTIFACT_POLICY.md
docs/decisions/
docs/reports/current/
docs/history/MILESTONES.md
```

#### Keep as current reports

Copy or move the following current scientific reports into `docs/reports/current/`:

```text
CODEX_MDOT5_GLOBAL_PHASE_DAE_PRODUCTION_RESULTS.md
CODEX_MDOT5_PHASE_DAE_EXIT_REFINEMENT_RESULTS.md
CODEX_MDOT5_PHASE_CRITICAL_GLOBALIZATION_RESULTS.md
CODEX_MDOT5_PHASE_CRITICAL_CLASSIFICATION_RESULTS.md
IMBH_QPE_PROJECT_BASELINE_HANDOFF_2026-07-10.md
CODEX_IMBH_PROJECT_REVIEW_AND_NEXT_ACTIONS_2026-07-10.md
this cleanup specification
```

`docs/PROJECT_STATUS.md` must label every conclusion as one of:

```text
CERTIFIED
SUPPORTED BUT NOT FULLY CERTIFIED
DIAGNOSTIC ONLY
REJECTED
PLANNED
```

It must clearly distinguish:

```text
- the mature no-wind slim branch;
- the stream-fed no-wind branch;
- the present steady positive-flux mass-loaded-wind closure;
- the finite-radius low-u endpoint result;
- the uncertified global nonexistence question;
- the proposed conservative signed-flux/time-dependent replacement model.
```

Replace `GPT_REPO_HANDOFF.md` with the new canonical status file, or reduce it to a short pointer to `docs/PROJECT_STATUS.md`. Do not maintain two competing handoffs.

---

### 3.5 A compact canonical result set

Create:

```text
results/
  README.md
  canonical/
  manifests/
```

Keep only the smallest data needed to reproduce important comparisons and regression tests.

Recommended canonical cases:

#### C1. Standard no-wind slim anchor

```text
results/canonical/no_wind_mdot5/
  state.npz
  config.json
  summary.csv
  provenance.json
```

#### C2. Stream-fed no-wind high-state anchor

```text
results/canonical/stream_no_wind_mdot5_fs080/
  state.npz
  config.json
  summary.csv
  provenance.json
```

#### C3. Phase-DAE entrance/interface state

```text
results/canonical/phase_dae_entry_N164/
  state.npz
  config.json
  summary.csv
  provenance.json
```

#### C4. Accepted positive-branch endpoint tail

Use the highest-quality accepted small-step run, not every step:

```text
results/canonical/phase_endpoint_positive_N164/
  tail_state_or_downsampled_trajectory.npz
  config.json
  scaling_fits.csv
  provenance.json
```

#### C5. Step-size comparison

Keep a compact table, not all trajectories:

```text
results/canonical/phase_endpoint_step_convergence/
  convergence.csv
  provenance.json
```

The table should contain:

```text
step or gauge
accepted steps
last positive R
last positive p_R
last positive logu
estimated R_*
fit-window definition
residual metrics
status
```

#### C6. Source-shape comparison

```text
results/canonical/source_shape_comparison/
  comparison.csv
  provenance.json
```

Include C2, C4, C-infinity, and wider-source results.

#### C7. Global-composite failure witness

Keep one compact state demonstrating that the phase block is accurate while the ordinary source-tail fails:

```text
results/canonical/global_composite_failure/
  interface_and_tail_snapshot.npz
  residual_profile.csv
  config.json
  provenance.json
```

#### Canonical artifact rules

Every canonical directory must contain:

```text
source_commit
generation command
configuration
scientific status
SHA-256 for every file
description of why the file is retained
description of what it can and cannot establish
```

Target size:

```text
5-10 NPZ files total
small CSV/JSON/Markdown summaries
1-3 figures per major certified result
```

Do not keep a file merely because it was expensive to generate.

---

### 3.6 Essential scripts

Keep scripts that are required to:

```text
- produce the canonical no-wind benchmark;
- produce the canonical stream-fed no-wind benchmark;
- reproduce the phase-DAE production trajectory;
- reproduce the endpoint classification;
- reproduce the globalization/failure audit;
- validate mass, angular momentum, and energy budgets;
- generate retained canonical figures/tables;
- build and validate artifact manifests.
```

Likely temporary keep candidates include:

```text
run_mdot5_global_phase_dae_production.py
run_mdot5_phase_dae_exit_refinement.py
run_mdot5_phase_critical_globalization.py
run_mdot5_phase_critical_classification.py
the production no-wind Mdot ladder runner
the production stream-fed no-wind runner
canonical plotting scripts
conservation audit scripts
```

The very large `run_mdot5_local_mdot_eta_continuation.py` should be kept temporarily because other workflows may depend on its helpers, but it should not remain a permanent monolith. Codex should first:

```text
- identify every externally used helper;
- move reusable logic into src/imri_qpe/...;
- add unit tests;
- replace sibling-script imports with package imports;
- prove numerical parity;
- only then retire the monolithic script.
```

---

## 4. ARCHIVE OFF MAIN

The following material should leave the default branch but remain recoverable in the tagged snapshot and verified artifact archive.

### 4.1 Bulk continuation checkpoints

Archive and remove from main:

```text
outputs/checkpoints/**
```

This includes, unless selected for the canonical set:

```text
arc_step_*.npz
stage_*.npz
every intermediate eta step
every continuation refresh
every patch/repolish state
every mesh scout state
every parameter-ladder intermediate
every failed Newton attempt
every duplicated state stored under a new run name
```

Retention rule:

```text
Keep the initial accepted state, final accepted state, one independent convergence comparator,
and any scientifically decisive rejected witness. Archive all intermediate breadcrumbs.
```

Do not keep every arclength step in Git.

---

### 4.2 Raw/per-run tables

Archive and remove from main:

```text
outputs/tables/**
```

except for the compact summaries copied into `results/canonical/`.

Particularly remove from main:

```text
*_newton_audit/
per-iteration residual dumps
per-cell Jacobian dumps
line-search traces
all repeated parameter-step summaries
tables that can be regenerated exactly from a retained canonical NPZ
duplicate Markdown/CSV versions with identical information
```

A useful table belongs in the main branch only when it is:

```text
small
human-readable
scientifically interpreted
referenced by current documentation
not trivially regenerated from another retained file
```

---

### 4.3 Exploratory figures

Archive and remove from main:

```text
outputs/figures/**
```

except for selected canonical figures.

Keep only figures that support current claims, for example:

```text
- no-wind Mdot sequence;
- stream-fed no-wind anchor;
- phase endpoint convergence/scaling;
- source-shape comparison;
- global-composite residual failure;
- future conservative-model validation.
```

Do not keep one PNG for every continuation refresh, patch, mesh, or parameter step.

For each retained figure, keep the small source table or canonical state needed to regenerate it.

---

### 4.4 Historical Codex plans and superseded reports

The `Note/` directory should not remain an unstructured chronological dump.

Archive or consolidate:

```text
old *_PLAN.md files
old *_NEXT_STEPS.md files
superseded attempt reports
retry-by-retry numerical diaries
reports whose conclusion is fully captured by a later milestone report
```

Before removing them from main:

1. Create `docs/history/MILESTONES.md`.
2. Summarize the meaningful sequence of accepted and rejected approaches.
3. Record the relevant historical commit/tag for each milestone.
4. Preserve scientifically decisive negative results.
5. Keep the full originals accessible through the immutable legacy tag.

Do not copy hundreds of old notes into a new archive directory inside the same default branch; that only moves the clutter.

---

### 4.5 Exploratory and superseded scripts

Many one-off scripts should be archived off main after a dependency audit.

Likely archive candidates include scripts whose names indicate:

```text
pilot
probe
scout
patch
refresh
repolish
best_polish
tiny
ultratiny
microgrid
one-off sensitivity
abandoned boundary condition
obsolete solver strategy
superseded continuation variant
```

Codex must not classify scripts by filename alone. For every script, determine:

```text
- imported by another tracked module?
- referenced by current documentation?
- covered by a test?
- needed to regenerate a canonical artifact?
- contains unique reusable logic?
- superseded by a later production runner?
```

Generate:

```text
docs/script_inventory.csv
```

with columns:

```text
path
lines
imports
imported_by
referenced_by_docs
test_coverage
canonical_artifact_dependency
classification
replacement
reason
```

Reusable logic must be moved into `src/` and tested before the wrapper is removed.

---

## 5. DELETE PERMANENTLY

The following do not need scientific archival treatment.

### 5.1 Python and tool caches

Delete wherever present:

```text
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
```

### 5.2 Operating-system/editor clutter

Delete:

```text
.DS_Store
Thumbs.db
*~
*.swp
*.tmp
```

### 5.3 Empty placeholder directories

Remove empty directories that exist only through `.gitkeep`, unless a user-facing layout truly requires them.

Examples:

```text
notebooks/.gitkeep
outputs/runs/.gitkeep
empty output subdirectories
```

Runtime code should create output directories as needed.

### 5.4 Duplicated or corrupted generated files

Permanently delete exact byte-for-byte duplicates after the manifest identifies them.

For near-duplicates, preserve only the canonical version and archive the rest if scientific provenance is uncertain.

### 5.5 Full-paper PDFs in the public source repository

Remove:

```text
Literature/*.pdf
Literature/full-paper copies without explicit redistribution permission
```

Replace with:

```text
references/references.bib
references/REFERENCES.md
```

Record:

```text
authors
title
journal
year
DOI
arXiv identifier when available
role in the project
```

Do not distribute full papers through the source repository merely for convenience. Researchers can retrieve papers through lawful publisher, arXiv, library, or author channels.

---

## 6. Proposed repository layout

Use the following target structure after the non-destructive cleanup:

```text
.
├── README.md
├── pyproject.toml
├── .gitignore
├── CITATION.cff
├── LICENSE
├── environment/
│   └── lock-or-explicit-environment-record
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── repository-hygiene.yml
├── src/
│   └── imri_qpe/
│       ├── layer1_hill_flow/
│       ├── layer2_scurve/
│       ├── layer3_minidisk_1d/
│       └── conservative_disk/          # new model, added later
├── tests/
│   ├── unit/
│   ├── regression/
│   └── data/                           # tiny fixtures only
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── MODEL_EQUATIONS.md
│   ├── REPRODUCIBILITY.md
│   ├── ARTIFACT_POLICY.md
│   ├── decisions/
│   ├── reports/
│   │   └── current/
│   └── history/
│       └── MILESTONES.md
├── experiments/
│   ├── legacy_steady_dae/
│   ├── signed_flux_bridge/
│   └── conservative_time_dependent/
├── results/
│   ├── README.md
│   ├── canonical/
│   └── manifests/
├── references/
│   ├── REFERENCES.md
│   └── references.bib
└── scripts/
    └── thin, documented entry points only
```

During the first cleanup, avoid moving the existing source package solely for aesthetic reasons.

---

## 7. `.gitignore` policy

Replace the current narrow output rules with a generated-artifact policy such as:

```gitignore
# Python/tool caches
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Environments/build products
.venv/
venv/
build/
dist/
*.egg-info/

# Local scratch and generated runs
/outputs/**
!/outputs/README.md
/scratch/
/tmp/
/runs/
/checkpoints/
/artifacts/raw/

# Logs/profiling
*.log
*.prof
*.trace

# OS/editor
.DS_Store
Thumbs.db
*~
*.swp
```

Do not rely on `.gitignore` alone. Add a CI hygiene check that fails when:

```text
- a generated checkpoint appears outside an approved canonical/test-fixture path;
- a new tracked file exceeds the project size threshold;
- a commit adds an unexpectedly large number of generated files;
- caches or compiled Python files are tracked;
- a report references a missing canonical artifact;
- a canonical artifact lacks provenance and SHA-256 metadata.
```

Suggested main-branch file-size threshold:

```text
5 MiB by default
```

Larger files require an explicit allow-list entry and written scientific justification.

---

## 8. Cleanup commit sequence

Use small, reviewable commits.

### Commit 1 — Inventory only

```text
Add repository inventory, duplicate report, largest-file report,
script dependency inventory, and proposed keep/archive/delete manifest.
No deletions.
```

### Commit 2 — Documentation/status consolidation

```text
Add docs/PROJECT_STATUS.md, MODEL_EQUATIONS.md,
REPRODUCIBILITY.md, ARTIFACT_POLICY.md, and MILESTONES.md.
Replace stale handoff pointers.
No numerical changes.
```

### Commit 3 — Canonical artifact set

```text
Add the selected compact canonical states, summaries, figures,
configs, provenance, and checksums.
Add regression tests that use them.
```

### Commit 4 — Archive verification record

```text
Add the external/archive manifest, archive checksum, tag name,
storage location, and extraction/verification log.
Do not include the full bulk archive in normal Git history.
```

### Commit 5 — Remove bulk generated outputs from main

```text
Remove outputs/checkpoints and raw/per-run tables/figures after
the archive and canonical set have been verified.
Update .gitignore.
```

### Commit 6 — Literature cleanup

```text
Remove full-paper copies.
Add references.bib and REFERENCES.md.
```

### Commit 7 — Script cleanup/refactor

```text
Move reusable code into src/, add tests, retain thin entry points,
and remove superseded exploratory wrappers.
This commit may be split into several smaller commits.
```

### Commit 8 — CI hygiene guard

```text
Add tests and repository-size/generated-file checks.
```

Keep scientific model changes out of these commits.

---

## 9. History rewrite policy

Initial decision:

```text
DO NOT REWRITE HISTORY.
```

Removing files in a new commit makes the current tree cleaner but does not remove old blobs from historical commits.

After the clean branch has been used successfully, measure:

```text
git count-objects -vH
fresh clone size
fresh clone time
largest historical blobs
pack size
```

Only then consider a separate history-rewrite work package.

A history rewrite must require:

```text
- explicit owner approval;
- a mirrored backup;
- verified legacy tag/archive outside the rewritten repository;
- a written mapping from old to new important SHAs;
- coordination with every active clone;
- force-push plan;
- post-rewrite integrity verification.
```

Do not casually rewrite history merely because the default tree is cluttered.

---

## 10. Acceptance criteria

The cleanup is accepted only if all conditions below pass.

### Scientific integrity

```text
[ ] No equation or closure changed.
[ ] No accepted numerical threshold changed.
[ ] Endpoint classification remains identical.
[ ] Global-composite certification status remains identical.
[ ] Negative results remain documented.
[ ] Canonical checkpoint values match pre-cleanup values.
```

### Reproducibility

```text
[ ] Fresh clone installs successfully.
[ ] Full unit test suite passes.
[ ] Compact regression suite passes.
[ ] Retained figures/tables regenerate from canonical inputs.
[ ] Every canonical artifact has config, provenance, status, and SHA-256.
[ ] Archive extracts and verifies successfully.
```

### Repository hygiene

```text
[ ] No cache or compiled files are tracked.
[ ] No bulk checkpoint ladder remains in main.
[ ] No raw Newton-audit directory remains in main.
[ ] No exploratory figure ladder remains in main.
[ ] No full-paper PDFs remain in the public source repository.
[ ] One canonical project-status document exists.
[ ] No competing stale handoff exists.
[ ] Generated-output policy is enforced in CI.
```

### Reviewability

```text
[ ] Cleanup commits are separated from physics changes.
[ ] Every removed path appears in the keep/archive/delete manifest.
[ ] Every archived path has a verified recovery location.
[ ] Script removals have dependency evidence.
[ ] The default branch contains no unexplained large binary.
```

---

## 11. Explicit instructions to Codex

1. Preserve `1e7438e` with an immutable annotated tag before changing the tree.
2. Inventory first; do not start with `git rm`.
3. Build and verify the full artifact archive.
4. Select the canonical result set according to scientific importance, not convenience.
5. Keep all source and tests in the first cleanup pass.
6. Replace stale documentation with one canonical status document.
7. Remove raw generated outputs from the default branch only after archive verification.
8. Replace literature PDFs with a bibliography.
9. Refactor one-off scripts only after dependency and numerical-parity checks.
10. Do not rewrite history.
11. Do not mix the repository cleanup with implementation of the new conservative disk equations.
12. End with a report containing:
    - before/after file counts and byte sizes;
    - before/after test results;
    - canonical artifact list and hashes;
    - archive location and verification result;
    - exact files kept, archived, and deleted;
    - unresolved cleanup risks;
    - exact commit SHA of the cleaned tree.

---

## 12. Final judgment

The project should keep the **code, tests, current scientific interpretation, and a compact set of decisive states**.

It should not keep thousands of intermediate solver states and raw diagnostic products in the normal source branch.

The steady positive-flux DAE work remains scientifically valuable and must be frozen as a reproducible legacy benchmark. The cleanup should create a clean foundation for the next physical model without erasing the route by which the current endpoint conclusion was reached.
