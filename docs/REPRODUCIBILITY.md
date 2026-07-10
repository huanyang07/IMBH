# Reproducibility

## Environment

```bash
python3 -m pip install -e '.[solver,dev]'
```

The recorded development environment is under `environment/`. Numerical
results depend on NumPy/SciPy and the platform BLAS/LAPACK implementation.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

Pre-cleanup baseline:

```text
182 passed, 4 subtests passed
```

Compact regression tests validate the canonical states, scientific-status
metadata, and SHA-256 manifests without requiring the historical checkpoint
ladder.

## Canonical Results

The retained evidence is under `results/canonical/`. Every case includes:

- configuration or compact state metadata;
- scientific status and limitations;
- source commit and tag;
- source paths and source hashes;
- payload hashes and `SHA256SUMS.txt`.

Validate all canonical cases with:

```bash
for d in results/canonical/*; do
  (cd "$d" && shasum -a 256 -c SHA256SUMS.txt)
done
```

## Full Legacy Evidence

The immutable Git tag `pre-cleanup-p0-2026-07-11` contains the complete
pre-cleanup tree. A separately verified local archive is recorded in
`docs/manifests/archive_verification.json`.

To regenerate a canonical artifact from raw checkpoints:

1. Check out the pre-cleanup tag in a separate worktree or extract the archive.
2. Use the generation command in the canonical case `provenance.json`.
3. Compare output hashes with `results/manifests/canonical_artifacts.csv`.

The archive itself is intentionally not committed to ordinary Git history.

## Production Output Metadata

New production outputs should record:

```text
Git SHA
Python, NumPy, and SciPy versions
platform and BLAS/LAPACK information when practical
command line and configuration
random seed when applicable
scientific status and acceptance gates
```
