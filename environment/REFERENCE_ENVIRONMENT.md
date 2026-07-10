# Tested Reference Environment

Recorded for the pre-cleanup parity suite on 2026-07-11.

```text
Python     3.12.13 (Clang 21.1.4)
platform   macOS 14.8.3, arm64
NumPy      2.3.5
SciPy      1.18.0
pytest     9.1.1
Pillow     12.2.0
BLAS       Apple Accelerate
LAPACK     Apple Accelerate
```

Matplotlib was not installed in the runtime used for the parity suite. Current
retained audit figures are generated with Pillow; Matplotlib remains an
optional analysis dependency for older plotting entry points.

Installation target:

```bash
python3 -m pip install -e '.[solver,dev]'
```

The exact Codex runtime path is intentionally not part of the portable
environment specification.
