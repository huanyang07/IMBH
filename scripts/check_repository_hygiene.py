"""Fail when the tracked tree violates the repository artifact policy."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
MAX_TRACKED_FILES = 600
BANNED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
BANNED_SUFFIXES = {".pyc", ".pyo", ".swp", ".tmp"}


def tracked_paths() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / piece.decode() for piece in raw.split(b"\0") if piece]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hygiene_errors(paths: list[Path] | None = None) -> list[str]:
    paths = tracked_paths() if paths is None else paths
    errors: list[str] = []
    if len(paths) >= MAX_TRACKED_FILES:
        errors.append(
            f"tracked file count {len(paths)} reaches policy limit "
            f"{MAX_TRACKED_FILES}"
        )
    for path in paths:
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix()
        if any(part in BANNED_PARTS for part in relative.parts):
            errors.append(f"tracked cache path: {relative_text}")
        if path.suffix.lower() in BANNED_SUFFIXES or path.name in {".DS_Store", "Thumbs.db"}:
            errors.append(f"tracked temporary file: {relative_text}")
        if relative.parts[0] == "outputs" and relative_text != "outputs/README.md":
            errors.append(f"tracked generated output: {relative_text}")
        if relative.parts[0] == "Literature":
            errors.append(f"tracked full-text literature file: {relative_text}")
        if path.exists() and path.stat().st_size > MAX_BYTES:
            errors.append(f"tracked file exceeds 5 MiB: {relative_text}")

    canonical_root = ROOT / "results/canonical"
    for case in sorted(path for path in canonical_root.iterdir() if path.is_dir()):
        provenance = case / "provenance.json"
        checksums = case / "SHA256SUMS.txt"
        if not provenance.is_file():
            errors.append(f"canonical case lacks provenance: {case.name}")
        if not checksums.is_file():
            errors.append(f"canonical case lacks checksums: {case.name}")
            continue
        for line in checksums.read_text().splitlines():
            expected, filename = line.split("  ", 1)
            payload = case / filename
            if not payload.is_file():
                errors.append(f"canonical checksum target missing: {payload.relative_to(ROOT)}")
            elif sha256(payload) != expected:
                errors.append(f"canonical checksum mismatch: {payload.relative_to(ROOT)}")

    if not (ROOT / "docs/PROJECT_STATUS.md").is_file():
        errors.append("canonical project status is missing")
    if not (ROOT / "GPT_REPO_HANDOFF.md").is_file():
        errors.append("GPT handoff pointer is missing")
    return errors


def main() -> None:
    errors = hygiene_errors()
    if errors:
        print("Repository hygiene failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Repository hygiene passed for {len(tracked_paths())} tracked files.")


if __name__ == "__main__":
    main()
