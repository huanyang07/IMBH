"""Build deterministic repository-cleanup and script-dependency inventories."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "docs" / "manifests"
ARCHIVE_NAME = "IMBH_legacy_steady_dae_artifacts_pre-cleanup-p0-2026-07-11.tar.gz"

CURRENT_NOTES = {
    "CODEX_IMBH_PROJECT_REVIEW_P0_RESULTS_2026-07-10.md",
    "CODEX_IMBH_REPOSITORY_CLEANUP_SPEC_2026-07-10.md",
    "CODEX_MDOT5_ANGULAR_MOMENTUM_LEDGER_RESULTS.md",
    "CODEX_MDOT5_ENDPOINT_VALIDITY_AND_EXPONENT_AUDIT_RESULTS.md",
    "CODEX_MDOT5_GLOBAL_PHASE_DAE_PRODUCTION_RESULTS.md",
    "CODEX_MDOT5_INDEPENDENT_OUTER_MANIFOLD_RESULTS.md",
    "CODEX_MDOT5_PHASE_CRITICAL_CLASSIFICATION_RESULTS.md",
    "CODEX_MDOT5_PHASE_CRITICAL_GLOBALIZATION_RESULTS.md",
    "CODEX_MDOT5_PHASE_DAE_EXIT_REFINEMENT_RESULTS.md",
}

ESSENTIAL_SCRIPTS = {
    "run_mdot5_angular_momentum_ledger_audit.py",
    "run_mdot5_endpoint_validity_audit.py",
    "run_mdot5_global_phase_dae_production.py",
    "run_mdot5_independent_outer_manifold_search.py",
    "run_mdot5_local_mdot_eta_continuation.py",
    "run_mdot5_phase_critical_classification.py",
    "run_mdot5_phase_critical_globalization.py",
    "run_mdot5_phase_dae_exit_refinement.py",
    "run_standard_slim_high_mdot_no_wind_ladder.py",
    "run_standard_slim_stream_anchor_regression.py",
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_paths() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / os.fsdecode(piece) for piece in raw.split(b"\0") if piece]


def _classification(relative: str) -> tuple[str, str, str, str]:
    path = Path(relative)
    top = path.parts[0]
    archive = f"{ARCHIVE_NAME}:{relative}"
    if top == "outputs":
        return (
            "generated_scientific_artifact",
            "ARCHIVE",
            "Bulk generated output; decisive cases will be copied into results/canonical.",
            archive,
        )
    if top == "Literature":
        return (
            "reference_full_text",
            "DELETE",
            "Do not distribute full-paper PDFs in the source tree; replace with bibliography.",
            archive,
        )
    if top == "Note":
        if path.name in CURRENT_NOTES:
            return (
                "current_scientific_report",
                "KEEP",
                "Current result or cleanup report; consolidate under docs/reports/current.",
                "docs/reports/current/",
            )
        return (
            "historical_scientific_note",
            "ARCHIVE",
            "Superseded numerical diary; preserve through tag/archive and summarize in milestones.",
            archive,
        )
    if top == "scripts":
        return (
            "scientific_runner",
            "KEEP",
            "Retain pending dependency audit; archive candidates require parity evidence.",
            "scripts/ or legacy tag",
        )
    if relative in {"notebooks/.gitkeep", "outputs/runs/.gitkeep"}:
        return (
            "empty_placeholder",
            "DELETE",
            "Runtime code creates directories as needed.",
            "not required",
        )
    if top == "tests":
        return ("test", "KEEP", "Regression or unit coverage.", relative)
    if top == "src":
        return ("scientific_source", "KEEP", "Production scientific source.", relative)
    return ("project_file", "KEEP", "Project configuration or handoff pointer.", relative)


def _read_texts(paths: list[Path], suffixes: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if path.suffix.lower() not in suffixes:
            continue
        try:
            result[path.relative_to(ROOT).as_posix()] = path.read_text(errors="replace")
        except OSError:
            continue
    return result


def _script_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tracked = _tracked_paths()
    commit = _git("rev-parse", "HEAD")
    inventory: list[dict[str, object]] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    total_size = 0
    for path in tracked:
        relative = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        digest = _sha256(path)
        category, action, reason, replacement = _classification(relative)
        total_size += size
        by_hash[digest].append(relative)
        inventory.append(
            {
                "path": relative,
                "file_type": path.suffix.lower().lstrip(".") or "none",
                "bytes": size,
                "sha256": digest,
                "git_status": "tracked",
                "scientific_category": category,
                "proposed_action": action,
                "reason": reason,
                "replacement_or_archive": replacement,
            }
        )
    inventory.sort(key=lambda row: str(row["path"]))
    _write_csv(
        MANIFEST_DIR / "repository_inventory.csv",
        list(inventory[0]),
        inventory,
    )
    largest = sorted(inventory, key=lambda row: int(row["bytes"]), reverse=True)
    _write_csv(
        MANIFEST_DIR / "largest_tracked_files.csv",
        list(inventory[0]),
        largest[:500],
    )

    duplicate_rows: list[dict[str, object]] = []
    for digest, paths in sorted(by_hash.items()):
        if len(paths) < 2:
            continue
        size = (ROOT / paths[0]).stat().st_size
        duplicate_rows.append(
            {
                "sha256": digest,
                "bytes_each": size,
                "copies": len(paths),
                "wasted_bytes": size * (len(paths) - 1),
                "paths": " | ".join(paths),
            }
        )
    duplicate_rows.sort(key=lambda row: int(row["wasted_bytes"]), reverse=True)
    _write_csv(
        MANIFEST_DIR / "duplicate_files.csv",
        ["sha256", "bytes_each", "copies", "wasted_bytes", "paths"],
        duplicate_rows,
    )

    scripts = sorted((ROOT / "scripts").glob("*.py"))
    script_modules = {path.stem: path for path in scripts}
    imports = {path.stem: _script_imports(path) for path in scripts}
    imported_by: dict[str, set[str]] = defaultdict(set)
    for importer, modules in imports.items():
        for module in modules:
            if module in script_modules:
                imported_by[module].add(importer)
    texts = _read_texts(tracked, {".md", ".py", ".toml"})
    script_rows: list[dict[str, object]] = []
    for path in scripts:
        module = path.stem
        name = path.name
        docs = sorted(
            source
            for source, text in texts.items()
            if source.endswith(".md") and name in text
        )
        tests = sorted(
            source
            for source, text in texts.items()
            if source.startswith("tests/") and (name in text or module in text)
        )
        essential = name in ESSENTIAL_SCRIPTS
        dependency = bool(imported_by[module])
        if essential:
            classification = "keep_essential"
            replacement = "scripts/"
            reason = "Reproduces a canonical/current result."
        elif dependency:
            classification = "keep_dependency_pending_refactor"
            replacement = "Move reusable helpers to src/ before retirement."
            reason = "Imported by another tracked runner."
        elif docs or tests:
            classification = "keep_referenced_pending_review"
            replacement = "scripts/ or documented replacement"
            reason = "Referenced by documentation or tests."
        else:
            classification = "archive_candidate_review"
            replacement = "legacy tag and verified artifact archive"
            reason = "No detected imports, tests, or documentation references."
        script_rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "lines": len(path.read_text(errors="replace").splitlines()),
                "imports": " | ".join(sorted(imports[module])),
                "imported_by": " | ".join(sorted(imported_by[module])),
                "referenced_by_docs": " | ".join(docs),
                "test_coverage": " | ".join(tests),
                "canonical_artifact_dependency": essential,
                "classification": classification,
                "replacement": replacement,
                "reason": reason,
            }
        )
    _write_csv(
        MANIFEST_DIR / "script_inventory.csv",
        list(script_rows[0]),
        script_rows,
    )

    metrics = {
        "source_commit": commit,
        "tracked_file_count": len(inventory),
        "tracked_bytes": total_size,
        "inventory_action_counts": {
            action: sum(row["proposed_action"] == action for row in inventory)
            for action in ("KEEP", "ARCHIVE", "DELETE")
        },
        "exact_duplicate_groups": len(duplicate_rows),
        "exact_duplicate_wasted_bytes": sum(
            int(row["wasted_bytes"]) for row in duplicate_rows
        ),
        "script_count": len(script_rows),
        "script_classification_counts": {
            category: sum(row["classification"] == category for row in script_rows)
            for category in sorted({str(row["classification"]) for row in script_rows})
        },
    }
    (MANIFEST_DIR / "pre_cleanup_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
