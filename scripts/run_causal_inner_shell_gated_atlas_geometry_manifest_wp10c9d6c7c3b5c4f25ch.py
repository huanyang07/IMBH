#!/usr/bin/env python3
"""Freeze geometry-first mixed-direction validation of the local atlas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_shell_gated_recentered_atlas_manifest_wp10c9d6c7c3b5c4f25cg as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ch"
PARENT_COMMIT = "7251f59456cef29363cb8b217a844e9c0805b2f7"
PARENT_PARENT = "3509fa0c8fba7ac7eb6bc931ef4494e963f458fa"
PARENT_TREE = "3c313cae02284e111b54974ab307e685d2e89e35"
CLASSIFICATION = "shell_gated_atlas_holdout_geometry_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ci"

ARTIFACT = (
    "causal_inner_shell_gated_atlas_geometry_manifest_"
    "wp10c9d6c7c3b5c4f25ch"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_geometry_manifest_"
    "wp10c9d6c7c3b5c4f25ch.py"
)
THIS_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_geometry_manifest_"
    "wp10c9d6c7c3b5c4f25ch.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ci.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25ci.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SHELL_GATED_ATLAS_"
    "GEOMETRY_MANIFEST_WP10C9D6C7C3B5C4F25CH_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_HOLDOUT = parent.CANONICAL_DIRECTORY / "holdout_design.npz"
PARENT_EXTENSION = parent.CANONICAL_DIRECTORY / "local_atlas_extension.npz"

_plain = parent._plain
_read = parent._read
_write_json = parent._write_json
_sha = parent._sha
_checksums = parent._checksums
_load_npz = parent._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("shell-gated atlas commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("shell-gated atlas lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("shell-gated atlas tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "design_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    holdout = _load_npz(PARENT_HOLDOUT)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != parent.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["old_inner_certificate_preserved_exactly"]
        or summary["planned_independent_geometry_candidates"] != 8
        or summary["new_truth_calls"] != 0
        or summary["trajectory_authorized"]
        or not all(metrics["checks"].values())
        or holdout["directions"].shape != (4, 28)
        or not np.array_equal(
            holdout["component_bounds"],
            np.asarray(parent.HOLDOUT_COMPONENT_BOUNDS),
        )
    ):
        raise RuntimeError("shell-gated geometry authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"shell-gated atlas source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("atlas geometry manifest requires a clean tracked tree")
    return {"summary": summary, "metrics": metrics, "hashes": hashes, "holdout": holdout}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "candidate_family": {
            "component_bounds": list(parent.HOLDOUT_COMPONENT_BOUNDS),
            "rung_order": "strictly_increasing_fail_fast",
            "direction_count": parent.HOLDOUT_DIRECTION_COUNT,
            "signs": [1],
            "candidates_per_rung": parent.HOLDOUT_DIRECTION_COUNT,
            "maximum_planned_candidates": parent.PLANNED_GEOMETRY_CANDIDATES,
        },
        "exact_geometric_retraction": {
            "equations": "C_phys_x_equals_C_phys_x_anchor",
            "state_normal": "exact_state_local_derivative_of_C_phys",
            "departure_seed": "frozen_positive_mixed_forward_sector_direction",
            "stable_memory_seed": "zero",
            "rate_reaction_lift_used": False,
            "maximum_Newton_iterations": 8,
            "maximum_line_search_iterations": 12,
            "maximum_radius_rescalings": 6,
            "coordinate_residual_tolerance": 1.0e-10,
        },
        "binding_per_rung_gates": {
            "completed_candidate_count_equal": parent.HOLDOUT_DIRECTION_COUNT,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 1.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 1.0e4,
            "minimum_departure_direction_alignment_cosine": 0.99,
            "maximum_departure_transverse_fraction": 0.05,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
        },
        "cost_budget": {
            "new_nonbase_continuous_rate_evaluations_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_fixed_Q_roots_equal": 0,
            "propagated_physical_states_equal": 0,
        },
        "decision": {
            "both_rungs_pass": {
                "classification": "shell_gated_atlas_mixed_geometry_valid_to_0p015",
                "authorizes_only": "definitions_only_full_mixed_holdout_exact_rate_manifest",
            },
            "only_first_rung_passes": {
                "classification": "shell_gated_atlas_mixed_geometry_valid_to_0p0125",
                "authorizes_only": "definitions_only_trigger_shell_exact_rate_manifest",
            },
            "first_rung_fails": {
                "classification": "shell_gated_atlas_mixed_geometry_failed_before_recenter_margin",
                "authorizes_only": "definitions_only_alternative_local_chart_basis_manifest",
            },
        },
        "authorization_boundaries": {
            "rate_truth_authorized_by_this_manifest": False,
            "geometry_candidate_may_become_atlas_center": False,
            "trajectory_authorized": False,
            "physical_microburst_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("atlas geometry manifest already canonicalized")
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {"parent_commit": PARENT_COMMIT, "parent_parent": PARENT_PARENT, "parent_tree": PARENT_TREE, "parent_hashes": frozen["hashes"], "holdout_design_sha256": _sha(PARENT_HOLDOUT), "extension_sha256": _sha(PARENT_EXTENSION)})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "planned_geometry_candidates": parent.PLANNED_GEOMETRY_CANDIDATES, "new_truth_calls": 0, "new_generator_assemblies": 0, "new_nonlinear_roots": 0, "propagated_states": 0, "trajectory_authorized": False, "predictive_cycle_authorized": False, "reduced_slow_evolution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "definition_commit": _git("rev-parse", "HEAD"), "definition_tree": _git("rev-parse", "HEAD^{tree}"), "tracked_worktree_clean_at_start": True, "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(("# Shell-gated atlas geometry manifest WP10c9d6c7c3b5c4f25ch", "", "## Classification", "", f"`{CLASSIFICATION}`", "", "Eight positive, mixed forward-sector geometry holdouts are frozen at component bounds 0.0125 and 0.015 in fail-fast order.", "", "All exact retraction, constraint, reconstruction, conditioning, direction-fidelity, height, and optical-depth gates are frozen before execution. No rate truth or trajectory is authorized by this package.", "")), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
