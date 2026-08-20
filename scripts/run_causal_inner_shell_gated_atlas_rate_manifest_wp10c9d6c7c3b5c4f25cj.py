#!/usr/bin/env python3
"""Freeze independent exact-rate validation of the shell-gated atlas."""

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

import run_causal_inner_shell_gated_atlas_geometry_preflight_wp10c9d6c7c3b5c4f25ci as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cj"
PARENT_COMMIT = "dddeed7d7358daaa1edf2965f8dcb22a354688ea"
PARENT_PARENT = "132737e71741e966babd48344004e317b1094ca4"
PARENT_TREE = "8e196f431c881e97fe641bc63d9df7713b35bcf9"
CLASSIFICATION = "shell_gated_atlas_independent_rate_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ck"
PLANNED_RATE_EVALUATIONS = 8

ARTIFACT = (
    "causal_inner_shell_gated_atlas_rate_manifest_"
    "wp10c9d6c7c3b5c4f25cj"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_rate_manifest_"
    "wp10c9d6c7c3b5c4f25cj.py"
)
THIS_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_rate_manifest_"
    "wp10c9d6c7c3b5c4f25cj.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_shell_gated_atlas_rate_validation_"
    "wp10c9d6c7c3b5c4f25ck.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_shell_gated_atlas_rate_validation_"
    "wp10c9d6c7c3b5c4f25ck.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SHELL_GATED_ATLAS_"
    "RATE_MANIFEST_WP10C9D6C7C3B5C4F25CJ_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

GEOMETRY_ARRAYS = parent.CANONICAL_DIRECTORY / "holdout_geometry.npz"
EXTENSION_ARRAYS = (
    parent.manifest.parent.CANONICAL_DIRECTORY / "local_atlas_extension.npz"
)

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
        raise RuntimeError("mixed atlas geometry commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("mixed atlas geometry lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("mixed atlas geometry tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = _read(parent.CANONICAL_DIRECTORY / "geometry_metrics.json")
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    atlas_hashes = _checksums(parent.manifest.parent.CANONICAL_DIRECTORY)
    arrays = _load_npz(GEOMETRY_ARRAYS)
    if (
        not summary["passed"]
        or summary["classification"] != parent.FULL_CLASSIFICATION
        or summary["largest_passing_component_bound"] != 0.015
        or summary["completed_candidate_count"] != PLANNED_RATE_EVALUATIONS
        or summary["failed_candidate_count"] != 0
        or summary["authorized_next"]
        != "definitions_only_full_mixed_holdout_exact_rate_manifest"
        or summary["new_truth_calls"] != 0
        or summary["geometry_candidate_became_atlas_center"]
        or arrays["candidate_primitive_states"].shape != (8, 112, 5)
        or arrays["candidate_scaled_deltas"].shape != (8, 560)
        or arrays["candidate_departure_coordinates"].shape != (8, 28)
        or not all(metrics["decision"]["budget_checks"].values())
    ):
        raise RuntimeError("mixed atlas exact-rate authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"mixed atlas geometry source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("atlas rate manifest requires a clean tracked tree")
    return {
        "summary": summary,
        "metrics": metrics,
        "hashes": hashes,
        "atlas_hashes": atlas_hashes,
        "arrays": arrays,
    }


def _contract() -> dict:
    inherited_model = parent.manifest.parent._contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "candidate_database": {
            "count": PLANNED_RATE_EVALUATIONS,
            "component_bounds": [0.0125, 0.015],
            "directions_per_bound": 4,
            "independent_of_extension_fit": True,
            "coefficient_refit_after_truth": False,
        },
        "binding_exact_rate_gates": {
            "completed_nonbase_rate_evaluations_equal": PLANNED_RATE_EVALUATIONS,
            "failed_rate_evaluations_equal": 0,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e6,
            "maximum_reaction_identity_defect": 1.0e-9,
            "maximum_rate_tangency_relative_defect": 1.0e-8,
            "maximum_coordinate_Jacobian_condition_number": 5.0e3,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_incoming_excision_characteristics_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "binding_independent_model_gates": inherited_model[
            "future_independent_rate_gates"
        ],
        "model_contract": {
            "inner_model": "frozen_certified_degree23_departure_field",
            "outer_correction": "frozen_full_state_degree4_even_degree5_odd",
            "shell_gate": inherited_model["shell_gate"],
            "decoder_and_rate_coefficients_hash_locked_before_truth": True,
            "online_truth_calls_per_macrostep": 0,
        },
        "decision": {
            "pass": {
                "classification": "shell_gated_degree45_atlas_field_independently_validated",
                "authorizes_only": "definitions_only_authentic_recentered_transition_forecast_manifest",
            },
            "fail": {
                "classification": "shell_gated_degree45_atlas_field_independent_validation_failed",
                "authorizes_only": "definitions_only_alternative_local_rate_extension_manifest",
            },
        },
        "authorization_boundaries": {
            "new_truth_calls_during_manifest": 0,
            "new_generator_assemblies": 0,
            "new_nonlinear_roots": 0,
            "propagated_states": 0,
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
        raise RuntimeError("atlas exact-rate manifest already canonicalized")
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "contract.json", contract)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {"parent_commit": PARENT_COMMIT, "parent_parent": PARENT_PARENT, "parent_tree": PARENT_TREE, "parent_hashes": frozen["hashes"], "atlas_hashes": frozen["atlas_hashes"], "holdout_geometry_sha256": _sha(GEOMETRY_ARRAYS), "extension_coefficients_sha256": _sha(EXTENSION_ARRAYS)})
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "planned_exact_rate_evaluations": PLANNED_RATE_EVALUATIONS, "coefficients_frozen_before_truth": True, "new_truth_calls": 0, "new_generator_assemblies": 0, "new_nonlinear_roots": 0, "propagated_states": 0, "trajectory_authorized": False, "predictive_cycle_authorized": False, "reduced_slow_evolution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST, parent.manifest.parent.THIS_RUNNER, parent.manifest.parent.THIS_TEST)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "definition_commit": _git("rev-parse", "HEAD"), "definition_tree": _git("rev-parse", "HEAD^{tree}"), "tracked_worktree_clean_at_start": True, "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(("# Shell-gated atlas rate manifest WP10c9d6c7c3b5c4f25cj", "", "## Classification", "", f"`{CLASSIFICATION}`", "", "Eight independent mixed forward-sector states are frozen for exact continuous-rate validation. All decoder and degree-4/5 full-state rate coefficients are hash-locked before any holdout rate is evaluated.", "", "A pass requires exact-rate admissibility plus maximum/median full-state errors 0.15/0.075, maximum a28 error 0.15, exact radial-sign agreement, and decoder errors below 0.005.", "", "No trajectory, nonlinear root, generator assembly, physical microburst, cycle evolution, or reduced slow evolution is authorized.", "")), encoding="utf-8")
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
