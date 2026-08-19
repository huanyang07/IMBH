#!/usr/bin/env python3
"""Freeze the exact geometric 28D departure-chart preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_explicit_nonlinear_470_architecture_audit_wp10c9d6c7c3b5c4f25aw as parent  # noqa: E402
import run_causal_inner_nonlinear_bundle_screen_wp10c9d6c7c3b5c4f25ak as failed_screen  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ax"
CLASSIFICATION = (
    "exact_geometric_28D_departure_chart_preflight_manifest_frozen_"
    "no_rate_database_execution_authorized"
)
PARENT_COMMIT = "8b4baa5011795f8ff5ab1649ea313347005d4814"
PARENT_PARENT = "6b43ad99d1893527227c1e67ea49e29e0136cfa9"
PARENT_TREE = "f68e330fb0eb96c8f2ff3711ccdb75f0a307c81d"

ARTIFACT = (
    "causal_inner_exact_geometric_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25ax"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25ax.py"
)
THIS_TEST = (
    "tests/test_causal_inner_exact_geometric_departure_chart_manifest_"
    "wp10c9d6c7c3b5c4f25ax.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25ay.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_exact_geometric_departure_chart_preflight_"
    "wp10c9d6c7c3b5c4f25ay.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXACT_GEOMETRIC_DEPARTURE_"
    "CHART_MANIFEST_WP10C9D6C7C3B5C4F25AX_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

GEOMETRY_PATH = parent.CANONICAL_DIRECTORY / "online_470_geometry.npz"
PREFLIGHT_PATH = (
    parent.preflight.CANONICAL_DIRECTORY / "preflight_diagnostics.npz"
)
FIBER_PATH = parent.FIBER_PATH

ANCHOR = "primary"
PHYSICAL_COORDINATE_DIMENSION = 162
DEPARTURE_DIMENSION = 28
ENERGY_DIRECTION_COUNT = 8
MAXIMUM_COMPONENT_BOUNDS = (2.5e-4, 1.0e-3, 5.0e-3)
SIGNS = (-1, 1)
PLANNED_CANDIDATES = (
    ENERGY_DIRECTION_COUNT * len(MAXIMUM_COMPONENT_BOUNDS) * len(SIGNS)
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("470-state architecture certificate commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("470-state architecture certificate lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("470-state architecture certificate tree changed")
    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    parent_summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    failed_hashes = _checksums(failed_screen.CANONICAL_DIRECTORY)
    failed_summary = _read(failed_screen.CANONICAL_DIRECTORY / "summary.json")
    failed_metrics = _read(failed_screen.CANONICAL_DIRECTORY / "metrics.json")
    if (
        not parent_summary["passed"]
        or parent_summary["online_coordinate_rank"] != 470
        or parent_summary["hidden_stable_remainder_dimension"] != 90
        or parent_summary["authorized_next"]
        != "definitions_only_exact_geometric_28D_departure_chart_and_database_manifest"
    ):
        raise RuntimeError("470-state architecture authorization changed")
    if (
        failed_metrics["completed_nonbase_continuous_rate_evaluations"] != 0
        or failed_metrics["fail_fast_coordinate_retraction"]["failure_kind"]
        != "rate_reaction_is_not_a_geometric_retraction"
    ):
        raise RuntimeError("failed reaction-lift diagnosis changed")
    return {
        "parent_summary": parent_summary,
        "parent_hashes": parent_hashes,
        "failed_summary": failed_summary,
        "failed_metrics": failed_metrics,
        "failed_hashes": failed_hashes,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "certify_a_finite_amplitude_chart_for_the_explicit_28D_"
            "departure_coordinates_before_spending_any_nonbase_rate_call"
        ),
        "candidate_family": {
            "anchor": ANCHOR,
            "coordinate_space": "scaled_560_primitive_coordinates",
            "departure_basis": (
                "certified_28D_basis_projected_against_the_162_physical_and_"
                "280_stable_memory_coordinates"
            ),
            "direction_selection": (
                "largest_eigenvectors_of_the_symmetric_part_of_the_"
                "similarity_transformed_28D_linear_departure_operator"
            ),
            "direction_count": ENERGY_DIRECTION_COUNT,
            "maximum_component_bounds": list(MAXIMUM_COMPONENT_BOUNDS),
            "signs": list(SIGNS),
            "planned_candidates": PLANNED_CANDIDATES,
            "heldout_anchor_used": False,
        },
        "exact_geometric_retraction": {
            "target": "C_phys_x_equals_C_phys_x0",
            "coordinate_definition": (
                "160_exact_integrated_mapped_only_R32_storage_coordinates_"
                "plus_2_frozen_stable_coordinates"
            ),
            "state_local_derivative": "exact_descriptor_coordinate_Jacobian",
            "correction": (
                "delta_x_equals_minus_JC_transpose_times_"
                "solve_JC_JC_transpose_coordinate_error"
            ),
            "globalization": "monotone_coordinate_residual_backtracking",
            "line_factors": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
            "maximum_Newton_iterations": 8,
            "maximum_radius_rescalings": 4,
            "rate_reaction_lift_used": False,
            "coordinate_target_or_trial_projection_used": False,
        },
        "binding_preflight_gates": {
            "completed_candidate_count_equal": PLANNED_CANDIDATES,
            "failed_candidate_count_equal": 0,
            "maximum_coordinate_residual_infinity": 1.0e-10,
            "maximum_normalized_Q3_defect": 1.0e-10,
            "maximum_final_scaled_component": max(MAXIMUM_COMPONENT_BOUNDS),
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_coordinate_Jacobian_condition_number": 1.0e4,
            "minimum_departure_direction_alignment_cosine": 0.99,
            "maximum_departure_transverse_fraction": 0.05,
            "maximum_coordinate_odd_symmetry_defect": 0.05,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "nonbase_continuous_rate_evaluations_equal": 0,
            "new_full_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "propagated_states_equal": 0,
        },
        "decision": {
            "pass": (
                "exact_geometric_28D_departure_chart_preflight_passed_"
                "guarded_nonlinear_rate_database_manifest_authorized"
            ),
            "fail": (
                "exact_geometric_28D_departure_chart_preflight_failed_"
                "nonlinear_rate_database_blocked"
            ),
            "pass_authorizes_only": (
                "definitions_only_guarded_nonlinear_28D_rate_database_manifest"
            ),
        },
        "claim_boundary": {
            "nonlinear_rate_database_executed": False,
            "nonlinear_28D_closure_identified": False,
            "online_integrator_implemented": False,
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
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parents = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("geometric chart manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("geometric chart manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor": ANCHOR,
        "departure_dimension": DEPARTURE_DIMENSION,
        "planned_candidate_count": PLANNED_CANDIDATES,
        "planned_nonbase_continuous_rate_evaluations": 0,
        "nonlinear_rate_database_executed": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ay",
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "contract.json", _contract())
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(
        ARTIFACT_DIRECTORY / "parent_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            "architecture_audit_hashes": parents["parent_hashes"],
            "failed_reaction_lift_screen_hashes": parents["failed_hashes"],
            "decisive_input_hashes": {
                "online_470_geometry": _sha(GEOMETRY_PATH),
                "exact_coordinate_preflight": _sha(PREFLIGHT_PATH),
                "exact_departure_fiber": _sha(FIBER_PATH),
            },
        },
    )
    _write(
        ARTIFACT_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PROSPECTIVE",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "authorized_next_runner": NEXT_RUNNER,
            "authorized_next_test": NEXT_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": parent.preflight.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Exact geometric departure-chart manifest WP10c9d6c7c3b5c4f25ax",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This package freezes a 48-candidate, primary-anchor geometry-only preflight for the explicit nonlinear 28-dimensional departure chart.",
                "",
                "Every finite-amplitude state is retracted with the exact state-local derivative of C_phys. The fixed-Q rate reaction lift is forbidden because it is a rate-control operator, not a geometric state normal.",
                "",
                "No nonbase continuous-rate evaluation, nonlinear root, propagated state, fitted coefficient, or predictive reduced evolution is authorized here.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("pass --freeze")
    print(json.dumps(_plain(_freeze()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
