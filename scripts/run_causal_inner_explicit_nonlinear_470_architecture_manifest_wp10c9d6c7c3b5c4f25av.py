#!/usr/bin/env python3
"""Freeze the explicit nonlinear 470-state reduced architecture revision."""

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

import run_causal_inner_coordinate_hessian_diagnosis_wp10c9d6c7c3b5c4f25au as parent  # noqa: E402
import run_causal_inner_stable_parametric_online_audit_wp10c9d6c7c3b5c4f25ai as stable  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25av"
CLASSIFICATION = (
    "explicit_nonlinear_470_state_conservative_IMEX_architecture_manifest_"
    "frozen_no_equilibrium_branch_assumption"
)
PARENT_COMMIT = "a69608a8bc43dfcd286ef6594f997cb060b76a5a"
PARENT_PARENT = "9ff35063ed18411dc660f0453ddac5e175ec5aaf"
PARENT_TREE = "30f465fb28d47086be4da9b28df0f4c9eceaa02c"

ARTIFACT = (
    "causal_inner_explicit_nonlinear_470_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25av"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_explicit_nonlinear_470_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25av.py"
)
THIS_TEST = (
    "tests/test_causal_inner_explicit_nonlinear_470_architecture_manifest_"
    "wp10c9d6c7c3b5c4f25av.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_explicit_nonlinear_470_architecture_audit_"
    "wp10c9d6c7c3b5c4f25aw.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_explicit_nonlinear_470_architecture_audit_"
    "wp10c9d6c7c3b5c4f25aw.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXPLICIT_NONLINEAR_470_"
    "ARCHITECTURE_MANIFEST_WP10C9D6C7C3B5C4F25AV_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PHYSICAL_DIMENSION = 162
STABLE_MEMORY_DIMENSION = 280
NONLINEAR_DEPARTURE_DIMENSION = 28
ONLINE_DIMENSION = 470
TRUNCATED_STABLE_DIMENSION = 90
FULL_DIMENSION = 560
FIDUCIAL_CYCLE_SECONDS = 6.7 * 86_400.0
WALL_BUDGET_SECONDS = 3.0 * 86_400.0
MAXIMUM_MACROSTEPS = 100_000


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


def _validate_parents() -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("470-architecture parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("470-architecture parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("470-architecture parent tree changed")
    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    parent_summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    stable_hashes = _checksums(stable.CANONICAL_DIRECTORY)
    stable_summary = _read(stable.CANONICAL_DIRECTORY / "summary.json")
    if (
        parent_summary["passed"]
        or parent_summary["physical_conditional_branch_found"]
        or parent_summary["classification"]
        != "coordinate_hessian_recovery_failed_branch_solver_architecture_requires_revision"
    ):
        raise RuntimeError("complete-KKT architecture revision trigger changed")
    if (
        not stable_summary["passed"]
        or stable_summary["total_architecture_dimension"] != ONLINE_DIMENSION
        or stable_summary["unstable_bundle_dimension"]
        != NONLINEAR_DEPARTURE_DIMENSION
        or stable_summary["unstable_bundle_linear_macro_propagation_authorized"]
    ):
        raise RuntimeError("inherited 470-state certificate changed")
    return {
        "parent_summary": parent_summary,
        "parent_hashes": parent_hashes,
        "stable_summary": stable_summary,
        "stable_hashes": stable_hashes,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "architecture": "explicit_nonlinear_conservative_IMEX_470",
        "state_partition": {
            "exact_physical_coordinates": PHYSICAL_DIMENSION,
            "strictly_stable_memory_coordinates": STABLE_MEMORY_DIMENSION,
            "explicit_nonlinear_departure_coordinates": NONLINEAR_DEPARTURE_DIMENSION,
            "online_continuous_dimension": ONLINE_DIMENSION,
            "balanced_truncated_stable_remainder": TRUNCATED_STABLE_DIMENSION,
            "full_dimension": FULL_DIMENSION,
            "identities": ["470_equals_162_plus_280_plus_28", "560_equals_470_plus_90"],
        },
        "why_the_branch_architecture_is_superseded": {
            "retained_memory_was_incorrectly_forced_to_stationarity": True,
            "complete_coordinate_curvature_was_required": True,
            "complete_KKT_homotopy_was_locally_ill_conditioned": True,
            "physical_branch_nonexistence_was_proved": False,
            "equilibrium_branches_are_required_for_reduced_evolution": False,
        },
        "coordinates": {
            "physical": (
                "160_exact_integrated_mapped_storage_coordinates_plus_"
                "2_explicit_stable_coordinates"
            ),
            "stable_memory": (
                "280_primary_Hermite_hidden_truth_trials_projected_into_"
                "the_kernel_of_the_exact_physical_coordinate_derivative"
            ),
            "departure": (
                "28_exact_positive_growth_fiber_directions_projected_"
                "against_the_442_physical_memory_coordinates"
            ),
            "finite_amplitude_unstable_chart": (
                "geometric_Newton_retraction_on_exact_C_phys_not_the_rate_reaction_lift"
            ),
        },
        "online_dynamics": {
            "physical_96_M_J_E": (
                "exact_finite_volume_flux_divergence_plus_physical_sources"
            ),
            "constitutive_and_stable_memory_346": (
                "inherited_energy_stable_parametric_descriptor_IMEX_or_exponential_update"
            ),
            "departure_28": (
                "explicitly_retained_nonlinear_reduced_vector_field_with_"
                "physical_guarded_offline_identification"
            ),
            "truncated_stable_90": (
                "inherited_balanced_Hermite_residualization_and_closure_error_budget"
            ),
            "truth_calls_per_online_macrostep": 0,
            "discrete_branch_label_required": False,
            "branch_events_may_be_derived_later_from_the_nonlinear_28D_dynamics": True,
        },
        "time_integrator": {
            "method": (
                "conservative_finite_volume_IMEX_with_L_stable_or_exponential_"
                "stable_memory_update_and_adaptive_nonlinear_departure_update"
            ),
            "step_controller": (
                "resolved_flux_error_departure_error_and_physical_guard_margin"
            ),
            "fast_full_order_microsteps_per_macrostep": 0,
        },
        "binding_structural_audit_gates": {
            "online_coordinate_rank_equal": ONLINE_DIMENSION,
            "online_coordinate_condition_number_max": 1.0e4,
            "stable_memory_projected_rank_equal": STABLE_MEMORY_DIMENSION,
            "stable_memory_projected_condition_number_max": 1.0e4,
            "departure_projected_rank_equal": NONLINEAR_DEPARTURE_DIMENSION,
            "hidden_remainder_dimension_equal": TRUNCATED_STABLE_DIMENSION,
            "Q3_rowspace_relative_defect_max": 1.0e-10,
            "anchor_hidden_rate_relative_fraction_max": 0.05,
            "inherited_stable_energy_amplification_max": 1.0,
            "projected_online_cycle_wall_seconds_max": 0.5 * WALL_BUDGET_SECONDS,
            "online_truth_calls_equal": 0,
        },
        "runtime_contract": {
            "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
            "wall_budget_seconds": WALL_BUDGET_SECONDS,
            "maximum_macrosteps": MAXIMUM_MACROSTEPS,
            "offline_database_cost_is_not_online_cycle_cost": True,
        },
        "decision": {
            "pass": (
                "explicit_nonlinear_470_architecture_structurally_certified_"
                "exact_geometric_departure_chart_manifest_authorized"
            ),
            "fail": (
                "explicit_nonlinear_470_architecture_structural_audit_failed_"
                "reduced_evolution_blocked"
            ),
            "pass_authorizes_only": (
                "definitions_only_exact_geometric_28D_departure_chart_and_database_manifest"
            ),
        },
        "claim_boundary": {
            "nonlinear_28D_closure_identified": False,
            "online_integrator_implemented": False,
            "physical_cycle_validated": False,
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
    parents = _validate_parents()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("470-architecture manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("470-architecture manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "selected_architecture": "explicit_nonlinear_conservative_IMEX_470",
        "online_dimension": ONLINE_DIMENSION,
        "hidden_stable_remainder_dimension": TRUNCATED_STABLE_DIMENSION,
        "equilibrium_branch_required": False,
        "nonlinear_closure_identified": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25aw",
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
            "KKT_diagnosis_package_hashes": parents["parent_hashes"],
            "stable_470_package_hashes": parents["stable_hashes"],
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
            "thread_environment": parent.manifest.parent.preflight.THREAD_ENVIRONMENT,
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
                "# Explicit nonlinear 470-state architecture manifest WP10c9d6c7c3b5c4f25av",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The reduced state is (c, eta, z, a): 162 exact physical coordinates, 280 certified stable-memory coordinates, and 28 explicit nonlinear departure coordinates. Only the remaining 90 stable directions are eliminated by the inherited balanced/Hermite closure.",
                "",
                "This restores the already certified 470-state algebraic architecture but replaces the failed reaction-lift chart with an exact C_phys geometric retraction. Equilibrium branches and discrete branch labels are not assumed; they may be derived later only if the explicit nonlinear dynamics supports them.",
                "",
                "A zero-new-truth structural audit is authorized. Nonlinear closure fitting, an online integrator, and a predictive cycle remain unauthorized.",
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
