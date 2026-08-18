#!/usr/bin/env python3
"""Freeze an intrinsic fixed-Q coordinate-geometry audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_bundle_screen_wp10c9d6c7c3b5c4f25ak as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25al"
CLASSIFICATION = (
    "intrinsic_constraint_geometry_manifest_frozen_"
    "orthogonal_chart_and_equilibrium_centered_architecture_audit_authorized"
)
PARENT_COMMIT = "a92c802ed01cb052cf13bde44332a5a87c8e263d"
PARENT_PARENT = "d9d23d644b1f3c068852507fe297403d261da262"
PARENT_TREE = "cc8c0537b1ef40228d386d448cd11c4900ba7f3c"
PARENT_DIRECTORY = parent.CANONICAL_DIRECTORY

ARTIFACT = (
    "causal_inner_intrinsic_constraint_geometry_manifest_"
    "wp10c9d6c7c3b5c4f25al"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_intrinsic_constraint_geometry_manifest_"
    "wp10c9d6c7c3b5c4f25al.py"
)
THIS_TEST = (
    "tests/test_causal_inner_intrinsic_constraint_geometry_manifest_"
    "wp10c9d6c7c3b5c4f25al.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_intrinsic_constraint_geometry_audit_"
    "wp10c9d6c7c3b5c4f25am.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_intrinsic_constraint_geometry_audit_"
    "wp10c9d6c7c3b5c4f25am.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_INTRINSIC_CONSTRAINT_GEOMETRY_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AL_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

ANCHORS = ("primary", "heldout")
PROJECTED_FIBER_DIRECTIONS = 8
MAXIMUM_COMPONENT_AMPLITUDE = 5.0e-3


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


def _validate_parent() -> tuple[dict, dict[str, str]]:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("intrinsic-geometry parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("intrinsic-geometry parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("intrinsic-geometry parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    metrics = _read(PARENT_DIRECTORY / "metrics.json")
    failure = metrics["fail_fast_coordinate_retraction"]
    if (
        summary["passed"]
        or summary["classification"] != parent.FAIL_CLASSIFICATION
        or failure["failure_kind"]
        != "rate_reaction_is_not_a_geometric_retraction"
        or metrics["completed_nonbase_continuous_rate_evaluations"] != 0
    ):
        raise RuntimeError("reaction-chart rejection changed")
    return summary, hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "preserved_rejection": parent.FAIL_CLASSIFICATION,
        "coordinate_geometry": {
            "constraint_rows": "Q_equals_DQ3_scaled_at_anchor",
            "minimum_norm_normal": "N_equals_Q_transpose_solve_Q_Q_transpose_I3",
            "orthogonal_tangent_projector": "P_equals_I_minus_N_Q",
            "orthonormal_tangent_basis": "Z_equals_null_space_Q",
            "state_retraction": "delta_next_equals_delta_minus_N_times_normalized_Q3_error",
            "physical_reaction_lift_used_for_state_retraction": False,
            "physical_reaction_lift_retained_for_rate_constraint": True,
        },
        "saved_generator_diagnostics": {
            "intrinsic_frozen_operator": "A_T_equals_Z_transpose_A_Z",
            "instantaneous_eigenvalues_are_normal_hyperbolicity_certificate": False,
            "reason": "both_committed_anchors_have_nonzero_base_rate_and_are_not_fast_equilibria",
            "old_28_fiber_tangent_projection": "W_equals_qr_P_V_old",
            "projected_operator": "A_W_equals_W_transpose_A_W",
            "new_full_generator_assemblies": 0,
        },
        "finite_amplitude_chart_audit": {
            "anchors": list(ANCHORS),
            "directions": "eight_largest_symmetric_growth_directions_of_A_W",
            "signs": [-1, 1],
            "maximum_scaled_component_amplitude": MAXIMUM_COMPONENT_AMPLITUDE,
            "nonlinear_rate_evaluations": 0,
            "nonlinear_roots": 0,
            "propagated_states": 0,
        },
        "binding_gates": {
            "constraint_rank_equal": 3,
            "constraint_gram_condition_number_max": 1.0e3,
            "normal_spectral_norm_max": 20.0,
            "projector_idempotence_defect_max": 1.0e-12,
            "projector_symmetry_defect_max": 1.0e-12,
            "tangent_annihilation_defect_max": 1.0e-12,
            "tangent_basis_orthogonality_defect_max": 1.0e-12,
            "projected_old_fiber_rank_equal": 28,
            "projected_old_fiber_condition_number_max": 10.0,
            "maximum_normalized_Q3_retraction_defect": 1.0e-10,
            "maximum_scaled_component_perturbation": MAXIMUM_COMPONENT_AMPLITUDE,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "incoming_excision_characteristics_equal": 0,
        },
        "decision": {
            "pass": "intrinsic_constraint_geometry_passed_equilibrium_centered_slow_fast_hybrid_manifest_authorized",
            "fail": "intrinsic_constraint_geometry_failed_reduced_architecture_blocked",
            "pass_authorizes_only": "definitions_only_constrained_equilibrium_branch_and_fast_transition_collocation_manifest",
        },
        "claim_boundary": {
            "old_28_modes_are_certified_physical_instabilities": False,
            "old_442_state_kernel_invalidated_as_transfer_approximation": False,
            "old_442_state_kernel_certified_as_equilibrium_centered_closure": False,
            "online_integrator_implementation_authorized": False,
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
    _, parent_hashes = _validate_parent()
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("intrinsic-geometry manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("intrinsic-geometry manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "preserved_parent_rejection": True,
        "anchor_count": len(ANCHORS),
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25am",
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
            "parent_package_hashes": parent_hashes,
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
            "thread_environment": parent.manifest.parent.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in ARTIFACT_DIRECTORY.iterdir()))
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Intrinsic constraint-geometry manifest WP10c9d6c7c3b5c4f25al",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The failed reaction-lift state chart is preserved. This definitions-only correction separates rate enforcement from state geometry: the physical ledger reaction remains the fixed-Q rate operator, while finite-amplitude state retraction uses the minimum-norm orthogonal normal of DQ3.",
                "",
                "Instantaneous eigenvalues at the two moving anchors will be reported but cannot define physical branches or normal hyperbolicity. A pass authorizes only an equilibrium-centered branch and transition-collocation architecture manifest.",
                "",
                "No root, propagation, nonlinear-rate sample, online solver, or predictive cycle is authorized.",
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
