#!/usr/bin/env python3
"""Freeze the nonlinear unstable-bundle architecture-selection screen."""

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

import run_causal_inner_stable_parametric_online_audit_wp10c9d6c7c3b5c4f25ai as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25aj"
CLASSIFICATION = (
    "nonlinear_unstable_bundle_database_manifest_frozen_"
    "finite_amplitude_architecture_selection_authorized"
)
PARENT_COMMIT = "ae5a7fbb36d472dc19648c2e10afbf253d3308a4"
PARENT_PARENT = "b5bca0cfd458d8aa2982828dc49a08ddd060b4e3"
PARENT_TREE = "6f45ac716890596e0e8bc1db42876b8278a88b79"

PARENT_DIRECTORY = parent.CANONICAL_DIRECTORY
FIBER_DIRECTORY = parent.manifest.FIBER_DIRECTORY
PRIMARY_GENERATOR_DIRECTORY = parent.manifest.PARENT_DIRECTORY
ARTIFACT = (
    "causal_inner_nonlinear_bundle_database_manifest_"
    "wp10c9d6c7c3b5c4f25aj"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_bundle_database_manifest_"
    "wp10c9d6c7c3b5c4f25aj.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_bundle_database_manifest_"
    "wp10c9d6c7c3b5c4f25aj.py"
)
NEXT_RUNNER = (
    "scripts/run_causal_inner_nonlinear_bundle_screen_"
    "wp10c9d6c7c3b5c4f25ak.py"
)
NEXT_TEST = (
    "tests/test_causal_inner_nonlinear_bundle_screen_"
    "wp10c9d6c7c3b5c4f25ak.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_BUNDLE_DATABASE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25AJ_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

ANCHORS = ("primary", "heldout")
ENERGY_DIRECTIONS = 8
MAXIMUM_COMPONENT_AMPLITUDES = (2.5e-4, 1.0e-3, 5.0e-3)
CONSERVATIVE_STORAGE_DIMENSION = 96
CONSTITUTIVE_STORAGE_DIMENSION = 64
EXPLICIT_STABLE_DIMENSION = 2
HIDDEN_STABLE_DIMENSION = 280
UNSTABLE_DIMENSION = 28


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
        raise RuntimeError("nonlinear-bundle parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("nonlinear-bundle parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("nonlinear-bundle parent tree changed")
    hashes = _checksums(PARENT_DIRECTORY)
    summary = _read(PARENT_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["stable_descriptor_dimension"] != 442
        or summary["unstable_bundle_dimension"] != UNSTABLE_DIMENSION
        or summary["authorized_next"]
        != "definitions_only_nonlinear_unstable_bundle_offline_database_manifest"
        or summary["predictive_cycle_authorized"]
    ):
        raise RuntimeError("stable-parametric certificate changed")
    _checksums(FIBER_DIRECTORY)
    return summary, hashes


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "parent_decisive_hashes": {
            name: _sha(PARENT_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "parametric_diagnostics.npz")
        },
        "fiber_decisive_hashes": {
            name: _sha(FIBER_DIRECTORY / name)
            for name in ("summary.json", "metrics.json", "decisive_fibers.npz")
        },
        "corrected_state_partition": {
            "coarse_true_conservative_M_J_E_storage": CONSERVATIVE_STORAGE_DIMENSION,
            "coarse_constitutive_storage": CONSTITUTIVE_STORAGE_DIMENSION,
            "explicit_stable_coordinates": EXPLICIT_STABLE_DIMENSION,
            "stable_hidden_memory": HIDDEN_STABLE_DIMENSION,
            "exact_unstable_bundle": UNSTABLE_DIMENSION,
            "total_local_dimension": 470,
            "note": "the inherited 162-coordinate block is 160 five-field storage coordinates plus two stable coordinates; only 96 storage coordinates are M/J/E conservative",
        },
        "nonlinear_fixed_Q_field": {
            "free_rate": "f_free_x_equals_minus_M_x_inverse_R_x",
            "constrained_rate": "F_Q_x_equals_f_free_x_minus_B_x_DQ_x_f_free_x",
            "identity": "DQ_x_F_Q_x_equals_zero",
            "finite_amplitude_chart": "x_of_a_equals_retract_to_base_Q_of_x0_plus_scaled_V_a_using_base_reaction_lift",
            "no_new_exact_generator_assembly": True,
            "no_nonlinear_root": True,
            "no_state_propagation": True,
        },
        "prospective_screen": {
            "anchors": list(ANCHORS),
            "directions": "eight_largest_eigenvectors_of_symmetric_part_of_exact_28_by_28_unstable_operator",
            "direction_count_per_anchor": ENERGY_DIRECTIONS,
            "signs": [-1, 1],
            "maximum_scaled_component_amplitudes": list(MAXIMUM_COMPONENT_AMPLITUDES),
            "total_nonbase_rate_evaluations": (
                len(ANCHORS)
                * ENERGY_DIRECTIONS
                * 2
                * len(MAXIMUM_COMPONENT_AMPLITUDES)
            ),
            "central_odd_radial_growth": "gamma_r_equals_d_transpose_g_plus_minus_g_minus_over_two_r",
            "linear_reference": "gamma_0_equals_d_transpose_U_d",
            "base_rate_is_subtracted_before_projection": True,
            "heldout_results_may_not_change_directions_or_amplitudes": True,
        },
        "binding_evaluator_gates": {
            "maximum_normalized_Q3_retraction_defect": 1.0e-10,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_reconstruction_factor": 1.0 + 1.0e-12,
            "maximum_raw_Schur_condition_number": 1.0e8,
            "maximum_reaction_identity_defect": 1.0e-10,
            "maximum_rate_tangency_relative_defect": 1.0e-10,
            "maximum_smallest_amplitude_linear_growth_relative_defect": 0.25,
            "maximum_H_over_R": 0.12,
            "minimum_scattering_optical_depth": 1.0,
            "maximum_scaled_component_perturbation": 5.0e-3,
        },
        "architecture_selection": {
            "local_saturation_requires": {
                "nonpositive_largest_amplitude_radial_growth_directions_per_anchor_min": 6,
                "largest_amplitude_direction_count": ENERGY_DIRECTIONS,
                "negative_fitted_cubic_directions_per_anchor_min": 6,
                "interpretation": "energy_bounded_cubic_or_port_Hamiltonian_normal_form_may_be_identified_offline",
            },
            "otherwise": {
                "selection": "conservative_hybrid_branch_and_event_map",
                "reason": "no_trustworthy_local_saturation_inside_the_certified_primitive_trust_region",
                "online_unstable_state": "eliminated_by_event_localization_and_conservative_reset",
            },
            "evaluator_failure": "stop_without_architecture_selection",
        },
        "hybrid_database_requirements": {
            "continuous_state": "c96_conservative_plus_eta66_constitutive_plus_z280_memory",
            "discrete_state": "branch_label_cold_hot_or_transition",
            "branch_construction": "fixed_invariant_pseudo_arclength_equilibrium_continuation",
            "transition_construction": "adaptive_orthogonal_collocation_boundary_value_solve_in_rescaled_fast_time",
            "event_surfaces": "separately_fitted_up_and_down_loss_of_stability_or_basin_boundaries",
            "reset_map": "global_Q3_preserving_map_for_c_eta_z_plus_integrated_boundary_flux_impulse",
            "online_truth_calls": 0,
            "heldout_branches_and_transitions_required": True,
        },
        "claim_boundary": {
            "branch_existence_assumed": False,
            "nonlinear_coefficients_identified": False,
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
        raise RuntimeError("nonlinear-bundle manifest requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("nonlinear-bundle manifest is already frozen")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "anchor_count": len(ANCHORS),
        "energy_direction_count_per_anchor": ENERGY_DIRECTIONS,
        "amplitude_count": len(MAXIMUM_COMPONENT_AMPLITUDES),
        "planned_nonbase_rate_evaluations": (
            len(ANCHORS)
            * ENERGY_DIRECTIONS
            * 2
            * len(MAXIMUM_COMPONENT_AMPLITUDES)
        ),
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": "WP10c9d6c7c3b5c4f25ak",
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
            "fiber_package_hashes": _checksums(FIBER_DIRECTORY),
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
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in parent.THREAD_ENVIRONMENT
            },
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
                "# Nonlinear unstable-bundle database manifest WP10c9d6c7c3b5c4f25aj",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "This definitions-only package corrects the state semantics: the 162-coordinate retained block contains 96 truly conservative M/J/E storage coordinates, 64 constitutive storage coordinates, and two explicit stable coordinates. It freezes a two-anchor finite-amplitude screen of the exact 28-dimensional positive-growth fiber.",
                "",
                "The screen will decide prospectively between an energy-bounded nonlinear normal form and a conservative hybrid branch/event map. It may evaluate the exact continuous fixed-Q vector field but may assemble no new full generator, solve no nonlinear root, and propagate no trajectory.",
                "",
                "Branch existence is not assumed. No online integrator, predictive cycle, or reduced slow evolution is authorized.",
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
