#!/usr/bin/env python3
"""Freeze the dual-consistent single-primary hidden-fast root execution."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_exact_geometric_470_chart_derivative_recovery_wp10c9d6c7c3b5c4f25de2 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25df"
PARENT_COMMIT = "d831aa88d7b47556c8380f181f11719295a80b78"
PARENT_PARENT = "5a01f9de30d3fd47f7e7cf6454dbb34b88e099e2"
PARENT_TREE = "946bfad75be5a59a3438b593020b02e628304e46"

CLASSIFICATION = (
    "dual_consistent_primary_hidden_fast_root_manifest_frozen_"
    "anchor_preflight_first"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dg"

PHYSICAL_DIMENSION = 560
COORDINATE_DIMENSION = 470
MACRO_DIMENSION = 82
HIDDEN_DIMENSION = 388
PRIMARY_INDEX = 5
SEALED_INDEX = 4
DUAL_GEOMETRY_GATE = 5.0e-12
MATERIAL_NAIVE_COUPLING_MIN = 1.0e-3

ARTIFACT = (
    "causal_inner_primary_hidden_fast_root_manifest_"
    "wp10c9d6c7c3b5c4f25df"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_primary_hidden_fast_root_manifest_"
    "wp10c9d6c7c3b5c4f25df.py"
)
THIS_TEST = (
    "tests/test_causal_inner_primary_hidden_fast_root_manifest_"
    "wp10c9d6c7c3b5c4f25df.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PRIMARY_HIDDEN_FAST_ROOT_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DF_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FIBER_DIRECTORY = parent.manifest.parent.manifest.CANONICAL_DIRECTORY
CHART_DIRECTORY = parent.manifest.parent.CANONICAL_DIRECTORY


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("primary hidden-root parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("primary hidden-root parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("primary hidden-root parent tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(parent.CANONICAL_DIRECTORY / "summary.json")
    payload = _read(
        parent.CANONICAL_DIRECTORY / "derivative_recovery_metrics.json"
    )
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["classification"] != parent.PASS_CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["primary_hidden_root_manifest_authorized"]
        or summary["branch_root_execution_authorized"]
        or summary["sealed_16ms_opened"]
        or not all(payload["checks"].values())
        or payload["metrics"]["new_exact_fixed_Q_rate_evaluations"] != 0
        or payload["metrics"]["new_intrinsic_hidden_roots"] != 0
    ):
        raise RuntimeError("derivative-recovery certificate changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"derivative-recovery source changed: {relative}")
    fiber_hashes = _checksums(FIBER_DIRECTORY)
    chart_hashes = _checksums(CHART_DIRECTORY)
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("primary hidden-root manifest requires a clean tracked tree")
    return {
        "parent_hashes": hashes,
        "fiber_hashes": fiber_hashes,
        "chart_hashes": chart_hashes,
    }


def _dual_geometry() -> tuple[dict[str, np.ndarray], dict]:
    fiber = _load_npz(FIBER_DIRECTORY / "fiber_geometry.npz")
    chart = _load_npz(CHART_DIRECTORY / "exact_chart_arrays.npz")
    restriction = np.asarray(fiber["macro_restriction_R82"], dtype=float)
    lifting = np.asarray(fiber["macro_lifting_L82"], dtype=float)
    hidden = np.asarray(fiber["hidden_orthonormal_basis_Z388"], dtype=float)
    identity = np.eye(COORDINATE_DIMENSION)
    projection = identity - lifting @ restriction
    hidden_dual = hidden.T @ projection
    anchor_coordinate = np.asarray(chart["anchor_coordinate_y470"], dtype=float)
    anchor_macro = restriction @ anchor_coordinate
    anchor_hidden = hidden_dual @ anchor_coordinate
    reconstructed = lifting @ anchor_macro + hidden @ anchor_hidden
    metrics = {
        "restriction_lifting_identity_infinity": float(
            np.linalg.norm(
                restriction @ lifting - np.eye(MACRO_DIMENSION), ord=np.inf
            )
        ),
        "restriction_hidden_annihilation_infinity": float(
            np.linalg.norm(restriction @ hidden, ord=np.inf)
        ),
        "hidden_orthonormality_infinity": float(
            np.linalg.norm(hidden.T @ hidden - np.eye(HIDDEN_DIMENSION), ord=np.inf)
        ),
        "fiber_projection_idempotence_infinity": float(
            np.linalg.norm(projection @ projection - projection, ord=np.inf)
        ),
        "fiber_projection_restriction_infinity": float(
            np.linalg.norm(restriction @ projection, ord=np.inf)
        ),
        "fiber_projection_hidden_identity_infinity": float(
            np.linalg.norm(projection @ hidden - hidden, ord=np.inf)
        ),
        "macro_hidden_decomposition_identity_infinity": float(
            np.linalg.norm(
                lifting @ restriction + hidden @ hidden_dual - identity,
                ord=np.inf,
            )
        ),
        "hidden_dual_macro_annihilation_infinity": float(
            np.linalg.norm(hidden_dual @ lifting, ord=np.inf)
        ),
        "hidden_dual_hidden_identity_infinity": float(
            np.linalg.norm(
                hidden_dual @ hidden - np.eye(HIDDEN_DIMENSION), ord=np.inf
            )
        ),
        "naive_hidden_projection_macro_coupling_infinity": float(
            np.linalg.norm(hidden.T @ lifting, ord=np.inf)
        ),
        "anchor_coordinate_reconstruction_relative_defect": float(
            np.linalg.norm(reconstructed - anchor_coordinate)
            / max(np.linalg.norm(anchor_coordinate), np.finfo(float).tiny)
        ),
        "naive_Z_transpose_F_is_the_hidden_coordinate_derivative": False,
        "correct_hidden_dual": "Qz_equals_Z_transpose_times_I_minus_LR",
    }
    arrays = {
        "macro_restriction_R82": restriction,
        "macro_lifting_L82": lifting,
        "hidden_basis_Z388": hidden,
        "fiber_projection_P470": projection,
        "hidden_dual_Q388": hidden_dual,
        "anchor_macro_X82": anchor_macro,
        "anchor_hidden_z388": anchor_hidden,
        "anchor_coordinate_y470": anchor_coordinate,
    }
    return arrays, metrics


def _geometry_checks(metrics: dict) -> dict[str, bool]:
    return {
        "restriction_lifting": metrics[
            "restriction_lifting_identity_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "restriction_hidden": metrics[
            "restriction_hidden_annihilation_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "hidden_orthonormal": metrics["hidden_orthonormality_infinity"]
        <= DUAL_GEOMETRY_GATE,
        "projection_idempotent": metrics[
            "fiber_projection_idempotence_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "projection_in_kernel": metrics[
            "fiber_projection_restriction_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "projection_preserves_hidden": metrics[
            "fiber_projection_hidden_identity_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "exact_decomposition": metrics[
            "macro_hidden_decomposition_identity_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "hidden_dual_annihilates_macro": metrics[
            "hidden_dual_macro_annihilation_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "hidden_dual_recovers_hidden": metrics[
            "hidden_dual_hidden_identity_infinity"
        ]
        <= DUAL_GEOMETRY_GATE,
        "anchor_reconstruction": metrics[
            "anchor_coordinate_reconstruction_relative_defect"
        ]
        <= DUAL_GEOMETRY_GATE,
        "naive_projection_materially_coupled": metrics[
            "naive_hidden_projection_macro_coupling_infinity"
        ]
        >= MATERIAL_NAIVE_COUPLING_MIN,
        "naive_projection_rejected": not metrics[
            "naive_Z_transpose_F_is_the_hidden_coordinate_derivative"
        ],
    }


def _contract(metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "mathematical_architecture": {
            "physical_state": "x_in_R560",
            "exact_coordinate": "y=C470(x)_in_R470",
            "macro_coordinate": "X=R82_y",
            "fiber_projection": "P=I470-L82_R82",
            "hidden_coordinate": "z=Qz_y_with_Qz=Z388_transpose_P",
            "exact_coordinate_decomposition": "y=L82_X+Z388_z",
            "coordinate_rate": "F(y)=DC470_at_chi(y)_times_fQ(chi(y))",
            "macro_rate": "G(X,z)=R82_F(y)",
            "hidden_rate": "H(X,z)=Qz_F(y)",
            "exact_rate_decomposition": "F=L82_G+Z388_H",
            "frozen_macro_critical_root": "H(Xstar,zstar)=0",
            "naive_residual_Z_transpose_F_forbidden": True,
            "reason": (
                "Z_transpose_L_is_nonzero_so_Z_transpose_F_mixes_macro_rate_"
                "into_the_hidden_coordinate_derivative"
            ),
            "measured_Z_transpose_L_infinity": metrics[
                "naive_hidden_projection_macro_coupling_infinity"
            ],
        },
        "complete_hidden_tangent": {
            "chart_lift": (
                "Dchi_y_solves_stack_DC470_and_N90_transpose_times_dx_"
                "equals_stack_dy_and_zero90"
            ),
            "coordinate_field_tangent": (
                "DF_y_dy=D2C470_x[Dchi_y_dy,fQ_x]+DC470_x_"
                "DfQ_x_Dchi_y_dy"
            ),
            "hidden_Jacobian": "Azz=Qz_DF_y_Z388",
            "macro_blocks": {
                "Axx": "R82_DF_y_L82",
                "Axz": "R82_DF_y_Z388",
                "Azx": "Qz_DF_y_L82",
                "Azz": "Qz_DF_y_Z388",
            },
            "coordinate_Hessian_term_required": True,
            "physical_reaction_derivative_required": True,
            "complete_tangent_independent_JVP_audit_required": True,
        },
        "prospective_execution": {
            "work_package": AUTHORIZED_NEXT,
            "candidate": "accepted_20ms_unclassified_primary",
            "sealed_candidate": "accepted_16ms_unclassified_sealed",
            "stage_order": [
                "load_hash_locked_anchor_chart_and_dual_geometry",
                "evaluate_one_fresh_exact_fixed_Q_rate_at_the_exact_20ms_anchor",
                "compute_F0_G0_H0_and_apply_initial_hidden_fraction_gate",
                "stop_without_generator_or_root_if_hidden_fraction_exceeds_0p25",
                "assemble_complete_coordinate_and_hidden_tangent_if_preflight_passes",
                "audit_complete_tangent_and_linear_predictor",
                "solve_one_fixed_macro_hidden_root_with_dense_Broyden_updates",
                "audit_physical_root_fast_stability_gap_and_critical_manifold_invariance",
            ],
            "budgets": {
                "new_exact_fixed_Q_rate_evaluations_max": 12,
                "new_complete_physical_generator_assemblies_max": 2,
                "new_intrinsic_hidden_roots_max": 1,
                "new_coordinate_chart_retractions_max": 12,
                "propagated_states_equal": 0,
                "sealed_16ms_truth_calls_equal": 0,
            },
            "anchor_truth_policy": {
                "fresh_anchor_rate_required": True,
                "saved_rate_or_generator_reuse_requires_bitwise_anchor_state_match": True,
                "mismatched_continuation_endpoint_descriptor_reuse_forbidden": True,
                "certified_reaction_modes_only": ["raw", "frozen_normalized"],
            },
            "root_solver": {
                "unknown": "z_in_R388_at_fixed_Xstar",
                "binding_residual": "H=Qz_F",
                "residual_scale": "RMS_norm_of_anchor_coordinate_rate_F0",
                "initial_exact_hidden_Jacobian_assemblies": 1,
                "optional_exact_refreshes_after_complete_line_failure_max": 1,
                "updates": "dense_good_Broyden_secant_updates",
                "line_search_factors": [1.0, 0.5, 0.25, 0.125],
                "no_post_candidate_projection": True,
                "accepted_exact_chart_states_only": True,
            },
        },
        "binding_gates": {
            "dual_geometry_all_defects_max": DUAL_GEOMETRY_GATE,
            "initial_hidden_coordinate_rate_fraction_max": 0.25,
            "complete_coordinate_tangent_JVP_relative_defect_max": 1.0e-6,
            "equilibrated_hidden_Jacobian_condition_number_max": 1.0e8,
            "linear_predictor_relative_residual_max": 1.0e-10,
            "linear_predictor_maximum_scaled_physical_component_max": 0.015,
            "normalized_hidden_residual_infinity_max": 1.0e-10,
            "exact_coordinate_closure_infinity_max": 1.0e-10,
            "exact_gauge_closure_infinity_max": 1.0e-10,
            "fixed_Q_relative_defect_max": 1.0e-12,
            "reaction_and_constraint_action_relative_defect_max": 1.0e-12,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "surrogate_hidden_rate_relative_defect_max": 0.10,
            "critical_manifold_invariance_relative_defect_max": 0.10,
            "fast_spectral_abscissa_max_per_second": 0.0,
            "fast_to_effective_slow_spectral_gap_ratio_min": 10.0,
        },
        "post_root_mathematics": {
            "critical_graph_derivative": "Dh=-Azz_inverse_Azx",
            "effective_slow_Jacobian": "Aslow=Axx-Axz_Azz_inverse_Azx",
            "critical_manifold_invariance_defect": (
                "norm_Z388_Dh_G_over_max_norm_F_and_tiny"
            ),
            "fast_stability": "maximum_real_eigenvalue_of_Azz",
            "spectral_gap": (
                "minimum_absolute_fast_real_decay_over_maximum_"
                "absolute_real_effective_slow_eigenvalue"
            ),
            "root_is_not_automatically_an_invariant_slow_manifold": True,
        },
        "decision": {
            "anchor_hidden_fraction_fail": {
                "classification": (
                    "primary_anchor_not_near_frozen_macro_critical_manifold_"
                    "root_not_attempted"
                ),
                "authorizes_only": (
                    "definitions_only_transition_or_macro_state_revision_manifest"
                ),
            },
            "tangent_or_root_fail": {
                "classification": "primary_hidden_root_not_established",
                "authorizes_only": "diagnosis_only",
            },
            "converged_unstable": {
                "classification": (
                    "primary_stationary_transition_or_fold_marker_not_slow_branch"
                ),
                "authorizes_only": "definitions_only_transition_capture_manifest",
            },
            "stable_but_invariance_or_gap_fail": {
                "classification": (
                    "stable_frozen_macro_critical_root_not_sufficient_for_"
                    "memoryless_slow_closure"
                ),
                "authorizes_only": (
                    "definitions_only_first_order_invariant_graph_or_memory_manifest"
                ),
            },
            "full_pass": {
                "classification": (
                    "primary_local_stable_critical_branch_seed_supported_unclassified"
                ),
                "authorizes_only": (
                    "definitions_only_one_direction_pseudo_arclength_"
                    "continuation_manifest"
                ),
            },
        },
        "authorization_boundaries": {
            "this_package_definitions_only": True,
            "new_truth_in_this_package": False,
            "branch_root_in_this_package": False,
            "next_execution_is_single_primary_only": True,
            "sealed_16ms_opened": False,
            "physical_microburst_authorized": False,
            "online_solver_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "derivative_certificate_summary": _sha(
                parent.CANONICAL_DIRECTORY / "summary.json"
            ),
            "derivative_certificate_metrics": _sha(
                parent.CANONICAL_DIRECTORY / "derivative_recovery_metrics.json"
            ),
            "exact_chart_arrays": _sha(CHART_DIRECTORY / "exact_chart_arrays.npz"),
            "fiber_geometry": _sha(FIBER_DIRECTORY / "fiber_geometry.npz"),
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
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
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
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
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
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("primary hidden-root manifest already exists")
    arrays, metrics = _dual_geometry()
    checks = _geometry_checks(metrics)
    if not all(checks.values()):
        raise RuntimeError(f"dual hidden geometry failed: {checks}")
    contract = _contract(metrics)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_npz(CANONICAL_DIRECTORY / "dual_hidden_geometry.npz", arrays)
    _write_json(
        CANONICAL_DIRECTORY / "dual_hidden_geometry_metrics.json",
        {"metrics": metrics, "checks": checks, "passed": True},
    )
    _write_json(CANONICAL_DIRECTORY / "primary_hidden_root_contract.json", contract)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            **frozen,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "dual_consistent_hidden_residual_frozen": True,
        "naive_Z_transpose_F_residual_rejected": True,
        "maximum_dual_geometry_defect": max(
            value
            for name, value in metrics.items()
            if isinstance(value, float)
            and name != "naive_hidden_projection_macro_coupling_infinity"
        ),
        "naive_hidden_projection_macro_coupling_infinity": metrics[
            "naive_hidden_projection_macro_coupling_infinity"
        ],
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "branch_root_in_this_package": False,
        "single_primary_root_execution_authorized_next": True,
        "physical_microburst_authorized": False,
        "online_solver_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in parent.manifest.parent.manifest.parent.field_manifest.training._thread_environment()
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Primary hidden-fast root manifest WP10c9d6c7c3b5c4f25df",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The exact 470-coordinate chart and its scale-aware implicit derivative are certified prospectively. Before authorizing a root, this package corrects the hidden-coordinate dual required by the non-orthogonal conservative lifting.",
                "",
                f"The frozen lifting has ||Z^T L||_inf = `{metrics['naive_hidden_projection_macro_coupling_infinity']:.6e}`. Therefore Z^T F is not dz/dt. The binding hidden rate is H=Z^T(I-LR)F, for which the exact coordinate decomposition closes at `{metrics['macro_hidden_decomposition_identity_infinity']:.6e}`.",
                "",
                "The next execution must first evaluate one exact fixed-Q rate at the bitwise 20 ms anchor and apply the frozen hidden-rate-fraction gate. If that gate fails, no generator or root may run. If it passes, one 388-dimensional fixed-macro root may use a complete coordinate tangent including both the physical reaction derivative and coordinate-Hessian term.",
                "",
                "A root is only a frozen-macro critical point. Fast stability, an effective slow Schur complement, a spectral gap, and the critical-manifold invariance defect remain independently binding before any slow-graph claim.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`. No microburst, online solver, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
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
