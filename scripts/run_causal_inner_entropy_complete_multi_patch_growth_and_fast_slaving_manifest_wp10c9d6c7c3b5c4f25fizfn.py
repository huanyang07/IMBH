#!/usr/bin/env python3
"""Freeze transported atlas growth and stable-bundle slaving diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.linalg import subspace_angles


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_second_pathwise_macro_patch_execution_wp10c9d6c7c3b5c4f25fizfm as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (  # noqa: E402
    ExactAffineMacroSystem,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    ThermodynamicAffineMacroAtlas,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfn_"
    "entropy_complete_multi_patch_growth_and_fast_slaving_manifest"
)
CLASSIFICATION = (
    "entropy_complete_transported_patch_growth_and_stable_bundle_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfo_"
    "entropy_complete_transported_third_macro_patch_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_multi_patch_growth_and_fast_slaving_manifest_"
    "wp10c9d6c7c3b5c4f25fizfn"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_MULTI_PATCH_"
    "GROWTH_AND_FAST_SLAVING_MANIFEST_WP10C9D6C7C3B5C4F25FIZFN_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_multi_patch_growth_and_fast_"
    "slaving_manifest_wp10c9d6c7c3b5c4f25fizfn.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_multi_patch_growth_and_fast_"
    "slaving_manifest_wp10c9d6c7c3b5c4f25fizfn.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "13018336521ef5baf21bea804804c9a81d2f19c3b67171de54650b9b3dc85535"
)
PATCH_1_ARRAYS = parent.parent.parent.CANONICAL_DIRECTORY / "macro_integrator_arrays.npz"
PATCH_2_ARRAYS = parent.CANONICAL_DIRECTORY / "pathwise_patch_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
COMPLETE_CYCLE_SECONDS = 578_880.0


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("second pathwise patch checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "pathwise_patch_metrics.json"
    )
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["two_patch_path_certified"]
        or summary["accepted_absolute_horizon_seconds"] != 8.0e-3
        or not summary["multi_patch_growth_and_fast_slaving_manifest_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != (
            "definitions_only_WP10c9d6c7c3b5c4f25fizfn_"
            "entropy_complete_multi_patch_growth_and_fast_slaving_manifest"
        )
        or metrics["new_truth_operator_calls"] != 39
        or not metrics["all_truth_physical_gates_passed"]
        or metrics["endpoint_maximum_macro_rate_relative_defect"] > 5.0e-2
    ):
        raise RuntimeError("transported patch authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"second patch source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transported patch manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _atlas(values: dict[str, np.ndarray], prefix: str) -> ThermodynamicAffineMacroAtlas:
    return ThermodynamicAffineMacroAtlas(
        anchor_macro_state=values[f"{prefix}anchor_macro_state"],
        macro_coordinate_scales=values[f"{prefix}macro_coordinate_scales"],
        base_normalized_output=values[f"{prefix}base_normalized_output"],
        normalized_output_jacobian=values[
            f"{prefix}normalized_output_chart_jacobian"
        ],
        output_component_scales=values[f"{prefix}output_component_scales"],
        trust_coordinate_infinity=1.5e-1,
        macro_coordinate_pullback=values[f"{prefix}macro_coordinate_pullbacks"],
    )


def _physical_rate_jacobian(system: ExactAffineMacroSystem) -> np.ndarray:
    scales = np.asarray(system.atlas.macro_coordinate_scales).ravel()
    return (
        scales[:, None]
        * np.asarray(system.normalized_rate_matrix)
        / scales[None, :]
    )


def _evidence_diagnostics() -> dict:
    with np.load(PATCH_1_ARRAYS) as archive:
        patch_1_values = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(PATCH_2_ARRAYS) as archive:
        patch_2_values = {name: np.asarray(archive[name]) for name in archive.files}
    patch_1 = _atlas(patch_1_values, "atlas_")
    patch_2 = _atlas(patch_2_values, "patch_2_")
    system_1 = ExactAffineMacroSystem.from_atlas(patch_1)
    system_2 = ExactAffineMacroSystem.from_atlas(patch_2)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(system_1.normalized_rate_matrix)
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(system_2.normalized_rate_matrix)
    selected_modes = 16
    selected_1 = np.argsort(np.real(eigenvalues_1))[::-1][:selected_modes]
    selected_2 = np.argsort(np.real(eigenvalues_2))[::-1][:selected_modes]
    angles = np.degrees(
        subspace_angles(
            eigenvectors_1[:, selected_1], eigenvectors_2[:, selected_2]
        )
    )
    physical_1 = _physical_rate_jacobian(system_1)
    physical_2 = _physical_rate_jacobian(system_2)
    drift = float(
        np.linalg.norm(physical_2 - physical_1, ord=np.inf)
        / np.linalg.norm(physical_2, ord=np.inf)
    )
    spectral_abscissa_1 = float(np.max(np.real(eigenvalues_1)))
    spectral_abscissa_2 = float(np.max(np.real(eigenvalues_2)))
    cycle_ratios = (
        1.0 / (abs(spectral_abscissa_1) * COMPLETE_CYCLE_SECONDS),
        1.0 / (abs(spectral_abscissa_2) * COMPLETE_CYCLE_SECONDS),
    )
    return {
        "complete_cycle_seconds": COMPLETE_CYCLE_SECONDS,
        "patch_1_spectral_abscissa_per_second": spectral_abscissa_1,
        "patch_2_spectral_abscissa_per_second": spectral_abscissa_2,
        "maximum_fast_efold_to_cycle_ratio": max(cycle_ratios),
        "selected_stable_bundle_modes": selected_modes,
        "stable_bundle_principal_angles_degrees": angles.tolist(),
        "maximum_stable_bundle_principal_angle_degrees": float(np.max(angles)),
        "physical_rate_jacobian_relative_infinity_drift": drift,
        "full_rebuild_truth_calls": 39,
        "transported_patch_truth_calls": 9,
        "transported_to_full_truth_call_fraction": 9.0 / 39.0,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_boundaries": {
            "generic_fixed_slow_global_root_rejection_preserved": True,
            "no_instantaneous_global_equilibrium_substitution": True,
            "all_80_macro_coordinates_remain_dynamic": True,
            "vertical_height_momentum_pair_only_is_analytically_eliminated": True,
            "complete_cycle_execution_authorized": False,
        },
        "stable_bundle_evidence_gates": {
            "both_local_spectral_abscissae_per_second_max": 0.0,
            "maximum_fast_efold_to_cycle_ratio": 1.0e-5,
            "selected_stable_bundle_modes": 16,
            "maximum_interpatch_stable_bundle_angle_degrees": 5.0,
            "maximum_physical_rate_jacobian_relative_infinity_drift": 1.0e-2,
            "interpretation": (
                "normally_stable_moving_full_macro_bundle_not_a_global_fixed_point"
            ),
        },
        "patch_3_anchor": {
            "macro_state": "certified_patch_2_exact_affine_8ms_endpoint",
            "primitive_charts": "certified_exact_thermodynamic_reconstruction_at_8ms",
            "base_output": "certified_full_truth_operator_at_8ms",
            "no_new_base_truth_call": True,
            "no_synthetic_or_projected_anchor": True,
        },
        "chain_rule_transport": {
            "old_chart": "z_2=P_2*diag(1/S_X2)*(X-X_2)",
            "new_chart_tangent": "dX=diag(S_X3)*T_3*dz_3",
            "coordinate_transport_per_cell": (
                "C_23=P_2*diag(S_X3/S_X2)*T_3"
            ),
            "physical_output_tangent": "dY_dz3=diag(S_Y2)*J_2*C_23",
            "new_normalized_tangent": (
                "J_3=diag(1/S_Y3)*diag(S_Y2)*J_2*C_23"
            ),
            "pullback_derivative_step": 1.0e-5,
            "maximum_pullback_condition_number": 1.0e5,
            "maximum_pullback_inverse_closure_infinity": 1.0e-10,
            "block_diagonal_transport_only": True,
            "transport_must_preserve_single_valued_flux_rows": True,
        },
        "independent_transport_validation": {
            "independent_JVP_directions": 4,
            "central_JVP_chart_step": 1.0e-2,
            "new_JVP_truth_calls": 8,
            "maximum_independent_JVP_relative_defect": 5.0e-2,
            "all_truth_physical_gates_binding": True,
            "transported_Jacobian_is_not_refit_from_the_validation_calls": True,
        },
        "overlap_and_dynamic_validation": {
            "overlap_witness": "certified_patch_2_7ms_state",
            "maximum_interpatch_output_relative_defect_per_block": 1.0e-1,
            "maximum_interpatch_macro_rate_relative_defect_per_field": 1.0e-1,
            "patch_3_fixed_macrostep_seconds": 1.0e-3,
            "patch_3_macrosteps": 4,
            "absolute_elapsed_endpoint_seconds": 1.2e-2,
            "atlas_trust_coordinate_infinity": 1.5e-1,
            "reserved_trust_coordinate_infinity": 1.2e-1,
            "one_new_dynamic_endpoint_truth_call": True,
            "maximum_endpoint_truth_output_relative_defect_per_block": 5.0e-2,
            "maximum_endpoint_truth_macro_rate_relative_defect_per_field": 5.0e-2,
            "maximum_endpoint_macro_roundtrip_relative_defect": 1.0e-10,
            "maximum_local_spectral_abscissa_per_second": 0.0,
            "exact_integrated_ledger_relative_defect_max": 1.0e-12,
        },
        "acquisition_cost": {
            "full_patch_truth_calls": 39,
            "transported_patch_truth_calls": 9,
            "transported_to_full_truth_call_fraction": 9.0 / 39.0,
            "maximum_transported_to_full_truth_call_fraction": 2.5e-1,
            "new_global_roots": 0,
            "online_truth_calls_per_macrostep": 0,
        },
        "decision": {
            "pass": (
                "authorize_definitions_only_adaptive_transported_atlas_cadence_"
                "and_heldout_readiness_manifest"
            ),
            "transport_failure": (
                "retain_full_colored_rebuilds_and_reassess_cycle_affordability"
            ),
            "physical_or_stability_failure": "stop_the_seven_field_cycle_path",
            "no_retrospective_gate_change": True,
        },
        "claim_boundary": {
            "one_transported_patch_execution_authorized": True,
            "unbounded_atlas_growth_authorized": False,
            "complete_cycle_execution_authorized": False,
            "reduced_slow_cycle_claim_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
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
                    "sha256": utils._sha256(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("transported patch manifest already exists")
    validated = _validate_parent(require_clean=True)
    diagnostics = _evidence_diagnostics()
    gates = _contract()["stable_bundle_evidence_gates"]
    passed = bool(
        diagnostics["patch_1_spectral_abscissa_per_second"]
        <= gates["both_local_spectral_abscissae_per_second_max"]
        and diagnostics["patch_2_spectral_abscissa_per_second"]
        <= gates["both_local_spectral_abscissae_per_second_max"]
        and diagnostics["maximum_fast_efold_to_cycle_ratio"]
        <= gates["maximum_fast_efold_to_cycle_ratio"]
        and diagnostics["maximum_stable_bundle_principal_angle_degrees"]
        <= gates["maximum_interpatch_stable_bundle_angle_degrees"]
        and diagnostics["physical_rate_jacobian_relative_infinity_drift"]
        <= gates["maximum_physical_rate_jacobian_relative_infinity_drift"]
    )
    if not passed:
        raise RuntimeError("saved stable-bundle evidence fails the frozen gates")
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(
        CANONICAL_DIRECTORY / "transported_patch_contract.json", _contract()
    )
    utils._write_json(
        CANONICAL_DIRECTORY / "stable_bundle_diagnostics.json", diagnostics
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "two_patch_path_preserved": True,
        "moving_full_macro_stable_bundle_supported": True,
        "generic_global_fixed_point_not_assumed": True,
        "transported_third_patch_execution_authorized": True,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "patch_1_arrays_sha256": utils._sha256(PATCH_1_ARRAYS),
            "patch_2_arrays_sha256": utils._sha256(PATCH_2_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete multi-patch growth and fast-slaving manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The two certified patches have negative spectral abscissae, "
                f"a maximum 16-mode bundle rotation of "
                f"`{diagnostics['maximum_stable_bundle_principal_angle_degrees']:.6e}` degrees, "
                "and physical rate-Jacobian drift "
                f"`{diagnostics['physical_rate_jacobian_relative_infinity_drift']:.6e}`.",
                "",
                "Fast slaving means transport of the complete stable 80-coordinate "
                "macro bundle, not an instantaneous global fixed point or a low-rank "
                "truncation. The next package tests exact chain-rule transport at 8 ms "
                "with eight blind JVP calls and one 12 ms endpoint call.",
                "",
                "No unbounded atlas growth or complete-cycle execution is authorized.",
                "",
                f"Authorized next: `{AUTHORIZED_NEXT}` only.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
