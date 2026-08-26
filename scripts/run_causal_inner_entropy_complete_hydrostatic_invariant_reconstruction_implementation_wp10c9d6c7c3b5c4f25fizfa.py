#!/usr/bin/env python3
"""Certify the hydrostatic invariant inverse at primary and held-out states."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_analytic_quasisteady_closure_manifest_wp10c9d6c7c3b5c4f25fizez as parent  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as fixed_q_implementation  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_execution_wp10c9d6c7c3b5c4f25fizey as adaptive_diagnosis  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_quasisteady import (  # noqa: E402
    hydrostatic_invariant_local_scaled_jacobian,
    reconstruct_hydrostatic_fixed_invariants,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (  # noqa: E402
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    generalized_maxwell_cattaneo_slow_targets,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfa_"
    "entropy_complete_hydrostatic_invariant_reconstruction_implementation"
)
PASS_CLASSIFICATION = (
    "entropy_complete_hydrostatic_Q3_plus_radial_stress_inverse_and_"
    "truth_samples_certified"
)
FAIL_CLASSIFICATION = (
    "entropy_complete_hydrostatic_Q3_plus_radial_stress_inverse_failed"
)
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizfb_"
    "entropy_complete_local_slow_flux_atlas_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hydrostatic_invariant_reconstruction_"
    "implementation_wp10c9d6c7c3b5c4f25fizfa"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_HYDROSTATIC_"
    "INVARIANT_RECONSTRUCTION_WP10C9D6C7C3B5C4F25FIZFA_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hydrostatic_invariant_"
    "reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hydrostatic_invariant_"
    "reconstruction_implementation_wp10c9d6c7c3b5c4f25fizfa.py"
)
SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_quasisteady.py"
)
SOURCE_TEST = "tests/test_causal_inner_generalized_maxwell_cattaneo_quasisteady.py"
SOURCE_SHA256 = "7eecba17418ab42d43158b6fdcaa332a8ec166ec475e61357ede9884e38e2ff2"
SOURCE_TEST_SHA256 = "85cd5601b1d5c77e23ca7742db714470a9c536be75cbf94d3f4714876f4e2423"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "8b75f41fbe17fea54eef71b56d5be987bb2ea36afa14c301b9dd072f9fefbfac"
)
AUDIT_ENVELOPE = ROOT / (
    "results/canonical/causal_inner_seven_field_physical_closure_local_"
    "audit_manifest_wp10c9d6c7c3b5c4f25fizeb/audit_envelope.npz"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHART_SCALES7 = np.asarray((1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03))
SLOW_ROWS = np.asarray((0, 2, 3), dtype=int)
SELECTED_CELLS = (0, 18, 36, 55, 74, 92, 111)
RANDOM_SEED = 20_260_830


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("analytic closure manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "analytic_quasisteady_closure_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["partial_equilibrium_Q3_plus_radial_stress_architecture_selected"]
        or not summary["local_reconstruction_implementation_authorized"]
        or summary["slow_flux_atlas_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("analytic reconstruction authorization changed")
    if utils._sha256(ROOT / SOURCE) != SOURCE_SHA256:
        raise RuntimeError("hydrostatic reconstruction source changed")
    if utils._sha256(ROOT / SOURCE_TEST) != SOURCE_TEST_SHA256:
        raise RuntimeError("hydrostatic reconstruction source test changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"analytic closure manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("hydrostatic reconstruction audit requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _embed(context, profile5: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            generalized_maxwell_cattaneo_hydrostatic_embedding(
                chart,
                proper_vertical_frequency=float(
                    context.vertical_frequency.frequency(float(radius))
                ),
            )
            for radius, chart in zip(context.grid.centers, profile5, strict=True)
        ]
    )


def _perturb_macro_state(
    targets: np.ndarray,
    radial: np.ndarray,
    stress: np.ndarray,
    *,
    invariant_amplitude: float,
    radial_amplitude: float,
    stress_amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perturbed_targets = np.array(targets, copy=True)
    perturbed_radial = np.array(radial, copy=True)
    perturbed_stress = np.array(stress, copy=True)
    for ordinal, cell in enumerate(SELECTED_CELLS):
        sign = -1.0 if ordinal % 2 else 1.0
        perturbed_targets[cell] *= 1.0 + invariant_amplitude * np.asarray(
            (sign, -sign, 0.5 * sign)
        )
        perturbed_radial[cell] += sign * radial_amplitude
        perturbed_stress[cell] *= 1.0 - sign * stress_amplitude
    return perturbed_targets, perturbed_radial, perturbed_stress


def _operator_record(operator) -> dict:
    return {
        "maximum_imaginary_speed_over_c": operator.maximum_imaginary_speed_over_c,
        "maximum_light_cone_excess_over_c": operator.maximum_light_cone_excess_over_c,
        "maximum_eigenvector_condition_number": operator.maximum_eigenvector_condition_number,
        "minimum_height_over_radius": operator.minimum_height_over_radius,
        "maximum_height_over_radius": operator.maximum_height_over_radius,
        "minimum_optical_depth": operator.minimum_optical_depth,
        "incoming_inner_characteristics": operator.incoming_inner_characteristics,
        "maximum_temporal_solve_relative_defect": float(
            np.max(operator.temporal_solve_relative_residuals)
        ),
    }


def _physical_checks(record: dict, gates: dict) -> dict:
    return {
        "imaginary_speed": record["maximum_imaginary_speed_over_c"]
        <= gates["maximum_imaginary_speed_over_c"],
        "light_cone": record["maximum_light_cone_excess_over_c"]
        <= gates["maximum_light_cone_excess_over_c"],
        "eigenbasis_condition": record["maximum_eigenvector_condition_number"]
        <= gates["eigenvector_condition_number_max"],
        "height": record["minimum_height_over_radius"]
        >= gates["minimum_height_over_radius"]
        and record["maximum_height_over_radius"]
        <= gates["maximum_height_over_radius"],
        "optical_depth": record["minimum_optical_depth"]
        >= gates["minimum_optical_depth"],
        "inner_excision": record["incoming_inner_characteristics"]
        == gates["inner_incoming_characteristics_equal"],
        "temporal_solve": record["maximum_temporal_solve_relative_defect"]
        <= gates["maximum_temporal_projection_solve_relative_defect"],
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    inverse_contract = contract["analytic_reconstruction"]
    validation_contract = contract["offline_validation"]
    gates = fixed_q_implementation.parent._contract()["binding_physical_gates"]
    context, _primary_profile, _primary_charts = fixed_q_implementation._primary_setup()
    with np.load(AUDIT_ENVELOPE) as archive:
        profiles = {
            "primary_20ms": np.asarray(archive["primary_20ms_base_charts5"]),
            "heldout_16ms": np.asarray(archive["heldout_16ms_base_charts5"]),
        }
    generator = np.random.default_rng(RANDOM_SEED)
    profile_records = {}
    arrays: dict[str, np.ndarray] = {}
    all_jvp_defects = []
    all_conditions = []
    operator_calls = 0
    start = time.perf_counter()
    for label, profile5 in profiles.items():
        charts = _embed(context, profile5)
        targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
        exact = reconstruct_hydrostatic_fixed_invariants(
            context,
            targets,
            charts[:, 1],
            charts[:, 4],
            template_charts=charts,
            constraint_tolerance=inverse_contract["maximum_constraint_relative_defect"],
            maximum_newton_corrections=inverse_contract["maximum_newton_corrections"],
        )
        exact_chart_defect = float(
            np.max(np.abs((exact.primitive_charts - charts) / CHART_SCALES7))
        )
        local_jvp_defects = []
        local_conditions = []
        for cell in SELECTED_CELLS:
            matrix = hydrostatic_invariant_local_scaled_jacobian(
                context, cell, charts[cell]
            )
            local_conditions.append(float(np.linalg.cond(matrix)))
            direction = generator.normal(size=3)
            direction /= np.linalg.norm(direction)
            step = 2.0e-6
            scale = np.abs(targets[cell])
            plus_targets = np.array(targets, copy=True)
            minus_targets = np.array(targets, copy=True)
            plus_targets[cell] += step * scale * direction
            minus_targets[cell] -= step * scale * direction
            plus = reconstruct_hydrostatic_fixed_invariants(
                context,
                plus_targets,
                charts[:, 1],
                charts[:, 4],
                template_charts=charts,
            )
            minus = reconstruct_hydrostatic_fixed_invariants(
                context,
                minus_targets,
                charts[:, 1],
                charts[:, 4],
                template_charts=charts,
            )
            finite = (
                plus.primitive_charts[cell, [0, 2, 3]]
                - minus.primitive_charts[cell, [0, 2, 3]]
            ) / (2.0 * step * np.asarray((1.0, 0.1, 1.0)))
            analytic = np.linalg.solve(matrix, direction)
            defect = float(
                np.max(np.abs(finite - analytic))
                / max(
                    float(np.max(np.abs(finite))),
                    float(np.max(np.abs(analytic))),
                    np.finfo(float).tiny,
                )
            )
            local_jvp_defects.append(defect)
        perturbed_targets, perturbed_radial, perturbed_stress = _perturb_macro_state(
            targets,
            charts[:, 1],
            charts[:, 4],
            invariant_amplitude=validation_contract["relative_invariant_perturbation"],
            radial_amplitude=validation_contract["radial_velocity_chart_perturbation"],
            stress_amplitude=validation_contract["stress_relative_perturbation"],
        )
        perturbed = reconstruct_hydrostatic_fixed_invariants(
            context,
            perturbed_targets,
            perturbed_radial,
            perturbed_stress,
            template_charts=charts,
            constraint_tolerance=inverse_contract["maximum_constraint_relative_defect"],
            maximum_newton_corrections=inverse_contract["maximum_newton_corrections"],
        )
        reconstructed_targets = generalized_maxwell_cattaneo_slow_targets(
            context, perturbed.primitive_charts
        )
        perturb_constraint_defect = float(
            np.max(
                np.abs(reconstructed_targets - perturbed_targets)
                / np.maximum(np.abs(perturbed_targets), np.finfo(float).tiny)
            )
        )
        prefilters = {}
        operators = {}
        for sample_label, sample_charts in (
            ("base", exact.primitive_charts),
            ("perturbed", perturbed.primitive_charts),
        ):
            prefilter = adaptive_diagnosis._midpoint_hyperbolicity_audit(
                context, sample_charts
            )
            prefilters[sample_label] = prefilter
            if (
                prefilter["maximum_eigenvalue_imaginary_ratio"]
                > validation_contract["midpoint_eigenvalue_imaginary_ratio_max"]
                or prefilter["maximum_eigenvector_imaginary_ratio"]
                > validation_contract["midpoint_eigenvalue_imaginary_ratio_max"]
            ):
                raise RuntimeError(
                    f"{label} {sample_label} leaves the seven-field hyperbolic branch"
                )
            operator = generalized_maxwell_cattaneo_radial_operator(
                context, sample_charts, quadrature_order=8
            )
            operator_calls += 1
            operators[sample_label] = operator
        operator_records = {
            name: _operator_record(operator) for name, operator in operators.items()
        }
        checks = {
            "exact_anchor_chart_reproduction": exact_chart_defect
            <= validation_contract["anchor_chart_reproduction_relative_defect"],
            "exact_anchor_constraint": exact.maximum_constraint_relative_defect
            <= inverse_contract["maximum_constraint_relative_defect"],
            "perturbed_constraint": perturb_constraint_defect
            <= inverse_contract["maximum_constraint_relative_defect"],
            "local_inverse_condition": max(local_conditions)
            <= inverse_contract["maximum_scaled_local_inverse_condition_number"],
            "local_inverse_JVP": max(local_jvp_defects)
            <= inverse_contract["maximum_local_inverse_JVP_relative_defect"],
            "base_prefilter": prefilters["base"]["maximum_eigenvalue_imaginary_ratio"]
            <= validation_contract["midpoint_eigenvalue_imaginary_ratio_max"],
            "perturbed_prefilter": prefilters["perturbed"]["maximum_eigenvalue_imaginary_ratio"]
            <= validation_contract["midpoint_eigenvalue_imaginary_ratio_max"],
            "base_physical": all(_physical_checks(operator_records["base"], gates).values()),
            "perturbed_physical": all(
                _physical_checks(operator_records["perturbed"], gates).values()
            ),
        }
        profile_records[label] = {
            "checks": checks,
            "exact_anchor_chart_reproduction_scaled_infinity": exact_chart_defect,
            "exact_anchor_constraint_relative_defect": exact.maximum_constraint_relative_defect,
            "perturbed_constraint_relative_defect": perturb_constraint_defect,
            "perturbed_maximum_newton_corrections": perturbed.maximum_newton_corrections,
            "perturbed_maximum_scaled_unknown_correction": perturbed.maximum_scaled_unknown_correction,
            "maximum_local_inverse_condition_number": max(local_conditions),
            "maximum_local_inverse_JVP_relative_defect": max(local_jvp_defects),
            "local_inverse_condition_numbers": local_conditions,
            "local_inverse_JVP_relative_defects": local_jvp_defects,
            "midpoint_prefilters": prefilters,
            "operator_records": operator_records,
            "operator_physical_checks": {
                name: _physical_checks(record, gates)
                for name, record in operator_records.items()
            },
        }
        all_jvp_defects.extend(local_jvp_defects)
        all_conditions.extend(local_conditions)
        prefix = label
        arrays.update(
            {
                f"{prefix}_base_charts5": profile5,
                f"{prefix}_base_charts7": charts,
                f"{prefix}_slow_targets_MJE": targets,
                f"{prefix}_perturbed_targets_MJE": perturbed_targets,
                f"{prefix}_perturbed_radial_velocity_over_c": perturbed_radial,
                f"{prefix}_perturbed_specific_shear_stress": perturbed_stress,
                f"{prefix}_perturbed_charts7": perturbed.primitive_charts,
            }
        )
        for sample_label, operator in operators.items():
            sample_prefix = f"{prefix}_{sample_label}"
            arrays.update(
                {
                    f"{sample_prefix}_weighted_shared_MJE_fluxes_over_c": operator.weighted_shared_exact_fluxes_over_c[:, SLOW_ROWS],
                    f"{sample_prefix}_weighted_MJE_sources_per_ct": operator.weighted_equation_sources_per_ct[:, SLOW_ROWS],
                    f"{sample_prefix}_MJE_integrated_states": operator.exact_integrated_states[:, SLOW_ROWS],
                    f"{sample_prefix}_radial_stress_rates_per_second": C
                    * operator.primitive_rates_per_ct[:, [1, 4]],
                    f"{sample_prefix}_MJE_boundary_source_rate_per_second": C
                    * operator.exact_global_boundary_source_rate_per_ct[[0, 2, 3]],
                }
            )
    passed = all(
        all(record["checks"].values()) for record in profile_records.values()
    ) and operator_calls == 4
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "generic_global_fixed_Q_Newton_rejection_preserved": True,
        "profiles": profile_records,
        "maximum_local_inverse_JVP_relative_defect": max(all_jvp_defects),
        "maximum_scaled_local_inverse_condition_number": max(all_conditions),
        "offline_seven_field_operator_calls": operator_calls,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "execution_wall_seconds": time.perf_counter() - start,
    }
    return metrics, arrays


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
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
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


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("hydrostatic reconstruction certificate already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "implementation_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "implementation_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "generic_global_fixed_Q_Newton_rejection_preserved": True,
        "hydrostatic_invariant_inverse_certified": bool(metrics["passed"]),
        "primary_and_heldout_truth_samples_certified": bool(metrics["passed"]),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "local_slow_flux_atlas_manifest_authorized": bool(metrics["passed"]),
        "slow_flux_atlas_execution_authorized": False,
        "online_macro_solver_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "source_sha256": SOURCE_SHA256,
            "source_test_sha256": SOURCE_TEST_SHA256,
            "audit_envelope_sha256": utils._sha256(AUDIT_ENVELOPE),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete hydrostatic invariant reconstruction",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Maximum inverse-JVP defect: `{metrics['maximum_local_inverse_JVP_relative_defect']:.6e}`; maximum scaled local condition number: `{metrics['maximum_scaled_local_inverse_condition_number']:.6e}`.",
                "",
                f"Seven-field operator samples: `{metrics['offline_seven_field_operator_calls']}`. No nonlinear root or physical state was propagated.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, SOURCE, SOURCE_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
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
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _execute()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
