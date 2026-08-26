#!/usr/bin/env python3
"""Execute the frozen adaptive-trust fixed-Q diagnosis without propagation."""

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

import run_causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_manifest_wp10c9d6c7c3b5c4f25fizex as parent  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_execution_wp10c9d6c7c3b5c4f25fizew as rejected_root  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_inexact_trust_trial_execution_wp10c9d6c7c3b5c4f25fizeu as physical_trial  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_equation_form_root_preflight_wp10c9d6c7c3b5c4f25fizes as preflight  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as implementation  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_slow_inexact_newton import (  # noqa: E402
    bounded_trf_inexact_direction,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    audit_specialized_nonlinear_causality,
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_fixed_slow_root import (  # noqa: E402
    equation_rate_parity_relative_defect,
    projected_fast_temporal_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    _outer_chart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    FAST_EQUATION_ROWS,
    directional_jacobian_relative_defect,
    generalized_maxwell_cattaneo_projected_fast_evaluation,
    generalized_maxwell_cattaneo_reconstruct_fixed_slow,
    radius_one_colored_jacobian,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    kerr_schild_column_geometry,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizey_"
    "entropy_complete_fixed_Q_adaptive_trust_diagnosis_execution"
)
PASS_CLASSIFICATION = (
    "entropy_complete_fixed_Q_adaptive_trust_useful_direction_certified"
)
NO_DIRECTION_CLASSIFICATION = (
    "entropy_complete_fixed_Q_generic_fixed_point_Newton_rejected"
)
LINEARIZATION_FAIL_CLASSIFICATION = (
    "entropy_complete_fixed_Q_adaptive_trust_linearization_diagnosis_failed"
)
AUTHORIZED_RETRY_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizez_"
    "entropy_complete_fixed_Q_adaptive_trust_primary_root_retry_manifest"
)
AUTHORIZED_ANALYTIC_ON_NO_DIRECTION = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizez_"
    "entropy_complete_fixed_Q_analytic_quasisteady_closure_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_adaptive_trust_diagnosis_execution_"
    "wp10c9d6c7c3b5c4f25fizey"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "ADAPTIVE_TRUST_DIAGNOSIS_EXECUTION_WP10C9D6C7C3B5C4F25FIZEY_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_adaptive_trust_"
    "diagnosis_execution_wp10c9d6c7c3b5c4f25fizey.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_adaptive_trust_"
    "diagnosis_execution_wp10c9d6c7c3b5c4f25fizey.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "706a4ba8ef497e4bf4a7d87458a2b037331332e211a0d4d4e7edde4d65cb4fed"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FORWARD_STEP = 1.0e-6
JVP_STEP = 2.0e-6
RANDOM_SEED = 20_260_829


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("adaptive-trust diagnosis manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "adaptive_trust_diagnosis_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["primary_root_rejection_preserved"]
        or not summary["adaptive_trust_diagnosis_authorized"]
        or summary["root_retry_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("adaptive-trust diagnosis authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"adaptive-trust manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive-trust diagnosis requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _midpoint_hyperbolicity_audit(context, charts: np.ndarray) -> dict:
    grid = context.grid
    exterior = _outer_chart(context, charts)
    points: list[tuple[int, float, np.ndarray]] = [
        (0, float(grid.edges[0]), np.asarray(charts[0], dtype=float))
    ]
    points.extend(
        (
            face,
            float(grid.edges[face]),
            0.5 * (charts[face - 1] + charts[face]),
        )
        for face in range(1, charts.shape[0])
    )
    points.append(
        (
            charts.shape[0],
            float(grid.edges[-1]),
            0.5 * (charts[-1] + exterior),
        )
    )
    maximum_eigenvalue_imaginary_ratio = 0.0
    maximum_eigenvector_imaginary_ratio = 0.0
    maximum_eigenvector_condition = 0.0
    minimum_specialized_margin = float("inf")
    worst_eigenvalue_face = -1
    worst_eigenvector_face = -1
    for face, radius, chart in points:
        principal = generalized_maxwell_cattaneo_principal(
            kerr_schild_column_geometry(radius, grid.gravitational_radius),
            chart,
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(radius)
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        eigenvalues = np.asarray(principal.eigenvalues_over_c)
        eigenvectors = np.asarray(principal.right_eigenvectors_scaled)
        value_ratio = float(
            np.max(np.abs(np.imag(eigenvalues)))
            / max(float(np.max(np.abs(eigenvalues))), 1.0)
        )
        vector_ratio = float(
            np.max(np.abs(np.imag(eigenvectors)))
            / max(float(np.max(np.abs(eigenvectors))), 1.0)
        )
        if value_ratio > maximum_eigenvalue_imaginary_ratio:
            maximum_eigenvalue_imaginary_ratio = value_ratio
            worst_eigenvalue_face = face
        if vector_ratio > maximum_eigenvector_imaginary_ratio:
            maximum_eigenvector_imaginary_ratio = vector_ratio
            worst_eigenvector_face = face
        maximum_eigenvector_condition = max(
            maximum_eigenvector_condition,
            float(principal.eigenvector_condition_number),
        )
        causality = audit_specialized_nonlinear_causality(principal.local_state)
        minimum_specialized_margin = min(
            minimum_specialized_margin,
            float(min(causality.inequality_margins)),
        )
    return {
        "evaluated_face_midpoints": len(points),
        "maximum_eigenvalue_imaginary_ratio": maximum_eigenvalue_imaginary_ratio,
        "maximum_eigenvector_imaginary_ratio": maximum_eigenvector_imaginary_ratio,
        "maximum_eigenvector_condition_number": maximum_eigenvector_condition,
        "minimum_specialized_causality_margin": minimum_specialized_margin,
        "worst_eigenvalue_face": worst_eigenvalue_face,
        "worst_eigenvector_face": worst_eigenvector_face,
    }


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    linear_contract = contract["fresh_equation_linearization"]
    candidate_contract = contract["adaptive_trust_candidates"]
    progress_contract = contract["useful_progress"]
    context, _profile, setup_charts = implementation._primary_setup()
    with np.load(rejected_root.CANONICAL_DIRECTORY / "root_arrays.npz") as archive:
        charts = archive["primary_lifted_base_charts7"]
        targets = archive["slow_targets_MJE"]
        chart_scales = archive["fast_chart_scales"]
        equation_scales = archive["equation_row_scales_per_cm"]
        coordinate = archive["final_coordinate"]
        fast = archive["final_fast_charts"]
        primitive_charts = archive["final_primitive_charts7"]
        saved_residual = archive["final_normalized_equation_residual"]
    with np.load(preflight.CANONICAL_DIRECTORY / "preflight_arrays.npz") as archive:
        original_fast = archive["base_fast_charts"]
    if not np.array_equal(setup_charts, charts):
        raise RuntimeError("primary setup does not reproduce the locked state")
    if not np.array_equal(fast, original_fast + coordinate * chart_scales):
        raise RuntimeError("locked fast state does not match its coordinate")
    physical_calls = 0

    def evaluate(values: np.ndarray):
        nonlocal physical_calls
        physical_calls += 1
        candidate_fast = fast + (np.asarray(values, dtype=float) - coordinate) * chart_scales
        evaluation = generalized_maxwell_cattaneo_projected_fast_evaluation(
            context,
            targets,
            candidate_fast,
            template_charts=charts,
            quadrature_order=8,
        )
        right = evaluation.radial_operator.equation_right_hand_sides_per_cm[
            :, FAST_EQUATION_ROWS
        ]
        residual = right / equation_scales
        checks = physical_trial._physical_checks(evaluation, {})
        temporal_blocks, _tangents = projected_fast_temporal_blocks(
            context, evaluation.reconstruction.primitive_charts
        )
        parity = equation_rate_parity_relative_defect(
            temporal_blocks,
            evaluation.projected_fast_rates_per_second,
            right,
        )
        checks["equation_rate_parity"] = parity <= preflight.parent._contract()[
            "root_equivalence"
        ]["maximum_equation_rate_parity_relative_defect"]
        return evaluation, residual, checks, parity

    start = time.perf_counter()
    base_evaluation, reproduced_residual, base_checks, base_parity = evaluate(coordinate)
    base_reproduced = bool(np.array_equal(reproduced_residual, saved_residual))
    primitive_reproduced = bool(
        np.array_equal(base_evaluation.reconstruction.primitive_charts, primitive_charts)
    )
    base_valid = base_reproduced and primitive_reproduced and all(base_checks.values())
    jacobian = None
    jvp_directions = None
    jvp_defects = None
    candidate_records: list[dict] = []
    directions: list[np.ndarray] = []
    predicted_residuals: list[np.ndarray] = []
    selected_index = None
    selected_evaluation = None
    selected_residual = None
    linearization_certified = False
    failure_reason = None
    if base_valid:
        def equation_field(values: np.ndarray) -> np.ndarray:
            if np.array_equal(values, coordinate):
                return np.array(saved_residual, copy=True)
            _evaluation, residual, checks, _parity = evaluate(values)
            if not all(checks.values()):
                raise RuntimeError("equation-linearization field failed a physical gate")
            return residual

        recovered, jacobian = radius_one_colored_jacobian(
            equation_field,
            coordinate,
            relative_step=FORWARD_STEP,
        )
        if not np.array_equal(recovered, saved_residual):
            raise RuntimeError("colored Jacobian changed the locked base residual")
        generator = np.random.default_rng(RANDOM_SEED)
        jvp_directions = generator.normal(
            size=(linear_contract["independent_central_JVP_directions"], *coordinate.shape)
        )
        jvp_defects = np.asarray(
            [
                directional_jacobian_relative_defect(
                    equation_field,
                    coordinate,
                    jacobian,
                    direction,
                    relative_step=JVP_STEP,
                )
                for direction in jvp_directions
            ]
        )
        linearization_certified = bool(
            np.max(jvp_defects)
            <= linear_contract["maximum_JVP_relative_defect"]
        )
    else:
        failure_reason = "locked_base_state_or_residual_reproduction_failed"

    if linearization_certified:
        old_two = float(np.linalg.norm(saved_residual))
        old_inf = float(np.max(np.abs(saved_residual)))
        gates = preflight.parent._contract()["binding_physical_gates"]
        for trust_radius in candidate_contract["ordered_trust_radii"]:
            direction = bounded_trf_inexact_direction(
                jacobian,
                saved_residual,
                maximum_absolute_step=float(trust_radius),
                maximum_backend_iterations=500,
                backend_tolerance=1.0e-12,
            )
            directions.append(np.array(direction.step, copy=True))
            predicted = saved_residual.ravel() + jacobian @ direction.step.ravel()
            predicted_residuals.append(predicted.reshape(saved_residual.shape))
            record = {
                "trust_radius": float(trust_radius),
                "forcing_two_norm": direction.forcing_two_norm,
                "forcing_infinity_norm": direction.forcing_infinity_norm,
                "normalized_directional_derivative": direction.normalized_directional_derivative,
                "active_bound_count": direction.active_bound_count,
                "backend_success": direction.backend_success,
                "backend_status": direction.backend_status,
                "backend_iterations": direction.backend_iterations,
                "predicted_two_norm_ratio": float(np.linalg.norm(predicted) / old_two),
                "predicted_infinity_norm_ratio": float(np.max(np.abs(predicted)) / old_inf),
                "full_physical_evaluation_performed": False,
                "selected": False,
            }
            candidate_coordinate = coordinate + direction.step.reshape(coordinate.shape)
            candidate_fast = fast + direction.step.reshape(coordinate.shape) * chart_scales
            try:
                reconstruction = generalized_maxwell_cattaneo_reconstruct_fixed_slow(
                    context,
                    targets,
                    candidate_fast,
                    template_charts=charts,
                )
                hyperbolicity = _midpoint_hyperbolicity_audit(
                    context, reconstruction.primitive_charts
                )
                reconstruction_passed = bool(
                    reconstruction.maximum_constraint_relative_defect
                    <= gates["maximum_fixed_slow_reconstruction_relative_defect"]
                )
                hyperbolicity_passed = bool(
                    hyperbolicity["maximum_eigenvalue_imaginary_ratio"]
                    <= candidate_contract["binding_midpoint_eigenvalue_imaginary_ratio"]
                )
                record.update(
                    {
                        "fixed_slow_reconstruction_relative_defect": reconstruction.maximum_constraint_relative_defect,
                        "fixed_slow_reconstruction_passed": reconstruction_passed,
                        "midpoint_hyperbolicity": hyperbolicity,
                        "midpoint_hyperbolicity_passed": hyperbolicity_passed,
                    }
                )
                if reconstruction_passed and hyperbolicity_passed:
                    evaluation, residual, checks, parity = evaluate(candidate_coordinate)
                    two_ratio = float(np.linalg.norm(residual) / old_two)
                    inf_ratio = float(np.max(np.abs(residual)) / old_inf)
                    useful = bool(
                        two_ratio <= progress_contract["maximum_actual_two_norm_ratio"]
                        and inf_ratio
                        <= progress_contract["maximum_actual_infinity_norm_ratio"]
                        and all(checks.values())
                    )
                    record.update(
                        {
                            "full_physical_evaluation_performed": True,
                            "actual_two_norm": float(np.linalg.norm(residual)),
                            "actual_infinity_norm": float(np.max(np.abs(residual))),
                            "actual_two_norm_ratio": two_ratio,
                            "actual_infinity_norm_ratio": inf_ratio,
                            "equation_rate_parity_relative_defect": parity,
                            "physical_checks": checks,
                            "useful_progress": useful,
                        }
                    )
                    if useful:
                        selected_index = len(candidate_records)
                        selected_evaluation = evaluation
                        selected_residual = residual
                        record["selected"] = True
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                record["exception"] = f"{type(error).__name__}: {error}"
            candidate_records.append(record)
            if selected_index is not None:
                break
        if selected_index is None:
            failure_reason = "no_feasible_candidate_achieved_frozen_useful_progress"
    elif failure_reason is None:
        failure_reason = "fresh_equation_linearization_JVP_gate_failed"

    wall = time.perf_counter() - start
    selected = selected_index is not None
    if selected:
        classification = PASS_CLASSIFICATION
        authorized_next = AUTHORIZED_RETRY_ON_PASS
    elif linearization_certified:
        classification = NO_DIRECTION_CLASSIFICATION
        authorized_next = AUTHORIZED_ANALYTIC_ON_NO_DIRECTION
    else:
        classification = LINEARIZATION_FAIL_CLASSIFICATION
        authorized_next = None
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": selected,
        "primary_root_policy_rejection_preserved": True,
        "base_residual_reproduced_bitwise": base_reproduced,
        "base_primitive_charts_reproduced_bitwise": primitive_reproduced,
        "base_equation_rate_parity_relative_defect": base_parity,
        "base_physical_checks": base_checks,
        "fresh_equation_linearization_certified": linearization_certified,
        "maximum_equation_JVP_relative_defect": (
            None if jvp_defects is None else float(np.max(jvp_defects))
        ),
        "candidate_records": candidate_records,
        "selected_candidate_index": selected_index,
        "selected_trust_radius": (
            None
            if selected_index is None
            else candidate_records[selected_index]["trust_radius"]
        ),
        "physical_field_calls": physical_calls,
        "maximum_physical_field_calls": 1 + 12 + 2 * linear_contract["independent_central_JVP_directions"] + candidate_contract["maximum_full_physical_candidate_evaluations"],
        "full_physical_candidate_evaluations": sum(
            bool(record["full_physical_evaluation_performed"])
            for record in candidate_records
        ),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "selected_candidate_is_a_root": False,
        "failure_reason": failure_reason,
        "execution_wall_seconds": wall,
        "authorized_next": authorized_next,
    }
    arrays = {
        "primary_lifted_base_charts7": charts,
        "slow_targets_MJE": targets,
        "last_accepted_coordinate": coordinate,
        "last_accepted_fast_charts": fast,
        "last_accepted_primitive_charts7": primitive_charts,
        "fast_chart_scales": chart_scales,
        "equation_row_scales_per_cm": equation_scales,
        "last_accepted_normalized_equation_residual": saved_residual,
    }
    if jacobian is not None:
        arrays["fresh_normalized_equation_jacobian"] = jacobian
        arrays["equation_JVP_directions"] = jvp_directions
        arrays["equation_JVP_relative_defects"] = jvp_defects
        arrays["adaptive_trust_directions"] = np.asarray(directions)
        arrays["adaptive_trust_predicted_residuals"] = np.asarray(predicted_residuals)
    if selected:
        selected_direction = directions[selected_index]
        arrays.update(
            {
                "selected_direction": selected_direction,
                "selected_candidate_coordinate": coordinate + selected_direction,
                "selected_candidate_fast_charts": fast + selected_direction * chart_scales,
                "selected_candidate_primitive_charts7": selected_evaluation.reconstruction.primitive_charts,
                "selected_candidate_normalized_equation_residual": selected_residual,
                "selected_candidate_projected_fast_rates_per_second": selected_evaluation.projected_fast_rates_per_second,
            }
        )
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": status,
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
        raise RuntimeError("adaptive-trust diagnosis result already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "diagnosis_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "diagnosis_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "primary_root_policy_rejection_preserved": True,
        "fresh_equation_linearization_certified": bool(
            metrics["fresh_equation_linearization_certified"]
        ),
        "useful_adaptive_trust_direction_certified": bool(metrics["passed"]),
        "generic_fixed_point_Newton_rejected": (
            metrics["classification"] == NO_DIRECTION_CLASSIFICATION
        ),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "adaptive_trust_primary_root_retry_manifest_authorized": bool(
            metrics["passed"]
        ),
        "analytic_quasisteady_closure_manifest_authorized": (
            metrics["classification"] == NO_DIRECTION_CLASSIFICATION
        ),
        "heldout_root_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "rejected_root_artifact": rejected_root.ARTIFACT,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete fixed-Q adaptive-trust diagnosis",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Fresh equation linearization certified: `{metrics['fresh_equation_linearization_certified']}`; selected trust radius: `{metrics['selected_trust_radius']}`.",
                "",
                "No nonlinear root was claimed and no physical state was propagated.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
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
    return 0 if summary["authorized_next"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
