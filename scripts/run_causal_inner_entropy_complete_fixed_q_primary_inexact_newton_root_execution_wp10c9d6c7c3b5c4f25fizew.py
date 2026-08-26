#!/usr/bin/env python3
"""Execute and classify the primary fixed-Q fast equilibrium."""

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

import run_causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_execution_manifest_wp10c9d6c7c3b5c4f25fizev as parent  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_inexact_trust_trial_execution_wp10c9d6c7c3b5c4f25fizeu as trial  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_equation_form_root_preflight_wp10c9d6c7c3b5c4f25fizes as preflight  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as implementation  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_slow_inexact_newton import (  # noqa: E402
    bounded_trf_inexact_direction,
    good_broyden_matrix_update,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_fixed_slow_root import (  # noqa: E402
    equation_rate_parity_relative_defect,
    physical_coordinate_rate_jacobian_at_root,
    projected_fast_temporal_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    FAST_EQUATION_ROWS,
    directional_jacobian_relative_defect,
    generalized_maxwell_cattaneo_projected_fast_evaluation,
    radius_one_colored_jacobian,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizew_"
    "entropy_complete_fixed_Q_primary_inexact_Newton_root_execution"
)
PASS_CLASSIFICATION = "entropy_complete_fixed_Q_primary_normally_attracting_root_certified"
NONATTRACTING_CLASSIFICATION = "entropy_complete_fixed_Q_primary_root_exists_but_is_not_normally_attracting"
TANGENT_FAIL_CLASSIFICATION = "entropy_complete_fixed_Q_primary_root_exists_but_physical_tangent_audit_failed"
FAIL_CLASSIFICATION = "entropy_complete_fixed_Q_primary_inexact_Newton_root_failed"
AUTHORIZED_HELDOUT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizex_"
    "entropy_complete_fixed_Q_heldout_root_replication_manifest"
)
AUTHORIZED_MEASURE_ON_NONATTRACTION = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizex_"
    "entropy_complete_fixed_Q_invariant_measure_diagnosis_manifest"
)
AUTHORIZED_TANGENT_DIAGNOSIS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizex_"
    "entropy_complete_fixed_Q_root_tangent_diagnosis_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_primary_inexact_newton_root_"
    "execution_wp10c9d6c7c3b5c4f25fizew"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "PRIMARY_INEXACT_NEWTON_ROOT_EXECUTION_WP10C9D6C7C3B5C4F25FIZEW_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_primary_inexact_"
    "newton_root_execution_wp10c9d6c7c3b5c4f25fizew.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_primary_inexact_"
    "newton_root_execution_wp10c9d6c7c3b5c4f25fizew.py"
)
SOLVER_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_slow_inexact_newton.py"
SOLVER_SOURCE_TEST = "tests/test_causal_inner_fixed_slow_inexact_newton.py"
SOLVER_SOURCE_SHA256 = "853f1b1a1c3d70ff31d90c9d6a6dd7b5e9887ae123c06f2a03236bb88b719e9d"
SOLVER_SOURCE_TEST_SHA256 = "5219da63b6cfbf545aac7911562e3c5c576a1f208a6e90e54af7f38bbfdbd80e"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "cb68359fc029e52d7816356ac736d250407fa499613e8d272784cf53d7e2f271"
)
PREFLIGHT_CHECKSUM_MANIFEST_SHA256 = (
    "dc55606d79dd93d141e9c9b0c574f201e48d55632e26d252b271f124bf7dd6d8"
)
TRIAL_CHECKSUM_MANIFEST_SHA256 = (
    "023951f1be0b91ff3905b0a0483a6aa5fc1abddd56f6333e63ca8164ade9c806"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FORWARD_STEP = 1.0e-6
PHYSICAL_JVP_STEP = 2.0e-6
PHYSICAL_JVP_DIRECTIONS = 4
RANDOM_SEED = 20_260_828


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    locks = (
        (parent.CANONICAL_DIRECTORY, PARENT_CHECKSUM_MANIFEST_SHA256, "root manifest"),
        (preflight.CANONICAL_DIRECTORY, PREFLIGHT_CHECKSUM_MANIFEST_SHA256, "equation preflight"),
        (trial.CANONICAL_DIRECTORY, TRIAL_CHECKSUM_MANIFEST_SHA256, "physical trial"),
    )
    hashes = {}
    for directory, expected, label in locks:
        if utils._sha256(directory / "SHA256SUMS.txt") != expected:
            raise RuntimeError(f"{label} checksum changed")
        hashes[label] = utils._validate_checksums(directory)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "primary_root_execution_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["primary_root_execution_authorized"]
        or summary["heldout_root_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("primary root execution authorization changed")
    if utils._sha256(ROOT / SOLVER_SOURCE) != SOLVER_SOURCE_SHA256:
        raise RuntimeError("inexact-Newton solver source changed")
    if utils._sha256(ROOT / SOLVER_SOURCE_TEST) != SOLVER_SOURCE_TEST_SHA256:
        raise RuntimeError("inexact-Newton solver source test changed")
    for artifact in (parent, preflight, trial):
        for relative, expected in utils._read_json(artifact.CANONICAL_DIRECTORY / "provenance.json")["source_hashes"].items():
            if utils._sha256(ROOT / relative) != expected:
                raise RuntimeError(f"locked primary-root source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("primary root execution requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    solver_contract = contract["inexact_Newton_solver"]
    root_contract = contract["root_gates"]
    certification_contract = contract["post_root_certification"]
    context, _profile, setup_charts = implementation._primary_setup()
    with np.load(preflight.CANONICAL_DIRECTORY / "preflight_arrays.npz") as archive:
        charts = archive["primary_lifted_charts7"]
        targets = archive["slow_targets_MJE"]
        fast_base = archive["base_fast_charts"]
        chart_scales = archive["fast_chart_scales"]
        equation_scales = archive["equation_row_scales_per_cm"]
        base_residual = archive["base_normalized_equation_residual"]
        base_jacobian = archive["colored_normalized_equation_jacobian"]
    with np.load(trial.CANONICAL_DIRECTORY / "trial_arrays.npz") as archive:
        trial_fast = archive["selected_trial_fast_charts"]
        trial_charts = archive["selected_trial_primitive_charts7"]
        trial_residual = archive["selected_trial_normalized_equation_residual"]
        trial_rates = archive["selected_trial_projected_fast_rates_per_second"]
    if not np.array_equal(setup_charts, charts):
        raise RuntimeError("primary setup does not reproduce the locked root state")
    base_coordinate = np.zeros_like(fast_base)
    coordinate = (trial_fast - fast_base) / chart_scales
    residual = np.array(trial_residual, copy=True)
    step_one = (coordinate - base_coordinate).ravel()
    jacobian = good_broyden_matrix_update(
        base_jacobian,
        step_one,
        residual.ravel() - base_residual.ravel(),
    )
    current_evaluation = None
    current_charts = np.array(trial_charts, copy=True)
    current_rates = np.array(trial_rates, copy=True)
    current_checks = None
    solver_refreshes = 0
    new_physical_calls = 0
    events = [
        {
            "correction": 1,
            "source": "hash_locked_fizeu_quarter_step",
            "accepted_factor": 0.25,
            "residual_infinity": float(np.max(np.abs(residual))),
            "residual_two_norm": float(np.linalg.norm(residual)),
        }
    ]
    accepted_coordinates = [np.array(coordinate, copy=True)]
    accepted_residuals = [np.array(residual, copy=True)]

    def evaluate(values: np.ndarray):
        nonlocal new_physical_calls
        new_physical_calls += 1
        candidate_fast = fast_base + np.asarray(values, dtype=float) * chart_scales
        evaluation = generalized_maxwell_cattaneo_projected_fast_evaluation(
            context,
            targets,
            candidate_fast,
            template_charts=charts,
            quadrature_order=8,
        )
        right = evaluation.radial_operator.equation_right_hand_sides_per_cm[:, FAST_EQUATION_ROWS]
        candidate_residual = right / equation_scales
        checks = trial._physical_checks(evaluation, {})
        return evaluation, candidate_residual, checks

    def exact_equation_jacobian(values, base_values, base_evaluation):
        nonlocal new_physical_calls
        point = np.asarray(values, dtype=float)

        def field(candidate):
            nonlocal new_physical_calls
            if np.array_equal(candidate, point):
                return np.array(base_values, copy=True)
            evaluation, candidate_residual, checks = evaluate(candidate)
            if not all(checks.values()):
                raise RuntimeError("colored-Jacobian candidate failed a physical gate")
            return candidate_residual

        recovered, matrix = radius_one_colored_jacobian(
            field, point, relative_step=FORWARD_STEP
        )
        if not np.array_equal(recovered, base_values):
            raise RuntimeError("colored Jacobian changed its base residual")
        return matrix

    def root_measures(values, rates):
        return (
            float(np.max(np.abs(values))),
            float(np.max(np.abs(rates / chart_scales))),
        )

    root_reached = False
    failure_reason = None
    solve_start = time.perf_counter()
    for correction in range(2, solver_contract["maximum_total_nonlinear_corrections"] + 1):
        residual_inf, rate_inf = root_measures(residual, current_rates)
        if residual_inf <= root_contract["maximum_normalized_equation_residual_infinity"] and rate_inf <= root_contract["maximum_physical_fast_coordinate_rate_infinity_per_second"]:
            root_reached = True
            break
        if correction == 10 and solver_refreshes == 0:
            if current_evaluation is None:
                current_evaluation, reproduced, current_checks = evaluate(coordinate)
                if not np.array_equal(reproduced, residual) or not all(current_checks.values()):
                    failure_reason = "iteration_reserve_base_reproduction_failed"
                    break
            jacobian = exact_equation_jacobian(coordinate, residual, current_evaluation)
            solver_refreshes += 1
            events.append({"correction": correction, "event": "iteration_reserve_exact_refresh"})
        accepted = False
        for local_attempt in range(2):
            direction = bounded_trf_inexact_direction(
                jacobian,
                residual,
                maximum_absolute_step=solver_contract["maximum_scaled_step"],
                maximum_backend_iterations=500,
                backend_tolerance=1.0e-12,
            )
            qualified = (
                np.all(np.isfinite(direction.step))
                and direction.forcing_two_norm <= solver_contract["maximum_inexact_forcing_two_norm"]
                and direction.normalized_directional_derivative < 0.0
            )
            direction_record = {
                "correction": correction,
                "local_attempt": local_attempt + 1,
                "forcing_two_norm": direction.forcing_two_norm,
                "forcing_infinity_norm": direction.forcing_infinity_norm,
                "normalized_directional_derivative": direction.normalized_directional_derivative,
                "maximum_absolute_step": direction.maximum_absolute_step,
                "active_bound_count": direction.active_bound_count,
                "backend_success": direction.backend_success,
                "backend_status": direction.backend_status,
                "backend_iterations": direction.backend_iterations,
                "qualified": bool(qualified),
                "line_trials": [],
            }
            if qualified:
                old_two = float(np.linalg.norm(residual))
                old_inf = float(np.max(np.abs(residual)))
                for factor in solver_contract["ordered_line_search_factors"]:
                    candidate_coordinate = coordinate + float(factor) * direction.step.reshape(coordinate.shape)
                    line_record = {"factor": float(factor)}
                    try:
                        candidate_evaluation, candidate_residual, candidate_checks = evaluate(candidate_coordinate)
                        candidate_two = float(np.linalg.norm(candidate_residual))
                        candidate_inf = float(np.max(np.abs(candidate_residual)))
                        armijo = candidate_two <= (1.0 - solver_contract["two_norm_Armijo_coefficient"] * float(factor)) * old_two
                        strict_inf = candidate_inf < old_inf
                        line_record.update({"residual_two_norm": candidate_two, "residual_infinity": candidate_inf, "Armijo_two_norm": armijo, "strict_infinity_decrease": strict_inf, "physical_checks": candidate_checks, "accepted": bool(armijo and strict_inf and all(candidate_checks.values()))})
                        if line_record["accepted"]:
                            displacement = (candidate_coordinate - coordinate).ravel()
                            change = candidate_residual.ravel() - residual.ravel()
                            jacobian = good_broyden_matrix_update(jacobian, displacement, change)
                            coordinate = candidate_coordinate
                            residual = candidate_residual
                            current_evaluation = candidate_evaluation
                            current_charts = candidate_evaluation.reconstruction.primitive_charts
                            current_rates = candidate_evaluation.projected_fast_rates_per_second
                            current_checks = candidate_checks
                            accepted_coordinates.append(np.array(coordinate, copy=True))
                            accepted_residuals.append(np.array(residual, copy=True))
                            direction_record["accepted_factor"] = float(factor)
                            accepted = True
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                        line_record.update({"accepted": False, "exception": f"{type(error).__name__}: {error}"})
                    direction_record["line_trials"].append(line_record)
                    if accepted:
                        break
            events.append(direction_record)
            if accepted:
                break
            if solver_refreshes == 0:
                if current_evaluation is None:
                    current_evaluation, reproduced, current_checks = evaluate(coordinate)
                    if not np.array_equal(reproduced, residual) or not all(current_checks.values()):
                        failure_reason = "refresh_base_reproduction_failed"
                        break
                jacobian = exact_equation_jacobian(coordinate, residual, current_evaluation)
                solver_refreshes += 1
                events.append({"correction": correction, "event": "failure_triggered_exact_refresh"})
                continue
            failure_reason = "inexact_direction_or_complete_line_search_failed_after_refresh"
            break
        if failure_reason is not None:
            break
        if not accepted:
            failure_reason = "correction_not_accepted"
            break
    solve_seconds = time.perf_counter() - solve_start
    residual_inf, rate_inf = root_measures(residual, current_rates)
    root_reached = root_reached or (
        residual_inf <= root_contract["maximum_normalized_equation_residual_infinity"]
        and rate_inf <= root_contract["maximum_physical_fast_coordinate_rate_infinity_per_second"]
    )
    equation_rate_parity = None
    certification = None
    equation_jacobian = None
    physical_jacobian = None
    eigenvalues = None
    jvp_directions = None
    jvp_defects = None
    slow_relative_rate = None
    spectral_abscissa = None
    attraction_ratio = None
    root_exists = False
    normally_attracting = False
    tangent_certified = False
    certification_calls_before = new_physical_calls
    certification_start = time.perf_counter()
    if root_reached and current_evaluation is not None and all(current_checks.values()):
        temporal_blocks, _tangents = projected_fast_temporal_blocks(context, current_charts)
        root_right = current_evaluation.radial_operator.equation_right_hand_sides_per_cm[:, FAST_EQUATION_ROWS]
        equation_rate_parity = equation_rate_parity_relative_defect(temporal_blocks, current_rates, root_right)
        root_exists = equation_rate_parity <= root_contract["maximum_equation_rate_parity_relative_defect"]
        if root_exists:
            equation_jacobian = exact_equation_jacobian(coordinate, residual, current_evaluation)
            physical_jacobian = physical_coordinate_rate_jacobian_at_root(
                temporal_blocks,
                equation_scales,
                equation_jacobian,
                fast_chart_scales=chart_scales,
            )
            root_physical_rate = current_rates / chart_scales

            def physical_field(candidate):
                if np.array_equal(candidate, coordinate):
                    return np.array(root_physical_rate, copy=True)
                evaluation, _candidate_residual, checks = evaluate(candidate)
                if not all(checks.values()):
                    raise RuntimeError("physical-tangent candidate failed a physical gate")
                return evaluation.projected_fast_rates_per_second / chart_scales

            generator = np.random.default_rng(RANDOM_SEED)
            jvp_directions = generator.normal(size=(PHYSICAL_JVP_DIRECTIONS, *coordinate.shape))
            jvp_defects = np.asarray([
                directional_jacobian_relative_defect(
                    physical_field,
                    coordinate,
                    physical_jacobian,
                    direction,
                    relative_step=PHYSICAL_JVP_STEP,
                )
                for direction in jvp_directions
            ])
            eigenvalues = np.linalg.eigvals(physical_jacobian)
            spectral_abscissa = float(np.max(eigenvalues.real))
            target_floor = np.maximum(
                np.max(np.abs(targets), axis=0) * 1.0e-14,
                np.finfo(float).tiny,
            )
            slow_relative_rate = float(np.max(
                np.abs(current_evaluation.slow_integrated_drift_per_second)
                / np.maximum(np.abs(targets), target_floor[None, :])
            ))
            attraction_ratio = float(-spectral_abscissa / max(slow_relative_rate, np.finfo(float).tiny))
            certification = {
                "maximum_physical_tangent_JVP_relative_defect": float(np.max(jvp_defects)),
                "spectral_abscissa_per_second": spectral_abscissa,
                "minimum_real_eigenvalue_per_second": float(np.min(eigenvalues.real)),
                "positive_real_eigenvalue_count": int(np.sum(eigenvalues.real > 0.0)),
                "slow_relative_rate_infinity_per_second": slow_relative_rate,
                "attraction_to_slow_rate_ratio": attraction_ratio,
                "all_eigenvalues_finite": bool(np.all(np.isfinite(eigenvalues))),
            }
            tangent_certified = (
                certification["maximum_physical_tangent_JVP_relative_defect"] <= certification_contract["maximum_physical_tangent_JVP_relative_defect"]
                and certification["all_eigenvalues_finite"]
            )
            normally_attracting = (
                tangent_certified
                and spectral_abscissa <= contract["normal_attraction"]["maximum_spectral_abscissa_per_second"]
                and attraction_ratio >= contract["normal_attraction"]["minimum_attraction_to_slow_relative_rate_ratio"]
            )
    certification_seconds = time.perf_counter() - certification_start
    if normally_attracting:
        classification = PASS_CLASSIFICATION
    elif root_exists and tangent_certified:
        classification = NONATTRACTING_CLASSIFICATION
    elif root_exists:
        classification = TANGENT_FAIL_CLASSIFICATION
    else:
        classification = FAIL_CLASSIFICATION
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": normally_attracting,
        "root_exists": root_exists,
        "normally_attracting": normally_attracting,
        "physical_tangent_certified": tangent_certified,
        "failure_reason": failure_reason,
        "new_nonlinear_roots": 1 if root_exists else 0,
        "propagated_states": 0,
        "accepted_nonlinear_corrections": len(accepted_coordinates),
        "solver_exact_colored_assemblies": 1 + solver_refreshes,
        "solver_optional_refreshes": solver_refreshes,
        "certification_colored_assemblies": 1 if root_exists else 0,
        "new_physical_field_calls": new_physical_calls,
        "certification_physical_field_calls": new_physical_calls - certification_calls_before,
        "solve_wall_seconds": solve_seconds,
        "certification_wall_seconds": certification_seconds,
        "final_normalized_equation_residual_infinity": residual_inf,
        "final_normalized_equation_residual_two_norm": float(np.linalg.norm(residual)),
        "final_physical_fast_coordinate_rate_infinity_per_second": rate_inf,
        "equation_rate_parity_relative_defect": equation_rate_parity,
        "events": events,
        "certification": certification,
    }
    arrays = {
        "primary_lifted_base_charts7": charts,
        "slow_targets_MJE": targets,
        "fast_chart_scales": chart_scales,
        "equation_row_scales_per_cm": equation_scales,
        "accepted_coordinates": np.asarray(accepted_coordinates),
        "accepted_normalized_equation_residuals": np.asarray(accepted_residuals),
        "final_coordinate": coordinate,
        "final_fast_charts": fast_base + coordinate * chart_scales,
        "final_primitive_charts7": current_charts,
        "final_normalized_equation_residual": residual,
        "final_projected_fast_rates_per_second": current_rates,
    }
    if root_exists:
        arrays.update({
            "root_slow_integrated_drift_per_second": current_evaluation.slow_integrated_drift_per_second,
            "root_normalized_equation_jacobian": equation_jacobian,
            "root_physical_coordinate_rate_jacobian_per_second": physical_jacobian,
            "root_physical_eigenvalues_per_second": eigenvalues,
            "physical_JVP_directions": jvp_directions,
            "physical_JVP_relative_defects": jvp_defects,
        })
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "SUPPORTED" if summary["root_exists"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("primary root certificate already exists")
    validated = _validate_parent(require_clean=True); utils = _utils(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "root_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "root_arrays.npz", **arrays)
    if metrics["passed"]: authorized = AUTHORIZED_HELDOUT_ON_PASS
    elif metrics["classification"] == NONATTRACTING_CLASSIFICATION: authorized = AUTHORIZED_MEASURE_ON_NONATTRACTION
    elif metrics["classification"] == TANGENT_FAIL_CLASSIFICATION: authorized = AUTHORIZED_TANGENT_DIAGNOSIS
    else: authorized = None
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": bool(metrics["passed"]), "root_exists": bool(metrics["root_exists"]), "physical_tangent_certified": bool(metrics["physical_tangent_certified"]), "normally_attracting": bool(metrics["normally_attracting"]), "new_nonlinear_roots": metrics["new_nonlinear_roots"], "propagated_states": 0, "heldout_root_replication_manifest_authorized": bool(metrics["passed"]), "invariant_measure_diagnosis_manifest_authorized": metrics["classification"] == NONATTRACTING_CLASSIFICATION, "root_tangent_diagnosis_manifest_authorized": metrics["classification"] == TANGENT_FAIL_CLASSIFICATION, "heldout_root_execution_authorized": False, "slow_flux_atlas_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": authorized}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "preflight_checksum_manifest_sha256": PREFLIGHT_CHECKSUM_MANIFEST_SHA256, "trial_checksum_manifest_sha256": TRIAL_CHECKSUM_MANIFEST_SHA256, "locked_hashes": validated["hashes"], "solver_source_sha256": SOLVER_SOURCE_SHA256, "solver_source_test_sha256": SOLVER_SOURCE_TEST_SHA256})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(("# Entropy-complete fixed-Q primary inexact-Newton root execution", "", f"Classification: `{metrics['classification']}`.", "", f"Root exists: `{metrics['root_exists']}`; normally attracting: `{metrics['normally_attracting']}`; final equation residual infinity: `{metrics['final_normalized_equation_residual_infinity']:.6e}`.", "", "No physical state was propagated.", "", f"Authorized next: `{summary['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, SOLVER_SOURCE, SOLVER_SOURCE_TEST, REPORT_RELATIVE); utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _execute(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
