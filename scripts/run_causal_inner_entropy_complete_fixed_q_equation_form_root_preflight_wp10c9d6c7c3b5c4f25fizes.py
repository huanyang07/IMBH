#!/usr/bin/env python3
"""Certify the conditioned equation-form primary fixed-Q root preflight."""

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
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_fixed_q_primary_root_execution_manifest_wp10c9d6c7c3b5c4f25fizer as parent  # noqa: E402
import run_causal_inner_entropy_complete_fixed_q_invariant_object_implementation_wp10c9d6c7c3b5c4f25fizeq as implementation  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_fixed_slow_root import (  # noqa: E402
    equation_rate_parity_relative_defect,
    fixed_slow_equation_row_scales_per_cm,
    projected_fast_temporal_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (  # noqa: E402
    FAST_CHART_SCALES,
    FAST_EQUATION_ROWS,
    directional_jacobian_relative_defect,
    generalized_maxwell_cattaneo_fast_charts,
    generalized_maxwell_cattaneo_projected_fast_evaluation,
    generalized_maxwell_cattaneo_slow_targets,
    radius_one_colored_jacobian,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizes_"
    "entropy_complete_fixed_Q_equation_form_root_preflight"
)
PASS_CLASSIFICATION = "entropy_complete_fixed_Q_equation_form_root_preflight_certified"
FAIL_CLASSIFICATION = "entropy_complete_fixed_Q_equation_form_root_preflight_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizet_"
    "entropy_complete_fixed_Q_primary_nonlinear_root_execution_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_fixed_q_equation_form_root_preflight_"
    "wp10c9d6c7c3b5c4f25fizes"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_FIXED_Q_"
    "EQUATION_FORM_ROOT_PREFLIGHT_WP10C9D6C7C3B5C4F25FIZES_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_fixed_q_equation_form_root_"
    "preflight_wp10c9d6c7c3b5c4f25fizes.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_fixed_q_equation_form_root_"
    "preflight_wp10c9d6c7c3b5c4f25fizes.py"
)
ROOT_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_generalized_maxwell_cattaneo_fixed_slow_root.py"
)
ROOT_SOURCE_TEST = (
    "tests/test_causal_inner_generalized_maxwell_cattaneo_fixed_slow_root.py"
)
ROOT_SOURCE_SHA256 = "006df082b8795961830d7eded1a199eabfede5e69fe9f44680ec04517234d0a9"
ROOT_SOURCE_TEST_SHA256 = "66fd81c2fa7d374ab3229ebe1c23f15966516bd21ac93b9e25611bbfe23db92f"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "c18e8b7f2ca9f73cd859afcff421af0445e07830b613b9415aa3288c8b63cc99"
)
IMPLEMENTATION_CHECKSUM_MANIFEST_SHA256 = (
    "be306c444fdde78fd414f23401613219c0841fabe19ee5984b0f75a7d8e87db4"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FORWARD_STEP = 1.0e-6
CENTRAL_STEP = 2.0e-6
JVP_DIRECTION_COUNT = 4
RANDOM_SEED = 20_260_827
EXPECTED_PROJECTED_FIELD_CALLS = 1 + 12 + 2 * JVP_DIRECTION_COUNT


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("equation-form root manifest checksum changed")
    if (
        utils._sha256(implementation.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != IMPLEMENTATION_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("projected-field certificate checksum changed")
    parent_hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    implementation_hashes = utils._validate_checksums(
        implementation.CANONICAL_DIRECTORY
    )
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "primary_root_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["equation_form_preflight_authorized"]
        or summary["primary_nonlinear_root_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
    ):
        raise RuntimeError("equation-form root preflight authorization changed")
    if utils._sha256(ROOT / ROOT_SOURCE) != ROOT_SOURCE_SHA256:
        raise RuntimeError("fixed-slow root source changed")
    if utils._sha256(ROOT / ROOT_SOURCE_TEST) != ROOT_SOURCE_TEST_SHA256:
        raise RuntimeError("fixed-slow root source test changed")
    for artifact in (parent, implementation):
        provenance = utils._read_json(
            artifact.CANONICAL_DIRECTORY / "provenance.json"
        )
        for relative, expected in provenance["source_hashes"].items():
            if utils._sha256(ROOT / relative) != expected:
                raise RuntimeError(f"locked preflight source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("equation-form root preflight requires clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "implementation_hashes": implementation_hashes,
        "summary": summary,
        "contract": contract,
    }


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    context, _profile, charts = implementation._primary_setup()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    fast = generalized_maxwell_cattaneo_fast_charts(charts)
    base_start = time.perf_counter()
    base = generalized_maxwell_cattaneo_projected_fast_evaluation(
        context,
        targets,
        fast,
        template_charts=charts,
        quadrature_order=8,
    )
    base_seconds = time.perf_counter() - base_start
    blocks_start = time.perf_counter()
    temporal_blocks, tangent_maps = projected_fast_temporal_blocks(
        context, base.reconstruction.primitive_charts
    )
    equation_scales = fixed_slow_equation_row_scales_per_cm(
        temporal_blocks,
        fast_chart_scales=FAST_CHART_SCALES,
        reference_time_seconds=1.0,
        relative_floor=contract["equation_row_scaling"][
            "floor_relative_to_global_maximum"
        ],
    )
    blocks_seconds = time.perf_counter() - blocks_start
    base_right = base.radial_operator.equation_right_hand_sides_per_cm[
        :, FAST_EQUATION_ROWS
    ]
    base_residual = base_right / equation_scales
    parity = equation_rate_parity_relative_defect(
        temporal_blocks,
        base.projected_fast_rates_per_second,
        base_right,
    )
    calls = 1

    def field(coordinates: np.ndarray) -> np.ndarray:
        nonlocal calls
        values = np.asarray(coordinates, dtype=float)
        if np.array_equal(values, np.zeros_like(values)):
            return np.array(base_residual, copy=True)
        calls += 1
        candidate_fast = fast + values * FAST_CHART_SCALES
        evaluation = generalized_maxwell_cattaneo_projected_fast_evaluation(
            context,
            targets,
            candidate_fast,
            template_charts=charts,
            quadrature_order=8,
        )
        return (
            evaluation.radial_operator.equation_right_hand_sides_per_cm[
                :, FAST_EQUATION_ROWS
            ]
            / equation_scales
        )

    coordinates = np.zeros_like(fast)
    jacobian_start = time.perf_counter()
    recovered_base, jacobian = radius_one_colored_jacobian(
        field, coordinates, relative_step=FORWARD_STEP
    )
    jacobian_seconds = time.perf_counter() - jacobian_start
    generator = np.random.default_rng(RANDOM_SEED)
    directions = generator.normal(size=(JVP_DIRECTION_COUNT, *coordinates.shape))
    jvp_start = time.perf_counter()
    jvp_defects = np.asarray(
        [
            directional_jacobian_relative_defect(
                field,
                coordinates,
                jacobian,
                direction,
                relative_step=CENTRAL_STEP,
            )
            for direction in directions
        ]
    )
    jvp_seconds = time.perf_counter() - jvp_start
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank_thresholds = np.asarray((1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14))
    ranks = np.asarray(
        [
            np.sum(singular_values > singular_values[0] * threshold)
            for threshold in rank_thresholds
        ],
        dtype=int,
    )
    trust = contract["prospective_linear_step"][
        "maximum_scaled_fast_chart_correction"
    ]
    linear_start = time.perf_counter()
    linear = lsq_linear(
        jacobian,
        -base_residual.ravel(),
        bounds=(-trust, trust),
        method="trf",
        lsq_solver="exact",
        tol=1.0e-12,
        max_iter=500,
    )
    linear_seconds = time.perf_counter() - linear_start
    step = np.asarray(linear.x, dtype=float)
    predicted = base_residual.ravel() + jacobian @ step
    base_inf = float(np.max(np.abs(base_residual)))
    base_two = float(np.linalg.norm(base_residual))
    predicted_inf_ratio = float(np.max(np.abs(predicted)) / base_inf)
    predicted_two_ratio = float(np.linalg.norm(predicted) / base_two)
    physical = {
        "maximum_imaginary_speed_over_c": base.radial_operator.maximum_imaginary_speed_over_c,
        "maximum_light_cone_excess_over_c": base.radial_operator.maximum_light_cone_excess_over_c,
        "maximum_eigenvector_condition_number": base.radial_operator.maximum_eigenvector_condition_number,
        "minimum_height_over_radius": base.radial_operator.minimum_height_over_radius,
        "maximum_height_over_radius": base.radial_operator.maximum_height_over_radius,
        "minimum_optical_depth": base.radial_operator.minimum_optical_depth,
        "incoming_inner_characteristics": base.radial_operator.incoming_inner_characteristics,
    }
    gates = contract["binding_physical_gates"]
    checks = {
        "constraint_reconstruction": base.reconstruction.maximum_constraint_relative_defect
        <= gates["maximum_fixed_slow_reconstruction_relative_defect"],
        "temporal_projection": base.maximum_temporal_projection_solve_relative_defect
        <= gates["maximum_temporal_projection_solve_relative_defect"],
        "equation_rate_parity": parity
        <= contract["root_equivalence"][
            "maximum_equation_rate_parity_relative_defect"
        ],
        "colored_base": np.array_equal(recovered_base, base_residual),
        "colored_shape": jacobian.shape == (448, 448),
        "colored_JVP": float(np.max(jvp_defects))
        <= contract["linearization_preflight"][
            "maximum_colored_JVP_relative_defect"
        ],
        "field_call_budget": calls == EXPECTED_PROJECTED_FIELD_CALLS,
        "bounded_linear_solver": bool(linear.success) and np.all(np.isfinite(step)),
        "bounded_linear_step": float(np.max(np.abs(step))) <= trust * (1.0 + 1.0e-12),
        "predicted_infinity_reduction": predicted_inf_ratio
        <= contract["prospective_linear_step"][
            "maximum_predicted_infinity_merit_ratio"
        ],
        "predicted_two_norm_reduction": predicted_two_ratio
        <= contract["prospective_linear_step"][
            "maximum_predicted_two_norm_merit_ratio"
        ],
        "imaginary_speed": physical["maximum_imaginary_speed_over_c"]
        <= gates["maximum_imaginary_speed_over_c"],
        "light_cone": physical["maximum_light_cone_excess_over_c"]
        <= gates["maximum_light_cone_excess_over_c"],
        "eigenbasis_condition": physical["maximum_eigenvector_condition_number"]
        <= gates["eigenvector_condition_number_max"],
        "height": physical["minimum_height_over_radius"]
        >= gates["minimum_height_over_radius"]
        and physical["maximum_height_over_radius"]
        <= gates["maximum_height_over_radius"],
        "optical_depth": physical["minimum_optical_depth"]
        >= gates["minimum_optical_depth"],
        "inner_excision": physical["incoming_inner_characteristics"]
        == gates["inner_incoming_characteristics_equal"],
    }
    passed = all(checks.values())
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "checks": checks,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "projected_field_calls": calls,
        "expected_projected_field_calls": EXPECTED_PROJECTED_FIELD_CALLS,
        "base_field_wall_seconds": base_seconds,
        "temporal_block_wall_seconds": blocks_seconds,
        "colored_jacobian_wall_seconds": jacobian_seconds,
        "independent_JVP_wall_seconds": jvp_seconds,
        "bounded_linear_step_wall_seconds": linear_seconds,
        "equation_rate_parity_relative_defect": parity,
        "base_normalized_equation_residual_infinity": base_inf,
        "base_normalized_equation_residual_two_norm": base_two,
        "maximum_colored_JVP_relative_defect": float(np.max(jvp_defects)),
        "individual_colored_JVP_relative_defects": jvp_defects.tolist(),
        "jacobian_condition_number_2": float(
            singular_values[0] / singular_values[-1]
        ),
        "numerical_rank_thresholds": rank_thresholds.tolist(),
        "numerical_ranks": ranks.tolist(),
        "bounded_linear_solver_status": int(linear.status),
        "bounded_linear_solver_message": str(linear.message),
        "bounded_linear_solver_iterations": int(linear.nit),
        "bounded_linear_step_maximum_absolute": float(np.max(np.abs(step))),
        "bounded_linear_step_active_bound_count": int(
            np.sum(np.abs(step) >= trust * (1.0 - 1.0e-8))
        ),
        "predicted_infinity_merit_ratio": predicted_inf_ratio,
        "predicted_two_norm_merit_ratio": predicted_two_ratio,
        "maximum_constraint_relative_defect": base.reconstruction.maximum_constraint_relative_defect,
        "maximum_temporal_projection_solve_relative_defect": base.maximum_temporal_projection_solve_relative_defect,
        "physical": physical,
    }
    arrays = {
        "primary_lifted_charts7": charts,
        "slow_targets_MJE": targets,
        "base_fast_charts": fast,
        "fast_chart_scales": FAST_CHART_SCALES,
        "projected_temporal_blocks": temporal_blocks,
        "fixed_slow_chart_tangent_maps": tangent_maps,
        "equation_row_scales_per_cm": equation_scales,
        "base_fast_equation_RHS_per_cm": base_right,
        "base_normalized_equation_residual": base_residual,
        "colored_normalized_equation_jacobian": jacobian,
        "singular_values": singular_values,
        "numerical_rank_thresholds": rank_thresholds,
        "numerical_ranks": ranks,
        "JVP_directions": directions,
        "JVP_relative_defects": jvp_defects,
        "bounded_linear_step": step.reshape(fast.shape),
        "predicted_normalized_equation_residual": predicted.reshape(fast.shape),
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
        raise RuntimeError("equation-form root preflight certificate already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "preflight_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "preflight_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "projected_field_certificate_preserved": True,
        "equation_form_linearization_certified": bool(metrics["passed"]),
        "bounded_linear_step_certified": bool(metrics["passed"]),
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "primary_nonlinear_root_execution_manifest_authorized": bool(
            metrics["passed"]
        ),
        "primary_nonlinear_root_execution_authorized": False,
        "heldout_root_execution_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["parent_hashes"],
            "implementation_artifact": implementation.ARTIFACT,
            "implementation_checksum_manifest_sha256": IMPLEMENTATION_CHECKSUM_MANIFEST_SHA256,
            "implementation_hashes": validated["implementation_hashes"],
            "root_source_sha256": ROOT_SOURCE_SHA256,
            "root_source_test_sha256": ROOT_SOURCE_TEST_SHA256,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete fixed-Q equation-form root preflight",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"Equation/rate parity defect: `{metrics['equation_rate_parity_relative_defect']:.6e}`; maximum independent colored-JVP defect: `{metrics['maximum_colored_JVP_relative_defect']:.6e}`.",
                "",
                f"The prospectively bounded linear step predicts infinity/two-norm merit ratios `{metrics['predicted_infinity_merit_ratio']:.6e}` and `{metrics['predicted_two_norm_merit_ratio']:.6e}`. No nonlinear candidate or state was propagated.",
                "",
                f"Authorized next: `{summary['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, ROOT_SOURCE, ROOT_SOURCE_TEST, REPORT_RELATIVE)
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
    metrics, arrays = _audit()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
