#!/usr/bin/env python3
"""Certify the monolithic descriptor-path DAE assembly (WP10c9d6a).

This is a small, production-neutral method preflight.  It does not solve a
physical timestep.  The package verifies one unified primitive residual,
records that responsive-height temporal storage is a non-exact one-form, and
tests whether the complete declared residual is a viable differentiable
target for the next manufactured-equilibrium/wave package.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_dae_scaling,
    causal_five_field_monolithic_storage_increment,
    causal_five_field_state_from_primitives,
    causal_five_field_temporal_storage_integrability_audit,
    causal_radial_high_order_directional_derivatives,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_monolithic_backward_euler,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6a"
ANALYZED_BASE_COMMIT = "e836df2e2c0e1d180f3a8c56383498578434762e"
ANALYZED_BASE_PARENT = "f409244f0f9b487b918d4e93f49e8bcf41049af1"
ANALYZED_BASE_TREE = "29ad6029a6eb3a1977b8395bbe9783358914cd07"
THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_dae_preflight_wp10c9d6a.py"
)

N_CELLS = 5
TIMESTEP_SECONDS = 1.0e-4
TEMPORAL_QUADRATURE_ORDER = 6
TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP = 1.0e-2
PATH_QUADRATURE_ORDER = 6
PATH_SUBDIVISION_FRACTION = 0.37
JVP_STEP = 1.0e-4
JVP_ORDERS = (4, 6)
HOMOGENEITY_FACTOR = -0.371

MAXIMUM_MAPPED_PATH_CLOSURE_DEFECT = 2.0e-8
MAXIMUM_TEMPORAL_REVERSAL_DEFECT = 2.0e-10
MAXIMUM_COLLINEAR_PATH_ADDITIVITY_DEFECT = 2.0e-10
MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE = 1.0e-14
MAXIMUM_BLOCK_LEDGER_DEFECT = 1.0e-12
MAXIMUM_SHARED_FLUX_TELESCOPE_DEFECT = 1.0e-12
MAXIMUM_CENTER_BROKEN_PATH_ADJUSTMENT = 2.0e-8
MAXIMUM_JVP_ORDER_DIFFERENCE = 1.0e-8
MAXIMUM_JVP_ADDITIVITY_DEFECT = 1.0e-8
MAXIMUM_JVP_HOMOGENEITY_DEFECT = 1.0e-8
MINIMUM_VERTICAL_EXTERIOR_DERIVATIVE = 1.0e-2
MINIMUM_COMPLETE_EXTERIOR_DERIVATIVE = 1.0e-7
MINIMUM_TEMPORAL_LOOP_DEFECT = 1.0e-5

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_extended_localization_wp10c9d5c1"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_dae_preflight_wp10c9d6a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "tests/test_causal_inner_monolithic_dae.py",
    "tests/test_causal_inner_monolithic_dae_preflight_wp10c9d6a.py",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _relative_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS.txt" or not path.is_file():
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    commit = _git("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (commit, parent, tree) != (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    ):
        raise RuntimeError("WP10c9d6a analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).exists()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _load_parent_evidence() -> dict:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        summary["classification"]
        != "D_no_recovery_or_stable_non_target_mechanism"
        or not summary["method_passed"]
        or not summary["monolithic_replacement_authorized"]
        or summary["nonlinear_candidate_authorized"]
        or summary["fixed_q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("WP10c9d5c1 binding status changed")
    if _sha256(PARENT_ARRAYS) != summary["decisive_arrays_sha256"]:
        raise RuntimeError("WP10c9d5c1 decisive archive changed")
    return summary


def _context_and_charts():
    from dataclasses import replace

    context = make_causal_five_field_regression_context(N_CELLS)
    charts = np.asarray(
        make_causal_five_field_seed(context).primitives,
        dtype=float,
    )
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            charts[-1],
            copy=True,
        ),
    ).validated()
    return context, charts


def _increment(charts: np.ndarray) -> np.ndarray:
    phase = np.sin(np.linspace(0.0, np.pi, charts.shape[0]))[:, None]
    scale = np.asarray(
        [1.0e-5, 0.0, 0.0, 1.0e-5, 0.0],
        dtype=float,
    )
    return phase * scale[None, :]


def _directions(charts: np.ndarray) -> dict[str, np.ndarray]:
    phase = np.sin(np.linspace(0.0, np.pi, charts.shape[0]))[:, None]
    first = phase * np.asarray(
        [1.0, 0.0, 0.0, 0.7, 0.0],
        dtype=float,
    )[None, :]
    second = phase * np.asarray(
        [0.0, 0.2, -0.15, 0.0, 0.0],
        dtype=float,
    )[None, :]
    return {
        "first": first,
        "second": second,
        "sum": first + second,
        "homogeneous": HOMOGENEITY_FACTOR * first,
    }


def _conservation_row_scales(context, charts: np.ndarray) -> np.ndarray:
    state = causal_five_field_state_from_primitives(context, charts)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    scaling = causal_five_field_dae_scaling(state, evaluation)
    return np.asarray(
        scaling.row_scales[: charts.size],
        dtype=float,
    ).reshape(charts.shape)


def _residual_function(
    context,
    old_charts: np.ndarray,
    row_scales: np.ndarray,
):
    shape = old_charts.shape

    def residual(new_values: np.ndarray) -> np.ndarray:
        new_charts = np.asarray(new_values, dtype=float).reshape(shape)
        evaluation = evaluate_causal_five_field_monolithic_backward_euler(
            old_charts,
            new_charts,
            TIMESTEP_SECONDS,
            context,
            temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
            reconstruction_directional_step=(
                TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
            ),
            path_quadrature_order=PATH_QUADRATURE_ORDER,
        )
        return np.asarray(
            evaluation.residual_rows / row_scales,
            dtype=float,
        ).ravel()

    return residual


def _jvp_report(
    context,
    charts: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    directions = _directions(charts)
    scales = _conservation_row_scales(context, charts)
    residual = _residual_function(context, charts, scales)
    base = np.asarray(charts, dtype=float).ravel()
    actions = {}
    for name, direction in directions.items():
        print(f"WP10c9d6a: JVP {name}", flush=True)
        actions[name] = causal_radial_high_order_directional_derivatives(
            residual,
            base,
            np.asarray(direction, dtype=float).ravel(),
            finite_difference_step=JVP_STEP,
            derivative_orders=JVP_ORDERS,
        )

    order_differences = {
        name: _relative_difference(values[4], values[6])
        for name, values in actions.items()
    }
    additivity = {
        str(order): _relative_difference(
            actions["sum"][order],
            actions["first"][order] + actions["second"][order],
        )
        for order in JVP_ORDERS
    }
    homogeneity = {
        str(order): _relative_difference(
            actions["homogeneous"][order],
            HOMOGENEITY_FACTOR * actions["first"][order],
        )
        for order in JVP_ORDERS
    }
    maximum_order = max(order_differences.values())
    maximum_additivity = max(additivity.values())
    maximum_homogeneity = max(homogeneity.values())
    report = {
        "finite_difference_step": JVP_STEP,
        "orders": JVP_ORDERS,
        "order_differences": order_differences,
        "additivity_defects": additivity,
        "homogeneity_defects": homogeneity,
        "maximum_order_difference": maximum_order,
        "maximum_additivity_defect": maximum_additivity,
        "maximum_homogeneity_defect": maximum_homogeneity,
        "passed": bool(
            maximum_order <= MAXIMUM_JVP_ORDER_DIFFERENCE
            and maximum_additivity <= MAXIMUM_JVP_ADDITIVITY_DEFECT
            and maximum_homogeneity <= MAXIMUM_JVP_HOMOGENEITY_DEFECT
        ),
    }
    arrays = {"conservation_row_scales": scales}
    for name, direction in directions.items():
        arrays[f"direction_{name}"] = np.asarray(direction, dtype=float)
        for order in JVP_ORDERS:
            arrays[f"jvp_{name}_order{order}"] = np.asarray(
                actions[name][order],
                dtype=float,
            )
    return report, arrays


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent = _load_parent_evidence()
    context, charts = _context_and_charts()
    increment = _increment(charts)
    new = charts + increment
    midpoint = charts + PATH_SUBDIVISION_FRACTION * increment

    zero = causal_five_field_monolithic_storage_increment(
        context,
        charts,
        charts,
        temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
        reconstruction_directional_step=(
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
    )
    forward = causal_five_field_monolithic_storage_increment(
        context,
        charts,
        new,
        temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
        reconstruction_directional_step=(
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
    )
    backward = causal_five_field_monolithic_storage_increment(
        context,
        new,
        charts,
        temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
        reconstruction_directional_step=(
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
    )
    first_segment = causal_five_field_monolithic_storage_increment(
        context,
        charts,
        midpoint,
        temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
        reconstruction_directional_step=(
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
    )
    second_segment = causal_five_field_monolithic_storage_increment(
        context,
        midpoint,
        new,
        temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
        reconstruction_directional_step=(
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
    )
    reversal_defect = _relative_difference(
        forward.total_storage_increment,
        -backward.total_storage_increment,
    )
    additivity_defect = _relative_difference(
        forward.total_storage_increment,
        (
            first_segment.total_storage_increment
            + second_segment.total_storage_increment
        ),
    )
    storage_reports = (zero, forward, backward, first_segment, second_segment)
    maximum_mapped_closure = max(
        item.maximum_mapped_path_closure_defect
        for item in storage_reports
    )
    maximum_factor_change = max(
        item.maximum_path_reconstruction_factor_change
        for item in storage_reports
    )
    minimum_factor = min(
        item.minimum_path_reconstruction_factor
        for item in storage_reports
    )

    integrability = (
        causal_five_field_temporal_storage_integrability_audit(
            context,
            float(context.grid.centers[0]),
            charts[0],
        )
    )
    evaluation = evaluate_causal_five_field_monolithic_backward_euler(
        charts,
        new,
        TIMESTEP_SECONDS,
        context,
        temporal_quadrature_order=TEMPORAL_QUADRATURE_ORDER,
        reconstruction_directional_step=(
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
        path_quadrature_order=PATH_QUADRATURE_ORDER,
    )
    faces = np.asarray(
        evaluation.stationary_ledger.interfaces
        .candidate_shared_face_fluxes_over_c,
        dtype=float,
    )
    flux_telescope = _relative_difference(
        evaluation.conservative_transport_rows,
        faces[1:] - faces[:-1],
    )
    jvp, jvp_arrays = _jvp_report(context, charts)

    temporal_product_detected = bool(
        integrability.relative_vertical_exterior_derivative
        >= MINIMUM_VERTICAL_EXTERIOR_DERIVATIVE
        and integrability.relative_complete_exterior_derivative
        >= MINIMUM_COMPLETE_EXTERIOR_DERIVATIVE
        and integrability.relative_loop_to_vertical_path
        >= MINIMUM_TEMPORAL_LOOP_DEFECT
    )
    assembly_passed = bool(
        np.array_equal(
            zero.total_storage_increment,
            np.zeros_like(zero.total_storage_increment),
        )
        and maximum_mapped_closure
        <= MAXIMUM_MAPPED_PATH_CLOSURE_DEFECT
        and reversal_defect <= MAXIMUM_TEMPORAL_REVERSAL_DEFECT
        and additivity_defect
        <= MAXIMUM_COLLINEAR_PATH_ADDITIVITY_DEFECT
        and maximum_factor_change
        <= MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        and minimum_factor > 0.0
        and forward.one_flux_reconstruction_for_space_and_storage
        and forward.mapped_storage_is_exact_endpoint_increment
        and forward.responsive_height_is_nonconservative_temporal_product
        and evaluation.maximum_block_ledger_defect
        <= MAXIMUM_BLOCK_LEDGER_DEFECT
        and flux_telescope <= MAXIMUM_SHARED_FLUX_TELESCOPE_DEFECT
        and evaluation.maximum_center_broken_path_adjustment
        <= MAXIMUM_CENTER_BROKEN_PATH_ADJUSTMENT
        and evaluation.incoming_excision_characteristics == 0
        and evaluation.stationary_ledger.source_double_count_defect == 0.0
        and not evaluation.uses_production_generator
        and not evaluation.uses_production_anchor_storage_derivative
        and np.all(np.isfinite(evaluation.residual_rows))
    )
    passed = bool(
        assembly_passed
        and temporal_product_detected
        and jvp["passed"]
    )
    classification = (
        "monolithic_descriptor_path_assembly_certified_"
        "manufactured_preflight_authorized"
        if passed
        else "monolithic_descriptor_path_preflight_failed"
    )

    decisive = {
        "base_primitive_charts": charts,
        "declared_increment": increment,
        "mapped_endpoint_increment": forward.mapped_endpoint_increment,
        "mapped_path_increment": forward.mapped_path_increment,
        "responsive_height_path_increment": (
            forward.responsive_height_path_increment
        ),
        "forward_total_storage_increment": (
            forward.total_storage_increment
        ),
        "backward_total_storage_increment": (
            backward.total_storage_increment
        ),
        "subdivided_total_storage_increment": (
            first_segment.total_storage_increment
            + second_segment.total_storage_increment
        ),
        "vertical_exterior_derivative": (
            integrability.vertical_exterior_derivative
        ),
        "complete_exterior_derivative": (
            integrability.complete_exterior_derivative
        ),
        "first_path_vertical_increment": (
            integrability.first_path_vertical_increment
        ),
        "second_path_vertical_increment": (
            integrability.second_path_vertical_increment
        ),
        "loop_vertical_increment": integrability.loop_vertical_increment,
        "candidate_shared_face_fluxes_over_c": faces,
        "conservative_transport_rows": (
            evaluation.conservative_transport_rows
        ),
        "center_broken_path_adjustment_rows": (
            evaluation.center_broken_path_adjustment_rows
        ),
        "complete_residual_rows": evaluation.residual_rows,
        **jvp_arrays,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "maximum_mapped_path_closure_defect": (
            MAXIMUM_MAPPED_PATH_CLOSURE_DEFECT
        ),
        "maximum_temporal_reversal_defect": (
            MAXIMUM_TEMPORAL_REVERSAL_DEFECT
        ),
        "maximum_collinear_path_additivity_defect": (
            MAXIMUM_COLLINEAR_PATH_ADDITIVITY_DEFECT
        ),
        "maximum_reconstruction_factor_change": (
            MAXIMUM_RECONSTRUCTION_FACTOR_CHANGE
        ),
        "maximum_block_ledger_defect": MAXIMUM_BLOCK_LEDGER_DEFECT,
        "maximum_shared_flux_telescope_defect": (
            MAXIMUM_SHARED_FLUX_TELESCOPE_DEFECT
        ),
        "maximum_center_broken_path_adjustment": (
            MAXIMUM_CENTER_BROKEN_PATH_ADJUSTMENT
        ),
        "maximum_jvp_order_difference": MAXIMUM_JVP_ORDER_DIFFERENCE,
        "maximum_jvp_additivity_defect": MAXIMUM_JVP_ADDITIVITY_DEFECT,
        "maximum_jvp_homogeneity_defect": (
            MAXIMUM_JVP_HOMOGENEITY_DEFECT
        ),
        "minimum_vertical_exterior_derivative": (
            MINIMUM_VERTICAL_EXTERIOR_DERIVATIVE
        ),
        "minimum_complete_exterior_derivative": (
            MINIMUM_COMPLETE_EXTERIOR_DERIVATIVE
        ),
        "minimum_temporal_loop_defect": MINIMUM_TEMPORAL_LOOP_DEFECT,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "n_cells": N_CELLS,
        "timestep_seconds": TIMESTEP_SECONDS,
        "temporal_quadrature_order": TEMPORAL_QUADRATURE_ORDER,
        "temporal_reconstruction_directional_step": (
            TEMPORAL_RECONSTRUCTION_DIRECTIONAL_STEP
        ),
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "path_subdivision_fraction": PATH_SUBDIVISION_FRACTION,
        "jvp_step": JVP_STEP,
        "jvp_orders": JVP_ORDERS,
        "homogeneity_factor": HOMOGENEITY_FACTOR,
        "reconstruction": "quadratic_admissible",
        "boundary_trace": "plm_one_sided",
        "cell_storage_quadrature": "gauss_legendre_4",
        "outer_boundary": "frozen_exterior_rusanov",
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "passed": passed,
        "assembly_passed": assembly_passed,
        "temporal_product_nonexactness_detected": (
            temporal_product_detected
        ),
        "parent_wp10c9d5c1_summary_path": _relative(PARENT_SUMMARY),
        "parent_wp10c9d5c1_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_branch_d_decision_preserved": bool(
            parent["monolithic_replacement_authorized"]
        ),
        "maximum_mapped_path_closure_defect": maximum_mapped_closure,
        "temporal_reversal_defect": reversal_defect,
        "collinear_path_additivity_defect": additivity_defect,
        "maximum_path_reconstruction_factor_change": (
            maximum_factor_change
        ),
        "minimum_path_reconstruction_factor": minimum_factor,
        "relative_vertical_exterior_derivative": (
            integrability.relative_vertical_exterior_derivative
        ),
        "relative_complete_exterior_derivative": (
            integrability.relative_complete_exterior_derivative
        ),
        "relative_temporal_loop_defect": (
            integrability.relative_loop_to_vertical_path
        ),
        "temporal_loop_fields": integrability.loop_fields,
        "temporal_loop_amplitude": integrability.loop_amplitude,
        "strict_endpoint_storage_potential_authorized": False,
        "declared_temporal_path_product_required": True,
        "maximum_block_ledger_defect": (
            evaluation.maximum_block_ledger_defect
        ),
        "shared_flux_telescope_defect": flux_telescope,
        "maximum_center_broken_path_adjustment": (
            evaluation.maximum_center_broken_path_adjustment
        ),
        "incoming_excision_characteristics": (
            evaluation.incoming_excision_characteristics
        ),
        "source_double_count_defect": (
            evaluation.stationary_ledger.source_double_count_defect
        ),
        "uses_production_generator": (
            evaluation.uses_production_generator
        ),
        "uses_production_anchor_storage_derivative": (
            evaluation.uses_production_anchor_storage_derivative
        ),
        "jvp_report": jvp,
        "manufactured_equilibrium_and_wave_preflight_authorized": passed,
        "physical_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_monolithic_dae_preflight_wp10c9d6a.py"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "parent_canonical_hashes": {
            _relative(PARENT_SUMMARY): _sha256(PARENT_SUMMARY),
            _relative(PARENT_ARRAYS): _sha256(PARENT_ARRAYS),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "authorization_status": (
            "manufactured_equilibrium_and_wave_preflight"
            if passed
            else "none"
        ),
        "establishes": (
            "One production-neutral monolithic descriptor-path residual, "
            "its temporal path contract, complete ledger, outgoing excision "
            "contract, and high-order directional differentiability on the "
            "declared regression context."
        ),
        "does_not_establish": (
            "An endpoint-only responsive-height storage potential, "
            "manufactured-wave convergence, physical-export convergence, "
            "nonlinear physical evolution, fixed-Q closure, or reduced "
            "evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    print(json.dumps(_plain(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
