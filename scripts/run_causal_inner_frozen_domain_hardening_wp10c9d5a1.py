#!/usr/bin/env python3
"""Run the WP10c9d5a1 inner-domain frozen-derivative audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_coordinate_principal_basis,
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_reduced_jacobian_pattern,
    causal_five_field_reconstruct_face_charts,
    causal_radial_dense_colored_audit,
    causal_radial_jvp_spatial_attribution,
    causal_radial_jvp_step_sweep,
    causal_radial_one_sided_jvp_sweep,
    causal_radial_project_jvp_actions,
    causal_radial_volume_weighted_scaled_direction,
)


ROOT = Path(__file__).resolve().parents[1]
THIS_RUNNER = (
    "scripts/run_causal_inner_frozen_domain_hardening_wp10c9d5a1.py"
)
PARENT_RUNNER = (
    ROOT
    / "scripts/run_causal_inner_frozen_hardening_wp10c9d5a.py"
)
PARENT_CANONICAL = (
    ROOT
    / "results/canonical/causal_inner_frozen_hardening_wp10c9d5a"
)
PARENT_DECISIVE_ARRAYS = PARENT_CANONICAL / "decisive_arrays.npz"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_domain_hardening_wp10c9d5a1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5a1"
ANALYZED_BASE_COMMIT = "155e18339076fd2b27d419173b92e1d5d608963b"
ANALYZED_BASE_PARENT = "038ba35659e76aff0605fffa5fb457e99362063d"
ANALYZED_BASE_TREE = "ec9432404f3e28e8505628b3e770923c254b56cf"
EMBEDDED_LABEL = "N128_exterior_N128_inner_c48"
N_FIELDS = 5
INNER_RADIUS_OVER_RG = 5.0
STENCIL_HALO_CELLS = 3
GENERATOR_RELATIVE_STEP = 4.0e-5
JVP_STEPS = (
    5.0e-6,
    1.0e-5,
    2.0e-5,
    4.0e-5,
    8.0e-5,
    1.6e-4,
    3.2e-4,
)
BRANCH_STEPS = (-4.0e-5, -2.0e-5, 0.0, 2.0e-5, 4.0e-5)
HELD_OUT_SEEDS = (91051, 91052, 91053, 91054)
STATIONARY_SPEED_TOLERANCE = 1.0e-12
MAXIMUM_DENSE_COLORED_DEFECT = 1.0e-10
MAXIMUM_OFF_PATTERN_ENTRY = 1.0e-10
MAXIMUM_SELECTED_JVP_DEFECT = 5.0e-5
MAXIMUM_PLATEAU_ADJACENT_CHANGE = 2.0e-5
MAXIMUM_REPLAY_BASE_DEFECT = 1.0e-13
MAXIMUM_BRANCH_SIGN_CHANGES = 0
MAXIMUM_ADMISSIBILITY_CHANGE = 1.0e-12
MAXIMUM_OUTER_BRANCH_CHANGES = 0

IMPLEMENTATION_SOURCES = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_domain_hardening.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_hardening.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_fluctuation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_characteristic_dissipation.py",
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    THIS_RUNNER,
    "tests/test_causal_inner_radial_hardening.py",
    "tests/test_causal_inner_frozen_domain_hardening_wp10c9d5a1.py",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


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


def _parent_module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5a_parent_runner",
        PARENT_RUNNER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("WP10c9d5a parent runner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    expected = (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    )
    if (commit, parent, tree) != expected:
        raise RuntimeError("WP10c9d5a1 analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _region_rows(first_cell: int, stop_cell: int) -> np.ndarray:
    return np.arange(
        N_FIELDS * int(first_cell),
        N_FIELDS * int(stop_cell),
        dtype=int,
    )


def _plateau_brackets_selected(audit) -> tuple[float, float]:
    selected = int(audit.selected_step_index)
    if selected <= 0 or selected >= audit.adjacent_relative_changes.size:
        raise ValueError("selected step lacks two adjacent plateau intervals")
    return (
        float(audit.adjacent_relative_changes[selected - 1]),
        float(audit.adjacent_relative_changes[selected]),
    )


def _projected_report(audit) -> dict:
    lower_change, upper_change = _plateau_brackets_selected(audit)
    passed = bool(
        audit.selected_matrix_relative_defect
        <= MAXIMUM_SELECTED_JVP_DEFECT
        and lower_change <= MAXIMUM_PLATEAU_ADJACENT_CHANGE
        and upper_change <= MAXIMUM_PLATEAU_ADJACENT_CHANGE
    )
    return {
        "selected_matrix_relative_defect": (
            audit.selected_matrix_relative_defect
        ),
        "lower_selected_adjacent_change": lower_change,
        "upper_selected_adjacent_change": upper_change,
        "passed": passed,
    }


def _step_key(step: float) -> str:
    return f"{float(step):+.1e}".replace("+", "p").replace("-", "m")


def _support_bump(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    clipped = np.clip(values, 0.0, 1.0)
    bump = np.sin(np.pi * clipped) ** 4
    return np.where((values > 0.0) & (values < 1.0), bump, 0.0)


def _held_out_direction(
    centers_over_rg: np.ndarray,
    cell_measures: np.ndarray,
    *,
    seed: int,
    kind: str,
) -> np.ndarray:
    radius = np.asarray(centers_over_rg, dtype=float)
    rng = np.random.default_rng(int(seed))
    if kind == "inner_smooth":
        lower = 1.8
        upper = INNER_RADIUS_OVER_RG
    elif kind == "near_excision":
        lower = 1.8
        upper = 3.5
    else:
        raise ValueError("unknown held-out direction kind")
    coordinate = (
        np.log(radius) - np.log(lower)
    ) / (np.log(upper) - np.log(lower))
    bump = _support_bump(coordinate)
    values = np.zeros((radius.size, N_FIELDS), dtype=float)
    for field in range(N_FIELDS):
        coefficients = rng.standard_normal((2, 3))
        profile = np.zeros(radius.size, dtype=float)
        for mode in range(1, 4):
            profile += (
                coefficients[0, mode - 1]
                * np.sin(mode * np.pi * coordinate)
                + coefficients[1, mode - 1]
                * np.cos(mode * np.pi * coordinate)
            )
        values[:, field] = bump * profile
    return causal_radial_volume_weighted_scaled_direction(
        values,
        cell_measures,
    ).ravel()


def _basis_rows(context, reconstruction) -> list:
    rows = [
        causal_five_field_coordinate_principal_basis(
            context,
            float(context.grid.edges[0]),
            reconstruction.right_face_charts[0],
        )
    ]
    n_cells = int(context.grid.centers.size)
    for face in range(1, n_cells):
        midpoint = 0.5 * (
            reconstruction.left_face_charts[face]
            + reconstruction.right_face_charts[face]
        )
        rows.append(
            causal_five_field_coordinate_principal_basis(
                context,
                float(context.grid.edges[face]),
                midpoint,
            )
        )
    if context.outer_boundary_flux_mode == "frozen_exterior_rusanov":
        exterior = np.asarray(
            context.outer_boundary_frozen_exterior_chart,
            dtype=float,
        )
        midpoint = 0.5 * (
            reconstruction.left_face_charts[-1] + exterior
        )
        rows.append(
            causal_five_field_coordinate_principal_basis(
                context,
                float(context.grid.edges[-1]),
                midpoint,
            )
        )
    return rows


def _branch_fingerprint(context, charts: np.ndarray) -> tuple[dict, dict]:
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
        purpose="flux",
    )
    bases = _basis_rows(context, reconstruction)
    speeds = np.asarray(
        [basis.numerical_speeds_over_c for basis in bases],
        dtype=float,
    )
    analytic = np.asarray(
        [basis.analytic_speeds_over_c for basis in bases],
        dtype=float,
    )
    conditions = np.asarray(
        [basis.descriptor_condition_number for basis in bases],
        dtype=float,
    )
    pairwise_gaps = []
    for row in speeds:
        pairwise_gaps.extend(
            abs(float(row[left] - row[right]))
            for left in range(N_FIELDS)
            for right in range(left)
        )
    ledger = causal_five_field_radial_candidate_ledger(
        context,
        charts,
        quadrature_order=6,
        stationary_speed_tolerance=STATIONARY_SPEED_TOLERANCE,
    )
    interfaces = ledger.interfaces
    summary = {
        "minimum_absolute_speed_over_c": float(
            np.min(np.abs(speeds))
        ),
        "minimum_pairwise_speed_gap_over_c": float(min(pairwise_gaps)),
        "maximum_analytic_speed_defect_over_c": float(
            np.max(np.abs(speeds - analytic))
        ),
        "maximum_descriptor_condition_number": float(
            np.max(conditions)
        ),
        "maximum_condition_face": int(np.argmax(conditions)),
        "minimum_admissibility_factor": float(
            np.min(reconstruction.admissibility_factors)
        ),
        "outer_boundary_choked": bool(
            interfaces.outer_boundary_choked
        ),
        "incoming_excision_characteristics": int(
            interfaces.incoming_excision_characteristics
        ),
        "outer_incoming_characteristics": int(
            interfaces.outer_incoming_characteristics
        ),
    }
    arrays = {
        "speeds": speeds,
        "analytic_speeds": analytic,
        "descriptor_conditions": conditions,
        "admissibility_factors": np.asarray(
            reconstruction.admissibility_factors,
            dtype=float,
        ),
    }
    return summary, arrays


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


def _environment() -> dict:
    blas = np.__config__.show(mode="dicts")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_lapack": blas,
        "float64_epsilon": np.finfo(np.float64).eps,
    }


def run(*, skip_dense: bool = False) -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent = _parent_module()
    replay_payload, replay_arrays = parent._load_replay_inputs()
    if EMBEDDED_LABEL not in replay_payload["contexts"]:
        raise RuntimeError("binding embedded replay context is absent")
    prefix = f"{EMBEDDED_LABEL}__"
    context = parent._context_from_payload(
        replay_payload["contexts"][EMBEDDED_LABEL],
        replay_arrays,
    )
    base = np.asarray(
        replay_arrays[prefix + "base_primitives"],
        dtype=float,
    )
    column_scales = np.asarray(
        replay_arrays[prefix + "primitive_column_scales"],
        dtype=float,
    )
    row_scales = np.asarray(
        replay_arrays[prefix + "conservation_row_scales"],
        dtype=float,
    )
    colored = np.asarray(
        replay_arrays[prefix + "colored_stationary_delta"],
        dtype=float,
    )
    function = parent._scaled_delta_function(
        context,
        base,
        column_scales,
        row_scales,
    )
    zero = np.zeros(base.size, dtype=float)
    replay_base = function(zero)
    expected_base = np.asarray(
        replay_arrays[prefix + "base_scaled_delta"],
        dtype=float,
    )
    replay_scale = max(
        float(np.max(np.abs(replay_base))),
        float(np.max(np.abs(expected_base))),
        np.finfo(float).tiny,
    )
    replay_defect = float(
        np.max(np.abs(replay_base - expected_base)) / replay_scale
    )

    centers_over_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / float(context.grid.gravitational_radius)
    )
    inner_cells = int(np.sum(centers_over_rg <= INNER_RADIUS_OVER_RG))
    if inner_cells < 3 or inner_cells >= base.shape[0]:
        raise RuntimeError("declared inner-domain cell count is invalid")
    halo_cells = min(
        base.shape[0],
        inner_cells + STENCIL_HALO_CELLS,
    )
    inner_rows = _region_rows(0, inner_cells)
    halo_rows = _region_rows(0, halo_cells)
    outer_three_rows = _region_rows(base.shape[0] - 3, base.shape[0])
    outer_last_rows = _region_rows(base.shape[0] - 1, base.shape[0])

    with np.load(PARENT_DECISIVE_ARRAYS, allow_pickle=False) as archive:
        original_direction = np.asarray(
            archive[prefix + "random_0__direction"],
            dtype=float,
        )
        original_direct = np.asarray(
            archive[prefix + "random_0__direct_actions"],
            dtype=float,
        )
        original_matrix = np.asarray(
            archive[prefix + "random_0__matrix_action"],
            dtype=float,
        )
    regions = {
        "inner_through_5rg": inner_rows,
        "inner_plus_stencil_halo": halo_rows,
        "outer_three_cells": outer_three_rows,
        "outermost_cell": outer_last_rows,
        "complete_grid": np.arange(base.size, dtype=int),
    }
    original_region_audits = {}
    decisive: dict[str, np.ndarray] = {
        "jvp_steps": np.asarray(JVP_STEPS, dtype=float),
        "original_random_0_direction": original_direction,
        "original_random_0_direct_actions": original_direct,
        "original_random_0_matrix_action": original_matrix,
        "grid_centers_over_rg": centers_over_rg,
    }
    for name, rows in regions.items():
        audit = causal_radial_project_jvp_actions(
            original_direct,
            original_matrix,
            JVP_STEPS,
            rows,
            selected_step=GENERATOR_RELATIVE_STEP,
        )
        original_region_audits[name] = _projected_report(audit)
        decisive[f"original_random_0__{name}__matrix_defects"] = (
            audit.matrix_relative_defects
        )
        decisive[f"original_random_0__{name}__adjacent_changes"] = (
            audit.adjacent_relative_changes
        )
    attribution = causal_radial_jvp_spatial_attribution(
        original_direct,
        JVP_STEPS,
        n_fields=N_FIELDS,
    )
    decisive["original_random_0__cell_squared_fractions"] = (
        attribution.cell_squared_fractions
    )
    selected_interval = int(
        np.flatnonzero(
            np.asarray(JVP_STEPS[1:]) == GENERATOR_RELATIVE_STEP
        )[0]
    )
    selected_fractions = attribution.cell_squared_fractions[
        selected_interval
    ]
    spatial_localization = {
        "selected_interval_lower_step": JVP_STEPS[selected_interval],
        "selected_interval_upper_step": JVP_STEPS[selected_interval + 1],
        "dominant_cell": int(
            attribution.dominant_cells[selected_interval]
        ),
        "dominant_cell_radius_over_rg": float(
            centers_over_rg[
                attribution.dominant_cells[selected_interval]
            ]
        ),
        "outermost_cell_squared_fraction": float(
            selected_fractions[-1]
        ),
        "outer_three_cells_squared_fraction": float(
            np.sum(selected_fractions[-3:])
        ),
        "inner_through_5rg_squared_fraction": float(
            np.sum(selected_fractions[:inner_cells])
        ),
        "first_three_cells_squared_fraction": float(
            np.sum(selected_fractions[:3])
        ),
    }

    print("WP10c9d5a1: one-sided failed-direction sweep", flush=True)
    one_sided = causal_radial_one_sided_jvp_sweep(
        function,
        zero,
        original_direction,
        JVP_STEPS,
    )
    decisive["original_random_0__forward_actions"] = (
        one_sided.forward_actions
    )
    decisive["original_random_0__backward_actions"] = (
        one_sided.backward_actions
    )
    decisive["original_random_0__one_sided_mismatches"] = (
        one_sided.one_sided_relative_mismatches
    )

    branch_summaries = {}
    branch_arrays = {}
    base_flat = base.ravel()
    for step in BRANCH_STEPS:
        print(f"WP10c9d5a1: branch fingerprint {step:+.1e}", flush=True)
        charts = (
            base_flat + column_scales * (step * original_direction)
        ).reshape(base.shape)
        summary, arrays = _branch_fingerprint(context, charts)
        key = _step_key(step)
        branch_summaries[key] = summary
        for name, values in arrays.items():
            branch_arrays[f"branch_{key}__{name}"] = values
    decisive.update(branch_arrays)
    base_key = _step_key(0.0)
    base_branch = branch_arrays[f"branch_{base_key}__speeds"]
    base_admissibility = branch_arrays[
        f"branch_{base_key}__admissibility_factors"
    ]
    sign_changes = 0
    maximum_admissibility_change = 0.0
    outer_branch_changes = 0
    base_summary = branch_summaries[base_key]
    for key, summary in branch_summaries.items():
        speeds = branch_arrays[f"branch_{key}__speeds"]
        sign_changes = max(
            sign_changes,
            int(
                np.sum(
                    np.sign(speeds) != np.sign(base_branch)
                )
            ),
        )
        maximum_admissibility_change = max(
            maximum_admissibility_change,
            float(
                np.max(
                    np.abs(
                        branch_arrays[
                            f"branch_{key}__admissibility_factors"
                        ]
                        - base_admissibility
                    )
                )
            ),
        )
        outer_branch_changes = max(
            outer_branch_changes,
            int(
                summary["outer_boundary_choked"]
                != base_summary["outer_boundary_choked"]
            )
            + int(
                summary["outer_incoming_characteristics"]
                != base_summary["outer_incoming_characteristics"]
            ),
        )
    branch_passed = bool(
        sign_changes <= MAXIMUM_BRANCH_SIGN_CHANGES
        and maximum_admissibility_change
        <= MAXIMUM_ADMISSIBILITY_CHANGE
        and outer_branch_changes <= MAXIMUM_OUTER_BRANCH_CHANGES
    )

    held_out_reports = {}
    held_out_specs = (
        ("inner_smooth_0", HELD_OUT_SEEDS[0], "inner_smooth"),
        ("inner_smooth_1", HELD_OUT_SEEDS[1], "inner_smooth"),
        ("near_excision_0", HELD_OUT_SEEDS[2], "near_excision"),
        ("near_excision_1", HELD_OUT_SEEDS[3], "near_excision"),
    )
    for name, seed, kind in held_out_specs:
        print(f"WP10c9d5a1: held-out JVP {name}", flush=True)
        direction = _held_out_direction(
            centers_over_rg,
            np.asarray(context.grid.cell_measures, dtype=float),
            seed=seed,
            kind=kind,
        )
        sweep = causal_radial_jvp_step_sweep(
            function,
            zero,
            colored,
            direction,
            JVP_STEPS,
            selected_step=GENERATOR_RELATIVE_STEP,
        )
        inner_audit = causal_radial_project_jvp_actions(
            sweep.direct_actions,
            sweep.matrix_action,
            JVP_STEPS,
            inner_rows,
            selected_step=GENERATOR_RELATIVE_STEP,
        )
        halo_audit = causal_radial_project_jvp_actions(
            sweep.direct_actions,
            sweep.matrix_action,
            JVP_STEPS,
            halo_rows,
            selected_step=GENERATOR_RELATIVE_STEP,
        )
        inner_report = _projected_report(inner_audit)
        halo_report = _projected_report(halo_audit)
        held_out_reports[name] = {
            "seed": seed,
            "kind": kind,
            "maximum_absolute_scaled_component": float(
                np.max(np.abs(direction))
            ),
            "inner_through_5rg": inner_report,
            "inner_plus_stencil_halo": halo_report,
            "passed": bool(
                inner_report["passed"] and halo_report["passed"]
            ),
        }
        decisive[f"{name}__direction"] = direction
        decisive[f"{name}__direct_actions"] = sweep.direct_actions
        decisive[f"{name}__matrix_action"] = sweep.matrix_action
        decisive[f"{name}__inner_matrix_defects"] = (
            inner_audit.matrix_relative_defects
        )
        decisive[f"{name}__inner_adjacent_changes"] = (
            inner_audit.adjacent_relative_changes
        )
        decisive[f"{name}__halo_matrix_defects"] = (
            halo_audit.matrix_relative_defects
        )
        decisive[f"{name}__halo_adjacent_changes"] = (
            halo_audit.adjacent_relative_changes
        )

    dense_report = {
        "skipped": bool(skip_dense),
        "selected_column_count": int(N_FIELDS * halo_cells),
        "maximum_dense_colored_relative_defect": None,
        "maximum_off_pattern_relative_entry": None,
        "passed": False,
    }
    if not skip_dense:
        print(
            "WP10c9d5a1: dense/colored inner-domain-plus-halo columns",
            flush=True,
        )
        pattern = causal_five_field_radial_reduced_jacobian_pattern(
            int(base.shape[0])
        )
        selected_columns = np.arange(
            N_FIELDS * halo_cells,
            dtype=int,
        )
        dense = causal_radial_dense_colored_audit(
            function,
            zero,
            colored,
            pattern,
            selected_columns,
            finite_difference_step=GENERATOR_RELATIVE_STEP,
        )
        dense_report = {
            "skipped": False,
            "selected_column_count": int(selected_columns.size),
            "maximum_dense_colored_relative_defect": (
                dense.maximum_relative_defect
            ),
            "maximum_off_pattern_relative_entry": (
                dense.maximum_off_pattern_relative_entry
            ),
            "maximum_per_column_relative_defect": float(
                np.max(dense.per_column_relative_defects)
            ),
            "passed": bool(
                dense.maximum_relative_defect
                <= MAXIMUM_DENSE_COLORED_DEFECT
                and dense.maximum_off_pattern_relative_entry
                <= MAXIMUM_OFF_PATTERN_ENTRY
            ),
        }
        decisive["dense_selected_columns"] = dense.selected_columns
        decisive["dense_columns"] = dense.dense_columns
        decisive["colored_columns"] = dense.colored_columns
        decisive["dense_per_column_defects"] = (
            dense.per_column_relative_defects
        )

    original_inner_passed = bool(
        original_region_audits["inner_through_5rg"]["passed"]
        and original_region_audits["inner_plus_stencil_halo"]["passed"]
    )
    held_out_passed = bool(
        all(report["passed"] for report in held_out_reports.values())
    )
    domain_scoped_passed = bool(
        not skip_dense
        and replay_defect <= MAXIMUM_REPLAY_BASE_DEFECT
        and original_inner_passed
        and held_out_passed
        and branch_passed
        and dense_report["passed"]
    )
    classification = (
        "inner_domain_derivative_certified_cache_first_localization_authorized_"
        "global_hardening_still_failed"
        if domain_scoped_passed
        else
        "inner_domain_derivative_not_certified_dynamic_localization_blocked"
    )

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "embedded_label": EMBEDDED_LABEL,
        "inner_radius_over_rg": INNER_RADIUS_OVER_RG,
        "stencil_halo_cells": STENCIL_HALO_CELLS,
        "generator_relative_step": GENERATOR_RELATIVE_STEP,
        "jvp_steps": JVP_STEPS,
        "branch_steps": BRANCH_STEPS,
        "held_out_seeds": HELD_OUT_SEEDS,
        "gates": {
            "maximum_dense_colored_defect": (
                MAXIMUM_DENSE_COLORED_DEFECT
            ),
            "maximum_off_pattern_entry": MAXIMUM_OFF_PATTERN_ENTRY,
            "maximum_selected_jvp_defect": (
                MAXIMUM_SELECTED_JVP_DEFECT
            ),
            "maximum_plateau_adjacent_change": (
                MAXIMUM_PLATEAU_ADJACENT_CHANGE
            ),
            "maximum_replay_base_defect": MAXIMUM_REPLAY_BASE_DEFECT,
            "maximum_branch_sign_changes": (
                MAXIMUM_BRANCH_SIGN_CHANGES
            ),
            "maximum_admissibility_change": (
                MAXIMUM_ADMISSIBILITY_CHANGE
            ),
            "maximum_outer_branch_changes": (
                MAXIMUM_OUTER_BRANCH_CHANGES
            ),
        },
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        **identity,
        "parent_wp10c9d5a_classification_preserved": True,
        "parent_wp10c9d5a_remains_globally_rejected": True,
        "parent_decisive_arrays_path": _relative(
            PARENT_DECISIVE_ARRAYS
        ),
        "parent_decisive_arrays_sha256": _sha256(
            PARENT_DECISIVE_ARRAYS
        ),
        "embedded_label": EMBEDDED_LABEL,
        "n_cells": int(base.shape[0]),
        "inner_cell_count": inner_cells,
        "inner_last_center_over_rg": float(
            centers_over_rg[inner_cells - 1]
        ),
        "halo_cell_count": halo_cells,
        "halo_last_center_over_rg": float(
            centers_over_rg[halo_cells - 1]
        ),
        "replay_base_relative_defect": replay_defect,
        "original_random_0_region_audits": original_region_audits,
        "spatial_localization": spatial_localization,
        "one_sided_relative_mismatches": (
            one_sided.one_sided_relative_mismatches
        ),
        "branch_summaries": branch_summaries,
        "branch_comparison": {
            "maximum_sign_changes": sign_changes,
            "maximum_admissibility_change": (
                maximum_admissibility_change
            ),
            "maximum_outer_branch_changes": outer_branch_changes,
            "passed": branch_passed,
        },
        "held_out_reports": held_out_reports,
        "dense_report": dense_report,
        "domain_scoped_derivative_passed": domain_scoped_passed,
        "wp10c9d5b_inner_localization_authorized": (
            domain_scoped_passed
        ),
        "global_frozen_candidate_recertification_authorized": False,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
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
            "scripts/run_causal_inner_frozen_domain_hardening_"
            "wp10c9d5a1.py"
        ),
        "method_scope": (
            "DOMAIN-SCOPED FROZEN DERIVATIVE HARDENING / "
            "PRODUCTION NEUTRAL"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "scientific_status": (
            "DIAGNOSTIC ONLY"
            if domain_scoped_passed
            else "REJECTED"
        ),
        "authorization_status": (
            "CERTIFIED FOR INNER LOCALIZATION ONLY"
            if domain_scoped_passed
            else "LOCALIZATION NOT AUTHORIZED"
        ),
        "source_input_hashes": {
            _relative(PARENT_DECISIVE_ARRAYS): _sha256(
                PARENT_DECISIVE_ARRAYS
            ),
            _relative(parent.REPLAY_INPUTS): _sha256(
                parent.REPLAY_INPUTS
            ),
            _relative(parent.REPLAY_CONTEXTS): _sha256(
                parent.REPLAY_CONTEXTS
            ),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "establishes": (
            "Whether the stored frozen derivative is stable on residual "
            "rows through 5 rg and the declared three-cell stencil halo, "
            "including held-out continuum directions and branch fingerprints."
        ),
        "does_not_establish": (
            "Global Jacobian hardening, a repaired physical operator, "
            "nonlinear convergence, fixed-Q closure, or reduced evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()
    result = run(skip_dense=args.skip_dense)
    print(json.dumps(_plain(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
