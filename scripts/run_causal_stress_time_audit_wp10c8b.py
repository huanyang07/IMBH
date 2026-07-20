"""Extend the certified causal reference to one stress-time rung for WP10c8b."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import run_causal_characteristic_extension_wp10c7l as wp10c7l
import run_causal_n128_reference_wp10c7n as wp10c7n
import run_causal_slow_mode_audit_wp10c8a as wp10c8a
import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveBDF2Restart,
    audit_causal_five_field_state_gates,
    causal_five_field_adaptive_bdf2_restarts_equal,
    causal_five_field_bdf_physical_ledger_from_restart,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_cell_states,
    causal_five_field_local_timescale_audit,
    causal_five_field_profile_fields,
    causal_five_field_state_summary,
    causal_restrict_cell_averages,
    causal_spatial_contraction_order,
    evaluate_causal_five_field_dae,
    evolve_causal_five_field_adaptive_bdf2_campaign,
    load_causal_five_field_adaptive_bdf2_restart,
    save_causal_five_field_adaptive_bdf2_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "eb0b161c0736ba971b3b16c0c80db07e262bc4a9"
WP10C8A_OUTPUT = (
    ROOT / "outputs/tables/causal_slow_mode_audit_wp10c8a.json"
)
WP10C7N_OUTPUT = (
    ROOT / "outputs/tables/causal_n128_reference_wp10c7n.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8b"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_stress_time_audit_wp10c8b.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_stress_time_audit_wp10c8b_arrays.npz"
)
RESOLUTIONS = (32, 64, 128)
TRAJECTORY_MODES = ("production", "temporal_control")
COMMON_OUTPUTS = (
    ("t_0p075", 7.5e-2),
    ("t_0p10", 1.0e-1),
    ("t_0p125", 1.25e-1),
    ("t_0p15", 1.5e-1),
)
START_TIME_SECONDS = 5.0e-2
TARGET_TIME_SECONDS = 1.5e-1
CONTROL_MAXIMUM_TIMESTEP_SECONDS = (
    0.5 * wp10c7k.SHARED_PASSING_CEILING_SECONDS
)
TARGET_TIME_RELATIVE_TOLERANCE = wp10c7l.TARGET_TIME_RELATIVE_TOLERANCE
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = (
    wp10c7l.MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
)
SPATIAL_RESPONSE_GATE = 5.0e-3
MAXIMUM_RICHARDSON_REMAINDER = 1.25e-3
MINIMUM_SPATIAL_ORDER = 1.8
MAXIMUM_PAIR_TEMPORAL_FRACTION = 0.25
REPLAY_FROM_LABEL = "t_0p125"
REPLAY_TO_LABEL = "t_0p15"
RADIAL_BANDS_RG = (
    ("horizon_to_6rg", 0.0, 6.0),
    ("6_to_60rg", 6.0, 60.0),
    ("60_to_200rg", 60.0, 200.0),
    ("200rg_to_outer", 200.0, np.inf),
)
TIMESCALE_THRESHOLDS_SECONDS = (0.05, 0.15, 1.0, 10.0, 100.0, 1000.0)
MINIMUM_BDF2_HISTORY_RATIO = 0.5
MAXIMUM_BDF2_HISTORY_RATIO = 2.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Load completed checkpoints and rewrite canonical evidence.",
    )
    parser.add_argument(
        "--reuse-replays",
        action="store_true",
        help="Resume missing trajectory checkpoints and load saved replays.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate evidence, parents, and projected spatial budget.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            wp10c7k._plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_authorization() -> tuple[dict, str, dict, str]:
    if not WP10C8A_OUTPUT.exists() or not WP10C7N_OUTPUT.exists():
        raise RuntimeError("WP10c8b requires WP10c8a and WP10c7n evidence")
    spectral = json.loads(WP10C8A_OUTPUT.read_text(encoding="utf-8"))
    spectral_arrays = ROOT / str(
        spectral.get("artifacts", {}).get("arrays_path", "")
    )
    reference = json.loads(WP10C7N_OUTPUT.read_text(encoding="utf-8"))
    reference_arrays = ROOT / str(
        reference.get("artifacts", {}).get("arrays_path", "")
    )
    if not (
        spectral.get("work_package") == "WP10c8a"
        and spectral.get("decision")
        == "wp10c8a_slow_manifold_not_authorized"
        and not spectral.get("gates", {}).get("wp10c8a_passed", True)
        and spectral.get("gates", {}).get(
            "low_mode_mesh_matching_passed",
            False,
        )
        and spectral_arrays.exists()
        and _sha256(spectral_arrays)
        == spectral.get("artifacts", {}).get("arrays_sha256")
        and reference.get("work_package") == "WP10c7n"
        and reference.get("decision")
        == "wp10c7n_n128_0p05_reference_certified"
        and reference.get("gates", {}).get("wp10c7n_passed", False)
        and reference_arrays.exists()
        and _sha256(reference_arrays)
        == reference.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8b authorization evidence differs")
    return (
        spectral,
        _sha256(WP10C8A_OUTPUT),
        reference,
        _sha256(WP10C7N_OUTPUT),
    )


def _initial_bundles(
    reference: dict,
) -> tuple[dict[int, dict], dict, str]:
    wp10c7k_evidence, wp10c7k_sha256 = wp10c7l._validate_wp10c7k()
    bundles = wp10c7l._initial_bundles(wp10c7k_evidence)
    bundles[128] = wp10c7n._initial_bundle()
    if (
        bundles[128]["vector_sha256"]
        != reference["initialization"]["n128_initial_state_sha256"]
    ):
        raise RuntimeError("WP10c8b N128 initial state differs")
    return bundles, wp10c7k_evidence, wp10c7k_sha256


def _controller_config(context, mode: str):
    production = wp10c7k._controller_config(context)
    if mode == "production":
        return production
    if mode == "temporal_control":
        return replace(
            production,
            maximum_dt=CONTROL_MAXIMUM_TIMESTEP_SECONDS,
        ).validated()
    raise ValueError(f"unknown WP10c8b trajectory mode {mode!r}")


def _controller_contract(mode: str) -> dict:
    contract = dict(wp10c7k._controller_contract())
    contract["role"] = mode
    if mode == "temporal_control":
        contract["maximum_timestep_seconds"] = (
            CONTROL_MAXIMUM_TIMESTEP_SECONDS
        )
        contract["reference_policy"] = (
            "same controller with only the maximum timestep halved"
        )
    else:
        contract["reference_policy"] = "unchanged WP10c7k controller"
    return contract


def _parent_restart(
    initial: dict,
    n_cells: int,
    mode: str,
    wp10c7k_evidence: dict,
    wp10c7k_sha256: str,
    reference: dict,
) -> tuple[CausalFiveFieldAdaptiveBDF2Restart, dict]:
    if n_cells in (32, 64):
        parent_label = (
            "t_0p0375"
            if n_cells == 64 and mode == "temporal_control"
            else "t_0p05"
        )
        parent_entry = wp10c7l._parent_checkpoint_entry(
            wp10c7k_evidence,
            n_cells,
        )
        restart = wp10c7l._load_snapshot(
            initial,
            wp10c7k_sha256,
            parent_entry,
            mode,
            parent_label,
        )
        path = wp10c7l._checkpoint_path(
            n_cells,
            mode,
            parent_label,
        )
        parent_package = "WP10c7l"
    else:
        authorization_sha256 = reference["wp10c7m_authorization"]["sha256"]
        restart = wp10c7n._load_snapshot(
            initial,
            authorization_sha256,
            mode,
            "t_0p05",
        )
        path = wp10c7n._checkpoint_path(mode, "t_0p05")
        parent_package = "WP10c7n"
    return restart, {
        "work_package": parent_package,
        "path": _relative(path),
        "sha256": _sha256(path),
        "state_vector_sha256": _array_sha256(restart.state_vector),
        "elapsed_time_seconds": restart.elapsed_time,
        "trajectory_mode": mode,
    }


def _extension_start(
    parent: CausalFiveFieldAdaptiveBDF2Restart,
    initial: dict,
    mode: str,
    parent_entry: dict,
    spectral_sha256: str,
    reference_sha256: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    maximum_dt = _controller_config(
        initial["context"],
        mode,
    ).maximum_dt
    return replace(
        parent,
        dt_next=min(parent.dt_next, maximum_dt),
        provenance={
            "work_package": "WP10c8b",
            "role": "matched_no_tide_stress_time_reference",
            "trajectory_mode": mode,
            "base_commit": BASE_COMMIT,
            "wp10c8a_evidence_sha256": spectral_sha256,
            "wp10c7n_evidence_sha256": reference_sha256,
            "parent_checkpoint": dict(parent_entry),
            "parent_elapsed_time_seconds": parent_entry[
                "elapsed_time_seconds"
            ],
            "n_cells": initial["state"].n_cells,
            "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
            "common_output_times_seconds": dict(COMMON_OUTPUTS),
            "spatial_options": dict(wp10c7l.SPATIAL_OPTIONS),
            "initial_state_sha256": initial["vector_sha256"],
            "controller": _controller_contract(mode),
            "segments": [],
        },
    )


def _checkpoint_path(n_cells: int, mode: str, label: str) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c8b_N{n_cells:03d}_{mode}_{label}.npz"
    )


def _replay_path(n_cells: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c8b_N{n_cells:03d}_production_replay_t_0p15.npz"
    )


def _output_target(label: str) -> float:
    return dict(COMMON_OUTPUTS)[label]


def _condition_multistep_start(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    ratio = (
        restart.history.previous_timestep_seconds
        / restart.older_timestep_seconds
    )
    if (
        MINIMUM_BDF2_HISTORY_RATIO
        <= ratio
        <= MAXIMUM_BDF2_HISTORY_RATIO
    ):
        return restart
    return replace(restart, next_order=1)


def _expected_labels(label: str) -> list[str]:
    labels = [name for name, _ in COMMON_OUTPUTS]
    return labels[: labels.index(label) + 1]


def _validate_restart(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
    initial: dict,
    mode: str,
    label: str,
    parent_entry: dict,
    spectral_sha256: str,
    reference_sha256: str,
) -> None:
    provenance = restart.provenance
    segments = provenance.get("segments", [])
    if not (
        provenance.get("work_package") == "WP10c8b"
        and provenance.get("role")
        == "matched_no_tide_stress_time_reference"
        and provenance.get("trajectory_mode") == mode
        and provenance.get("wp10c8a_evidence_sha256")
        == spectral_sha256
        and provenance.get("wp10c7n_evidence_sha256")
        == reference_sha256
        and provenance.get("parent_checkpoint") == parent_entry
        and provenance.get("parent_elapsed_time_seconds")
        == parent_entry["elapsed_time_seconds"]
        and provenance.get("n_cells") == initial["state"].n_cells
        and provenance.get("target_elapsed_time_seconds")
        == TARGET_TIME_SECONDS
        and provenance.get("common_output_times_seconds")
        == dict(COMMON_OUTPUTS)
        and provenance.get("spatial_options")
        == wp10c7l.SPATIAL_OPTIONS
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
        and provenance.get("controller") == _controller_contract(mode)
        and [row.get("label") for row in segments]
        == _expected_labels(label)
        and all(row.get("passed", False) for row in segments)
        and restart.elapsed_time == _output_target(label)
        and restart.next_order == 2
        and audit_causal_five_field_state_gates(
            initial["context"],
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(
            f"WP10c8b N{initial['state'].n_cells} {mode} {label} differs"
        )


def _load_snapshot(
    initial: dict,
    mode: str,
    label: str,
    parent_entry: dict,
    spectral_sha256: str,
    reference_sha256: str,
    *,
    path: Path | None = None,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    restart = load_causal_five_field_adaptive_bdf2_restart(
        (
            _checkpoint_path(initial["state"].n_cells, mode, label)
            if path is None
            else path
        ),
        initial["context"],
    )
    _validate_restart(
        restart,
        initial,
        mode,
        label,
        parent_entry,
        spectral_sha256,
        reference_sha256,
    )
    return restart


def _progress(n_cells: int, mode: str, label: str):
    def progress(relative_step, restart, result) -> None:
        at_target = abs(restart.elapsed_time - _output_target(label)) <= (
            TARGET_TIME_RELATIVE_TOLERANCE
            * max(1.0, abs(_output_target(label)))
        )
        if relative_step != 1 and relative_step % 8 != 0 and not at_target:
            return
        print(
            json.dumps(
                {
                    "mode": f"n{n_cells}_wp10c8b_{mode}_{label}",
                    "relative_accepted_step": int(relative_step),
                    "accepted_steps": int(restart.accepted_steps),
                    "order": int(result.order),
                    "dt_used": float(result.dt_used),
                    "dt_next": float(result.dt_next),
                    "elapsed_time": float(restart.elapsed_time),
                    "audits": int(restart.audit_count),
                    "rejected_attempts": int(restart.rejected_attempts),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return progress


def _run_segment(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    mode: str,
    label: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    started = time.perf_counter()
    campaign = evolve_causal_five_field_adaptive_bdf2_campaign(
        initial["context"],
        start,
        _output_target(label),
        _controller_config(initial["context"], mode),
        target_time_relative_tolerance=TARGET_TIME_RELATIVE_TOLERANCE,
        progress=_progress(initial["state"].n_cells, mode, label),
    )
    wall_seconds = time.perf_counter() - started
    if not campaign.passed:
        raise RuntimeError(
            f"WP10c8b N{initial['state'].n_cells} {mode} {label} "
            f"failed: {campaign.message}"
        )
    summary = wp10c7k._segment_summary(label, start, campaign)
    summary["wall_seconds"] = wall_seconds
    return wp10c7k._append_segment(campaign.restart, summary)


def _run_or_load_segment(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    mode: str,
    label: str,
    parent_entry: dict,
    spectral_sha256: str,
    reference_sha256: str,
    *,
    force: bool,
) -> tuple[CausalFiveFieldAdaptiveBDF2Restart, bool]:
    path = _checkpoint_path(initial["state"].n_cells, mode, label)
    if path.exists() and not force:
        return (
            _load_snapshot(
                initial,
                mode,
                label,
                parent_entry,
                spectral_sha256,
                reference_sha256,
            ),
            True,
        )
    state = _run_segment(initial, start, mode, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_causal_five_field_adaptive_bdf2_restart(
        path,
        initial["context"],
        state,
    )
    restored = _load_snapshot(
        initial,
        mode,
        label,
        parent_entry,
        spectral_sha256,
        reference_sha256,
    )
    bitwise = causal_five_field_adaptive_bdf2_restarts_equal(
        state,
        restored,
    )
    if not bitwise:
        raise RuntimeError(
            f"WP10c8b N{initial['state'].n_cells} {mode} {label} "
            "is not bitwise"
        )
    return restored, bitwise


def _profile_response(initial: dict, vector: np.ndarray) -> dict:
    baseline = causal_five_field_profile_fields(
        initial["context"],
        initial["vector"],
    )
    current = causal_five_field_profile_fields(
        initial["context"],
        vector,
    )
    return {
        name: np.asarray(current[name] - baseline[name], dtype=float)
        for name in baseline
    }


def _normalized_balance_rows(
    flux_divergence: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    tiny = np.finfo(float).tiny
    return np.abs(flux_divergence - source) / np.maximum(
        np.abs(flux_divergence) + np.abs(source),
        tiny,
    )


def _weighted_summary(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    normalized = weights / float(np.sum(weights))
    return {
        "maximum": float(np.max(values)),
        "weighted_mean": float(np.sum(normalized * values)),
        "weighted_rms": float(
            np.sqrt(np.sum(normalized * values**2))
        ),
        "median": float(np.median(values)),
    }


def _off_manifold_diagnostics(
    initial: dict,
    vector: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    cells = causal_five_field_cell_states(context, vector)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    clocks = causal_five_field_local_timescale_audit(context, vector)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    targets = np.asarray(
        [
            cell.closure.target_specific_stress(shear)
            for cell, shear in zip(
                cells,
                evaluation.proper_shear_rates,
                strict=True,
            )
        ],
        dtype=float,
    )
    stresses = np.asarray(
        [cell.stress.specific_stress for cell in cells],
        dtype=float,
    )
    stress_scale = np.maximum.reduce(
        (
            np.abs(targets),
            np.abs(stresses),
            np.full(n_cells, np.finfo(float).tiny),
        )
    )
    stress_departure = np.abs(stresses - targets) / stress_scale
    flux_divergence = (
        state.weighted_face_fluxes_over_c[1:]
        - state.weighted_face_fluxes_over_c[:-1]
    )
    radial_balance = _normalized_balance_rows(
        flux_divergence[:, 1],
        evaluation.integrated_sources_per_ct[:, 1],
    )
    stress_balance = _normalized_balance_rows(
        flux_divergence[:, 4],
        evaluation.integrated_sources_per_ct[:, 4],
    )
    bands = {}
    for name, lower, upper in RADIAL_BANDS_RG:
        mask = (radius_rg >= lower) & (radius_rg < upper)
        if not np.any(mask):
            continue
        bands[name] = {
            "cell_count": int(np.count_nonzero(mask)),
            "radius_range_rg": [
                float(np.min(radius_rg[mask])),
                float(np.max(radius_rg[mask])),
            ],
            "stress_target_departure": _weighted_summary(
                stress_departure[mask],
                measures[mask],
            ),
            "radial_momentum_stationary_balance": _weighted_summary(
                radial_balance[mask],
                measures[mask],
            ),
            "stress_stationary_balance": _weighted_summary(
                stress_balance[mask],
                measures[mask],
            ),
            "stress_relaxation_seconds": {
                "minimum": float(
                    np.min(clocks.stress_relaxation_seconds[mask])
                ),
                "maximum": float(
                    np.max(clocks.stress_relaxation_seconds[mask])
                ),
                "median": float(
                    np.median(clocks.stress_relaxation_seconds[mask])
                ),
            },
        }
    timescale_coverage = {}
    total_measure = float(np.sum(measures))
    for threshold in TIMESCALE_THRESHOLDS_SECONDS:
        mask = clocks.stress_relaxation_seconds <= threshold
        timescale_coverage[str(threshold)] = {
            "cell_fraction": float(np.mean(mask)),
            "measure_fraction": float(
                np.sum(measures[mask]) / total_measure
            ),
            "outermost_radius_rg": (
                float(np.max(radius_rg[mask])) if np.any(mask) else None
            ),
        }
    arrays = {
        "radius_rg": radius_rg,
        "specific_stress": stresses,
        "target_specific_stress": targets,
        "stress_target_relative_departure": stress_departure,
        "radial_momentum_stationary_balance": radial_balance,
        "stress_stationary_balance": stress_balance,
        "stress_relaxation_seconds": (
            clocks.stress_relaxation_seconds
        ),
        "radial_advection_seconds": clocks.radial_advection_seconds,
        "luminosity_response_seconds": (
            clocks.luminosity_response_seconds
        ),
        "thermal_response_seconds": clocks.thermal_response_seconds,
    }
    return {
        "full_domain": {
            "stress_target_departure": _weighted_summary(
                stress_departure,
                measures,
            ),
            "radial_momentum_stationary_balance": _weighted_summary(
                radial_balance,
                measures,
            ),
            "stress_stationary_balance": _weighted_summary(
                stress_balance,
                measures,
            ),
        },
        "radial_bands": bands,
        "stress_timescale_coverage": timescale_coverage,
        "clock_minima_seconds": {
            "characteristic_crossing": float(
                np.min(clocks.characteristic_crossing_seconds)
            ),
            "stress_relaxation": float(
                np.min(clocks.stress_relaxation_seconds)
            ),
            "radial_advection": float(
                np.min(clocks.radial_advection_seconds)
            ),
            "luminosity_response": float(
                np.min(clocks.luminosity_response_seconds)
            ),
            "thermal_response": float(
                np.min(clocks.thermal_response_seconds)
            ),
        },
    }, arrays


def _common_time_row(
    label: str,
    initial: dict[int, dict],
    snapshots: dict[tuple[int, str], dict[str, CausalFiveFieldAdaptiveBDF2Restart]],
) -> tuple[dict, dict[str, np.ndarray]]:
    temporal = {}
    arrays = {}
    responses = {}
    for n_cells in RESOLUTIONS:
        row, differences = wp10c7l._accumulated_temporal_row(
            initial[n_cells],
            snapshots[(n_cells, "production")][label],
            snapshots[(n_cells, "temporal_control")][label],
        )
        temporal[n_cells] = row
        responses[n_cells] = _profile_response(
            initial[n_cells],
            snapshots[(n_cells, "production")][label].state_vector,
        )
        arrays.update(
            {
                f"{label}_n{n_cells}_production_control_{name}": values
                for name, values in differences.items()
            }
        )

    pair_rows = {}
    pair_differences = {}
    for coarse_n, fine_n in ((32, 64), (64, 128)):
        differences = {}
        for name, coarse_values in responses[coarse_n].items():
            restricted = causal_restrict_cell_averages(
                initial[coarse_n]["context"].grid,
                initial[fine_n]["context"].grid,
                responses[fine_n][name],
            )
            differences[name] = coarse_values - restricted
            arrays[f"{label}_n{coarse_n}_{name}"] = coarse_values
            arrays[
                f"{label}_restricted_n{fine_n}_to_n{coarse_n}_{name}"
            ] = restricted
            arrays[
                f"{label}_n{coarse_n}_n{fine_n}_difference_{name}"
            ] = differences[name]
        pair_differences[(coarse_n, fine_n)] = differences
        pair_rows[f"n{coarse_n}_n{fine_n}"] = (
            wp10c7k._profile_difference_rows(
                initial[coarse_n]["context"],
                differences,
            )
        )

    raw_32_64 = float(
        pair_rows["n32_n64"]["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
    )
    raw_64_128 = float(
        pair_rows["n64_n128"]["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
    )
    order = causal_spatial_contraction_order(raw_32_64, raw_64_128)
    richardson = raw_64_128 / (2.0**order - 1.0)
    temporal_h = {
        n_cells: float(
            temporal[n_cells]["estimated_production_temporal_error"][
                "maximum_log_h_over_r_profile"
            ]
        )
        for n_cells in RESOLUTIONS
    }
    pair_temporal = {
        "n32_n64": temporal_h[32] + temporal_h[64],
        "n64_n128": temporal_h[64] + temporal_h[128],
    }
    pair_separated = {
        "n32_n64": bool(
            pair_temporal["n32_n64"]
            <= MAXIMUM_PAIR_TEMPORAL_FRACTION * raw_32_64
        ),
        "n64_n128": bool(
            pair_temporal["n64_n128"]
            <= MAXIMUM_PAIR_TEMPORAL_FRACTION * raw_64_128
        ),
    }
    conservative = (
        raw_64_128 + temporal_h[64] + temporal_h[128]
    )
    row = {
        "label": label,
        "elapsed_time_seconds": _output_target(label),
        "extension_duration_seconds": (
            _output_target(label) - START_TIME_SECONDS
        ),
        "temporal_control": {
            str(n_cells): temporal[n_cells]
            for n_cells in RESOLUTIONS
        },
        "spatial_profile_differences": pair_rows,
        "raw_n32_n64_log_h_over_r_difference": raw_32_64,
        "raw_n64_n128_log_h_over_r_difference": raw_64_128,
        "observed_spatial_order": order,
        "richardson_n128_to_continuum_remainder": richardson,
        "temporal_log_h_over_r_uncertainty": {
            str(n_cells): temporal_h[n_cells]
            for n_cells in RESOLUTIONS
        },
        "pair_temporal_uncertainty": pair_temporal,
        "pair_temporal_separation": {
            "maximum_fraction": MAXIMUM_PAIR_TEMPORAL_FRACTION,
            **pair_separated,
        },
        "conservative_n64_n128_log_h_over_r_total": conservative,
        "gates": {
            "spatial_response": SPATIAL_RESPONSE_GATE,
            "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
            "maximum_richardson_remainder": (
                MAXIMUM_RICHARDSON_REMAINDER
            ),
        },
        "temporal_passed": bool(
            all(
                temporal[n_cells]["gate_audit"]["passed"]
                for n_cells in RESOLUTIONS
            )
        ),
        "pair_temporal_separation_passed": bool(
            all(pair_separated.values())
        ),
        "spatial_gate_passed": bool(
            conservative <= SPATIAL_RESPONSE_GATE
        ),
        "order_passed": bool(order >= MINIMUM_SPATIAL_ORDER),
        "richardson_passed": bool(
            richardson <= MAXIMUM_RICHARDSON_REMAINDER
        ),
    }
    row["passed"] = bool(
        row["temporal_passed"]
        and row["pair_temporal_separation_passed"]
        and row["spatial_gate_passed"]
        and row["order_passed"]
        and row["richardson_passed"]
    )
    return row, arrays


def _campaign_summary(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    final: CausalFiveFieldAdaptiveBDF2Restart,
) -> dict:
    segments = list(final.provenance["segments"])
    attempts = [
        attempt
        for segment in segments
        for record in segment["records"]
        for attempt in record["attempts"]
    ]
    accepted = [attempt for attempt in attempts if attempt["accepted"]]
    independent = [
        attempt["independent_audit"]
        for attempt in attempts
        if attempt["independent_audit"] is not None
    ]
    local_passed = all(
        attempt["local_gate_audit"] is None
        or attempt["local_gate_audit"]["passed"]
        for attempt in accepted
    )
    ledger = causal_five_field_bdf_physical_ledger_from_restart(final)
    relative = causal_five_field_bdf_physical_ledger_relative_defects(
        ledger
    )
    maximum_ledger = float(np.max(np.abs(relative)))
    state_gates = audit_causal_five_field_state_gates(
        initial["context"],
        final.state_vector,
    )
    return {
        "segments": segments,
        "extension_accepted_steps": int(
            final.accepted_steps - start.accepted_steps
        ),
        "extension_accepted_bdf2_steps": int(
            final.accepted_bdf2_steps - start.accepted_bdf2_steps
        ),
        "extension_rejected_attempts": int(
            final.rejected_attempts - start.rejected_attempts
        ),
        "extension_audit_count": int(
            final.audit_count - start.audit_count
        ),
        "minimum_dt_used": min(
            segment["minimum_dt_used"] for segment in segments
        ),
        "maximum_dt_used": max(
            segment["maximum_dt_used"] for segment in segments
        ),
        "work": wp10c7k._sum_work(segments),
        "wall_seconds": float(
            sum(segment["wall_seconds"] for segment in segments)
        ),
        "all_segments_passed": all(
            segment["passed"] for segment in segments
        ),
        "local_estimator_passed": local_passed,
        "independent_audits": {
            "count": len(independent),
            "maximum_normalized_error": max(
                float(
                    row["temporal_gate_audit"][
                        "maximum_normalized_error"
                    ]
                )
                for row in independent
            ),
            "passed": bool(
                independent
                and all(row["passed"] for row in independent)
            ),
        },
        "physical_ledger": {
            "component_relative_defects": wp10c7k._plain(relative),
            "maximum_relative_defect": maximum_ledger,
            "gate": MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT,
            "passed": bool(
                maximum_ledger
                <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
            ),
        },
        "state_gates": state_gates,
        "state_summary": causal_five_field_state_summary(
            initial["context"],
            final.state_vector,
        ),
    }


def _run_replay(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    expected: CausalFiveFieldAdaptiveBDF2Restart,
    parent_entry: dict,
    spectral_sha256: str,
    reference_sha256: str,
) -> dict:
    replay = _run_segment(
        initial,
        _condition_multistep_start(start),
        "production",
        REPLAY_TO_LABEL,
    )
    path = _replay_path(initial["state"].n_cells)
    save_causal_five_field_adaptive_bdf2_restart(
        path,
        initial["context"],
        replay,
    )
    restored = _load_snapshot(
        initial,
        "production",
        REPLAY_TO_LABEL,
        parent_entry,
        spectral_sha256,
        reference_sha256,
        path=path,
    )

    def deterministic_restart(
        restart: CausalFiveFieldAdaptiveBDF2Restart,
    ) -> CausalFiveFieldAdaptiveBDF2Restart:
        provenance = dict(restart.provenance)
        segments = []
        for row in provenance.get("segments", []):
            segment = dict(row)
            segment.pop("wall_seconds", None)
            segments.append(segment)
        provenance["segments"] = segments
        return replace(restart, provenance=provenance)

    bitwise = causal_five_field_adaptive_bdf2_restarts_equal(
        deterministic_restart(expected),
        deterministic_restart(restored),
    )
    return {
        "from_label": REPLAY_FROM_LABEL,
        "to_label": REPLAY_TO_LABEL,
        "path": _relative(path),
        "sha256": _sha256(path),
        "bitwise": bitwise,
    }


def _load_replay(
    initial: dict,
    expected: CausalFiveFieldAdaptiveBDF2Restart,
    parent_entry: dict,
    spectral_sha256: str,
    reference_sha256: str,
) -> dict:
    path = _replay_path(initial["state"].n_cells)
    restored = _load_snapshot(
        initial,
        "production",
        REPLAY_TO_LABEL,
        parent_entry,
        spectral_sha256,
        reference_sha256,
        path=path,
    )

    def deterministic_restart(
        restart: CausalFiveFieldAdaptiveBDF2Restart,
    ) -> CausalFiveFieldAdaptiveBDF2Restart:
        provenance = dict(restart.provenance)
        segments = []
        for row in provenance.get("segments", []):
            segment = dict(row)
            segment.pop("wall_seconds", None)
            segments.append(segment)
        provenance["segments"] = segments
        return replace(restart, provenance=provenance)

    return {
        "from_label": REPLAY_FROM_LABEL,
        "to_label": REPLAY_TO_LABEL,
        "path": _relative(path),
        "sha256": _sha256(path),
        "bitwise": causal_five_field_adaptive_bdf2_restarts_equal(
            deterministic_restart(expected),
            deterministic_restart(restored),
        ),
    }


def _aggregate(
    spectral: dict,
    spectral_sha256: str,
    reference: dict,
    reference_sha256: str,
    initial: dict[int, dict],
    starts: dict[tuple[int, str], CausalFiveFieldAdaptiveBDF2Restart],
    parent_entries: dict[tuple[int, str], dict],
    snapshots: dict[
        tuple[int, str],
        dict[str, CausalFiveFieldAdaptiveBDF2Restart],
    ],
    roundtrips: dict[tuple[int, str, str], bool],
    replays: dict[int, dict],
    *,
    output_path: Path,
    arrays_path: Path,
) -> dict:
    common = {}
    arrays = {}
    off_manifold = {}
    for label, _ in COMMON_OUTPUTS:
        row, row_arrays = _common_time_row(label, initial, snapshots)
        common[label] = row
        arrays.update(row_arrays)
        off_manifold[label] = {}
        for n_cells in (64, 128):
            diagnostic, diagnostic_arrays = _off_manifold_diagnostics(
                initial[n_cells],
                snapshots[(n_cells, "production")][label].state_vector,
            )
            off_manifold[label][str(n_cells)] = diagnostic
            for name, values in diagnostic_arrays.items():
                arrays[f"{label}_n{n_cells}_{name}"] = values

    campaigns = {
        str(n_cells): {
            mode: _campaign_summary(
                initial[n_cells],
                starts[(n_cells, mode)],
                snapshots[(n_cells, mode)][COMMON_OUTPUTS[-1][0]],
            )
            for mode in TRAJECTORY_MODES
        }
        for n_cells in RESOLUTIONS
    }
    campaign_passed = all(
        campaigns[str(n_cells)][mode]["all_segments_passed"]
        and campaigns[str(n_cells)][mode]["local_estimator_passed"]
        and campaigns[str(n_cells)][mode]["independent_audits"]["passed"]
        and campaigns[str(n_cells)][mode]["physical_ledger"]["passed"]
        and campaigns[str(n_cells)][mode]["state_gates"]["passed"]
        for n_cells in RESOLUTIONS
        for mode in TRAJECTORY_MODES
    )
    latest_certified = START_TIME_SECONDS
    for label, elapsed in COMMON_OUTPUTS:
        if common[label]["passed"]:
            latest_certified = elapsed
        else:
            break
    target_passed = bool(
        latest_certified == TARGET_TIME_SECONDS
        and campaign_passed
        and all(roundtrips.values())
        and all(row["bitwise"] for row in replays.values())
    )
    certified_operator_audit_authorized = bool(
        latest_certified >= 0.125
        and campaign_passed
        and all(roundtrips.values())
        and all(row["bitwise"] for row in replays.values())
    )
    initial_off_manifold = {}
    for n_cells in (64, 128):
        diagnostic, diagnostic_arrays = _off_manifold_diagnostics(
            initial[n_cells],
            starts[(n_cells, "production")].state_vector,
        )
        initial_off_manifold[str(n_cells)] = diagnostic
        for name, values in diagnostic_arrays.items():
            arrays[f"t_0p05_n{n_cells}_{name}"] = values
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)

    payload = {
        "schema_version": 1,
        "work_package": "WP10c8b",
        "generated_at_utc": "2026-07-19T00:00:00Z",
        "base_commit": BASE_COMMIT,
        "scope": (
            "matched N32/N64/N128 full-causal no-tide extension from "
            "0.05 s to the 0.15 s stress-time rung, with independent "
            "half-ceiling temporal controls and region-aware off-manifold "
            "diagnostics"
        ),
        "wp10c8a_evidence": {
            "path": _relative(WP10C8A_OUTPUT),
            "sha256": spectral_sha256,
            "decision": spectral["decision"],
        },
        "wp10c7n_evidence": {
            "path": _relative(WP10C7N_OUTPUT),
            "sha256": reference_sha256,
            "decision": reference["decision"],
        },
        "start_elapsed_time_seconds": START_TIME_SECONDS,
        "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
        "common_output_times_seconds": dict(COMMON_OUTPUTS),
        "spatial_options": dict(wp10c7l.SPATIAL_OPTIONS),
        "initialization": {
            str(n_cells): {
                "state_vector_sha256": initial[n_cells]["vector_sha256"],
                "parent_checkpoints": {
                    mode: parent_entries[(n_cells, mode)]
                    for mode in TRAJECTORY_MODES
                },
            }
            for n_cells in RESOLUTIONS
        },
        "controller_contracts": {
            mode: _controller_contract(mode)
            for mode in TRAJECTORY_MODES
        },
        "campaigns": campaigns,
        "checkpoint_roundtrips_bitwise": {
            f"n{n_cells}_{mode}_{label}": value
            for (n_cells, mode, label), value in roundtrips.items()
        },
        "restart_replay": {
            str(n_cells): replays[n_cells]
            for n_cells in replays
        },
        "common_time_contract": common,
        "off_manifold_diagnostics": {
            "t_0p05": initial_off_manifold,
            **off_manifold,
        },
        "gates": {
            "all_campaigns_passed": campaign_passed,
            "all_checkpoint_roundtrips_bitwise": all(
                roundtrips.values()
            ),
            "all_replays_bitwise": all(
                row["bitwise"] for row in replays.values()
            ),
            "target_spatially_certified": target_passed,
            "wp10c8b_passed": target_passed,
            "certified_operator_audit_authorized": (
                certified_operator_audit_authorized
            ),
        },
        "latest_spatially_certified_time_seconds": latest_certified,
        "decision": (
            "wp10c8b_stress_time_full_reference_certified"
            if target_passed
            else "wp10c8b_stress_time_spatial_stop"
        ),
        "next_authorization": (
            "wp10c8c_region_selective_operator_only_reduction_audit"
            if target_passed
            else (
                "wp10c8c_certified_state_operator_only_closure_audit"
                if certified_operator_audit_authorized
                else "retain_full_causal_dae_and_repair_spatial_duration"
            )
        ),
        "hard_stops": [
            "global algebraic P_R/chi elimination remains rejected",
            "no loading-time macrosteps",
            "no tide or wind",
            "no hot-state or cycle claim",
        ],
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    (
        spectral,
        spectral_sha256,
        reference,
        reference_sha256,
    ) = _validate_authorization()
    initial, wp10c7k_evidence, wp10c7k_sha256 = _initial_bundles(
        reference
    )
    parents = {}
    parent_entries = {}
    starts = {}
    reference_endpoint = reference["common_time_contract"]["t_0p05"]
    projected_raw = (
        reference_endpoint["raw_n64_n128_log_h_over_r_difference"]
        * TARGET_TIME_SECONDS
        / START_TIME_SECONDS
    )
    projected_total = (
        projected_raw
        + 3.0
        * reference_endpoint["n64_temporal_uncertainty"]
        + 3.0
        * reference_endpoint["n128_temporal_uncertainty"]
    )
    projected_authorized = bool(
        projected_total <= SPATIAL_RESPONSE_GATE
    )
    for n_cells in RESOLUTIONS:
        for mode in TRAJECTORY_MODES:
            parent, entry = _parent_restart(
                initial[n_cells],
                n_cells,
                mode,
                wp10c7k_evidence,
                wp10c7k_sha256,
                reference,
            )
            parents[(n_cells, mode)] = parent
            parent_entries[(n_cells, mode)] = entry
            starts[(n_cells, mode)] = _extension_start(
                parent,
                initial[n_cells],
                mode,
                entry,
                spectral_sha256,
                reference_sha256,
            )
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8b",
                    "preflight_passed": projected_authorized,
                    "projected_raw_n64_n128_at_0p15": projected_raw,
                    "projected_conservative_total_at_0p15": (
                        projected_total
                    ),
                    "spatial_gate": SPATIAL_RESPONSE_GATE,
                    "parents": {
                        f"n{n_cells}_{mode}": parent_entries[
                            (n_cells, mode)
                        ]
                        for n_cells in RESOLUTIONS
                        for mode in TRAJECTORY_MODES
                    },
                },
                sort_keys=True,
            )
        )
        return
    if not projected_authorized:
        raise RuntimeError("WP10c8b projected stress-time budget failed")

    snapshots = {
        (n_cells, mode): {}
        for n_cells in RESOLUTIONS
        for mode in TRAJECTORY_MODES
    }
    roundtrips = {}
    if args.aggregate_only:
        for n_cells in RESOLUTIONS:
            for mode in TRAJECTORY_MODES:
                for label, _ in COMMON_OUTPUTS:
                    snapshots[(n_cells, mode)][label] = _load_snapshot(
                        initial[n_cells],
                        mode,
                        label,
                        parent_entries[(n_cells, mode)],
                        spectral_sha256,
                        reference_sha256,
                    )
                    roundtrips[(n_cells, mode, label)] = True
    else:
        for n_cells in RESOLUTIONS:
            for mode in TRAJECTORY_MODES:
                current = starts[(n_cells, mode)]
                for label, _ in COMMON_OUTPUTS:
                    current, bitwise = _run_or_load_segment(
                        initial[n_cells],
                        _condition_multistep_start(current),
                        mode,
                        label,
                        parent_entries[(n_cells, mode)],
                        spectral_sha256,
                        reference_sha256,
                        force=args.force,
                    )
                    snapshots[(n_cells, mode)][label] = current
                    roundtrips[(n_cells, mode, label)] = bitwise

    replays = {}
    for n_cells in (64, 128):
        if args.aggregate_only or args.reuse_replays:
            replays[n_cells] = _load_replay(
                initial[n_cells],
                snapshots[(n_cells, "production")][REPLAY_TO_LABEL],
                parent_entries[(n_cells, "production")],
                spectral_sha256,
                reference_sha256,
            )
        else:
            replays[n_cells] = _run_replay(
                initial[n_cells],
                snapshots[(n_cells, "production")][REPLAY_FROM_LABEL],
                snapshots[(n_cells, "production")][REPLAY_TO_LABEL],
                parent_entries[(n_cells, "production")],
                spectral_sha256,
                reference_sha256,
            )
    payload = _aggregate(
        spectral,
        spectral_sha256,
        reference,
        reference_sha256,
        initial,
        starts,
        parent_entries,
        snapshots,
        roundtrips,
        replays,
        output_path=output_path,
        arrays_path=arrays_path,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "latest_spatially_certified_time_seconds": payload[
                    "latest_spatially_certified_time_seconds"
                ],
                "target_contract": payload["common_time_contract"][
                    COMMON_OUTPUTS[-1][0]
                ],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
