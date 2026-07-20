"""Audit exact secular ledgers and stable observable reduction for WP10c8f."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import run_causal_mixed_mode_reduction_audit_wp10c8d as wp10c8d
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    causal_five_field_loading_time,
    causal_linear_initial_response,
    causal_linear_transfer_response,
    causal_lyapunov_metric_audit,
    causal_stable_rational_krylov_rom,
    causal_stable_rom_initial_response,
    causal_stable_rom_transfer_response,
    evaluate_causal_five_field_dae,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "d6fc8e3cf6b8d45803a6f0111a70726f47c60457"
WP10C8D_OUTPUT = (
    ROOT / "outputs/tables/causal_mixed_mode_reduction_audit_wp10c8d.json"
)
WP10C8E_OUTPUT = (
    ROOT / "outputs/tables/causal_stationary_branch_preflight_wp10c8e.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_stable_observable_reduction_audit_wp10c8f.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_stable_observable_reduction_audit_wp10c8f_arrays.npz"
)
RESOLUTIONS = (64, 128)
REDUCTION_ANCHORS = ("t_0", "t_0p05", "t_0p125")
LEDGER_ANCHORS = (
    ("t_0", 0.0),
    ("t_0p05", 5.0e-2),
    ("t_0p10", 1.0e-1),
    ("t_0p125", 1.25e-1),
)
ORDERS = (8, 16, 24, 32, 48, 64, 96)
RATIONAL_TIMESCALES_SECONDS = (
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
)
CERTIFIED_RESPONSE_TIMES_SECONDS = (
    1.0e-2,
    3.0e-2,
    5.0e-2,
    1.0e-1,
    1.25e-1,
)
MAXIMUM_REAL_EIGENVALUE_PER_S = 1.0e-8
MAXIMUM_STABILIZATION_CORRECTION_FRACTION = 5.0e-2
MAXIMUM_TRAINING_RESPONSE_ERROR = 1.0e-1
MAXIMUM_HELD_OUT_RESPONSE_ERROR = 2.5e-1
MAXIMUM_CROSS_MESH_TRANSFER_EXCESS = 2.5e-1
MAXIMUM_PROTECTED_DEFECT = 2.0e-9
MAXIMUM_LINEAR_ONLINE_COST_FRACTION = 5.0e-2
PREFERRED_MAXIMUM_ORDER = 64
RESPONSE_ACTIVITY_FRACTION = 1.0e-8
LEDGER_COMPONENTS = (
    (0, "rest_mass"),
    (2, "angular_momentum"),
    (3, "killing_energy"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate evidence and descriptor caches without auditing them.",
    )
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


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_authorization() -> tuple[dict, str, dict, str]:
    if not WP10C8D_OUTPUT.exists() or not WP10C8E_OUTPUT.exists():
        raise RuntimeError("WP10c8f requires canonical WP10c8d/e evidence")
    mixed = json.loads(WP10C8D_OUTPUT.read_text(encoding="utf-8"))
    branch = json.loads(WP10C8E_OUTPUT.read_text(encoding="utf-8"))
    mixed_arrays = ROOT / str(
        mixed.get("artifacts", {}).get("arrays_path", "")
    )
    if not (
        mixed.get("work_package") == "WP10c8d"
        and mixed.get("decision")
        == "wp10c8d_compact_cross_mesh_markovian_basis_not_found"
        and not mixed.get("gates", {}).get(
            "nonlinear_rom_authorized",
            True,
        )
        and mixed_arrays.exists()
        and _sha256(mixed_arrays)
        == mixed.get("artifacts", {}).get("arrays_sha256")
    ):
        raise RuntimeError("WP10c8d evidence is not canonical")
    if not (
        branch.get("work_package") == "WP10c8e"
        and branch.get("decision")
        == "wp10c8e_stationary_anchor_solve_not_authorized"
        and not branch.get("gates", {}).get(
            "matched_stationary_anchor_solve_authorized",
            True,
        )
    ):
        raise RuntimeError("WP10c8e did not authorize the alternative audit")
    return (
        mixed,
        _sha256(WP10C8D_OUTPUT),
        branch,
        _sha256(WP10C8E_OUTPUT),
    )


def _load_descriptor_cache(
    mixed: dict,
    n_cells: int,
    label: str,
) -> tuple[dict, dict]:
    cache = mixed["descriptor_caches"][str(n_cells)][label]
    path = ROOT / cache["path"]
    if not path.exists() or _sha256(path) != cache["sha256"]:
        raise RuntimeError(f"WP10c8f descriptor cache mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        arrays = {
            name: np.asarray(saved[name], dtype=float)
            for name in saved.files
            if name != "metadata_json"
        }
    if not (
        metadata.get("work_package") == "WP10c8d"
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor") == label
        and metadata.get("descriptor_rank") == 5 * n_cells
    ):
        raise RuntimeError("WP10c8f descriptor metadata is invalid")
    return arrays, {
        "path": cache["path"],
        "sha256": cache["sha256"],
        "metadata": metadata,
    }


def _finite_timescale(storage: float, rate: float, scale: float):
    if abs(rate) <= 1.0e-14 * max(abs(scale), np.finfo(float).tiny):
        return None
    return float(abs(storage) / abs(rate))


def _global_ledger_row(initial: dict, vector: np.ndarray) -> dict:
    context = initial["context"]
    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    storage = np.sum(measures[:, None] * state.conserved, axis=0)
    boundary = C * (
        state.weighted_face_fluxes_over_c[-1]
        - state.weighted_face_fluxes_over_c[0]
    )
    total_source = C * np.sum(
        evaluation.integrated_sources_per_ct,
        axis=0,
    )
    net_rate = total_source - boundary
    source_components = {
        name: C * np.sum(values, axis=0)
        for name, values in (
            evaluation.integrated_source_components_per_ct.items()
        )
    }
    stream = np.zeros(5, dtype=float)
    if context.stream_sources is not None:
        stream[:4] = np.sum(context.stream_sources.matrix, axis=0)
    loading_time = causal_five_field_loading_time(context, vector)
    components = {}
    for component, name in LEDGER_COMPONENTS:
        source_scale = max(
            abs(float(stream[component])),
            abs(float(total_source[component])),
            abs(float(boundary[component])),
            np.finfo(float).tiny,
        )
        timescale = _finite_timescale(
            float(storage[component]),
            float(net_rate[component]),
            source_scale,
        )
        required = float(boundary[component] - total_source[component])
        components[name] = {
            "storage": float(storage[component]),
            "boundary_transport_rate": float(boundary[component]),
            "total_source_rate": float(total_source[component]),
            "stream_source_rate": float(stream[component]),
            "net_storage_plus_vertical_rate": float(net_rate[component]),
            "net_rate_over_stream": float(
                net_rate[component] / source_scale
            ),
            "accumulation_timescale_seconds": timescale,
            "accumulation_timescale_over_loading_time": (
                None if timescale is None else timescale / loading_time
            ),
            "required_stationary_external_rate": required,
            "required_stationary_external_over_stream": float(
                required / source_scale
            ),
            "source_components": {
                source_name: float(values[component])
                for source_name, values in source_components.items()
            },
        }
    return {
        "loading_time_seconds": float(loading_time),
        "components": components,
    }


def _mesh_agreement(coarse: float | None, fine: float | None):
    if coarse is None or fine is None:
        return None
    return float(
        abs(coarse - fine)
        / max(abs(coarse), abs(fine), np.finfo(float).tiny)
    )


def _global_ledger_audit(initial: dict, vectors: dict) -> dict:
    rows = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for label, _time_seconds in LEDGER_ANCHORS:
            rows[str(n_cells)][label] = _global_ledger_row(
                initial[n_cells],
                vectors[n_cells][label],
            )
    cross_mesh = {}
    for label, _time_seconds in LEDGER_ANCHORS:
        cross_mesh[label] = {}
        for _component, name in LEDGER_COMPONENTS:
            coarse = rows["64"][label]["components"][name]
            fine = rows["128"][label]["components"][name]
            cross_mesh[label][name] = {
                "net_rate_relative_difference": _mesh_agreement(
                    coarse["net_storage_plus_vertical_rate"],
                    fine["net_storage_plus_vertical_rate"],
                ),
                "accumulation_timescale_relative_difference": (
                    _mesh_agreement(
                        coarse["accumulation_timescale_seconds"],
                        fine["accumulation_timescale_seconds"],
                    )
                ),
                "required_external_rate_relative_difference": (
                    _mesh_agreement(
                        coarse["required_stationary_external_rate"],
                        fine["required_stationary_external_rate"],
                    )
                ),
            }
    drift = {}
    for n_cells in RESOLUTIONS:
        drift[str(n_cells)] = {}
        for _component, name in LEDGER_COMPONENTS:
            early = rows[str(n_cells)]["t_0p05"]["components"][name][
                "net_storage_plus_vertical_rate"
            ]
            late = rows[str(n_cells)]["t_0p125"]["components"][name][
                "net_storage_plus_vertical_rate"
            ]
            drift[str(n_cells)][name] = {
                "rate_change_fraction": float(
                    abs(late - early)
                    / max(abs(early), abs(late), np.finfo(float).tiny)
                ),
                "sign_preserved": bool(early * late >= 0.0),
            }
    return {
        "resolutions": rows,
        "cross_mesh": cross_mesh,
        "startup_rate_drift": drift,
        "interpretation": (
            "Rates close the exact instantaneous global finite-volume "
            "ledger for conserved storage plus responsive-height storage."
        ),
    }


def _relative_response_error(
    full: np.ndarray,
    reduced: np.ndarray,
) -> dict:
    reference = np.asarray(full, dtype=float)
    approximation = np.asarray(reduced, dtype=float)
    if reference.shape != approximation.shape or reference.ndim != 3:
        raise ValueError("response arrays must share (sample, output, input)")
    denominator = np.max(np.abs(reference), axis=(0, 2))
    global_scale = max(float(np.max(denominator)), np.finfo(float).tiny)
    active = denominator >= RESPONSE_ACTIVITY_FRACTION * global_scale
    errors = np.zeros(denominator.shape, dtype=float)
    errors[active] = (
        np.max(np.abs(approximation - reference), axis=(0, 2))[active]
        / denominator[active]
    )
    return {
        "maximum_relative_error": float(np.max(errors)),
        "per_output_relative_error": errors,
        "active_outputs": active,
        "active_output_count": int(np.count_nonzero(active)),
    }


def _profile_response_error(
    full: np.ndarray,
    reduced: np.ndarray,
) -> float:
    return float(
        np.max(np.abs(np.asarray(reduced) - np.asarray(full)))
        / max(float(np.max(np.abs(full))), np.finfo(float).tiny)
    )


def _rom_response_audit(arrays: dict, rom) -> tuple[dict, dict]:
    dynamic = arrays["dynamic"]
    training = np.column_stack(
        (arrays["basis_inputs"], arrays["training_directions"])
    )
    held_out = arrays["held_out_directions"]
    selected_outputs = arrays["balanced_output_matrix"]
    profile_gate = CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1[
        "maximum_log_h_over_r_profile"
    ]
    profile_outputs = arrays["log_h_over_r_profile"] / profile_gate
    timescales = np.asarray(RATIONAL_TIMESCALES_SECONDS, dtype=float)
    transient_times = np.asarray(
        CERTIFIED_RESPONSE_TIMES_SECONDS,
        dtype=float,
    )

    full_training_transfer = causal_linear_transfer_response(
        dynamic,
        training,
        selected_outputs,
        timescales,
    )
    reduced_training_transfer = causal_stable_rom_transfer_response(
        rom,
        training,
        timescales,
    )
    full_held_transfer = causal_linear_transfer_response(
        dynamic,
        held_out,
        selected_outputs,
        timescales,
    )
    reduced_held_transfer = causal_stable_rom_transfer_response(
        rom,
        held_out,
        timescales,
    )
    full_training_profile = causal_linear_transfer_response(
        dynamic,
        training,
        profile_outputs,
        timescales,
    )
    reduced_training_profile = causal_stable_rom_transfer_response(
        rom,
        training,
        timescales,
        outputs=profile_outputs,
    )
    full_held_profile = causal_linear_transfer_response(
        dynamic,
        held_out,
        profile_outputs,
        timescales,
    )
    reduced_held_profile = causal_stable_rom_transfer_response(
        rom,
        held_out,
        timescales,
        outputs=profile_outputs,
    )

    full_training_state = causal_linear_initial_response(
        dynamic,
        training,
        transient_times,
    )
    reduced_training_state = causal_stable_rom_initial_response(
        rom,
        training,
        transient_times,
    )
    full_held_state = causal_linear_initial_response(
        dynamic,
        held_out,
        transient_times,
    )
    reduced_held_state = causal_stable_rom_initial_response(
        rom,
        held_out,
        transient_times,
    )
    full_training_transient = np.einsum(
        "on,tnd->tod",
        selected_outputs,
        full_training_state,
        optimize=True,
    )
    reduced_training_transient = np.einsum(
        "on,tnd->tod",
        selected_outputs,
        reduced_training_state,
        optimize=True,
    )
    full_held_transient = np.einsum(
        "on,tnd->tod",
        selected_outputs,
        full_held_state,
        optimize=True,
    )
    reduced_held_transient = np.einsum(
        "on,tnd->tod",
        selected_outputs,
        reduced_held_state,
        optimize=True,
    )
    full_training_transient_profile = np.einsum(
        "hn,tnd->thd",
        profile_outputs,
        full_training_state,
        optimize=True,
    )
    reduced_training_transient_profile = np.einsum(
        "hn,tnd->thd",
        profile_outputs,
        reduced_training_state,
        optimize=True,
    )
    full_held_transient_profile = np.einsum(
        "hn,tnd->thd",
        profile_outputs,
        full_held_state,
        optimize=True,
    )
    reduced_held_transient_profile = np.einsum(
        "hn,tnd->thd",
        profile_outputs,
        reduced_held_state,
        optimize=True,
    )
    row = {
        "training_transfer": _relative_response_error(
            full_training_transfer,
            reduced_training_transfer,
        ),
        "held_out_transfer": _relative_response_error(
            full_held_transfer,
            reduced_held_transfer,
        ),
        "training_transfer_profile_relative_error": (
            _profile_response_error(
                full_training_profile,
                reduced_training_profile,
            )
        ),
        "held_out_transfer_profile_relative_error": (
            _profile_response_error(
                full_held_profile,
                reduced_held_profile,
            )
        ),
        "training_transient": _relative_response_error(
            full_training_transient,
            reduced_training_transient,
        ),
        "held_out_transient": _relative_response_error(
            full_held_transient,
            reduced_held_transient,
        ),
        "training_transient_profile_relative_error": (
            _profile_response_error(
                full_training_transient_profile,
                reduced_training_transient_profile,
            )
        ),
        "held_out_transient_profile_relative_error": (
            _profile_response_error(
                full_held_transient_profile,
                reduced_held_transient_profile,
            )
        ),
    }
    traces = {
        "full_training_transfer": full_training_transfer,
        "reduced_training_transfer": reduced_training_transfer,
    }
    return row, traces


def _local_reduction_audit(arrays: dict) -> tuple[dict, dict, dict]:
    started = time.perf_counter()
    metric = causal_lyapunov_metric_audit(
        arrays["dynamic"],
        state_weights=arrays["state_weights"],
    )
    metric_row = {
        "minimum_eigenvalue": metric.minimum_eigenvalue,
        "maximum_eigenvalue": metric.maximum_eigenvalue,
        "normalized_minimum_eigenvalue": (
            metric.normalized_minimum_eigenvalue
        ),
        "relative_residual": metric.relative_residual,
        "positive_definite": metric.positive_definite,
        "residual_passed": metric.residual_passed,
        "accepted": metric.accepted,
    }
    rows = {}
    roms = {}
    traces = {}
    for order in ORDERS:
        order_started = time.perf_counter()
        rom = causal_stable_rational_krylov_rom(
            arrays["dynamic"],
            arrays["basis_inputs"],
            arrays["balanced_output_matrix"],
            arrays["protected_operators"],
            order=order,
            timescales_seconds=np.asarray(
                RATIONAL_TIMESCALES_SECONDS,
                dtype=float,
            ),
            state_weights=arrays["state_weights"],
            initial_directions=arrays["training_directions"],
            stability_tolerance=MAXIMUM_REAL_EIGENVALUE_PER_S,
        )
        response, trace = _rom_response_audit(arrays, rom)
        linear_cost = float((order / arrays["dynamic"].shape[0]) ** 2)
        training_maximum = max(
            response["training_transfer"]["maximum_relative_error"],
            response[
                "training_transfer_profile_relative_error"
            ],
            response["training_transient"]["maximum_relative_error"],
            response[
                "training_transient_profile_relative_error"
            ],
        )
        held_maximum = max(
            response["held_out_transfer"]["maximum_relative_error"],
            response["held_out_transfer_profile_relative_error"],
            response["held_out_transient"]["maximum_relative_error"],
            response["held_out_transient_profile_relative_error"],
        )
        passed = bool(
            rom.stabilization_succeeded
            and rom.maximum_real_eigenvalue
            <= MAXIMUM_REAL_EIGENVALUE_PER_S
            and rom.stabilization_correction_fraction
            <= MAXIMUM_STABILIZATION_CORRECTION_FRACTION
            and rom.biorthogonality_defect <= MAXIMUM_PROTECTED_DEFECT
            and rom.protected_value_defect <= MAXIMUM_PROTECTED_DEFECT
            and rom.protected_dynamics_defect <= MAXIMUM_PROTECTED_DEFECT
            and training_maximum <= MAXIMUM_TRAINING_RESPONSE_ERROR
            and held_maximum <= MAXIMUM_HELD_OUT_RESPONSE_ERROR
            and linear_cost <= MAXIMUM_LINEAR_ONLINE_COST_FRACTION
        )
        rows[str(order)] = {
            "order": order,
            "raw_maximum_real_eigenvalue_per_s": (
                rom.raw_maximum_real_eigenvalue
            ),
            "maximum_real_eigenvalue_per_s": (
                rom.maximum_real_eigenvalue
            ),
            "stabilization_succeeded": rom.stabilization_succeeded,
            "stabilization_penalty": rom.stabilization_penalty,
            "stabilization_correction_fraction": (
                rom.stabilization_correction_fraction
            ),
            "biorthogonality_defect": rom.biorthogonality_defect,
            "protected_value_defect": rom.protected_value_defect,
            "protected_dynamics_defect": rom.protected_dynamics_defect,
            "linear_online_cost_fraction_estimate": linear_cost,
            "training_maximum_response_error": training_maximum,
            "held_out_maximum_response_error": held_maximum,
            "responses": response,
            "candidate_singular_values": (
                rom.candidate_singular_values
            ),
            "wall_seconds": time.perf_counter() - order_started,
            "local_gate_passed": passed,
        }
        roms[order] = rom
        traces[order] = trace
    return {
        "lyapunov_metric": metric_row,
        "orders": rows,
        "wall_seconds": time.perf_counter() - started,
    }, roms, traces


def _cross_mesh_transfer(
    coarse: dict,
    fine: dict,
) -> dict:
    full_coarse = coarse["full_training_transfer"]
    full_fine = fine["full_training_transfer"]
    reduced_coarse = coarse["reduced_training_transfer"]
    reduced_fine = fine["reduced_training_transfer"]
    scale = max(
        float(np.max(np.abs(full_coarse))),
        float(np.max(np.abs(full_fine))),
        np.finfo(float).tiny,
    )
    full_difference = full_coarse - full_fine
    reduced_difference = reduced_coarse - reduced_fine
    excess = float(
        np.max(np.abs(reduced_difference - full_difference)) / scale
    )
    return {
        "full_transfer_relative_difference": float(
            np.max(np.abs(full_difference)) / scale
        ),
        "reduced_transfer_relative_difference": float(
            np.max(np.abs(reduced_difference)) / scale
        ),
        "rom_cross_mesh_excess": excess,
        "passed": bool(excess <= MAXIMUM_CROSS_MESH_TRANSFER_EXCESS),
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    mixed, mixed_sha256, branch, branch_sha256 = (
        _validate_authorization()
    )
    initial, vectors, state_provenance = wp10c8d._load_states()
    descriptor_arrays = {n_cells: {} for n_cells in RESOLUTIONS}
    descriptor_provenance = {str(n_cells): {} for n_cells in RESOLUTIONS}
    for n_cells in RESOLUTIONS:
        for label in REDUCTION_ANCHORS:
            arrays, provenance = _load_descriptor_cache(
                mixed,
                n_cells,
                label,
            )
            descriptor_arrays[n_cells][label] = arrays
            descriptor_provenance[str(n_cells)][label] = provenance
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c8f",
                    "preflight_passed": True,
                    "wp10c8d_evidence_sha256": mixed_sha256,
                    "wp10c8e_evidence_sha256": branch_sha256,
                    "descriptor_count": 6,
                },
                sort_keys=True,
            )
        )
        return

    ledger = _global_ledger_audit(initial, vectors)
    print(
        json.dumps(
            {
                "work_package": "WP10c8f",
                "phase": "global_ledger",
                "n128_t_0p125": {
                    name: ledger["resolutions"]["128"]["t_0p125"][
                        "components"
                    ][name]["accumulation_timescale_seconds"]
                    for _component, name in LEDGER_COMPONENTS
                },
            },
            sort_keys=True,
        )
    )

    reductions = {str(n_cells): {} for n_cells in RESOLUTIONS}
    roms = {n_cells: {} for n_cells in RESOLUTIONS}
    traces = {n_cells: {} for n_cells in RESOLUTIONS}
    array_payload = {}
    for label in REDUCTION_ANCHORS:
        for n_cells in RESOLUTIONS:
            summary, local_roms, local_traces = _local_reduction_audit(
                descriptor_arrays[n_cells][label]
            )
            reductions[str(n_cells)][label] = summary
            roms[n_cells][label] = local_roms
            traces[n_cells][label] = local_traces
            for order in ORDERS:
                array_payload[
                    f"n{n_cells}_{label}_r{order}_candidate_singular_values"
                ] = summary["orders"][str(order)][
                    "candidate_singular_values"
                ]
            print(
                json.dumps(
                    {
                        "work_package": "WP10c8f",
                        "phase": "stable_rational_ladder",
                        "n_cells": n_cells,
                        "anchor": label,
                        "lyapunov_metric_accepted": summary[
                            "lyapunov_metric"
                        ]["accepted"],
                        "passing_orders": [
                            int(order)
                            for order, row in summary["orders"].items()
                            if row["local_gate_passed"]
                        ],
                    },
                    sort_keys=True,
                )
            )

    cross_mesh = {}
    compact_orders = []
    for label in REDUCTION_ANCHORS:
        cross_mesh[label] = {}
        for order in ORDERS:
            row = _cross_mesh_transfer(
                traces[64][label][order],
                traces[128][label][order],
            )
            cross_mesh[label][str(order)] = row
    for order in ORDERS:
        if (
            order <= PREFERRED_MAXIMUM_ORDER
            and all(
                reductions[str(n_cells)][label]["orders"][str(order)][
                    "local_gate_passed"
                ]
                for n_cells in RESOLUTIONS
                for label in REDUCTION_ANCHORS
            )
            and all(
                cross_mesh[label][str(order)]["passed"]
                for label in REDUCTION_ANCHORS
            )
        ):
            compact_orders.append(order)

    stable_observable_model_found = bool(compact_orders)
    decision = (
        "wp10c8f_stable_cross_mesh_observable_model_found"
        if stable_observable_model_found
        else "wp10c8f_stable_cross_mesh_observable_model_not_found"
    )
    next_authorization = (
        "bounded_nonlinear_microburst_reconstruction_audit"
        if stable_observable_model_found
        else "ledger_driven_equation_free_closure_preflight"
    )
    payload = {
        "work_package": "WP10c8f",
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "next_authorization": next_authorization,
        "scope": {
            "description": (
                "Exact global secular ledger and stability-preserving "
                "observable-specific rational Krylov audit"
            ),
            "resolutions": RESOLUTIONS,
            "reduction_anchors": REDUCTION_ANCHORS,
            "ledger_anchors": dict(LEDGER_ANCHORS),
            "orders": ORDERS,
            "rational_timescales_seconds": (
                RATIONAL_TIMESCALES_SECONDS
            ),
            "certified_response_times_seconds": (
                CERTIFIED_RESPONSE_TIMES_SECONDS
            ),
            "nonlinear_rom_implemented": False,
            "memory_model_implemented": False,
            "new_full_dae_trajectory_run": False,
        },
        "authorization": {
            "wp10c8d_decision": mixed["decision"],
            "wp10c8d_evidence_sha256": mixed_sha256,
            "wp10c8e_decision": branch["decision"],
            "wp10c8e_evidence_sha256": branch_sha256,
        },
        "state_provenance": state_provenance,
        "descriptor_provenance": descriptor_provenance,
        "global_ledger_audit": ledger,
        "reductions": reductions,
        "cross_mesh_transfer": cross_mesh,
        "compact_passing_orders": compact_orders,
        "gates": {
            "maximum_real_eigenvalue_per_s": (
                MAXIMUM_REAL_EIGENVALUE_PER_S
            ),
            "maximum_stabilization_correction_fraction": (
                MAXIMUM_STABILIZATION_CORRECTION_FRACTION
            ),
            "maximum_training_response_error": (
                MAXIMUM_TRAINING_RESPONSE_ERROR
            ),
            "maximum_held_out_response_error": (
                MAXIMUM_HELD_OUT_RESPONSE_ERROR
            ),
            "maximum_cross_mesh_transfer_excess": (
                MAXIMUM_CROSS_MESH_TRANSFER_EXCESS
            ),
            "maximum_protected_defect": MAXIMUM_PROTECTED_DEFECT,
            "maximum_linear_online_cost_fraction": (
                MAXIMUM_LINEAR_ONLINE_COST_FRACTION
            ),
            "preferred_maximum_order": PREFERRED_MAXIMUM_ORDER,
            "stable_cross_mesh_observable_model_found": (
                stable_observable_model_found
            ),
            "nonlinear_rom_authorized": False,
            "memory_model_authorized": False,
        },
        "interpretation": {
            "lyapunov_metric": (
                "A failed dense Lyapunov metric is reported as a numerical "
                "non-certificate; its spectrum is never clipped."
            ),
            "stabilization": (
                "LQR corrections act only in the null space of the exact "
                "global ledger coordinates. Correction size and response "
                "damage remain binding authorization gates."
            ),
            "stationarity": (
                "Required external rates quantify the torque or power that "
                "the no-tide state lacks; they are not added to the model."
            ),
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
        },
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)
    payload["artifacts"]["arrays_sha256"] = _sha256(arrays_path)
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "work_package": "WP10c8f",
                "decision": decision,
                "compact_passing_orders": compact_orders,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
