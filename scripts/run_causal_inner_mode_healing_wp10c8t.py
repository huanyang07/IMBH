"""Run the WP10c8t extended inner-mode healing decision.

WP10c8s found an exact finite-amplitude equal-q34 pair whose complete
loading-time-normalized slow-rate response is localized in the innermost
retained shell and remains above the healing gate at 0.025 s.  Its original
coarse/fine decay curve was not temporally resolved well enough to infer a
relaxation law.

This package keeps the production physics, q34 coordinate map, five-shell
layout, spatial operator, and BDF formula fixed.  It:

* replays the committed N64 mode-0 fine trajectory to 0.025 s and persists
  the exact increment-primary BDF history;
* constructs a nested h/h2 pair at 1.25e-3/6.25e-4 s;
* continues both plus/minus pairs without a second BDF1 startup to 0.125 s;
* bounds the complete slow-rate decay and all 34 accumulated initial-slip
  components; and
* classifies only an N64 fast initial layer, a persistent localized mode, or
  a numerically inconclusive result.

No reduced evolution, relaxation fit, augmented coordinate, tide, wind,
hot-state search, stability claim, or loading-time macrostep is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_complete_rate_healing_wp10c8s as wp10c8s
import run_causal_extended_healing_wp10c8q as wp10c8q
import run_causal_interface_state_sufficiency_wp10c8r as wp10c8r
import run_causal_natural_healing_wp10c8p as wp10c8p
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o
import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldBDFRestart,
    causal_cumulative_trapezoid,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_loading_time,
    causal_five_field_observable_snapshot,
    causal_refined_spread_upper_bound,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_adaptive_bdf2_restart,
    load_causal_five_field_bdf_restart,
    save_causal_five_field_bdf_restart,
)


BASE_COMMIT = "4a54eb547b5c9f1663ce2480367110d195c0b4bd"
WORK_PACKAGE = "WP10c8t"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_inner_mode_healing_wp10c8t.py"
CASE_ID = "mode_0_inner_stress_existing"

PARENT_JSON = (
    ROOT / "outputs/tables/causal_complete_rate_healing_wp10c8s.json"
)
PARENT_ARRAYS = (
    ROOT / "outputs/tables/causal_complete_rate_healing_wp10c8s_arrays.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8t"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_mode_healing_wp10c8t.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_inner_mode_healing_wp10c8t_arrays.npz"
)

PARENT_DURATION_SECONDS = 0.025
TARGET_DURATION_SECONDS = 0.125
OUTPUT_OFFSETS_SECONDS = (
    0.0,
    0.0025,
    0.005,
    0.010,
    0.025,
    0.050,
    0.075,
    0.100,
    0.125,
)
SEGMENT_TARGETS_SECONDS = (0.050, 0.075, 0.100, 0.125)
TIMESTEP_SECONDS = {
    "coarse": 1.25e-3,
    "fine": 6.25e-4,
}
TOTAL_SUBDIVISIONS = {
    label: int(round(TARGET_DURATION_SECONDS / timestep))
    for label, timestep in TIMESTEP_SECONDS.items()
}
PARENT_SUBDIVISIONS = {
    label: int(round(PARENT_DURATION_SECONDS / timestep))
    for label, timestep in TIMESTEP_SECONDS.items()
}

NONLINEAR_SIGNIFICANCE_GATE = wp10c8o.INSTANTANEOUS_SCREEN_GATE
HEALING_FINAL_SPREAD_GATE = 0.10
MAXIMUM_INITIAL_SLIP = 0.10
MINIMUM_HEALING_FACTOR = 2.0
MINIMUM_RELAXATION_FIT_EFOLDS = 2.0
TEMPORAL_UNCERTAINTY_GATE = 0.025
TEMPORAL_RELATIVE_UNCERTAINTY_GATE = 0.10
TEMPORAL_RELATIVE_SPREAD_FLOOR = 0.10
MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT = 1.0e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return wp10c8p._array_sha256(values)


def _plain(value):
    return wp10c8p._plain(value)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _case_arrays(parent_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prefix = f"{CASE_ID}_"
    arrays = {
        name.removeprefix(prefix): np.asarray(values)
        for name, values in parent_arrays.items()
        if name.startswith(prefix)
        and not name.startswith(f"{CASE_ID}_healing_")
        and not name.startswith(f"{CASE_ID}_localization_")
        and not name.startswith(f"{CASE_ID}_linear_healing_")
    }
    required = {
        "minus_state_vector",
        "plus_state_vector",
        "coordinate_names",
        "coordinate_scales",
        "interface_flux_scales",
    }
    if not required.issubset(arrays):
        missing = ", ".join(sorted(required - set(arrays)))
        raise RuntimeError(f"WP10c8t parent case arrays are missing: {missing}")
    return arrays


def _validate_schedule() -> None:
    for label, timestep in TIMESTEP_SECONDS.items():
        if not np.isclose(
            TOTAL_SUBDIVISIONS[label] * timestep,
            TARGET_DURATION_SECONDS,
            rtol=0.0,
            atol=16.0 * np.finfo(float).eps,
        ):
            raise RuntimeError(f"WP10c8t {label} target is not commensurate")
        if not np.isclose(
            PARENT_SUBDIVISIONS[label] * timestep,
            PARENT_DURATION_SECONDS,
            rtol=0.0,
            atol=16.0 * np.finfo(float).eps,
        ):
            raise RuntimeError(f"WP10c8t {label} parent is not commensurate")
        for target in SEGMENT_TARGETS_SECONDS:
            if not np.isclose(
                round(target / timestep) * timestep,
                target,
                rtol=0.0,
                atol=16.0 * np.finfo(float).eps,
            ):
                raise RuntimeError(
                    f"WP10c8t {label} segment is not commensurate"
                )
    if TIMESTEP_SECONDS["coarse"] != 2.0 * TIMESTEP_SECONDS["fine"]:
        raise RuntimeError("WP10c8t timesteps are not a nested h/h2 pair")


def _load_contract() -> tuple[dict, dict, dict[str, np.ndarray], dict]:
    if not PARENT_JSON.exists() or not PARENT_ARRAYS.exists():
        raise FileNotFoundError("required WP10c8s evidence is missing")
    parent = json.loads(PARENT_JSON.read_text(encoding="utf-8"))
    if not (
        parent.get("work_package") == "WP10c8s"
        and parent.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and CASE_ID in parent.get("cases", {})
    ):
        raise RuntimeError("WP10c8t parent provenance failed")
    parent_arrays = _load_npz(PARENT_ARRAYS)
    case = _case_arrays(parent_arrays)
    operator_arrays, operator_metadata = wp10c8r._load_operator_cache(
        wp10c8s.OPERATOR_N64
    )
    # WP10c8t is deliberately N64-only until its endpoint changes the
    # architecture decision.  Reconstructing the generic WP10c8q contracts
    # would also build an unused N128 tangent and dominate preflight cost.
    context = wp10c7k._context(64)
    anchor_path = wp10c8q.wp10c8i._t_0p025_path(64)
    anchor_restart = load_causal_five_field_adaptive_bdf2_restart(
        anchor_path,
        context,
    )
    if not (
        anchor_restart.elapsed_time == PARENT_DURATION_SECONDS
        and anchor_restart.provenance.get("work_package") == "WP10c7l"
        and anchor_restart.provenance.get("trajectory_mode") == "production"
        and anchor_restart.provenance.get("n_cells") == 64
    ):
        raise RuntimeError("WP10c8t N64 anchor checkpoint differs")
    contract = {
        "context": context,
        "anchor_vector": np.asarray(
            anchor_restart.state_vector,
            dtype=float,
        ),
        "shell_edges_rg": np.asarray(
            operator_arrays["shell_edges_rg"],
            dtype=float,
        ),
        "anchor_path": anchor_path,
        "anchor_sha256": _sha256(anchor_path),
    }
    return parent, contract, case, {
        "arrays": operator_arrays,
        "metadata": operator_metadata,
    }


def _parent_trajectory(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    side: str,
) -> dict:
    return wp10c8s._run_or_load_trajectory(
        case_id=CASE_ID,
        resolution="fine",
        side=side,
        context=contract["context"],
        initial_vector=np.asarray(case[f"{side}_state_vector"], dtype=float),
        force=False,
    )


def _contract_n_cells(contract: dict) -> int:
    return int(contract["context"].grid.centers.size)


def _contract_checkpoint_directory(contract: dict) -> Path:
    return Path(contract.get("checkpoint_directory", CHECKPOINT_DIRECTORY))


def _restart_path(
    contract: dict,
    resolution: str,
    side: str,
    target: float,
) -> Path:
    label = f"t{target:.3f}".replace(".", "p")
    return _contract_checkpoint_directory(contract) / (
        f"N{_contract_n_cells(contract):03d}_{resolution}_{side}_"
        f"{label}_restart.npz"
    )


def _initial_segment_path(
    contract: dict,
    resolution: str,
    side: str,
) -> Path:
    return _contract_checkpoint_directory(contract) / (
        f"N{_contract_n_cells(contract):03d}_{resolution}_{side}_"
        "initial_t0p025.npz"
    )


def _segment_path(
    contract: dict,
    resolution: str,
    side: str,
    target: float,
) -> Path:
    label = f"t{target:.3f}".replace(".", "p")
    return _contract_checkpoint_directory(contract) / (
        f"N{_contract_n_cells(contract):03d}_{resolution}_{side}_"
        f"segment_{label}.npz"
    )


def _trajectory_path(
    contract: dict,
    resolution: str,
    side: str,
) -> Path:
    return _contract_checkpoint_directory(contract) / (
        f"N{_contract_n_cells(contract):03d}_{resolution}_{side}_"
        "trajectory_t0p125.npz"
    )


def _restart_evidence(
    restart_path: Path,
    restart: CausalFiveFieldBDFRestart,
) -> dict:
    return {
        "path": _relative(restart_path),
        "sha256": _sha256(restart_path),
        "state_sha256": _array_sha256(restart.state_vector),
        "previous_physical_increment_sha256": _array_sha256(
            restart.history.previous_physical_increment
        ),
        "previous_vertical_killing_increment_sha256": _array_sha256(
            restart.history.previous_vertical_killing_increment
        ),
        "previous_timestep_seconds": (
            restart.history.previous_timestep_seconds
        ),
        "elapsed_time_seconds": restart.elapsed_time,
    }


def _save_checked_restart(
    *,
    path: Path,
    context,
    restart: CausalFiveFieldBDFRestart,
) -> CausalFiveFieldBDFRestart:
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    if not causal_five_field_bdf_restarts_equal(restart, restored):
        raise RuntimeError("WP10c8t restart did not round-trip bitwise")
    return restored


def _result_row(result, wall_seconds: float) -> dict:
    return wp10c8p._result_row(result, wall_seconds)


def _run_initial_segment(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    resolution: str,
    side: str,
    force: bool,
) -> dict:
    context = contract["context"]
    timestep = TIMESTEP_SECONDS[resolution]
    subdivisions = PARENT_SUBDIVISIONS[resolution]
    initial = np.asarray(case[f"{side}_state_vector"], dtype=float)
    path = _initial_segment_path(contract, resolution, side)
    restart_path = _restart_path(
        contract,
        resolution,
        side,
        PARENT_DURATION_SECONDS,
    )
    parent = (
        _parent_trajectory(contract=contract, case=case, side=side)
        if resolution == "coarse"
        and contract.get("bitwise_parent_replay_required", True)
        else None
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "purpose": "exact_inner_mode_terminal_bdf_history",
        "case_id": CASE_ID,
        "resolution": resolution,
        "side": side,
        "duration_seconds": PARENT_DURATION_SECONDS,
        "subdivisions": subdivisions,
        "timestep_seconds": timestep,
        "initial_state_sha256": _array_sha256(initial),
        "parent_trajectory_sha256": (
            parent["sha256"] if parent is not None else None
        ),
        "startup": "one_bdf1_with_zero_predictor_then_fixed_bdf2",
    }
    if path.exists() and restart_path.exists() and not force:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            states = np.asarray(source["states"], dtype=float)
        if not all(metadata.get(key) == value for key, value in provenance.items()):
            raise RuntimeError(f"stale WP10c8t initial segment: {path}")
        restart = load_causal_five_field_bdf_restart(restart_path, context)
        if not (
            restart.provenance == provenance
            and np.array_equal(restart.state_vector, states[-1])
            and states.shape == (subdivisions + 1, initial.size)
        ):
            raise RuntimeError("cached WP10c8t initial state/history differ")
        if parent is not None and not np.array_equal(states, parent["states"]):
            raise RuntimeError("cached WP10c8t parent replay is not bitwise")
        return {
            "states": states,
            "summary": metadata["summary"],
            "restart": restart,
            "restart_evidence": _restart_evidence(restart_path, restart),
            "path": path,
            "sha256": _sha256(path),
            "bitwise_parent_replay": bool(parent is not None),
            "cached": True,
        }

    snapshots = [initial.copy()]

    def progress(completed, total, state, _history) -> None:
        snapshots.append(np.asarray(state, dtype=float).copy())
        print(
            f"WP10c8t N{_contract_n_cells(contract)} "
            f"{resolution} {side} history: "
            f"step {completed}/{total}",
            flush=True,
        )

    started = time.perf_counter()
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        initial,
        np.zeros_like(initial),
        timestep,
        PARENT_DURATION_SECONDS,
        subdivisions,
        wp10c8p._step_config(),
        startup_with_bdf1=True,
        progress=progress,
    )
    wall_seconds = time.perf_counter() - started
    states = np.asarray(snapshots, dtype=float)
    if not (
        result.passed
        and result.history is not None
        and states.shape == (subdivisions + 1, initial.size)
    ):
        raise RuntimeError("WP10c8t initial history trajectory failed")
    bitwise_parent = bool(
        parent is not None
        and states.shape == parent["states"].shape
        and np.array_equal(states, parent["states"])
    )
    if parent is not None and not bitwise_parent:
        raise RuntimeError("WP10c8t replay did not reproduce WP10c8s bitwise")
    restart = CausalFiveFieldBDFRestart(
        state_vector=result.state_vector,
        history=result.history,
        elapsed_time=PARENT_DURATION_SECONDS,
        dt_next=timestep,
        next_order=2,
        accepted_steps=subdivisions,
        rejected_attempts=0,
        provenance=provenance,
    )
    restart = _save_checked_restart(
        path=restart_path,
        context=context,
        restart=restart,
    )
    summary = _result_row(result, wall_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        metadata_json=np.asarray(
            json.dumps(
                {**provenance, "summary": _plain(summary)},
                sort_keys=True,
                allow_nan=False,
            )
        ),
    )
    return {
        "states": states,
        "summary": summary,
        "restart": restart,
        "restart_evidence": _restart_evidence(restart_path, restart),
        "path": path,
        "sha256": _sha256(path),
        "bitwise_parent_replay": bitwise_parent,
        "cached": False,
    }


def _run_continuation_segment(
    *,
    contract: dict,
    resolution: str,
    side: str,
    start: float,
    target: float,
    restart: CausalFiveFieldBDFRestart,
    force: bool,
) -> dict:
    context = contract["context"]
    timestep = TIMESTEP_SECONDS[resolution]
    subdivisions = int(round((target - start) / timestep))
    duration = subdivisions * timestep
    path = _segment_path(contract, resolution, side, target)
    restart_path = _restart_path(contract, resolution, side, target)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "purpose": "history_preserving_inner_mode_continuation",
        "case_id": CASE_ID,
        "resolution": resolution,
        "side": side,
        "segment_start_seconds": start,
        "segment_target_seconds": target,
        "segment_subdivisions": subdivisions,
        "timestep_seconds": timestep,
        "starting_restart_state_sha256": _array_sha256(
            restart.state_vector
        ),
        "starting_history_increment_sha256": _array_sha256(
            restart.history.previous_physical_increment
        ),
        "startup": "continued_bdf2_without_new_bdf1",
    }
    if subdivisions < 1 or not np.isclose(
        duration,
        target - start,
        rtol=0.0,
        atol=16.0 * np.finfo(float).eps,
    ):
        raise RuntimeError("WP10c8t continuation segment is invalid")
    if path.exists() and restart_path.exists() and not force:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            states = np.asarray(source["states"], dtype=float)
        if not all(metadata.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale WP10c8t segment: {path}")
        terminal = load_causal_five_field_bdf_restart(
            restart_path,
            context,
        )
        if not (
            states.shape == (subdivisions, restart.state_vector.size)
            and np.array_equal(terminal.state_vector, states[-1])
        ):
            raise RuntimeError("cached WP10c8t segment state/history differ")
        return {
            "states": states,
            "summary": metadata["summary"],
            "restart": terminal,
            "restart_evidence": _restart_evidence(
                restart_path,
                terminal,
            ),
            "path": path,
            "sha256": _sha256(path),
            "cached": True,
        }

    snapshots: list[np.ndarray] = []

    def progress(completed, total, state, _history) -> None:
        snapshots.append(np.asarray(state, dtype=float).copy())
        print(
            f"WP10c8t N{_contract_n_cells(contract)} "
            f"{resolution} {side} to {target:.3f} s: "
            f"step {completed}/{total}",
            flush=True,
        )

    started = time.perf_counter()
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        restart.state_vector,
        restart.history.previous_physical_increment,
        timestep,
        duration,
        subdivisions,
        wp10c8p._step_config(),
        startup_with_bdf1=False,
        initial_history=restart.history,
        progress=progress,
    )
    wall_seconds = time.perf_counter() - started
    states = np.asarray(snapshots, dtype=float)
    if not (
        result.passed
        and result.history is not None
        and states.shape == (subdivisions, restart.state_vector.size)
    ):
        raise RuntimeError(
            f"WP10c8t {resolution} {side} continuation failed"
        )
    terminal = CausalFiveFieldBDFRestart(
        state_vector=result.state_vector,
        history=result.history,
        elapsed_time=target,
        dt_next=timestep,
        next_order=2,
        accepted_steps=int(round(target / timestep)),
        rejected_attempts=0,
        provenance=expected,
    )
    terminal = _save_checked_restart(
        path=restart_path,
        context=context,
        restart=terminal,
    )
    summary = _result_row(result, wall_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        metadata_json=np.asarray(
            json.dumps(
                {**expected, "summary": _plain(summary)},
                sort_keys=True,
                allow_nan=False,
            )
        ),
    )
    return {
        "states": states,
        "summary": summary,
        "restart": terminal,
        "restart_evidence": _restart_evidence(
            restart_path,
            terminal,
        ),
        "path": path,
        "sha256": _sha256(path),
        "cached": False,
    }


def _run_or_load_trajectory(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    resolution: str,
    side: str,
    force: bool,
) -> dict:
    path = _trajectory_path(contract, resolution, side)
    initial = _run_initial_segment(
        contract=contract,
        case=case,
        resolution=resolution,
        side=side,
        force=force,
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": CASE_ID,
        "resolution": resolution,
        "side": side,
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "total_subdivisions": TOTAL_SUBDIVISIONS[resolution],
        "timestep_seconds": TIMESTEP_SECONDS[resolution],
        "initial_segment_sha256": initial["sha256"],
        "continuation": "exact_history_bdf2_without_new_bdf1_startup",
    }
    final_restart_path = _restart_path(
        contract,
        resolution,
        side,
        TARGET_DURATION_SECONDS,
    )
    if path.exists() and final_restart_path.exists() and not force:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            states = np.asarray(source["states"], dtype=float)
        if not all(metadata.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale WP10c8t trajectory: {path}")
        restart = load_causal_five_field_bdf_restart(
            final_restart_path,
            contract["context"],
        )
        if not (
            states.shape
            == (
                TOTAL_SUBDIVISIONS[resolution] + 1,
                restart.state_vector.size,
            )
            and np.array_equal(restart.state_vector, states[-1])
        ):
            raise RuntimeError("cached WP10c8t trajectory state/history differ")
        return {
            "states": states,
            "summary": metadata["summary"],
            "segments": metadata["segments"],
            "initial_history": initial["restart_evidence"],
            "bitwise_parent_replay": initial["bitwise_parent_replay"],
            "final_restart": restart,
            "final_restart_evidence": _restart_evidence(
                final_restart_path,
                restart,
            ),
            "path": path,
            "sha256": _sha256(path),
            "cached": True,
        }

    state_parts = [np.asarray(initial["states"], dtype=float)]
    rows = [initial["summary"]]
    segment_evidence = []
    current_restart = initial["restart"]
    current_time = PARENT_DURATION_SECONDS
    for target in SEGMENT_TARGETS_SECONDS:
        segment = _run_continuation_segment(
            contract=contract,
            resolution=resolution,
            side=side,
            start=current_time,
            target=target,
            restart=current_restart,
            force=force,
        )
        state_parts.append(segment["states"])
        rows.append(segment["summary"])
        segment_evidence.append(
            {
                "target_time_seconds": target,
                "path": _relative(segment["path"]),
                "sha256": segment["sha256"],
                "restart": segment["restart_evidence"],
                "cached": segment["cached"],
            }
        )
        current_restart = segment["restart"]
        current_time = target
    states = np.concatenate(state_parts, axis=0)
    expected_shape = (
        TOTAL_SUBDIVISIONS[resolution] + 1,
        current_restart.state_vector.size,
    )
    if states.shape != expected_shape:
        raise RuntimeError("WP10c8t complete trajectory shape differs")
    summary = wp10c8q._merge_fixed_result_rows(rows)
    metadata = {
        **expected,
        "summary": _plain(summary),
        "segments": segment_evidence,
        "initial_history": initial["restart_evidence"],
        "bitwise_parent_replay": initial["bitwise_parent_replay"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        ),
    )
    return {
        "states": states,
        "summary": summary,
        "segments": segment_evidence,
        "initial_history": initial["restart_evidence"],
        "bitwise_parent_replay": initial["bitwise_parent_replay"],
        "final_restart": current_restart,
        "final_restart_evidence": _restart_evidence(
            final_restart_path,
            current_restart,
        ),
        "path": path,
        "sha256": _sha256(path),
        "cached": False,
    }


def _trajectory_diagnostics_with_rates(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    operator_arrays: dict[str, np.ndarray],
    states: np.ndarray,
    subdivisions: int,
    rate_cache: dict[str, tuple[np.ndarray, dict, dict[str, np.ndarray]]],
    compute_fresh_rates: bool,
    duration_seconds: float = TARGET_DURATION_SECONDS,
    output_offsets_seconds: tuple[float, ...] = OUTPUT_OFFSETS_SECONDS,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = contract["context"]
    baseline = causal_five_field_observable_snapshot(
        context,
        contract["anchor_vector"],
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    radius_rg = context.grid.centers / context.grid.gravitational_radius
    edges_rg = context.grid.edges / context.grid.gravitational_radius
    _common, interpolation = wp10c8q.wp10c8i._common_log_h_interpolation(
        radius_rg,
        edges_rg,
    )
    coordinate_names = tuple(
        str(value) for value in case["coordinate_names"]
    )
    coordinate_scales = np.asarray(case["coordinate_scales"], dtype=float)
    summary, arrays = wp10c8p._trajectory_diagnostics(
        context=context,
        states=states,
        subdivisions=subdivisions,
        shell_edges_rg=contract["shell_edges_rg"],
        baseline_snapshot=baseline,
        anchor_interface_scales=np.asarray(
            case["interface_flux_scales"],
            dtype=float,
        ),
        coordinate_names=coordinate_names,
        coordinate_scales=coordinate_scales,
        primitive_scales=np.asarray(
            operator_arrays["primitive_column_scales"],
            dtype=float,
        ),
        conservation_scales=np.asarray(
            operator_arrays["conservation_row_scales"],
            dtype=float,
        ),
        common_interpolation=interpolation,
        compute_fresh_rates=False,
        rate_cache={},
        duration_seconds=duration_seconds,
        output_offsets_seconds=output_offsets_seconds,
    )
    if not compute_fresh_rates:
        arrays["scaled_primitive_rates_per_s"] = np.zeros(
            (
                arrays["output_times"].size,
                context.grid.centers.size * 5,
            ),
            dtype=float,
        )
        return summary, arrays

    coordinate_map = wp10c8o._coordinate_evaluator(
        context,
        contract["shell_edges_rg"],
    )
    rates = []
    scaled_primitive_rates = []
    audits = []
    for output_time, primitives in zip(
        arrays["output_times"],
        arrays["output_primitives"],
        strict=True,
    ):
        key = _array_sha256(primitives)
        if key not in rate_cache:
            print(
                f"WP10c8t N{_contract_n_cells(contract)} fresh-rate audit: "
                f"t={float(output_time):.6g} s",
                flush=True,
            )
            rate_cache[key] = wp10c8o._fresh_coordinate_rate(
                context=context,
                primitives=np.asarray(primitives, dtype=float).ravel(),
                coordinate_evaluator=coordinate_map,
                primitive_scales=np.asarray(
                    operator_arrays["primitive_column_scales"],
                    dtype=float,
                ),
                conservation_scales=np.asarray(
                    operator_arrays["conservation_row_scales"],
                    dtype=float,
                ),
                coordinate_scales=coordinate_scales,
                binding_dae_storage_audit=False,
            )
        rate, audit, details = rate_cache[key]
        rates.append(np.asarray(rate, dtype=float))
        audits.append(audit)
        scaled_primitive_rates.append(
            np.asarray(details["scaled_primitive_rate_per_s"], dtype=float)
        )
    arrays["normalized_coordinate_rates"] = np.asarray(rates, dtype=float)
    arrays["scaled_primitive_rates_per_s"] = np.asarray(
        scaled_primitive_rates,
        dtype=float,
    )
    summary = {
        **summary,
        "fresh_rates_evaluated": True,
        "fresh_rate_audits": audits,
        "all_fresh_rate_audits_passed": bool(
            all(row["passed"] for row in audits)
        ),
    }
    return summary, arrays


def _pair_arrays(
    *,
    minus: dict[str, np.ndarray],
    plus: dict[str, np.ndarray],
    coordinate_scales: np.ndarray,
    loading_time_seconds: float,
) -> dict[str, np.ndarray]:
    spread = wp10c8s._pair_spread_arrays(
        minus=minus,
        plus=plus,
        coordinate_scales=coordinate_scales,
        loading_time_seconds=loading_time_seconds,
    )
    signed_slip = 0.5 * (
        np.asarray(plus["coordinates"], dtype=float)
        - np.asarray(minus["coordinates"], dtype=float)
    ) / coordinate_scales[None, :]
    signed_slow_rate = (
        0.5
        * (
            np.asarray(plus["normalized_coordinate_rates"], dtype=float)
            - np.asarray(minus["normalized_coordinate_rates"], dtype=float)
        )
        * float(loading_time_seconds)
        / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
    )
    slow_times = np.asarray(spread["times"], dtype=float) / float(
        loading_time_seconds
    )
    rate_integrated_slip = causal_cumulative_trapezoid(
        slow_times,
        signed_slow_rate,
    )
    state_change_slip = signed_slip - signed_slip[0]
    spread.update(
        {
            "signed_coordinate_slip": signed_slip,
            "signed_slow_rate_half_difference": signed_slow_rate,
            "rate_integrated_slip": rate_integrated_slip,
            "state_change_slip": state_change_slip,
            "rate_integral_reconciliation_defect": (
                state_change_slip - rate_integrated_slip
            ),
        }
    )
    return spread


def _localization(
    *,
    context,
    minus: dict[str, np.ndarray],
    plus: dict[str, np.ndarray],
    operator_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    n_cells = int(context.grid.centers.size)
    radius_rg = context.grid.centers / context.grid.gravitational_radius
    shell_edges = np.asarray(operator_arrays["shell_edge_indices"], dtype=int)
    amplitudes = np.asarray(
        operator_arrays["physical_input_amplitudes"],
        dtype=float,
    ).reshape(n_cells, 5)
    state_half = 0.5 * (
        np.asarray(plus["output_primitives"], dtype=float)
        - np.asarray(minus["output_primitives"], dtype=float)
    ) / amplitudes[None, :, :]
    rate_half = 0.5 * (
        np.asarray(plus["scaled_primitive_rates_per_s"], dtype=float)
        - np.asarray(minus["scaled_primitive_rates_per_s"], dtype=float)
    ).reshape((-1, n_cells, 5))
    state_support = [
        wp10c8s._dominant_support(
            values,
            radius_rg=radius_rg,
            shell_edge_indices=shell_edges,
        )
        for values in state_half
    ]
    rate_support = [
        wp10c8s._dominant_support(
            values,
            radius_rg=radius_rg,
            shell_edge_indices=shell_edges,
        )
        for values in rate_half
    ]
    return {
        "times_seconds": minus["output_times"],
        "state_support": state_support,
        "primitive_rate_support": rate_support,
        "all_state_outputs_localized_in_one_shell": bool(
            all(row["localized_in_one_shell"] for row in state_support)
        ),
        "all_rate_outputs_localized_in_one_shell": bool(
            all(row["localized_in_one_shell"] for row in rate_support)
        ),
        "final_state_controlling_shell": state_support[-1][
            "controlling_shell"
        ],
        "final_rate_controlling_shell": rate_support[-1][
            "controlling_shell"
        ],
    }, {
        "times": np.asarray(minus["output_times"], dtype=float),
        "radius_rg": np.asarray(radius_rg, dtype=float),
        "shell_edge_indices": shell_edges,
        "state_half_difference_over_amplitude": state_half,
        "scaled_primitive_rate_half_difference_per_s": rate_half,
    }


def _decision(
    *,
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    all_contracts_passed: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    if not (
        np.array_equal(coarse["times"], fine["times"])
        and np.array_equal(coarse["full_names"], fine["full_names"])
    ):
        raise RuntimeError("WP10c8t coarse/fine schemas differ")
    uncertainty, upper = causal_refined_spread_upper_bound(
        coarse["full_spreads"],
        fine["full_spreads"],
    )
    lower = np.maximum(fine["full_spreads"] - uncertainty, 0.0)
    slip_uncertainty = np.abs(
        fine["signed_coordinate_slip"]
        - coarse["signed_coordinate_slip"]
    )
    slip_upper = (
        np.abs(fine["signed_coordinate_slip"]) + slip_uncertainty
    )
    relative_mask = (
        fine["full_spreads"] >= TEMPORAL_RELATIVE_SPREAD_FLOOR
    )
    maximum_relative = float(
        np.max(
            uncertainty[relative_mask]
            / np.maximum(
                fine["full_spreads"][relative_mask],
                np.finfo(float).tiny,
            )
        )
        if np.any(relative_mask)
        else 0.0
    )
    temporal_passed = bool(
        float(np.max(uncertainty)) <= TEMPORAL_UNCERTAINTY_GATE
        and maximum_relative <= TEMPORAL_RELATIVE_UNCERTAINTY_GATE
    )
    significant = upper[0] >= NONLINEAR_SIGNIFICANCE_GATE
    factor_two = bool(
        np.any(significant)
        and np.all(
            upper[-1, significant]
            <= upper[0, significant] / MINIMUM_HEALING_FACTOR
        )
    )
    late_difference = np.diff(upper[-3:, significant], axis=0)
    late_uncertainty = (
        uncertainty[-2:, significant]
        + uncertainty[-3:-1, significant]
    )
    no_regrowth = bool(
        np.any(significant)
        and np.all(late_difference <= late_uncertainty)
    )
    final_gate = bool(
        np.any(significant)
        and np.all(upper[-1, significant] <= HEALING_FINAL_SPREAD_GATE)
    )
    persistent = bool(
        np.any(significant)
        and np.any(lower[-1, significant] > HEALING_FINAL_SPREAD_GATE)
    )
    slip_small = bool(float(np.max(slip_upper[-1])) <= MAXIMUM_INITIAL_SLIP)
    if np.any(significant):
        initial_maximum = max(
            float(np.max(upper[0, significant])),
            np.finfo(float).tiny,
        )
        final_maximum = max(
            float(np.max(upper[-1, significant])),
            np.finfo(float).tiny,
        )
        e_folds = float(np.log(initial_maximum / final_maximum))
    else:
        initial_maximum = 0.0
        final_maximum = 0.0
        e_folds = 0.0
    healed = bool(
        all_contracts_passed
        and temporal_passed
        and factor_two
        and no_regrowth
        and final_gate
        and slip_small
    )
    if healed:
        classification = "n64_fast_initial_layer_with_small_slip_supported"
    elif all_contracts_passed and persistent:
        classification = "n64_persistent_localized_inner_mode_through_0p125s"
    else:
        classification = "n64_inner_mode_healing_numerically_inconclusive"
    relaxation_fit_prerequisites = {
        "temporally_resolved_curve": temporal_passed,
        "minimum_two_apparent_e_folds": bool(
            e_folds >= MINIMUM_RELAXATION_FIT_EFOLDS
        ),
        "no_late_regrowth": no_regrowth,
        "multiple_amplitudes_tested": False,
        "n128_nonlinear_confirmation": False,
    }
    return {
        "classification": classification,
        "all_n64_contracts_passed": all_contracts_passed,
        "maximum_temporal_uncertainty": float(np.max(uncertainty)),
        "maximum_relative_temporal_uncertainty": maximum_relative,
        "temporal_curve_passed": temporal_passed,
        "initial_maximum_uncertainty_inclusive_spread": float(
            np.max(upper[0])
        ),
        "final_maximum_uncertainty_inclusive_spread": float(
            np.max(upper[-1])
        ),
        "final_maximum_uncertainty_exclusive_lower_spread": float(
            np.max(lower[-1])
        ),
        "factor_two_decay_passed": factor_two,
        "no_late_regrowth_passed": no_regrowth,
        "final_healing_gate_passed": final_gate,
        "persistence_separated_from_healing_gate": persistent,
        "measured_minimum_controlling_e_folds": e_folds,
        "final_maximum_accumulated_slip_upper_bound": float(
            np.max(slip_upper[-1])
        ),
        "accumulated_slip_gate": MAXIMUM_INITIAL_SLIP,
        "small_accumulated_slip_passed": slip_small,
        "natural_healing_with_small_slip_passed": healed,
        "relaxation_fit_prerequisites": relaxation_fit_prerequisites,
        "relaxation_fit_authorized": bool(
            all(relaxation_fit_prerequisites.values())
        ),
        "n128_architecture_confirmation_required": bool(
            classification
            != "n64_inner_mode_healing_numerically_inconclusive"
        ),
    }, {
        "temporal_uncertainty": uncertainty,
        "uncertainty_inclusive_spreads": upper,
        "uncertainty_exclusive_lower_spreads": lower,
        "significant_initial_output_mask": significant,
        "coordinate_slip_temporal_uncertainty": slip_uncertainty,
        "coordinate_slip_upper_bounds": slip_upper,
    }


def _run_diagnostics(
    *,
    contract: dict,
    case: dict[str, np.ndarray],
    operator_arrays: dict[str, np.ndarray],
    trajectories: dict[str, dict],
    compute_fresh_rates: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    loading_time = causal_five_field_loading_time(
        contract["context"],
        contract["anchor_vector"],
    )
    diagnostics = {}
    all_arrays: dict[str, np.ndarray] = {}
    rate_cache: dict[
        str,
        tuple[np.ndarray, dict, dict[str, np.ndarray]],
    ] = {}
    for resolution in ("coarse", "fine"):
        for side in ("minus", "plus"):
            label = f"{resolution}_{side}"
            summary, arrays = _trajectory_diagnostics_with_rates(
                contract=contract,
                case=case,
                operator_arrays=operator_arrays,
                states=trajectories[label]["states"],
                subdivisions=TOTAL_SUBDIVISIONS[resolution],
                rate_cache=rate_cache,
                compute_fresh_rates=compute_fresh_rates,
            )
            diagnostics[label] = summary
            all_arrays.update(
                {f"{label}_{name}": values for name, values in arrays.items()}
            )

    pair_arrays = {}
    pair_ledgers = {}
    localizations = {}
    for resolution in ("coarse", "fine"):
        minus = {
            name.removeprefix(f"{resolution}_minus_"): values
            for name, values in all_arrays.items()
            if name.startswith(f"{resolution}_minus_")
        }
        plus = {
            name.removeprefix(f"{resolution}_plus_"): values
            for name, values in all_arrays.items()
            if name.startswith(f"{resolution}_plus_")
        }
        pair = _pair_arrays(
            minus=minus,
            plus=plus,
            coordinate_scales=np.asarray(case["coordinate_scales"], dtype=float),
            loading_time_seconds=loading_time,
        )
        ledger_summary, ledger_arrays = wp10c8p._pair_diagnostics(
            minus=minus,
            plus=plus,
            coordinate_scales=np.asarray(case["coordinate_scales"], dtype=float),
            coordinate_names=tuple(
                str(value) for value in case["coordinate_names"]
            ),
        )
        localization_summary, localization_arrays = _localization(
            context=contract["context"],
            minus=minus,
            plus=plus,
            operator_arrays=operator_arrays,
        )
        pair_arrays[resolution] = pair
        pair_ledgers[resolution] = ledger_summary
        localizations[resolution] = localization_summary
        all_arrays.update(
            {
                f"{resolution}_pair_{name}": values
                for name, values in pair.items()
            }
        )
        all_arrays.update(
            {
                f"{resolution}_ledger_{name}": values
                for name, values in ledger_arrays.items()
            }
        )
        all_arrays.update(
            {
                f"{resolution}_localization_{name}": values
                for name, values in localization_arrays.items()
            }
        )

    trajectory_contracts = bool(
        all(row["summary"]["passed"] for row in trajectories.values())
        and trajectories["coarse_minus"]["bitwise_parent_replay"]
        and trajectories["coarse_plus"]["bitwise_parent_replay"]
    )
    diagnostic_contracts = bool(
        compute_fresh_rates
        and all(
            row["maximum_physical_mje_shell_ledger_relative_defect"]
            <= MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT
            and row["maximum_flux_reconstruction_defect"]
            <= wp10c8p.MAXIMUM_FLUX_RECONSTRUCTION_DEFECT
            and row["all_output_state_gates_passed"]
            and row["all_fresh_rate_audits_passed"]
            for row in diagnostics.values()
        )
    )
    decision, decision_arrays = _decision(
        coarse=pair_arrays["coarse"],
        fine=pair_arrays["fine"],
        all_contracts_passed=bool(
            trajectory_contracts and diagnostic_contracts
        ),
    )
    all_arrays.update(
        {f"decision_{name}": values for name, values in decision_arrays.items()}
    )
    maximum_reconciliation = float(
        np.max(
            np.abs(
                pair_arrays["fine"]["rate_integral_reconciliation_defect"]
            )
        )
    )
    return {
        "loading_time_seconds": loading_time,
        "trajectory_contracts_passed": trajectory_contracts,
        "diagnostic_contracts_passed": diagnostic_contracts,
        "trajectory_diagnostics": diagnostics,
        "pair_ledgers": pair_ledgers,
        "localization": localizations,
        "maximum_rate_integral_slip_reconciliation_defect": (
            maximum_reconciliation
        ),
        "decision": decision,
    }, all_arrays


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-only",
        choices=(
            "coarse-minus",
            "coarse-plus",
            "fine-minus",
            "fine-plus",
        ),
        default=None,
        help="Populate exactly one expensive trajectory and exit.",
    )
    parser.add_argument(
        "--history-only",
        choices=(
            "coarse-minus",
            "coarse-plus",
            "fine-minus",
            "fine-plus",
        ),
        default=None,
        help="Populate only the exact t=0.025 s state/history artifact.",
    )
    parser.add_argument(
        "--skip-fresh-rates",
        action="store_true",
        help="Development-only: assemble without binding fresh rates.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the selected WP10c8t caches.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def main() -> None:
    _validate_schedule()
    args = _arguments()
    parent, contract, case, operator = _load_contract()

    if args.history_only is not None:
        resolution, side = args.history_only.split("-", maxsplit=1)
        result = _run_initial_segment(
            contract=contract,
            case=case,
            resolution=resolution,
            side=side,
            force=args.force,
        )
        print(
            json.dumps(
                _plain(
                    {
                        "work_package": WORK_PACKAGE,
                        "history_only": args.history_only,
                        "path": _relative(result["path"]),
                        "sha256": result["sha256"],
                        "restart": result["restart_evidence"],
                        "bitwise_parent_replay": (
                            result["bitwise_parent_replay"]
                        ),
                        "cached": result["cached"],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.trajectory_only is not None:
        resolution, side = args.trajectory_only.split("-", maxsplit=1)
        result = _run_or_load_trajectory(
            contract=contract,
            case=case,
            resolution=resolution,
            side=side,
            force=args.force,
        )
        print(
            json.dumps(
                _plain(
                    {
                        "work_package": WORK_PACKAGE,
                        "trajectory_only": args.trajectory_only,
                        "path": _relative(result["path"]),
                        "sha256": result["sha256"],
                        "final_restart": result["final_restart_evidence"],
                        "bitwise_parent_replay": (
                            result["bitwise_parent_replay"]
                        ),
                        "summary": result["summary"],
                        "cached": result["cached"],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    trajectories = {}
    for resolution in ("coarse", "fine"):
        for side in ("minus", "plus"):
            label = f"{resolution}_{side}"
            trajectories[label] = _run_or_load_trajectory(
                contract=contract,
                case=case,
                resolution=resolution,
                side=side,
                force=args.force,
            )
    diagnostics, arrays = _run_diagnostics(
        contract=contract,
        case=case,
        operator_arrays=operator["arrays"],
        trajectories=trajectories,
        compute_fresh_rates=not args.skip_fresh_rates,
    )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    arrays_path = args.arrays if args.arrays.is_absolute() else ROOT / args.arrays
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT / "scripts/run_causal_complete_rate_healing_wp10c8s.py",
        ROOT / "scripts/run_causal_extended_healing_wp10c8q.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_bdf_restart.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_bdf_evolution.py",
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "case_id": CASE_ID,
        "scope": {
            "production_physics_changed": False,
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "reduced_evolution_run": False,
            "relaxation_law_fit": False,
            "n128_nonlinear_confirmation_run": False,
        },
        "contract": {
            "parent_duration_seconds": PARENT_DURATION_SECONDS,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "output_offsets_seconds": OUTPUT_OFFSETS_SECONDS,
            "segment_targets_seconds": SEGMENT_TARGETS_SECONDS,
            "timestep_seconds": TIMESTEP_SECONDS,
            "total_subdivisions": TOTAL_SUBDIVISIONS,
            "parent_subdivisions": PARENT_SUBDIVISIONS,
        },
        "gates": {
            "minimum_significant_initial_spread": (
                NONLINEAR_SIGNIFICANCE_GATE
            ),
            "maximum_final_healing_spread": HEALING_FINAL_SPREAD_GATE,
            "maximum_accumulated_initial_slip": MAXIMUM_INITIAL_SLIP,
            "minimum_healing_factor": MINIMUM_HEALING_FACTOR,
            "maximum_temporal_uncertainty": TEMPORAL_UNCERTAINTY_GATE,
            "maximum_relative_temporal_uncertainty": (
                TEMPORAL_RELATIVE_UNCERTAINTY_GATE
            ),
            "maximum_shell_ledger_relative_defect": (
                MAXIMUM_SHELL_LEDGER_RELATIVE_DEFECT
            ),
        },
        "authorization": {
            "wp10c8s_json_path": _relative(PARENT_JSON),
            "wp10c8s_json_sha256": _sha256(PARENT_JSON),
            "wp10c8s_arrays_path": _relative(PARENT_ARRAYS),
            "wp10c8s_arrays_sha256": _sha256(PARENT_ARRAYS),
        },
        "trajectory_provenance": {
            label: {
                "path": _relative(row["path"]),
                "sha256": row["sha256"],
                "initial_history": row["initial_history"],
                "final_restart": row["final_restart_evidence"],
                "bitwise_parent_replay": row["bitwise_parent_replay"],
                "segments": row["segments"],
                "cached": row["cached"],
                "summary": row["summary"],
            }
            for label, row in trajectories.items()
        },
        "diagnostics": diagnostics,
        "decision": diagnostics["decision"]["classification"],
        "next_action": (
            "run_exact_n128_architecture_confirmation_at_0p125s"
            if diagnostics["decision"][
                "n128_architecture_confirmation_required"
            ]
            else "repair_only_the_failed_n64_temporal_or_physical_gate"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": output["decision"],
                "next_action": output["next_action"],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
