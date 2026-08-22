#!/usr/bin/env python3
"""Acquire and independently replay one original-free-field physical cycle.

The execution starts from the accepted 20 ms full-model primitive state.  It
never glues a fixed-Q trajectory into physical time.  Exact original-free-field
witnesses create local conservative patches; the patches then advance the
82-coordinate macro ledger and a mode-local hidden amplitude without a truth
call.  Every candidate is exactly retracted and physically audited before it
can become history.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import (  # noqa: E402
    ConservativeCoordinateSplit,
    ConservativeHeunEngine,
    ConservativeHiddenAmplitudeModel,
    HiddenAmplitudeState,
    HystereticModeSelector,
    LocalAffineReducedPatch,
    canonical_rate_basis,
    relative_projection_defects,
)
import run_causal_inner_adaptive_complete_cycle_manifest_wp10c9d6c7c3b5c4f25fd as manifest  # noqa: E402
import run_causal_inner_arclength_segment_wp10c9d6c7c3b5c4f25f5 as arclength  # noqa: E402
import run_causal_inner_hot_free_field_rom_preflight_wp10c9d6c7c3b5c4f25f8 as hot  # noqa: E402
import run_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3 as cold_engine  # noqa: E402
import run_causal_inner_phase_collocation_preflight_manifest_wp10c9d6c7c3b5c4f25e6 as cold_rates  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fe"
PASS_CLASSIFICATION = "complete_cycle_atlas_acquired_and_independently_replayed"
BUDGET_CLASSIFICATION = "complete_cycle_inconclusive_acquisition_budget_exhausted"
NO_RETURN_CLASSIFICATION = "complete_cycle_not_observed_within_frozen_horizon"
PHYSICAL_CLASSIFICATION = "complete_cycle_original_free_field_physical_gate_failed"
VALIDATION_CLASSIFICATION = "complete_cycle_atlas_blind_or_step_halving_validation_failed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ff_multi_anchor_cycle_map_manifest"
ARTIFACT = "causal_inner_adaptive_complete_cycle_execution_wp10c9d6c7c3b5c4f25fe"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_COMPLETE_CYCLE_EXECUTION_"
    "WP10C9D6C7C3B5C4F25FE_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_complete_cycle_execution_"
    "wp10c9d6c7c3b5c4f25fe.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_complete_cycle_execution_"
    "wp10c9d6c7c3b5c4f25fe.py"
)

METHOD_COMPONENT_TOLERANCE = 1.0e-12
CENTERED_STORAGE_ACTION_TOLERANCE = 1.0e-7
COORDINATE_CONDITION_TOLERANCE = 2.5e3
COORDINATE_RETRACTION_TOLERANCE = 5.0e-10
GAUGE_RETRACTION_TOLERANCE = 5.0e-10
LOCAL_AXIS_RATE_TOLERANCE = 5.0e-2
LOCAL_HIDDEN_RATE_TOLERANCE = 5.0e-2
DIAGONAL_HIDDEN_FRACTION = 0.25
SECTION_DEPARTURE_FRACTION = 1.0e-6


def _helper():
    return manifest._helper()


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _relative_increment(
    candidate: np.ndarray,
    reference: np.ndarray,
    anchor: np.ndarray,
) -> float:
    return float(
        np.linalg.norm((np.asarray(candidate) - anchor) - (np.asarray(reference) - anchor))
        / max(
            float(np.linalg.norm(np.asarray(reference) - anchor)),
            np.finfo(float).tiny,
        )
    )


def _fit_affine_axis(
    eta: np.ndarray,
    rates: np.ndarray,
    training: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit f(eta)=intercept+eta*slope from prospectively training witnesses."""

    x = np.asarray(eta, dtype=float)
    values = np.asarray(rates, dtype=float)
    mask = np.asarray(training, dtype=bool)
    if x.ndim != 1 or values.ndim != 2 or len(x) != len(values) or mask.shape != x.shape:
        raise ValueError("axis fit arrays disagree")
    design = np.column_stack((np.ones(np.count_nonzero(mask)), x[mask]))
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("axis fit needs two distinct training coordinates")
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design, values[mask], rcond=None
    )
    return np.asarray(coefficients[0]), np.asarray(coefficients[1])


def _axis_prediction(
    intercept: np.ndarray, slope: np.ndarray, eta: float
) -> np.ndarray:
    return np.asarray(intercept) + float(eta) * np.asarray(slope)


def _section_crossing_fraction(left: float, right: float) -> float | None:
    """Return the exact fraction for a negative-to-nonnegative linear crossing."""

    a = float(left)
    b = float(right)
    if not (a < 0.0 <= b) or b == a:
        return None
    return float(-a / (b - a))


def _bitwise_state(left: HiddenAmplitudeState, right: HiddenAmplitudeState) -> bool:
    return bool(
        np.array_equal(left.macro, right.macro)
        and np.array_equal(left.amplitudes, right.amplitudes)
        and left.forcing_phase == right.forcing_phase
        and left.elapsed_seconds == right.elapsed_seconds
        and left.mode == right.mode
    )


def _state_roundtrip(state: HiddenAmplitudeState) -> HiddenAmplitudeState:
    stream = io.BytesIO()
    np.savez(
        stream,
        macro=state.macro,
        amplitudes=state.amplitudes,
        forcing_phase=np.asarray(state.forcing_phase),
        mode=np.asarray(state.mode),
        elapsed_seconds=np.asarray(state.elapsed_seconds),
    )
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as payload:
        return HiddenAmplitudeState(
            macro=payload["macro"],
            amplitudes=payload["amplitudes"],
            forcing_phase=float(payload["forcing_phase"]),
            mode=str(payload["mode"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
        )


def _coordinate_roundtrip(coordinate: np.ndarray) -> np.ndarray:
    stream = io.BytesIO()
    np.savez(stream, coordinate=np.asarray(coordinate))
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as payload:
        return np.asarray(payload["coordinate"])


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "complete_cycle_execution_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["mathematical_architecture_verified"]
        or not summary["complete_cycle_execution_authorized"]
        or summary["complete_cycle_executed"]
        or summary["authorized_next"] != f"{WORK_PACKAGE}_complete_cycle_execution"
        or contract["authorized_execution"] != summary["authorized_next"]
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("adaptive complete-cycle execution authorization changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("complete-cycle execution requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
        ROOT / hot.THIS_RUNNER,
        Path(arclength.__file__).resolve(),
        Path(arclength._transport().__file__).resolve(),
        Path(arclength._exact_chart().__file__).resolve(),
        Path(arclength._source()._post().exact_rate.rate_source.__file__).resolve(),
    )
    return {str(path.relative_to(ROOT)): helper._sha(path) for path in paths}


def _prepare_scratch(parent_lock: dict) -> dict:
    helper = _helper()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": _source_hashes(),
        "manifest_hashes": parent_lock["manifest_hashes"],
    }
    identity_path = SCRATCH_DIRECTORY / "execution_identity.json"
    if SCRATCH_DIRECTORY.exists():
        if not identity_path.exists() or helper._read(identity_path) != identity:
            raise RuntimeError("complete-cycle scratch identity changed")
    else:
        SCRATCH_DIRECTORY.mkdir(parents=True)
        helper._write_json(identity_path, identity)
    return identity


def _initial_inputs() -> dict:
    helper = _helper()
    candidates_path = (
        cold_engine.manifest.architecture.manifest.CANDIDATE_DIRECTORY
        / "candidate_geometry_arrays.npz"
    )
    candidates = helper._load_npz(candidates_path)
    base = hot.manifest.parent.parent._source()._base_inputs()
    state = np.asarray(candidates["candidate_primitive_states"][-1], dtype=float)
    coordinate = np.asarray(
        candidates["candidate_absolute_y470_coordinates"][-1], dtype=float
    )
    recovered, factors = base["model"].coordinate(state)
    np.testing.assert_array_equal(recovered, coordinate)
    if float(np.min(factors)) < 1.0 - 1.0e-12:
        raise RuntimeError("accepted cold terminal coordinate activated reconstruction")
    geometry = base["geometry"]
    split = ConservativeCoordinateSplit(
        macro_restriction=geometry["R"],
        macro_lift=geometry["L"],
        hidden_dual=geometry["Q"],
        hidden_lift=geometry["Z"],
        tolerance=5.0e-11,
    )
    return {
        "base": base,
        "model": base["model"],
        "split": split,
        "state": state,
        "coordinate": coordinate,
        "candidate_times_seconds": np.asarray(candidates["candidate_times_seconds"]),
        "candidate_coordinates": np.asarray(
            candidates["candidate_absolute_y470_coordinates"]
        ),
        "candidate_states": np.asarray(candidates["candidate_primitive_states"]),
        "candidates_path": candidates_path,
    }


def _cold_mode_basis(split: ConservativeCoordinateSplit) -> tuple[np.ndarray, dict]:
    helper = _helper()
    payload = helper._load_npz(cold_rates.COLD_RATE_ARRAYS)
    labels = ("02", "05", "08", "12")
    hidden = []
    coordinate_rates = []
    for label in labels:
        jacobian = np.asarray(payload[f"candidate_{label}ms__coordinate_jacobian470x560"])
        free = np.asarray(payload[f"candidate_{label}ms__scaled_free_rate560_per_s"])
        rate = jacobian @ free
        coordinate_rates.append(rate)
        hidden.append(split.split_rate(rate)[1])
    hidden = np.asarray(hidden)
    basis, singular, energy = canonical_rate_basis(hidden[[0, 2]], 2)
    defects = relative_projection_defects(hidden, basis)
    if float(np.max(defects[[1, 3]])) > LOCAL_HIDDEN_RATE_TOLERANCE:
        raise RuntimeError("accepted cold free-rate basis no longer passes held-out data")
    return basis, {
        "coordinate_rates": np.asarray(coordinate_rates),
        "hidden_rates": hidden,
        "projection_defects": defects,
        "singular_values": singular,
        "cumulative_energy": energy,
    }


def _hot_mode_basis() -> tuple[np.ndarray, dict]:
    helper = _helper()
    arrays = helper._load_npz(
        manifest.OFF_AXIS_DIRECTORY / "hot_mode_off_axis_arrays.npz"
    )
    basis = np.asarray(arrays["extended_hidden_rate_basis388xr"])
    return basis, {
        "hidden_rates": np.vstack((
            np.asarray(arrays["off_axis_hidden_free_rates3x388_per_s"]),
            np.zeros((0, basis.shape[0])),
        )),
    }


def _build_retraction_context(inputs: dict, state: np.ndarray, coordinate: np.ndarray) -> dict:
    exact_chart = arclength._exact_chart()
    model = inputs["model"]
    recovered, _factors = model.coordinate(state)
    np.testing.assert_allclose(recovered, coordinate, rtol=0.0, atol=1.0e-13)
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(
        model, state
    )
    gauge_basis = exact_chart._canonical_null_basis(coordinate_jacobian)
    began = time.perf_counter()
    augmented, augmented_metrics = exact_chart._augmented_jacobian(
        model, state, gauge_basis
    )
    wall = float(time.perf_counter() - began)
    if (
        coordinate_metrics["rank"] != exact_chart.COORDINATE_DIMENSION
        or coordinate_metrics["condition_number"] > COORDINATE_CONDITION_TOLERANCE
        or augmented_metrics["augmented_rank"] != exact_chart.PHYSICAL_DIMENSION
    ):
        raise RuntimeError("local exact coordinate anchor lost rank or conditioning")
    return {
        "state": np.asarray(state),
        "coordinate": np.asarray(coordinate),
        "model_state": np.asarray(model.decoded_state(coordinate)),
        "gauge_basis": gauge_basis,
        "anchor_delta": exact_chart._delta(model, state),
        "augmented": augmented,
        "coordinate_metrics": coordinate_metrics,
        "augmented_metrics": augmented_metrics,
        "assembly_wall_seconds": wall,
    }


def _retract(inputs: dict, context: dict, target: np.ndarray) -> tuple[np.ndarray, dict]:
    model = inputs["model"]
    target_coordinate = np.asarray(target, dtype=float)
    initial = (
        context["state"]
        + np.asarray(model.decoded_state(target_coordinate))
        - context["model_state"]
    )
    state, _matrix, metrics = arclength._transport()._transport_retract(
        model=model,
        initial_state=initial,
        target=target_coordinate,
        gauge_basis=context["gauge_basis"],
        anchor_delta=context["anchor_delta"],
        anchor_augmented=context["augmented"],
    )
    recovered, factors = model.coordinate(state)
    result = {
        **metrics,
        "recovered_coordinate_relative_defect": _relative(
            recovered, target_coordinate
        ),
        "minimum_decoder_reconstruction_factor": float(np.min(factors)),
    }
    return np.asarray(state), result


def _free_field(inputs: dict, state: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    base = inputs["base"]
    model = inputs["model"]
    exact_chart = arclength._exact_chart()
    rate_source = arclength._source()._post().exact_rate.rate_source
    configuration = base["configuration"]
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(state.shape)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    began = time.perf_counter()
    tangent = rate_source.causal_five_field_monolithic_frozen_tangent(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    tangent_wall = float(time.perf_counter() - began)
    began = time.perf_counter()
    coordinate_jacobian, coordinate_metrics = exact_chart._coordinate_jacobian(
        model, state
    )
    jacobian_wall = float(time.perf_counter() - began)
    coordinate_rate = coordinate_jacobian @ tangent.scaled_base_rate_per_s
    physical = rate_source._state_audit(context, state)
    spatial = tangent.spatial_tangent
    ledger_values = {
        "maximum_node_reconstruction_relative_defect": float(
            tangent.maximum_node_reconstruction_relative_defect
        ),
        "maximum_node_partition_of_unity_defect": float(
            tangent.maximum_node_partition_of_unity_defect
        ),
        "maximum_descriptor_component_defect": float(
            tangent.maximum_descriptor_component_defect
        ),
        "maximum_storage_rate_component_defect": float(
            tangent.maximum_storage_rate_component_defect
        ),
        "maximum_base_rate_balance_defect": float(
            tangent.maximum_base_rate_balance_defect
        ),
        "maximum_generator_factorization_defect": float(
            tangent.maximum_generator_factorization_defect
        ),
        "maximum_centered_storage_action_relative_defect": float(
            tangent.maximum_centered_storage_action_relative_defect
        ),
        "maximum_spatial_block_ledger_relative_defect": float(
            spatial.maximum_block_ledger_relative_defect
        ),
        "incoming_excision_characteristics": int(
            tangent.incoming_excision_characteristics
        ),
    }
    ledger_passed = bool(
        max(
            ledger_values[name]
            for name in (
                "maximum_node_reconstruction_relative_defect",
                "maximum_node_partition_of_unity_defect",
                "maximum_descriptor_component_defect",
                "maximum_storage_rate_component_defect",
                "maximum_base_rate_balance_defect",
                "maximum_generator_factorization_defect",
                "maximum_spatial_block_ledger_relative_defect",
            )
        )
        <= METHOD_COMPONENT_TOLERANCE
        and ledger_values["maximum_centered_storage_action_relative_defect"]
        <= CENTERED_STORAGE_ACTION_TOLERANCE
        and ledger_values["incoming_excision_characteristics"] == 0
        and tangent.uses_center_broken_within_cell_paths
        and not tangent.uses_production_generator
        and not tangent.uses_production_anchor_storage_derivative
    )
    metrics = {
        "tangent_wall_seconds": tangent_wall,
        "coordinate_jacobian_wall_seconds": jacobian_wall,
        "total_free_evaluation_wall_seconds": tangent_wall + jacobian_wall,
        "coordinate_jacobian_rank": int(coordinate_metrics["rank"]),
        "coordinate_jacobian_condition_number": float(
            coordinate_metrics["condition_number"]
        ),
        "coordinate_reconstruction_relative_defect": float(
            coordinate_metrics["coordinate_reconstruction_relative_defect"]
        ),
        "minimum_reconstruction_factor": float(
            physical["minimum_reconstruction_factor"]
        ),
        "maximum_height_ratio": float(physical["maximum_h_over_r"]),
        "minimum_scattering_optical_depth": float(
            physical["minimum_scattering_optical_depth"]
        ),
        "scaled_free_rate_norm_per_second": float(
            np.linalg.norm(tangent.scaled_base_rate_per_s)
        ),
        "coordinate_free_rate_norm_per_second": float(
            np.linalg.norm(coordinate_rate)
        ),
        "reaction_free_ledger_values": ledger_values,
        "reaction_free_ledger_passed": ledger_passed,
    }
    arrays = {
        "primitive_state": np.asarray(state),
        "scaled_free_rate560_per_s": np.asarray(tangent.scaled_base_rate_per_s),
        "coordinate_free_rate470_per_s": coordinate_rate,
    }
    return metrics, arrays


def _physical_passed(metrics: dict, retraction: dict | None) -> bool:
    retraction_ok = True
    if retraction is not None:
        retraction_ok = bool(
            retraction["passed"]
            and retraction["coordinate_residual_infinity"]
            <= COORDINATE_RETRACTION_TOLERANCE
            and retraction["gauge_residual_infinity"] <= GAUGE_RETRACTION_TOLERANCE
        )
    return bool(
        retraction_ok
        and metrics["coordinate_jacobian_rank"] == 470
        and metrics["coordinate_jacobian_condition_number"]
        <= COORDINATE_CONDITION_TOLERANCE
        and metrics["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
        and metrics["maximum_height_ratio"] <= 0.5
        and metrics["minimum_scattering_optical_depth"] >= 1.0
        and metrics["reaction_free_ledger_passed"]
    )


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _load_witness(
    patch_directory: Path,
    *,
    label: str,
    target: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    helper = _helper()
    metrics_path = patch_directory / f"witness_{label}.json"
    arrays_path = patch_directory / f"witness_{label}.npz"
    if metrics_path.exists() != arrays_path.exists():
        raise RuntimeError(f"partial exact witness scratch: {label}")
    if not metrics_path.exists():
        return None
    metrics = helper._read(metrics_path)
    arrays = helper._load_npz(arrays_path)
    np.testing.assert_array_equal(arrays["target_coordinate470"], target)
    return metrics, arrays


def _exact_witness(
    inputs: dict,
    context: dict,
    patch_directory: Path,
    *,
    label: str,
    target: np.ndarray,
    witness_index: int,
    eta: float,
    kind: str,
    anchor_state: np.ndarray | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    cached = _load_witness(patch_directory, label=label, target=target)
    if cached is not None:
        print(f"patch witness {label}: reused exact scratch", flush=True)
        return cached
    began = time.perf_counter()
    if anchor_state is None:
        state, retraction = _retract(inputs, context, target)
    else:
        state = np.asarray(anchor_state)
        retraction = None
    retraction_wall = float(time.perf_counter() - began)
    if retraction is not None and not (
        retraction["passed"]
        and retraction["coordinate_residual_infinity"]
        <= COORDINATE_RETRACTION_TOLERANCE
        and retraction["gauge_residual_infinity"]
        <= GAUGE_RETRACTION_TOLERANCE
    ):
        helper._write_json(
            patch_directory / f"witness_{label}_failure.json",
            {
                "label": label,
                "kind": kind,
                "witness_index": int(witness_index),
                "eta": float(eta),
                "classification": "exact_retraction_failed_before_free_field",
                "retraction": retraction,
                "retraction_wall_seconds": retraction_wall,
            },
        )
        raise RuntimeError("physical_gate")
    metrics, arrays = _free_field(inputs, state)
    withheld = bool(witness_index % 4 == 0 or kind == "diagonal")
    metrics.update({
        "label": label,
        "kind": kind,
        "witness_index": int(witness_index),
        "eta": float(eta),
        "withheld_from_fit": withheld,
        "retraction": retraction,
        "retraction_wall_seconds": retraction_wall,
    })
    metrics["physical_passed"] = _physical_passed(metrics, retraction)
    arrays.update({
        "target_coordinate470": np.asarray(target),
        "primitive_state": np.asarray(state),
    })
    helper._write_json(patch_directory / f"witness_{label}.json", metrics)
    _save_npz(patch_directory / f"witness_{label}.npz", arrays)
    print(
        f"patch witness {label}: |f|={metrics['coordinate_free_rate_norm_per_second']:.6e}/s "
        f"retract={retraction_wall:.3f}s field={metrics['total_free_evaluation_wall_seconds']:.3f}s "
        f"withheld={withheld}",
        flush=True,
    )
    return metrics, arrays


def _select_mode_basis(
    hidden_rates: np.ndarray,
    training: np.ndarray,
    heldout: np.ndarray,
) -> tuple[np.ndarray, int, dict]:
    attempts = {}
    selected = None
    selected_rank = 0
    train_count = int(np.count_nonzero(training))
    for rank in manifest.HIDDEN_RANK_CANDIDATES:
        if rank > min(train_count, hidden_rates.shape[1]):
            continue
        basis, singular, energy = canonical_rate_basis(hidden_rates[training], rank)
        defects = relative_projection_defects(hidden_rates, basis)
        heldout_max = float(np.max(defects[heldout])) if np.any(heldout) else 0.0
        attempts[str(rank)] = {
            "singular_values": singular.tolist(),
            "cumulative_energy": energy.tolist(),
            "maximum_training_defect": float(np.max(defects[training])),
            "maximum_holdout_defect": heldout_max,
        }
        selected = basis
        selected_rank = rank
        if (
            attempts[str(rank)]["maximum_training_defect"]
            <= LOCAL_HIDDEN_RATE_TOLERANCE
            and heldout_max <= LOCAL_HIDDEN_RATE_TOLERANCE
        ):
            break
    if selected is None:
        raise RuntimeError("no admissible local hidden basis could be built")
    final_attempt = attempts[str(selected_rank)]
    if max(
        final_attempt["maximum_training_defect"],
        final_attempt["maximum_holdout_defect"],
    ) > LOCAL_HIDDEN_RATE_TOLERANCE:
        raise RuntimeError("hidden_rate")
    return selected, selected_rank, attempts


def _recent_training_hidden_rates(
    inputs: dict,
    patch_index: int,
    *,
    maximum_samples: int = 14,
) -> np.ndarray:
    helper = _helper()
    samples = []
    for prior_index in range(max(0, patch_index - 7), patch_index):
        directory = SCRATCH_DIRECTORY / f"patch_{prior_index:04d}"
        for label in ("anchor", "physical_full", "physical_half"):
            metrics_path = directory / f"witness_{label}.json"
            arrays_path = directory / f"witness_{label}.npz"
            if not metrics_path.exists():
                continue
            metrics = helper._read(metrics_path)
            if metrics["withheld_from_fit"]:
                continue
            arrays = helper._load_npz(arrays_path)
            samples.append(
                inputs["split"].split_rate(
                    arrays["coordinate_free_rate470_per_s"]
                )[1]
            )
    if not samples:
        return np.empty((0, inputs["split"].hidden_dimension))
    return np.asarray(samples[-maximum_samples:])


def _mode_distances(
    hidden_rates: np.ndarray,
    mode_bases: dict[str, np.ndarray],
) -> dict[str, float]:
    result = {}
    for name, basis in mode_bases.items():
        result[name] = float(
            np.max(relative_projection_defects(hidden_rates, basis))
            / LOCAL_HIDDEN_RATE_TOLERANCE
        )
    return result


def _patch_training(
    inputs: dict,
    coordinate: np.ndarray,
    state: np.ndarray,
    patch_index: int,
    witness_start: int,
    mode_bases: dict[str, np.ndarray],
    current_mode: str,
    selector_state: tuple[str | None, int],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], tuple[str | None, int]]:
    helper = _helper()
    patch_directory = SCRATCH_DIRECTORY / f"patch_{patch_index:04d}"
    patch_directory.mkdir(exist_ok=True)
    context = _build_retraction_context(inputs, state, coordinate)
    anchor_metrics, anchor_arrays = _exact_witness(
        inputs,
        context,
        patch_directory,
        label="anchor",
        target=coordinate,
        witness_index=witness_start,
        eta=0.0,
        kind="anchor",
        anchor_state=state,
    )
    f0 = np.asarray(anchor_arrays["coordinate_free_rate470_per_s"])
    step = manifest.MACRO_STEP_SECONDS
    full_target = coordinate + step * f0
    full_metrics, full_arrays = _exact_witness(
        inputs,
        context,
        patch_directory,
        label="physical_full",
        target=full_target,
        witness_index=witness_start + 1,
        eta=1.0,
        kind="physical_full",
    )
    use_diagonal = bool((patch_index + 1) % 4 == 0)
    if use_diagonal:
        _macro0, hidden0 = inputs["split"].split_rate(f0)
        cold_basis = mode_bases[current_mode]
        direction = np.asarray(cold_basis[:, patch_index % cold_basis.shape[1]])
        transverse = direction - hidden0 * float(hidden0 @ direction) / max(
            float(hidden0 @ hidden0), np.finfo(float).tiny
        )
        transverse /= max(float(np.linalg.norm(transverse)), np.finfo(float).tiny)
        hidden_delta = (
            DIAGONAL_HIDDEN_FRACTION
            * step
            * max(float(np.linalg.norm(hidden0)), np.finfo(float).tiny)
            * transverse
        )
        validation_target = full_target + inputs["split"].compose(
            np.zeros(inputs["split"].macro_dimension), hidden_delta
        )
        validation_eta = 1.0
        validation_kind = "diagonal"
    else:
        validation_target = coordinate + 0.5 * step * f0
        validation_eta = 0.5
        validation_kind = "physical_half"
    validation_metrics, validation_arrays = _exact_witness(
        inputs,
        context,
        patch_directory,
        label=validation_kind,
        target=validation_target,
        witness_index=witness_start + 2,
        eta=validation_eta,
        kind=validation_kind,
    )
    records = (anchor_metrics, full_metrics, validation_metrics)
    arrays = (anchor_arrays, full_arrays, validation_arrays)
    if not all(record["physical_passed"] for record in records):
        helper._write_json(
            patch_directory / "patch_failure.json",
            {
                "patch_index": patch_index,
                "classification": "physical_gate",
                "witness_physical_passed": [
                    record["physical_passed"] for record in records
                ],
            },
        )
        raise RuntimeError("physical_gate")

    etas = np.asarray([record["eta"] for record in records])
    rates = np.stack(
        [item["coordinate_free_rate470_per_s"] for item in arrays]
    )
    training = np.asarray(
        [not record["withheld_from_fit"] and record["kind"] != "diagonal" for record in records]
    )
    heldout = ~training
    intercept, slope = _fit_affine_axis(etas, rates, training)
    predictions = np.stack(
        [_axis_prediction(intercept, slope, eta) for eta in etas]
    )
    defects = np.asarray(
        [_relative(prediction, exact) for prediction, exact in zip(predictions, rates, strict=True)]
    )
    blind_defect = float(np.max(defects[heldout])) if np.any(heldout) else 0.0
    if blind_defect > manifest.MAXIMUM_BLIND_RATE_DEFECT:
        helper._write_json(
            patch_directory / "patch_failure.json",
            {
                "patch_index": patch_index,
                "classification": "blind_rate",
                "maximum_blind_rate_defect": blind_defect,
                "axis_rate_defects": defects.tolist(),
                "training_flags": training.tolist(),
            },
        )
        raise RuntimeError("blind_rate")

    hidden_rates = np.stack(
        [inputs["split"].split_rate(rate)[1] for rate in rates]
    )
    distances = _mode_distances(hidden_rates, mode_bases)
    best_name = min(distances, key=distances.get)
    if distances[best_name] > 1.0:
        new_name = f"acquired_{len(mode_bases) - 1:03d}"
        prior_training = _recent_training_hidden_rates(inputs, patch_index)
        pooled_rates = np.vstack((prior_training, hidden_rates))
        pooled_training = np.concatenate((
            np.ones(len(prior_training), dtype=bool),
            training,
        ))
        pooled_heldout = np.concatenate((
            np.zeros(len(prior_training), dtype=bool),
            heldout,
        ))
        try:
            basis, rank, basis_attempts = _select_mode_basis(
                pooled_rates, pooled_training, pooled_heldout
            )
        except RuntimeError as error:
            helper._write_json(
                patch_directory / "patch_failure.json",
                {
                    "patch_index": patch_index,
                    "classification": str(error),
                },
            )
            raise
        mode_bases[new_name] = basis
        basis_attempts["training_pool"] = {
            "prior_training_samples": int(len(prior_training)),
            "current_training_samples": int(np.count_nonzero(training)),
            "current_holdout_samples": int(np.count_nonzero(heldout)),
        }
        distances[new_name] = float(
            np.max(relative_projection_defects(hidden_rates, basis))
            / LOCAL_HIDDEN_RATE_TOLERANCE
        )
        best_name = new_name
    else:
        basis = mode_bases[best_name]
        rank = basis.shape[1]
        basis_attempts = {"inherited": {"mode": best_name}}

    selector = HystereticModeSelector(
        relative_switch_margin=manifest.MODE_SWITCH_MARGIN,
        persistence_steps=manifest.MODE_SWITCH_PERSISTENCE,
    )
    decision = selector.update(
        current_mode=current_mode,
        normalized_distances=distances,
        pending_mode=selector_state[0],
        pending_count=selector_state[1],
    )
    selected_mode = decision.mode
    local_basis = mode_bases[best_name]
    macro0, hidden0 = inputs["split"].split(coordinate)
    model = ConservativeHiddenAmplitudeModel(
        split=inputs["split"],
        hidden_origin=hidden0,
        hidden_basis=local_basis,
    )
    macro_intercept, hidden_intercept = inputs["split"].split_rate(intercept)
    macro_slope, hidden_slope = inputs["split"].split_rate(slope)
    patch = LocalAffineReducedPatch(
        anchor_macro=macro0,
        anchor_amplitudes=np.zeros(local_basis.shape[1]),
        anchor_reduced_rate=np.concatenate(
            (macro_intercept, local_basis.T @ hidden_intercept)
        ),
        physical_rate_delta=np.concatenate(
            (macro_slope, local_basis.T @ hidden_slope)
        ),
        macro_step_seconds=step,
        mode=selected_mode,
        anchor_id=f"patch_{patch_index:04d}",
        maximum_absolute_eta=manifest.MAXIMUM_PATCH_COORDINATE,
    )
    engine = ConservativeHeunEngine(
        model=model,
        patch=patch,
        forcing_angular_frequency=0.0,
        maximum_embedded_error_fraction=manifest.MAXIMUM_EMBEDDED_ERROR,
    )
    initial = HiddenAmplitudeState(
        macro=macro0,
        amplitudes=np.zeros(local_basis.shape[1]),
        forcing_phase=0.0,
        mode=selected_mode,
        elapsed_seconds=patch_index * step,
    )
    result = engine.step(initial, step)
    candidate_coordinate = model.decode(result.candidate)
    candidate_state, candidate_retraction = _retract(
        inputs, context, candidate_coordinate
    )
    candidate_coordinate_exact, candidate_factors = inputs["model"].coordinate(
        candidate_state
    )
    candidate_physical = arclength._exact_chart()._physical_audit(
        inputs["model"], candidate_state, candidate_factors
    )
    candidate_passed = bool(
        result.accepted
        and candidate_retraction["passed"]
        and candidate_retraction["coordinate_residual_infinity"]
        <= COORDINATE_RETRACTION_TOLERANCE
        and candidate_retraction["gauge_residual_infinity"]
        <= GAUGE_RETRACTION_TOLERANCE
        and candidate_physical["passed"]
    )
    patch_metrics = {
        "patch_index": patch_index,
        "anchor_assembly_wall_seconds": context["assembly_wall_seconds"],
        "witness_indices": [record["witness_index"] for record in records],
        "witness_kinds": [record["kind"] for record in records],
        "withheld_flags": [record["withheld_from_fit"] for record in records],
        "training_flags": training.tolist(),
        "axis_rate_defects": defects.tolist(),
        "maximum_blind_rate_defect": blind_defect,
        "mode_distances": distances,
        "mode_before": current_mode,
        "nearest_mode": best_name,
        "mode_after": selected_mode,
        "mode_switched": decision.switched,
        "pending_mode": decision.pending_mode,
        "pending_count": decision.pending_count,
        "hidden_rank": int(local_basis.shape[1]),
        "hidden_basis_attempts": basis_attempts,
        "embedded_error_fraction": result.embedded_error_fraction,
        "macro_ledger_defect": result.macro_ledger_defect,
        "eta_start": result.start_eta,
        "eta_predictor": result.predictor_eta,
        "eta_endpoint": result.endpoint_eta,
        "candidate_retraction": candidate_retraction,
        "candidate_physical": candidate_physical,
        "accepted": candidate_passed,
    }
    patch_arrays = {
        "anchor_coordinate470": coordinate,
        "anchor_primitive_state": state,
        "axis_etas": etas,
        "exact_coordinate_rates3x470_per_s": rates,
        "axis_rate_predictions3x470_per_s": predictions,
        "axis_rate_defects": defects,
        "training_flags": training,
        "hidden_basis388xr": local_basis,
        "affine_intercept470_per_s": intercept,
        "affine_slope470_per_s": slope,
        "candidate_coordinate470": candidate_coordinate_exact,
        "candidate_primitive_state": candidate_state,
    }
    helper._write_json(patch_directory / "patch_metrics.json", patch_metrics)
    _save_npz(patch_directory / "patch_arrays.npz", patch_arrays)
    return (
        patch_metrics,
        patch_arrays,
        mode_bases,
        (decision.pending_mode, decision.pending_count),
    )


def _load_completed_patch(index: int) -> tuple[dict, dict[str, np.ndarray]] | None:
    helper = _helper()
    directory = SCRATCH_DIRECTORY / f"patch_{index:04d}"
    metrics_path = directory / "patch_metrics.json"
    arrays_path = directory / "patch_arrays.npz"
    if metrics_path.exists() != arrays_path.exists():
        raise RuntimeError("partial completed patch scratch")
    if not metrics_path.exists():
        return None
    return helper._read(metrics_path), helper._load_npz(arrays_path)


def _scratch_inventory() -> dict:
    helper = _helper()
    patch_directories = sorted(SCRATCH_DIRECTORY.glob("patch_[0-9][0-9][0-9][0-9]"))
    witnesses = []
    witness_failures = []
    patch_failures = []
    for directory in patch_directories:
        for metrics_path in sorted(directory.glob("witness_*.json")):
            if metrics_path.name.endswith("_failure.json"):
                witness_failures.append(helper._read(metrics_path))
                continue
            arrays_path = metrics_path.with_suffix(".npz")
            if not arrays_path.exists():
                raise RuntimeError(f"partial witness evidence: {metrics_path}")
            witnesses.append((metrics_path, arrays_path))
        failure_path = directory / "patch_failure.json"
        if failure_path.exists():
            patch_failures.append(helper._read(failure_path))
    return {
        "patch_directories": patch_directories,
        "witnesses": witnesses,
        "witness_failures": witness_failures,
        "patch_failures": patch_failures,
    }


def _replay_patch(
    split: ConservativeCoordinateSplit,
    patch_metrics: dict,
    patch_arrays: dict[str, np.ndarray],
    coordinate: np.ndarray,
    *,
    half_step: bool,
) -> tuple[np.ndarray, HiddenAmplitudeState]:
    coordinate_path, state = _replay_patch_path(
        split,
        patch_metrics,
        patch_arrays,
        coordinate,
        half_step=half_step,
    )
    return np.asarray(coordinate_path[-1]), state


def _replay_patch_path(
    split: ConservativeCoordinateSplit,
    patch_metrics: dict,
    patch_arrays: dict[str, np.ndarray],
    coordinate: np.ndarray,
    *,
    half_step: bool,
) -> tuple[list[np.ndarray], HiddenAmplitudeState]:
    basis = np.asarray(patch_arrays["hidden_basis388xr"])
    anchor = np.asarray(patch_arrays["anchor_coordinate470"])
    macro_anchor, hidden_anchor = split.split(anchor)
    macro, hidden = split.split(coordinate)
    model = ConservativeHiddenAmplitudeModel(split, hidden_anchor, basis)
    mode = str(patch_metrics["mode_after"])
    patch = LocalAffineReducedPatch(
        anchor_macro=macro_anchor,
        anchor_amplitudes=np.zeros(basis.shape[1]),
        anchor_reduced_rate=np.concatenate((
            split.split_rate(patch_arrays["affine_intercept470_per_s"])[0],
            basis.T @ split.split_rate(patch_arrays["affine_intercept470_per_s"])[1],
        )),
        physical_rate_delta=np.concatenate((
            split.split_rate(patch_arrays["affine_slope470_per_s"])[0],
            basis.T @ split.split_rate(patch_arrays["affine_slope470_per_s"])[1],
        )),
        macro_step_seconds=manifest.MACRO_STEP_SECONDS,
        mode=mode,
        anchor_id=f"replay_{patch_metrics['patch_index']:04d}",
        maximum_absolute_eta=manifest.MAXIMUM_PATCH_COORDINATE,
    )
    engine = ConservativeHeunEngine(
        model,
        patch,
        forcing_angular_frequency=0.0,
        maximum_embedded_error_fraction=manifest.MAXIMUM_EMBEDDED_ERROR,
    )
    state = HiddenAmplitudeState(
        macro=macro,
        amplitudes=basis.T @ (hidden - hidden_anchor),
        forcing_phase=0.0,
        mode=mode,
        elapsed_seconds=patch_metrics["patch_index"] * manifest.MACRO_STEP_SECONDS,
    )
    count = 2 if half_step else 1
    timestep = manifest.MACRO_STEP_SECONDS / count
    path = []
    for _index in range(count):
        result = engine.step(state, timestep)
        if not result.accepted:
            raise RuntimeError("truth-free atlas replay left a frozen patch")
        state = result.candidate
        path.append(model.decode(state))
    return path, state


def _replay_atlas(
    split: ConservativeCoordinateSplit,
    patches: list[tuple[dict, dict[str, np.ndarray]]],
    start: np.ndarray,
    section_normal: np.ndarray,
    *,
    half_step: bool,
) -> dict:
    coordinate = np.asarray(start).copy()
    previous_section = 0.0
    seen_negative = False
    mode_switches = 0
    previous_mode = str(patches[0][0]["mode_before"])
    event = None
    states = [coordinate.copy()]
    state = None
    substeps = 2 if half_step else 1
    for patch_metrics, patch_arrays in patches:
        coordinate_path, state = _replay_patch_path(
            split, patch_metrics, patch_arrays, coordinate, half_step=half_step
        )
        mode = str(patch_metrics["mode_after"])
        if mode != previous_mode:
            mode_switches += 1
            previous_mode = mode
        for substep_index, next_coordinate in enumerate(coordinate_path):
            current_section = float(section_normal @ (next_coordinate - start))
            if current_section < 0.0:
                seen_negative = True
            fraction = _section_crossing_fraction(
                previous_section, current_section
            )
            if seen_negative and fraction is not None and mode_switches >= 2:
                event_coordinate = coordinate + fraction * (
                    next_coordinate - coordinate
                )
                event = {
                    "coordinate": event_coordinate,
                    "elapsed_seconds": (
                        patch_metrics["patch_index"]
                        + (substep_index + fraction) / substeps
                    )
                    * manifest.MACRO_STEP_SECONDS,
                    "mode_switches": mode_switches,
                }
                states.append(event_coordinate)
                break
            coordinate = next_coordinate
            previous_section = current_section
            states.append(coordinate.copy())
        if event is not None:
            break
    return {
        "event": event,
        "coordinates": np.asarray(states),
        "terminal_coordinate": np.asarray(states[-1]),
        "terminal_state": state,
    }


def _restart_replay_bitwise(
    split: ConservativeCoordinateSplit,
    patches: list[tuple[dict, dict[str, np.ndarray]]],
    start: np.ndarray,
) -> bool:
    midpoint = len(patches) // 2
    coordinate = np.asarray(start).copy()
    state = None
    for metrics, arrays in patches[:midpoint]:
        coordinate, state = _replay_patch(
            split, metrics, arrays, coordinate, half_step=False
        )
    if state is None:
        return False
    restored = _state_roundtrip(state)
    restored_coordinate = _coordinate_roundtrip(coordinate)
    if not (
        _bitwise_state(state, restored)
        and np.array_equal(coordinate, restored_coordinate)
    ):
        return False
    left_coordinate = coordinate.copy()
    right_coordinate = restored_coordinate.copy()
    left_state = state
    right_state = restored
    for metrics, arrays in patches[midpoint:]:
        left_coordinate, left_state = _replay_patch(
            split, metrics, arrays, left_coordinate, half_step=False
        )
        right_coordinate, right_state = _replay_patch(
            split, metrics, arrays, right_coordinate, half_step=False
        )
    return bool(
        np.array_equal(left_coordinate, right_coordinate)
        and _bitwise_state(left_state, right_state)
    )


def _execute(parent_lock: dict, identity: dict) -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    inputs = _initial_inputs()
    cold_basis, cold_evidence = _cold_mode_basis(inputs["split"])
    hot_basis, _hot_evidence = _hot_mode_basis()
    mode_bases: dict[str, np.ndarray] = {
        "cold_recovery": cold_basis,
        "hot_geometry": hot_basis,
    }
    current_mode = "cold_recovery"
    selector_state: tuple[str | None, int] = (None, 0)
    coordinate = np.asarray(inputs["coordinate"]).copy()
    state = np.asarray(inputs["state"]).copy()
    start_coordinate = coordinate.copy()
    start_hidden = inputs["split"].split(start_coordinate)[1]
    cold_hidden = np.stack(
        [inputs["split"].split(value)[1] for value in inputs["candidate_coordinates"]]
    )
    cold_hidden_path = float(
        np.sum(np.linalg.norm(np.diff(cold_hidden, axis=0), axis=1))
    )
    patches: list[tuple[dict, dict[str, np.ndarray]]] = []
    trajectory = [coordinate.copy()]
    trajectory_states = [state.copy()]
    section_normal = None
    previous_section = 0.0
    seen_departure = False
    seen_negative = False
    event = None
    mode_switches = 0
    execution_started = time.perf_counter()
    cumulative_prior_wall = 0.0
    elapsed_path = SCRATCH_DIRECTORY / "cumulative_wall_seconds.json"
    if elapsed_path.exists():
        cumulative_prior_wall = float(helper._read(elapsed_path)["wall_seconds"])

    stop_reason = None
    for patch_index in range(manifest.MAXIMUM_PATCHES):
        prior = _load_completed_patch(patch_index)
        if prior is None:
            try:
                patch_metrics, patch_arrays, mode_bases, selector_state = _patch_training(
                    inputs,
                    coordinate,
                    state,
                    patch_index,
                    3 * patch_index + 1,
                    mode_bases,
                    current_mode,
                    selector_state,
                )
            except RuntimeError as error:
                if str(error) == "physical_gate":
                    stop_reason = PHYSICAL_CLASSIFICATION
                elif str(error) in ("blind_rate", "hidden_rate"):
                    stop_reason = VALIDATION_CLASSIFICATION
                else:
                    raise
        else:
            patch_metrics, patch_arrays = prior
            nearest_mode = str(patch_metrics["nearest_mode"])
            if nearest_mode not in mode_bases:
                mode_bases[nearest_mode] = np.asarray(
                    patch_arrays["hidden_basis388xr"]
                )
            selector_state = (
                patch_metrics["pending_mode"], patch_metrics["pending_count"]
            )
            print(f"patch {patch_index:04d}: reused completed scratch", flush=True)
        if stop_reason is not None:
            break
        if not patch_metrics["accepted"]:
            stop_reason = PHYSICAL_CLASSIFICATION
            break
        patches.append((patch_metrics, patch_arrays))
        if patch_metrics["mode_switched"]:
            mode_switches += 1
        current_mode = str(patch_metrics["mode_after"])
        coordinate = np.asarray(patch_arrays["candidate_coordinate470"])
        state = np.asarray(patch_arrays["candidate_primitive_state"])
        if section_normal is None:
            f0 = np.asarray(patch_arrays["exact_coordinate_rates3x470_per_s"])[0]
            section_normal = f0 / max(float(np.linalg.norm(f0)), np.finfo(float).tiny)
        current_section = float(section_normal @ (coordinate - start_coordinate))
        section_scale = max(
            float(np.linalg.norm(coordinate - start_coordinate)),
            np.finfo(float).tiny,
        )
        if current_section > SECTION_DEPARTURE_FRACTION * section_scale:
            seen_departure = True
        if seen_departure and current_section < 0.0:
            seen_negative = True
        fraction = _section_crossing_fraction(previous_section, current_section)
        trajectory.append(coordinate.copy())
        trajectory_states.append(state.copy())
        if seen_negative and fraction is not None and mode_switches >= 2:
            left = trajectory[-2]
            event_target = left + fraction * (coordinate - left)
            event_context = _build_retraction_context(
                inputs, trajectory_states[-2], left
            )
            event_state, event_retraction = _retract(inputs, event_context, event_target)
            event_coordinate, event_factors = inputs["model"].coordinate(event_state)
            event_physical = arclength._exact_chart()._physical_audit(
                inputs["model"], event_state, event_factors
            )
            hidden_return = float(
                np.linalg.norm(inputs["split"].split(event_coordinate)[1] - start_hidden)
                / max(cold_hidden_path, np.finfo(float).tiny)
            )
            event = {
                "patch_index": patch_index,
                "crossing_fraction": fraction,
                "elapsed_seconds": (patch_index + fraction) * manifest.MACRO_STEP_SECONDS,
                "mode_switches": mode_switches,
                "hidden_section_return_defect": hidden_return,
                "retraction": event_retraction,
                "physical": event_physical,
                "passed": bool(
                    event_retraction["passed"]
                    and event_physical["passed"]
                    and hidden_return <= manifest.MAXIMUM_HIDDEN_SECTION_RETURN_DEFECT
                ),
            }
            trajectory[-1] = np.asarray(event_coordinate)
            trajectory_states[-1] = np.asarray(event_state)
            if event["passed"]:
                break
            stop_reason = PHYSICAL_CLASSIFICATION
            break
        previous_section = current_section
        cumulative = cumulative_prior_wall + float(
            time.perf_counter() - execution_started
        )
        helper._write_json(elapsed_path, {"wall_seconds": cumulative})
        print(
            f"patch {patch_index + 1:02d}/{manifest.MAXIMUM_PATCHES}: "
            f"mode={current_mode} section={current_section:.6e} "
            f"switches={mode_switches} elapsed={coordinate.shape[0] and (patch_index + 1) * manifest.MACRO_STEP_SECONDS:.6e}s",
            flush=True,
        )
        if cumulative / 3600.0 >= manifest.MAXIMUM_COMPLETE_EXECUTION_WALL_HOURS:
            stop_reason = BUDGET_CLASSIFICATION
            break

    total_wall = cumulative_prior_wall + float(time.perf_counter() - execution_started)
    helper._write_json(elapsed_path, {"wall_seconds": total_wall})
    if event is None and stop_reason is None:
        stop_reason = BUDGET_CLASSIFICATION if len(patches) >= manifest.MAXIMUM_PATCHES else NO_RETURN_CLASSIFICATION

    validation = {
        "performed": False,
        "blind_passed": all(
            item[0]["maximum_blind_rate_defect"]
            <= manifest.MAXIMUM_BLIND_RATE_DEFECT
            for item in patches
        ),
        "restart_bitwise": False,
        "step_halving_cycle_map_defect": None,
        "step_halving_cycle_duration_defect": None,
        "passed": False,
    }
    full_replay = None
    half_replay = None
    if event is not None and event["passed"]:
        full_replay = _replay_atlas(
            inputs["split"], patches, start_coordinate, section_normal, half_step=False
        )
        half_replay = _replay_atlas(
            inputs["split"], patches, start_coordinate, section_normal, half_step=True
        )
        restart_bitwise = _restart_replay_bitwise(
            inputs["split"], patches, start_coordinate
        )
        if full_replay["event"] is not None and half_replay["event"] is not None:
            start_macro = inputs["split"].split(start_coordinate)[0]
            full_macro = inputs["split"].split(full_replay["event"]["coordinate"])[0]
            half_macro = inputs["split"].split(half_replay["event"]["coordinate"])[0]
            map_defect = _relative_increment(half_macro, full_macro, start_macro)
            duration_defect = abs(
                half_replay["event"]["elapsed_seconds"]
                - full_replay["event"]["elapsed_seconds"]
            ) / max(full_replay["event"]["elapsed_seconds"], np.finfo(float).tiny)
        else:
            map_defect = float("inf")
            duration_defect = float("inf")
        validation = {
            "performed": True,
            "blind_passed": validation["blind_passed"],
            "restart_bitwise": restart_bitwise,
            "step_halving_cycle_map_defect": map_defect,
            "step_halving_cycle_duration_defect": duration_defect,
            "passed": bool(
                validation["blind_passed"]
                and restart_bitwise
                and map_defect <= manifest.MAXIMUM_STEP_HALVING_CYCLE_MAP_DEFECT
                and duration_defect
                <= manifest.MAXIMUM_STEP_HALVING_CYCLE_DURATION_DEFECT
            ),
        }
        if not validation["passed"]:
            stop_reason = VALIDATION_CLASSIFICATION

    passed = bool(event is not None and event["passed"] and validation["passed"])
    classification = PASS_CLASSIFICATION if passed else str(stop_reason)
    final_coordinate = np.asarray(trajectory[-1])
    start_macro = inputs["split"].split(start_coordinate)[0]
    final_macro = inputs["split"].split(final_coordinate)[0]
    inventory = _scratch_inventory()
    exact_witness_count = len(inventory["witnesses"])
    if exact_witness_count > manifest.MAXIMUM_EXACT_FREE_FIELD_WITNESSES:
        raise RuntimeError("exact free-field witness budget exceeded")
    gate_values = {
        "completed_patches": len(patches),
        "attempted_patches": len(inventory["patch_directories"]),
        "exact_free_field_witnesses": exact_witness_count,
        "failed_exact_retractions": len(inventory["witness_failures"]),
        "accepted_macrosteps": len(patches),
        "mode_switches": mode_switches,
        "cycle_observed": event is not None,
        "cycle_duration_seconds": event["elapsed_seconds"] if event else None,
        "cycle_delta_q82_norm": float(np.linalg.norm(final_macro - start_macro)),
        "execution_wall_seconds": total_wall,
        "maximum_blind_rate_defect": max(
            (item[0]["maximum_blind_rate_defect"] for item in patches),
            default=0.0,
        ),
        "maximum_embedded_error_fraction": max(
            (item[0]["embedded_error_fraction"] for item in patches),
            default=0.0,
        ),
        "maximum_macro_ledger_defect": max(
            (item[0]["macro_ledger_defect"] for item in patches),
            default=0.0,
        ),
        "minimum_hidden_rank": min(
            (item[0]["hidden_rank"] for item in patches), default=0
        ),
        "maximum_hidden_rank": max(
            (item[0]["hidden_rank"] for item in patches), default=0
        ),
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "gate_values": gate_values,
        "event": event,
        "validation": validation,
        "patch_metrics": [item[0] for item in patches],
        "partial_patch_failures": inventory["patch_failures"],
        "partial_witness_failures": inventory["witness_failures"],
        "mode_names": sorted(mode_bases),
        "fixed_Q_physical_rate_calls": 0,
        "fixed_Q_reaction_calls": 0,
        "nonlinear_roots": 0,
        "BDF_microsteps": 0,
        "input_lock": parent_lock,
        "execution_identity": identity,
        "cold_basis_projection_defects": cold_evidence["projection_defects"].tolist(),
    }
    arrays = {
        "start_coordinate470": start_coordinate,
        "start_primitive_state": inputs["state"],
        "trajectory_coordinates": np.asarray(trajectory),
        "trajectory_primitive_states": np.asarray(trajectory_states),
        "trajectory_macro82": np.stack(
            [inputs["split"].split(value)[0] for value in trajectory]
        ),
        "section_normal470": np.asarray(section_normal) if section_normal is not None else np.zeros(470),
        "final_coordinate470": final_coordinate,
        "delta_q82": final_macro - start_macro,
        "full_replay_coordinates": (
            full_replay["coordinates"] if full_replay is not None else np.empty((0, 470))
        ),
        "half_replay_coordinates": (
            half_replay["coordinates"] if half_replay is not None else np.empty((0, 470))
        ),
    }
    return metrics, arrays


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("complete-cycle canonical result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "cycle_execution_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "cycle_execution_arrays.npz", arrays)
    patch_payload = {}
    witness_coordinates = []
    witness_rates = []
    witness_states = []
    witness_indices = []
    witness_withheld = []
    witness_kinds = []
    inventory = _scratch_inventory()
    for directory in inventory["patch_directories"]:
        patch_index = int(directory.name.rsplit("_", 1)[1])
        completed = _load_completed_patch(patch_index)
        if completed is not None:
            _patch_metrics, patch_arrays = completed
            for name, value in patch_arrays.items():
                if name not in (
                    "anchor_primitive_state",
                    "candidate_primitive_state",
                ):
                    patch_payload[f"patch_{patch_index:04d}__{name}"] = value
        for label in ("anchor", "physical_full", "physical_half", "diagonal"):
            metrics_path = directory / f"witness_{label}.json"
            arrays_path = directory / f"witness_{label}.npz"
            if not metrics_path.exists():
                continue
            item = helper._read(metrics_path)
            payload = helper._load_npz(arrays_path)
            witness_coordinates.append(payload["target_coordinate470"])
            witness_rates.append(payload["coordinate_free_rate470_per_s"])
            witness_states.append(payload["primitive_state"])
            witness_indices.append(item["witness_index"])
            witness_withheld.append(item["withheld_from_fit"])
            witness_kinds.append(item["kind"])
    _save_npz(CANONICAL_DIRECTORY / "patch_atlas_arrays.npz", patch_payload)
    _save_npz(CANONICAL_DIRECTORY / "exact_witness_arrays.npz", {
        "coordinates": np.asarray(witness_coordinates),
        "coordinate_rates_per_s": np.asarray(witness_rates),
        "primitive_states": np.asarray(witness_states),
        "indices": np.asarray(witness_indices),
        "withheld": np.asarray(witness_withheld),
        "kinds": np.asarray(witness_kinds),
    })
    passed = bool(metrics["passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "complete_cycle_execution_attempted": True,
        "complete_cycle_observed": bool(metrics["gate_values"]["cycle_observed"]),
        "complete_cycle_atlas_verified": passed,
        "multi_anchor_cycle_map_manifest_authorized": passed,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": metrics["execution_identity"]["implementation_commit"],
        "implementation_tree": metrics["execution_identity"]["implementation_tree"],
        "source_hashes": metrics["execution_identity"]["source_hashes"],
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    values = metrics["gate_values"]
    REPORT_PATH.write_text(
        "\n".join((
            "# Adaptive original-free-field complete-cycle execution",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The execution acquired `{values['completed_patches']}` patches with `{values['exact_free_field_witnesses']}` exact original-free-field witnesses in `{values['execution_wall_seconds'] / 3600.0:.3f}` wall hours.",
            "",
            f"Cycle observed: `{values['cycle_observed']}`; persistent mode switches: `{values['mode_switches']}`; cycle duration: `{values['cycle_duration_seconds']}` s.",
            "",
            f"Maximum blind rate defect: `{values['maximum_blind_rate_defect']:.6e}`; maximum embedded correction: `{values['maximum_embedded_error_fraction']:.6e}`; restart bitwise: `{metrics['validation']['restart_bitwise']}`.",
            "",
            "All fixed-Q physical-rate/reaction calls, nonlinear roots, and BDF microsteps remained zero. Reduced slow evolution remains separately gated.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = manifest.parent.manifest.parent.arclength._source()._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("complete-cycle execution result already exists")
    parent_lock = _validate_parent(require_clean=True)
    identity = _prepare_scratch(parent_lock)
    metrics, arrays = _execute(parent_lock, identity)
    return _canonicalize(metrics, arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
