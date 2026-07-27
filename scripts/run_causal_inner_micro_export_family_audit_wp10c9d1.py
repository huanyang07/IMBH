"""Run the WP10c9d1 characteristic-family audit of failed micro exports.

WP10c9d0 established that the conservative embedded inner patch does not
produce contracting instantaneous or cumulative physical exports.  This
cache-only follow-up decomposes the unchanged patch state histories with the
exact five-family primitive projector partition and maps every component
through the same nonlinear M/J/E, cooling, and responsive-height observable
operator used by WP10c9d0.

The package can select a complete characteristic subsystem for redesign only
when one family is significant, dominant, persistent, and stable across both
refinement pairs.  It cannot authorize a one-block repair, production change,
truth trajectory, constrained average, or reduced slow evolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_micro_export_preflight_wp10c9d0 as wp10c9d0

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    causal_five_field_characteristic_family_decomposition,
    causal_five_field_characteristic_family_projectors,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9d1"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_micro_export_family_audit_wp10c9d1.py"
)
WP10C8Z_ARRAYS = wp10c9d0.WP10C8Z_ARRAYS
WP10C9D0_OUTPUT = wp10c9d0.DEFAULT_OUTPUT
WP10C9D0_ARRAYS = wp10c9d0.DEFAULT_ARRAYS
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_family_audit_wp10c9d1.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_family_audit_wp10c9d1_arrays.npz"
)
CACHE_ROOT = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_micro_export_family_wp10c9d1"
)

PATCH_LABELS = (
    "N128_exterior_N128_inner_c48",
    "N128_exterior_N256_inner_c48",
    "N128_exterior_N512_inner_c48",
)
FAMILY_SAMPLE_STRIDE = 5
MAXIMUM_PROJECTOR_DEFECT = 1.0e-10
MAXIMUM_FAMILY_SUM_CLOSURE_DEFECT = 2.0e-3
MINIMUM_FAMILY_ACTIVITY_FRACTION = 0.50
MINIMUM_FAMILY_PERSISTENCE_FRACTION = 0.50
MINIMUM_PAIR_ACTIVITY_COSINE = 0.90
MINIMUM_ACTIVE_ERROR_NORM = 1.0e-6


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
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _family_cache_contract(
    *,
    label: str,
    base_primitives: np.ndarray,
    amplitudes: np.ndarray,
    state_history: np.ndarray,
    selected_indices: np.ndarray,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "label": label,
        "base_sha256": _array_sha256(base_primitives),
        "amplitudes_sha256": _array_sha256(amplitudes),
        "state_history_sha256": _array_sha256(state_history),
        "selected_indices_sha256": _array_sha256(selected_indices),
        "finite_difference_step": wp10c9d0.FINITE_DIFFERENCE_STEP,
    }


def _build_or_load_family_signals(
    *,
    label: str,
    configuration: dict,
    state_history: np.ndarray,
    selected_indices: np.ndarray,
    times: np.ndarray,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = CACHE_ROOT / f"{label}.json"
    arrays_path = CACHE_ROOT / f"{label}_arrays.npz"
    base = np.asarray(configuration["base_primitives"], dtype=float)
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    contract = _family_cache_contract(
        label=label,
        base_primitives=base,
        amplitudes=amplitudes,
        state_history=state_history,
        selected_indices=selected_indices,
    )
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            return payload, _load_npz(arrays_path)

    started = time.perf_counter()
    projector_contract, _bases = (
        causal_five_field_characteristic_family_projectors(
            configuration["context"],
            base,
            amplitudes,
        )
    )
    selected_state = np.asarray(
        state_history[selected_indices],
        dtype=float,
    )
    components = causal_five_field_characteristic_family_decomposition(
        selected_state,
        projector_contract,
    )
    state_closure = float(
        np.max(np.abs(np.sum(components, axis=0) - selected_state))
        / max(float(np.max(np.abs(selected_state))), np.finfo(float).tiny)
    )
    layout = configuration["layout"]
    interface_face = int(layout.coupling_face_index)
    family_signals = np.empty(
        (
            len(CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES),
            selected_indices.size,
            len(wp10c9d0.OBSERVABLE_NAMES),
        ),
        dtype=float,
    )
    for family, family_name in enumerate(
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    ):
        for output_index in range(selected_indices.size):
            if output_index % 5 == 0 or output_index + 1 == selected_indices.size:
                print(
                    f"WP10c9d1: {label} {family_name} "
                    f"{output_index + 1}/{selected_indices.size}",
                    flush=True,
                )
            family_signals[family, output_index] = (
                wp10c9d0._directional_observables(
                    configuration["context"],
                    base,
                    amplitudes * components[family, output_index],
                    interface_face=interface_face,
                    active_cells=interface_face,
                    step=wp10c9d0.FINITE_DIFFERENCE_STEP,
                )
            )
    family_cumulative = np.asarray(
        [
            wp10c9d0._cumulative_trapezoid(
                times,
                family_signals[family],
            )
            for family in range(family_signals.shape[0])
        ],
        dtype=float,
    )
    arrays = {
        "selected_indices": selected_indices,
        "times": times,
        "components": components,
        "family_signals": family_signals,
        "family_cumulative_signals": family_cumulative,
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **contract,
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "maximum_projector_identity_defect": float(
            projector_contract.maximum_identity_closure_defect
        ),
        "maximum_projector_idempotence_defect": float(
            projector_contract.maximum_idempotence_defect
        ),
        "maximum_cross_projector_defect": float(
            projector_contract.maximum_cross_projector_defect
        ),
        "maximum_basis_condition_number": float(
            projector_contract.maximum_basis_condition_number
        ),
        "maximum_eigenpair_defect": float(
            projector_contract.maximum_eigenpair_defect
        ),
        "maximum_state_reconstruction_defect": state_closure,
        "wall_seconds": time.perf_counter() - started,
    }
    payload["passed"] = bool(
        max(
            payload["maximum_projector_identity_defect"],
            payload["maximum_projector_idempotence_defect"],
            payload["maximum_cross_projector_defect"],
            payload["maximum_eigenpair_defect"],
            payload["maximum_state_reconstruction_defect"],
        )
        <= MAXIMUM_PROJECTOR_DEFECT
    )
    json_path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def _closure_defect(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    scales: np.ndarray,
    significant: np.ndarray,
) -> float:
    if not np.any(significant):
        return 0.0
    return float(
        np.max(
            np.abs(
                np.asarray(reconstructed, dtype=float)[:, significant]
                - np.asarray(reference, dtype=float)[:, significant]
            )
            / scales[significant]
        )
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= np.finfo(float).tiny or b_norm <= np.finfo(float).tiny:
        return None
    return float(np.dot(a, b) / (a_norm * b_norm))


def _error_attribution(
    family_histories: dict[str, np.ndarray],
    total_histories: dict[str, np.ndarray],
    total_baselines: dict[str, np.ndarray],
    labels: tuple[str, str, str],
    *,
    selected_observables: np.ndarray,
) -> dict:
    declared = np.asarray(selected_observables, dtype=int)
    all_total = np.asarray(
        [total_histories[label][:, declared] for label in labels],
        dtype=float,
    )
    response_scales = np.max(np.abs(all_total), axis=(0, 1))
    baseline_scales = np.max(
        np.abs(
            np.asarray(
                [total_baselines[label][declared] for label in labels],
                dtype=float,
            )
        ),
        axis=0,
    )
    significant = response_scales >= (
        wp10c9d0.MINIMUM_RELATIVE_ACTIVITY
        * np.maximum(baseline_scales, 1.0)
    )
    selected_observables = declared[significant]
    if selected_observables.size == 0:
        return {
            "available": False,
            "reason": "no_absolutely_significant_observables",
            "single_family_selected": False,
            "selected_family": None,
        }
    scales = response_scales[significant]
    pair_rows = {}
    normalized_profiles = {}
    for pair_label, first_label, second_label in (
        ("coarse_medium", labels[0], labels[1]),
        ("medium_fine", labels[1], labels[2]),
    ):
        family_error = (
            family_histories[second_label][
                :, :, selected_observables
            ]
            - family_histories[first_label][
                :, :, selected_observables
            ]
        ) / scales[None, None, :]
        total_error = (
            total_histories[second_label][:, selected_observables]
            - total_histories[first_label][:, selected_observables]
        ) / scales[None, :]
        reconstructed = np.sum(family_error, axis=0)
        closure = float(
            np.max(np.abs(reconstructed - total_error))
        )
        activity = np.sum(np.abs(family_error), axis=(1, 2))
        fractions = activity / max(
            float(np.sum(activity)),
            np.finfo(float).tiny,
        )
        leading_index = int(np.argmax(fractions))
        per_time_activity = np.linalg.norm(family_error, axis=2)
        per_time_total = np.sum(per_time_activity, axis=0)
        active = np.linalg.norm(total_error, axis=1) >= (
            MINIMUM_ACTIVE_ERROR_NORM
        )
        dominant = np.zeros(active.shape, dtype=bool)
        dominant[active] = (
            per_time_activity[leading_index, active]
            / np.maximum(
                per_time_total[active],
                np.finfo(float).tiny,
            )
            >= MINIMUM_FAMILY_ACTIVITY_FRACTION
        )
        persistence = (
            float(np.mean(dominant[active])) if np.any(active) else 0.0
        )
        profile = np.linalg.norm(family_error[leading_index], axis=1)
        normalized_profiles[pair_label] = profile
        controlling = np.unravel_index(
            int(np.argmax(np.abs(total_error))),
            total_error.shape,
        )
        component_family_values = family_error[
            :,
            controlling[0],
            controlling[1],
        ]
        component_activity = np.abs(component_family_values)
        component_fractions = component_activity / max(
            float(np.sum(component_activity)),
            np.finfo(float).tiny,
        )
        pair_rows[pair_label] = {
            "maximum_family_sum_closure_defect": closure,
            "family_activity_fractions": {
                name: float(fractions[index])
                for index, name in enumerate(
                    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
                )
            },
            "leading_family": (
                CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES[
                    leading_index
                ]
            ),
            "leading_activity_fraction": float(
                fractions[leading_index]
            ),
            "leading_persistence_fraction": persistence,
            "controlling_time_index": int(controlling[0]),
            "controlling_observable": wp10c9d0.OBSERVABLE_NAMES[
                int(selected_observables[controlling[1]])
            ],
            "controlling_normalized_error": float(
                total_error[controlling]
            ),
            "controlling_component_family_values": {
                name: float(component_family_values[index])
                for index, name in enumerate(
                    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
                )
            },
            "controlling_component_activity_fractions": {
                name: float(component_fractions[index])
                for index, name in enumerate(
                    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
                )
            },
        }
    profile_cosine = _cosine(
        normalized_profiles["coarse_medium"],
        normalized_profiles["medium_fine"],
    )
    first = pair_rows["coarse_medium"]
    second = pair_rows["medium_fine"]
    selected = bool(
        first["leading_family"] == second["leading_family"]
        and first["leading_activity_fraction"]
        >= MINIMUM_FAMILY_ACTIVITY_FRACTION
        and second["leading_activity_fraction"]
        >= MINIMUM_FAMILY_ACTIVITY_FRACTION
        and first["leading_persistence_fraction"]
        >= MINIMUM_FAMILY_PERSISTENCE_FRACTION
        and second["leading_persistence_fraction"]
        >= MINIMUM_FAMILY_PERSISTENCE_FRACTION
        and profile_cosine is not None
        and profile_cosine >= MINIMUM_PAIR_ACTIVITY_COSINE
        and max(
            first["maximum_family_sum_closure_defect"],
            second["maximum_family_sum_closure_defect"],
        )
        <= MAXIMUM_FAMILY_SUM_CLOSURE_DEFECT
    )
    return {
        **pair_rows,
        "leading_activity_profile_cosine": profile_cosine,
        "single_family_selected": selected,
        "selected_family": (
            second["leading_family"] if selected else None
        ),
    }


def run(*, force: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (WP10C8Z_ARRAYS, WP10C9D0_OUTPUT, WP10C9D0_ARRAYS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "WP10c9d1 requires prior evidence: " + ", ".join(missing)
        )
    d0 = json.loads(WP10C9D0_OUTPUT.read_text(encoding="utf-8"))
    if (
        d0.get("classification")
        != "conservative_micro_exports_fail_spatial_gate"
        or not bool(d0.get("method_contract_passed"))
        or bool(d0.get("fixed_q_micro_solver_authorized"))
    ):
        raise RuntimeError("WP10c9d0 authorization changed")

    c8z = wp10c9d0._load_npz(WP10C8Z_ARRAYS)
    d0_arrays = wp10c9d0._load_npz(WP10C9D0_ARRAYS)
    configurations = wp10c9d0._patch_configurations(c8z)
    d0_indices = np.asarray(
        d0_arrays[f"patch_{PATCH_LABELS[0]}_indices"],
        dtype=int,
    )
    d0_times = np.asarray(
        d0_arrays[f"patch_{PATCH_LABELS[0]}_times"],
        dtype=float,
    )
    selected_positions = np.arange(
        0,
        d0_indices.size,
        FAMILY_SAMPLE_STRIDE,
        dtype=int,
    )
    if selected_positions[-1] != d0_indices.size - 1:
        selected_positions = np.concatenate(
            (selected_positions, np.asarray([d0_indices.size - 1]))
        )
    selected_indices = d0_indices[selected_positions]
    selected_times = d0_times[selected_positions]

    arrays: dict[str, np.ndarray] = {
        "selected_positions": selected_positions,
        "selected_history_indices": selected_indices,
        "times": selected_times,
    }
    contracts = {}
    family_signals = {}
    family_cumulative = {}
    total_signals = {}
    total_cumulative = {}
    total_baselines = {}
    signal_closures = {}
    for label in PATCH_LABELS:
        print(f"WP10c9d1: preparing {label}", flush=True)
        state_history = np.asarray(
            c8z[f"{label}_state_history"],
            dtype=float,
        )
        report, cached = _build_or_load_family_signals(
            label=label,
            configuration=configurations[label],
            state_history=state_history,
            selected_indices=selected_indices,
            times=selected_times,
            force=force,
        )
        contracts[label] = report
        family_signals[label] = np.asarray(
            cached["family_signals"],
            dtype=float,
        )
        family_cumulative[label] = np.asarray(
            cached["family_cumulative_signals"],
            dtype=float,
        )
        total_signals[label] = np.asarray(
            d0_arrays[f"patch_{label}_signals"],
            dtype=float,
        )[selected_positions]
        total_baselines[label] = np.asarray(
            d0_arrays[f"patch_{label}_base_observable"],
            dtype=float,
        )
        total_cumulative[label] = wp10c9d0._cumulative_trapezoid(
            selected_times,
            total_signals[label],
        )
        response_scale = np.maximum(
            np.max(np.abs(total_signals[label]), axis=0),
            np.finfo(float).tiny,
        )
        significant = response_scale >= (
            wp10c9d0.MINIMUM_RELATIVE_ACTIVITY
            * np.maximum(
                np.abs(
                    d0_arrays[
                        f"patch_{label}_base_observable"
                    ]
                ),
                1.0,
            )
        )
        signal_closures[label] = {
            "instantaneous": _closure_defect(
                np.sum(family_signals[label], axis=0),
                total_signals[label],
                response_scale,
                significant,
            ),
            "cumulative": _closure_defect(
                np.sum(family_cumulative[label], axis=0),
                total_cumulative[label],
                np.maximum(
                    np.max(np.abs(total_cumulative[label]), axis=0),
                    np.finfo(float).tiny,
                ),
                significant,
            ),
        }
        for key, values in cached.items():
            arrays[f"{label}_{key}"] = values
        arrays[f"{label}_total_signals"] = total_signals[label]
        arrays[f"{label}_total_cumulative_signals"] = (
            total_cumulative[label]
        )

    attributions = {}
    for group, indices in wp10c9d0.GROUPS.items():
        attributions[group] = {
            "instantaneous": _error_attribution(
                family_signals,
                total_signals,
                total_baselines,
                PATCH_LABELS,
                selected_observables=indices,
            ),
            "cumulative": _error_attribution(
                family_cumulative,
                total_cumulative,
                {
                    label: np.zeros_like(total_baselines[label])
                    for label in PATCH_LABELS
                },
                PATCH_LABELS,
                selected_observables=indices,
            ),
        }
    method_contract_passed = bool(
        all(row["passed"] for row in contracts.values())
        and max(
            value
            for row in signal_closures.values()
            for value in row.values()
        )
        <= MAXIMUM_FAMILY_SUM_CLOSURE_DEFECT
    )
    instantaneous_selection = attributions["exported"][
        "instantaneous"
    ]["single_family_selected"]
    cumulative_selection = attributions["exported"]["cumulative"][
        "single_family_selected"
    ]
    subsystem_selected = bool(
        method_contract_passed
        and instantaneous_selection
        and cumulative_selection
        and attributions["exported"]["instantaneous"]["selected_family"]
        == attributions["exported"]["cumulative"]["selected_family"]
    )
    if subsystem_selected:
        selected_family = attributions["exported"]["cumulative"][
            "selected_family"
        ]
        classification = (
            "conservative_export_defect_selects_complete_"
            f"{selected_family}_subsystem"
        )
        next_action = (
            f"design one production-neutral complete {selected_family} "
            "subsystem operator while preserving all coupled conservative "
            "and descriptor ledgers"
        )
    else:
        selected_family = None
        classification = (
            "conservative_export_defect_is_multifamily_full_coupled_"
            "operator_required"
        )
        next_action = (
            "design a production-neutral well-balanced full five-field "
            "near-horizon spatial operator; no one-family or one-block "
            "candidate is authorized"
        )

    arrays_path = DEFAULT_ARRAYS
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "runner": THIS_RUNNER,
        "classification": classification,
        "method_contract_passed": method_contract_passed,
        "single_subsystem_selected": subsystem_selected,
        "selected_family": selected_family,
        "next_action": next_action,
        "family_labels": CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        "observable_names": wp10c9d0.OBSERVABLE_NAMES,
        "sample_count": int(selected_times.size),
        "gates": {
            "maximum_projector_defect": MAXIMUM_PROJECTOR_DEFECT,
            "maximum_family_sum_closure_defect": (
                MAXIMUM_FAMILY_SUM_CLOSURE_DEFECT
            ),
            "minimum_family_activity_fraction": (
                MINIMUM_FAMILY_ACTIVITY_FRACTION
            ),
            "minimum_family_persistence_fraction": (
                MINIMUM_FAMILY_PERSISTENCE_FRACTION
            ),
            "minimum_pair_activity_cosine": (
                MINIMUM_PAIR_ACTIVITY_COSINE
            ),
        },
        "contracts": contracts,
        "family_sum_closures": signal_closures,
        "attributions": attributions,
        "input_evidence": {
            _relative(path): _sha256(path) for path in required
        },
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    payload, _arrays = run(force=arguments.force)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
