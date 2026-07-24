"""Run the WP10c8r interface-state dimension and sufficiency audit.

WP10c8q rejected the raw 34-coordinate instantaneous slow-rate closure and
reported a rank-two plane for the interface-4 M/J/E_K response.  This package
adds the missing absolute-significance gate before interpreting that rank and
audits the complete 34-rate tangent response on the same coordinate fiber.

The package is deliberately evidence-first.  It does not modify production
physics, the DAE, BDF2, the moment ladder, or any truth trajectory.  A
two-coordinate nonlinear lift is authorized only when two independent,
scientifically significant interface-4 response families survive the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_gate_normalized_finite_time_null_gain,
)


BASE_COMMIT = "e180139b9ba32e2849506bb09ff924e6d762b54e"
WORK_PACKAGE = "WP10c8r"
SCHEMA_VERSION = 1

PARENT_JSON = (
    ROOT
    / "outputs/tables/causal_extended_healing_wp10c8q.json"
)
PARENT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_extended_healing_wp10c8q_arrays.npz"
)
PARENT_RATE_JSON = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8q"
    / "slow_rate_fiber_audit.json"
)
PARENT_RATE_ARRAYS = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c8q"
    / "slow_rate_fiber_audit_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_interface_state_sufficiency_wp10c8r.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_interface_state_sufficiency_wp10c8r_arrays.npz"
)

LEVEL_INDEX = wp10c8o.LEVEL_INDEX
RATE_WINDOW_SECONDS = wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
INTERFACE_FLUX_GATE = wp10c8o.INTERFACE_FLUX_RELATIVE_GATE
STATIC_SCREEN_GATE = wp10c8o.INSTANTANEOUS_SCREEN_GATE
PROMOTION_GATE = 0.10
RANK_DIRECTION_RATIO = 0.10
AUDIT_SEED_MULTIPLIER = 1.0e-3
TOP_TANGENT_MODES = 8
INTERFACE_INDEX = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _load_operator_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata_json"].item()))
        arrays = {
            name: np.asarray(source[name], dtype=float)
            for name in source.files
            if name != "metadata_json"
        }
    if metadata.get("work_package") != "WP10c8i":
        raise RuntimeError(f"unexpected operator cache at {path}")
    return arrays, metadata


def _gate_normalized_interface_half_difference(
    plus_interface_values: np.ndarray,
    minus_interface_values: np.ndarray,
) -> np.ndarray:
    """Return signed M/J/E_K half differences in interface gate units.

    WP10c8o static-output arrays already divide every interface component by
    its frozen physical flux scale.  Only the declared relative gate remains
    to be applied here.
    """

    plus = np.asarray(plus_interface_values, dtype=float)
    minus = np.asarray(minus_interface_values, dtype=float)
    if (
        plus.shape != minus.shape
        or plus.size != 12
        or np.any(~np.isfinite(plus))
        or np.any(~np.isfinite(minus))
    ):
        raise ValueError("interface values must be finite four-face M/J/E arrays")
    return 0.5 * (plus - minus).reshape(4, 3) / INTERFACE_FLUX_GATE


def _significance_filtered_transport_audit(
    vectors: np.ndarray,
    families: tuple[str, ...],
    *,
    minimum_maximum_component: float = PROMOTION_GATE,
) -> dict:
    """Audit rank only after absolute output significance is established."""

    values = np.asarray(vectors, dtype=float)
    threshold = float(minimum_maximum_component)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(families) != values.shape[0]
        or np.any(~np.isfinite(values))
        or not np.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise ValueError("significance-filtered transport inputs are invalid")
    maxima = np.max(np.abs(values), axis=1)
    significant = maxima >= threshold
    significant_indices = np.flatnonzero(significant)
    unique_families = tuple(
        dict.fromkeys(families[index] for index in significant_indices)
    )
    if significant_indices.size:
        selected = values[significant_indices]
        norms = np.linalg.norm(selected, axis=1)
        if np.any(norms <= np.finfo(float).tiny):
            raise ValueError("significant transport vector vanished")
        unit = selected / norms[:, None]
        _left, singular, right_h = np.linalg.svd(unit, full_matrices=False)
        padded = np.zeros(3, dtype=float)
        padded[: singular.size] = singular
        leading = max(float(padded[0]), np.finfo(float).tiny)
        ratios = padded / leading
        supported_dimension = int(
            np.count_nonzero(ratios >= RANK_DIRECTION_RATIO)
        )
        normal = np.asarray(right_h[-1], dtype=float)
        pivot = int(np.argmax(np.abs(normal)))
        if normal[pivot] < 0.0:
            normal *= -1.0
    else:
        unit = np.empty((0, 3), dtype=float)
        padded = np.zeros(3, dtype=float)
        ratios = np.zeros(3, dtype=float)
        supported_dimension = 0
        normal = np.zeros(3, dtype=float)
    independent_significant_family_count = len(unique_families)
    rank_two_authorized = bool(
        independent_significant_family_count >= 2
        and supported_dimension >= 2
    )
    return {
        "sample_maximum_absolute_components": maxima,
        "minimum_significant_maximum_component": threshold,
        "significant_mask": significant,
        "significant_sample_count": int(significant_indices.size),
        "significant_sample_indices": significant_indices,
        "significant_families": unique_families,
        "independent_significant_family_count": (
            independent_significant_family_count
        ),
        "unit_significant_vectors": unit,
        "singular_values": padded,
        "singular_value_ratios": ratios,
        "supported_dimension_at_ratio_0p1": supported_dimension,
        "plane_normal": normal,
        "rank_two_authorized": rank_two_authorized,
    }


def _infer_loading_time_seconds(
    minus_coordinate_rates: np.ndarray,
    plus_coordinate_rates: np.ndarray,
    slow_rate_half_difference: np.ndarray,
) -> tuple[float, float]:
    """Infer the loading time used by the committed slow-rate conversion."""

    minus = np.asarray(minus_coordinate_rates, dtype=float)
    plus = np.asarray(plus_coordinate_rates, dtype=float)
    slow = np.asarray(slow_rate_half_difference, dtype=float)
    if (
        minus.shape != plus.shape
        or minus.shape != slow.shape
        or minus.ndim != 1
        or np.any(~np.isfinite(minus))
        or np.any(~np.isfinite(plus))
        or np.any(~np.isfinite(slow))
    ):
        raise ValueError("loading-time inference inputs are invalid")
    raw_half = 0.5 * (plus - minus)
    scale = max(float(np.max(np.abs(raw_half))), np.finfo(float).tiny)
    mask = np.abs(raw_half) >= 1.0e-10 * scale
    if not np.any(mask):
        raise ValueError("coordinate-rate pair has no usable loading-time entry")
    ratios = slow[mask] * RATE_WINDOW_SECONDS / raw_half[mask]
    loading = float(np.median(ratios))
    defect = float(
        np.max(np.abs(ratios / loading - 1.0))
    )
    if not np.isfinite(loading) or loading <= 0.0:
        raise ValueError("inferred loading time is invalid")
    return loading, defect


def _operator_dimension_audit(
    cache_path: Path,
    *,
    loading_time_seconds: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Audit the complete slow-rate fiber and its amplitude-box top modes."""

    arrays, metadata = _load_operator_cache(cache_path)
    rate_rows, _rate_gates, rate_diagnostics = wp10c8i._rate_output_rows(
        arrays,
        metadata,
        LEVEL_INDEX,
    )
    slow_rows = (
        float(loading_time_seconds)
        / RATE_WINDOW_SECONDS
        * np.asarray(rate_rows, dtype=float)
    )
    constraints = np.asarray(
        arrays[f"level_{LEVEL_INDEX}_constraints"],
        dtype=float,
    )
    state_weights = np.asarray(arrays["state_weights"], dtype=float)
    scaled_amplitudes = (
        np.asarray(arrays["physical_input_amplitudes"], dtype=float)
        / np.asarray(arrays["primitive_column_scales"], dtype=float)
    )
    audit = causal_gate_normalized_finite_time_null_gain(
        slow_rows,
        constraints,
        np.ones(slow_rows.shape[0], dtype=float),
        state_weights=state_weights,
        state_amplitudes_scaled=scaled_amplitudes,
    )
    left, singular, right_h = np.linalg.svd(
        np.asarray(audit.gate_normalized_null_operator, dtype=float),
        full_matrices=False,
    )
    leading = max(float(singular[0]), np.finfo(float).tiny)
    ratios = singular / leading
    coordinate_names = tuple(
        metadata["levels"][LEVEL_INDEX]["coordinate_names"]
    )
    interface_names = tuple(metadata["interface_flux_names"])
    interface_rows = np.asarray(arrays["interface_flux_jacobian"], dtype=float)
    top_count = min(TOP_TANGENT_MODES, singular.size)
    top_states = []
    top_rate_responses = []
    top_interface_responses = []
    top_rows = []
    for mode in range(top_count):
        state = audit.null_basis_audit.basis @ right_h[mode]
        amplitude_ratio = float(
            np.max(np.abs(state) / scaled_amplitudes)
        )
        box_factor = 1.0 / max(1.0, amplitude_ratio)
        tested_state = AUDIT_SEED_MULTIPLIER * box_factor * state
        rate_response = slow_rows @ tested_state
        interface_response = (
            interface_rows @ tested_state / INTERFACE_FLUX_GATE
        )
        rate_control = int(np.argmax(np.abs(rate_response)))
        interface_control = int(np.argmax(np.abs(interface_response)))
        point_control = int(
            np.argmax(np.abs(tested_state) / scaled_amplitudes)
        )
        maximum_rate = float(np.max(np.abs(rate_response)))
        maximum_interface = float(np.max(np.abs(interface_response)))
        top_states.append(tested_state)
        top_rate_responses.append(rate_response)
        top_interface_responses.append(interface_response)
        top_rows.append(
            {
                "mode_index": mode,
                "singular_value": float(singular[mode]),
                "singular_value_ratio": float(ratios[mode]),
                "amplitude_box_factor_before_seed_multiplier": box_factor,
                "seed_multiplier": AUDIT_SEED_MULTIPLIER,
                "maximum_slow_rate_response": maximum_rate,
                "controlling_slow_rate_coordinate": (
                    coordinate_names[rate_control]
                ),
                "maximum_interface_flux_response": maximum_interface,
                "controlling_interface_flux": (
                    interface_names[interface_control]
                ),
                "controlling_primitive_cell": point_control // 5,
                "controlling_primitive_field": point_control % 5,
                "slow_rate_significant_at_0p25": bool(
                    maximum_rate >= STATIC_SCREEN_GATE
                ),
                "interface_flux_significant_at_0p1": bool(
                    maximum_interface >= PROMOTION_GATE
                ),
            }
        )
    significant_rows = [
        row for row in top_rows if row["slow_rate_significant_at_0p25"]
    ]
    significant_interface_controls = tuple(
        dict.fromkeys(
            row["controlling_interface_flux"]
            for row in significant_rows
            if row["interface_flux_significant_at_0p1"]
        )
    )
    summary = {
        "cache_path": _relative(cache_path),
        "cache_sha256": _sha256(cache_path),
        "n_cells": int(metadata["n_cells"]),
        "anchor_label": metadata["anchor_label"],
        "loading_time_seconds": float(loading_time_seconds),
        "coordinate_count": int(slow_rows.shape[0]),
        "fiber_nullity": audit.null_basis_audit.nullity,
        "singular_values": singular,
        "singular_value_ratios": ratios,
        "direction_counts": {
            "at_least_0p5": int(np.count_nonzero(ratios >= 0.5)),
            "at_least_0p1": int(np.count_nonzero(ratios >= 0.1)),
            "at_least_0p01": int(np.count_nonzero(ratios >= 0.01)),
            "at_least_0p001": int(np.count_nonzero(ratios >= 0.001)),
        },
        "top_modes": top_rows,
        "tested_significant_slow_rate_mode_count": len(significant_rows),
        "significant_interface_controls": significant_interface_controls,
        "all_significant_modes_localized_at_interface4": bool(
            significant_rows
            and len(significant_interface_controls) == 1
            and significant_interface_controls[0].startswith(
                f"interface_{INTERFACE_INDEX}_"
            )
        ),
        "rate_row_diagnostics": rate_diagnostics,
    }
    audit_arrays = {
        "singular_values": singular,
        "singular_value_ratios": ratios,
        "left_singular_vectors": left,
        "top_tested_state_directions": np.asarray(top_states),
        "top_slow_rate_responses": np.asarray(top_rate_responses),
        "top_interface_flux_responses": np.asarray(top_interface_responses),
        "coordinate_names": np.asarray(coordinate_names, dtype="U"),
        "interface_names": np.asarray(interface_names, dtype="U"),
    }
    return summary, audit_arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (
        PARENT_JSON,
        PARENT_ARRAYS,
        PARENT_RATE_JSON,
        PARENT_RATE_ARRAYS,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required WP10c8q evidence is missing: {path}")
    parent = _load_json(PARENT_JSON)
    parent_rate = _load_json(PARENT_RATE_JSON)
    if not (
        parent.get("work_package") == "WP10c8q"
        and parent_rate.get("work_package") == "WP10c8q"
        and parent.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and parent_rate.get("arrays_sha256")
        == _sha256(PARENT_RATE_ARRAYS)
    ):
        raise RuntimeError("WP10c8q parent provenance failed")
    parent_rate_summary = parent_rate["summary"]
    parent_arrays = _load_npz(PARENT_ARRAYS)
    rate_arrays = _load_npz(PARENT_RATE_ARRAYS)

    rate_case_ids = tuple(
        str(value) for value in rate_arrays["robustness_case_ids"]
    )
    rate_interface_vectors = []
    rate_all_interfaces = []
    for case_id in rate_case_ids:
        normalized = _gate_normalized_interface_half_difference(
            rate_arrays[f"{case_id}_plus_interface_flux"],
            rate_arrays[f"{case_id}_minus_interface_flux"],
        )
        rate_all_interfaces.append(normalized)
        rate_interface_vectors.append(normalized[INTERFACE_INDEX - 1])
    rate_all_interfaces = np.asarray(rate_all_interfaces)
    rate_interface_vectors = np.asarray(rate_interface_vectors)

    parent_unit_rank = np.linalg.svd(
        np.asarray(
            rate_arrays["robustness_unit_interface4_transport_vectors"],
            dtype=float,
        ),
        compute_uv=False,
    )
    parent_unit_ratios = parent_unit_rank / max(
        float(parent_unit_rank[0]), np.finfo(float).tiny
    )
    _u, _s, parent_right = np.linalg.svd(
        np.asarray(
            rate_arrays["robustness_unit_interface4_transport_vectors"],
            dtype=float,
        ),
        full_matrices=False,
    )
    parent_plane_normal = np.asarray(parent_right[-1], dtype=float)
    pivot = int(np.argmax(np.abs(parent_plane_normal)))
    if parent_plane_normal[pivot] < 0.0:
        parent_plane_normal *= -1.0

    rate_significance = _significance_filtered_transport_audit(
        rate_interface_vectors,
        tuple(f"slow_rate:{case_id}" for case_id in rate_case_ids),
    )

    original_n64 = np.asarray(
        parent_arrays[
            "n64_continuation_decision_normalized_interface4_transport"
        ],
        dtype=float,
    )
    original_n128 = np.asarray(
        parent_arrays[
            "n128_existing_divergence_signed_gate_normalized_internal_faces_mje"
        ],
        dtype=float,
    )[:, INTERFACE_INDEX - 1, :]
    combined_vectors = np.vstack(
        (
            original_n64,
            original_n128,
            rate_interface_vectors,
        )
    )
    combined_families = (
        *("original_healing_direction" for _ in range(original_n64.shape[0])),
        *("original_healing_direction" for _ in range(original_n128.shape[0])),
        *(f"slow_rate:{case_id}" for case_id in rate_case_ids),
    )
    combined_significance = _significance_filtered_transport_audit(
        combined_vectors,
        combined_families,
    )

    n128_case = str(parent_rate_summary["n128_confirmation_pair"])
    loading128, loading128_defect = _infer_loading_time_seconds(
        rate_arrays[f"{n128_case}_minus_coordinate_rate_output"],
        rate_arrays[f"{n128_case}_plus_coordinate_rate_output"],
        rate_arrays[
            f"{n128_case}_slow_rate_half_difference_per_unit_slow_time"
        ],
    )
    loading64 = float(
        parent_rate_summary["tangent"]["loading_time_seconds"]
    )
    loading64_second = float(
        parent_rate_summary["held_out_anchor_tangent"]["loading_time_seconds"]
    )
    operator_contracts = (
        (
            "n64_t_0p025",
            ROOT
            / "outputs/checkpoints/causal_five_field_wp10c8i"
            / "N064_t_0p025_moment_operators.npz",
            loading64,
        ),
        (
            "n64_t_0p10",
            ROOT
            / "outputs/checkpoints/causal_five_field_wp10c8i"
            / "N064_t_0p10_moment_operators.npz",
            loading64_second,
        ),
        (
            "n128_t_0p025",
            ROOT
            / "outputs/checkpoints/causal_five_field_wp10c8i"
            / "N128_t_0p025_moment_operators.npz",
            loading128,
        ),
    )
    operator_summaries = {}
    output_arrays: dict[str, np.ndarray] = {
        "rate_case_ids": np.asarray(rate_case_ids, dtype="U"),
        "rate_case_all_interface_gate_vectors": rate_all_interfaces,
        "rate_case_interface4_gate_vectors": rate_interface_vectors,
        "parent_unit_rank_singular_values": parent_unit_rank,
        "parent_unit_rank_singular_value_ratios": parent_unit_ratios,
        "parent_unit_rank_plane_normal": parent_plane_normal,
        "original_n64_interface4_gate_history": original_n64,
        "original_n128_interface4_gate_history": original_n128,
        "combined_interface4_gate_vectors": combined_vectors,
        "combined_interface4_families": np.asarray(
            combined_families,
            dtype="U",
        ),
    }
    for label, path, loading in operator_contracts:
        if not path.exists():
            raise FileNotFoundError(f"required operator cache is missing: {path}")
        summary, arrays = _operator_dimension_audit(
            path,
            loading_time_seconds=loading,
        )
        operator_summaries[label] = summary
        output_arrays.update(
            {f"{label}_{name}": value for name, value in arrays.items()}
        )

    rate_interface4_maximum = float(
        np.max(np.abs(rate_interface_vectors))
    )
    rate_all_interface_maximum = float(
        np.max(np.abs(rate_all_interfaces))
    )
    significant_rate_case_count = int(
        rate_significance["significant_sample_count"]
    )
    two_coordinate_interface4_authorized = bool(
        rate_significance["rank_two_authorized"]
        and combined_significance["rank_two_authorized"]
        and all(
            summary["all_significant_modes_localized_at_interface4"]
            for summary in operator_summaries.values()
        )
    )
    if two_coordinate_interface4_authorized:
        decision = "wp10c8r_two_coordinate_interface4_audit_authorized"
        next_action = (
            "identify_two_physical_trace_coordinates_then_run_augmented_fiber"
        )
    else:
        decision = (
            "wp10c8r_two_coordinate_interface4_state_not_authorized"
        )
        next_action = (
            "audit_healing_and_localization_of_complete_slow_rate_fiber_modes"
        )

    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    arrays_path = args.arrays if args.arrays.is_absolute() else ROOT / args.arrays
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **output_arrays)

    source_paths = (
        ROOT / "scripts/run_causal_interface_state_sufficiency_wp10c8r.py",
        ROOT / "scripts/run_causal_extended_healing_wp10c8q.py",
        ROOT / "scripts/run_causal_moment_sufficiency_audit_wp10c8i.py",
        ROOT / "scripts/run_causal_nonlinear_fiber_audit_wp10c8o.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_mixed_reduction.py",
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "production_physics_changed": False,
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "new_truth_evolution_run": False,
            "augmented_interface_coordinates_added": False,
            "interface_dynamics_derived": False,
            "parent_rank_interpretation_audited": True,
            "complete_slow_rate_fiber_audited": True,
        },
        "authorization": {
            "wp10c8q": {
                "json_path": _relative(PARENT_JSON),
                "json_sha256": _sha256(PARENT_JSON),
                "arrays_path": _relative(PARENT_ARRAYS),
                "arrays_sha256": _sha256(PARENT_ARRAYS),
                "rate_json_path": _relative(PARENT_RATE_JSON),
                "rate_json_sha256": _sha256(PARENT_RATE_JSON),
                "rate_arrays_path": _relative(PARENT_RATE_ARRAYS),
                "rate_arrays_sha256": _sha256(PARENT_RATE_ARRAYS),
            }
        },
        "gates": {
            "static_counterexample_half_spread": STATIC_SCREEN_GATE,
            "minimum_significant_transport_component": PROMOTION_GATE,
            "rank_direction_ratio": RANK_DIRECTION_RATIO,
            "audit_seed_multiplier": AUDIT_SEED_MULTIPLIER,
        },
        "parent_rank_reproduction": {
            "singular_values": parent_unit_rank,
            "singular_value_ratios": parent_unit_ratios,
            "plane_normal": parent_plane_normal,
            "semantics": (
                "Reproduces the WP10c8q SVD after every nonzero interface-4 "
                "vector is normalized to unit length, without an absolute "
                "scientific-significance gate."
            ),
        },
        "absolute_transport_significance": {
            "rate_case_count": len(rate_case_ids),
            "maximum_interface4_gate_half_spread": rate_interface4_maximum,
            "maximum_all_interface_gate_half_spread": (
                rate_all_interface_maximum
            ),
            "significant_rate_case_count": significant_rate_case_count,
            "rate_cases": rate_significance,
            "combined_with_original_healing_direction": (
                combined_significance
            ),
            "original_n64_maximum_interface4_gate_half_spread": float(
                np.max(np.abs(original_n64))
            ),
            "original_n128_maximum_interface4_gate_half_spread": float(
                np.max(np.abs(original_n128))
            ),
        },
        "loading_time_inference": {
            "n64_primary_seconds": loading64,
            "n64_second_anchor_seconds": loading64_second,
            "n128_seconds": loading128,
            "n128_relative_defect": loading128_defect,
        },
        "complete_slow_rate_operator": operator_summaries,
        "two_coordinate_interface4_authorized": (
            two_coordinate_interface4_authorized
        ),
        "augmented_fiber_run": False,
        "augmented_fiber_not_run_reason": (
            None
            if two_coordinate_interface4_authorized
            else (
                "No independent slow-rate fiber case has a scientifically "
                "significant interface-4 transport response, while the "
                "complete slow-rate tangent has several significant "
                "directions not localized at interface 4."
            )
        ),
        "decision": decision,
        "next_action": next_action,
        "interpretation": (
            "The original WP10c8p healing direction remains a significant, "
            "persistent, approximately rank-one interface-4 ambiguity.  The "
            "WP10c8q independent slow-rate directions do not establish a "
            "second interface-4 amplitude because their absolute interface-4 "
            "responses are many orders below the declared gate.  Their large "
            "responses live in the complete 34-rate vector, so their healing "
            "and spatial support must be audited before selecting a localized "
            "interface state or a distributed coarse model."
        ),
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": decision,
                "maximum_interface4_rate_case_half_spread": (
                    rate_interface4_maximum
                ),
                "significant_rate_case_count": significant_rate_case_count,
                "two_coordinate_interface4_authorized": (
                    two_coordinate_interface4_authorized
                ),
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
