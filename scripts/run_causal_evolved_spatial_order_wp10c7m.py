"""Audit evolved-state N32/N64/N128 spatial order for WP10c7m."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

import run_causal_characteristic_extension_wp10c7l as wp10c7l
from imri_qpe.layer3_minidisk_1d import (
    audit_causal_five_field_state_gates,
    causal_five_field_constraint_manifold_jvp,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_dae_scaling,
    causal_five_field_profile_fields,
    causal_five_field_residual_terms,
    causal_five_field_state_from_primitives,
    causal_restrict_cell_averages,
    causal_spatial_contraction_order,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "f51aeee5c5e474a978e16c22f008b1898136e27d"
WP10C7L_OUTPUT = (
    ROOT
    / "outputs/tables/causal_characteristic_extension_wp10c7l.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_evolved_spatial_order_wp10c7m.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_evolved_spatial_order_wp10c7m_arrays.npz"
)
RESOLUTIONS = (32, 64, 128)
ORACLE_METHODS = ("pchip", "natural_cubic")
TARGET_TIME_SECONDS = 5.0e-2
MINIMUM_SPATIAL_ORDER = 1.8
MAXIMUM_AUTHORIZATION_BUDGET = 2.5e-3
PLANNED_COMBINED_TEMPORAL_RESERVE = 5.0e-4
JVP_DIRECTIONAL_SECONDS = 2.0e-4
DIAGNOSED_INNER_RADIUS_RG = 15.0
DIAGNOSED_OUTER_RADIUS_RG = 60.0
REQUIRED_ORDER_FIELDS = {
    "full_log_h_over_r_tangent": "full_domain",
    "full_log_temperature_tangent": "diagnosed_15_60_rg",
    "full_scaled_killing_energy_tangent": "diagnosed_15_60_rg",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate WP10c7l endpoints and state transfers only.",
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


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
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


def _context(n_cells: int):
    return make_causal_five_field_regression_context(
        n_cells,
        **wp10c7l.SPATIAL_OPTIONS,
    )


def _load_inputs() -> tuple[dict, str, dict, dict]:
    if not WP10C7L_OUTPUT.exists():
        raise RuntimeError("WP10c7m requires canonical WP10c7l evidence")
    wp10c7l_evidence = json.loads(
        WP10C7L_OUTPUT.read_text(encoding="utf-8")
    )
    if not (
        wp10c7l_evidence.get("work_package") == "WP10c7l"
        and wp10c7l_evidence.get("decision")
        == "wp10c7l_characteristic_rung_spatial_stop"
        and wp10c7l_evidence.get("next_authorization")
        == "diagnose_spatial_error_growth_before_extension"
        and wp10c7l_evidence.get("target_elapsed_time_seconds")
        == TARGET_TIME_SECONDS
        and wp10c7l_evidence.get("spatial_options")
        == wp10c7l.SPATIAL_OPTIONS
    ):
        raise RuntimeError("WP10c7l did not authorize WP10c7m")
    wp10c7k_evidence, wp10c7k_sha256 = wp10c7l._validate_wp10c7k()
    initial = wp10c7l._initial_bundles(wp10c7k_evidence)
    endpoints = {}
    endpoint_provenance = {}
    for n_cells in (32, 64):
        parent = wp10c7l._parent_checkpoint_entry(
            wp10c7k_evidence,
            n_cells,
        )
        endpoint = wp10c7l._load_snapshot(
            initial[n_cells],
            wp10c7k_sha256,
            parent,
            "production",
            "t_0p05",
        )
        endpoints[n_cells] = endpoint
        path = wp10c7l._checkpoint_path(
            n_cells,
            "production",
            "t_0p05",
        )
        endpoint_provenance[n_cells] = {
            "path": _relative(path),
            "sha256": _sha256(path),
            "state_vector_sha256": _array_sha256(
                endpoint.state_vector
            ),
        }
    return (
        wp10c7l_evidence,
        _sha256(WP10C7L_OUTPUT),
        initial,
        {
            "endpoints": endpoints,
            "provenance": endpoint_provenance,
        },
    )


def _interpolate(
    source_radius: np.ndarray,
    source_values: np.ndarray,
    target_radius: np.ndarray,
    method: str,
) -> np.ndarray:
    source_x = np.log(np.asarray(source_radius, dtype=float))
    target_x = np.log(np.asarray(target_radius, dtype=float))
    values = np.asarray(source_values, dtype=float)
    if method == "pchip":
        interpolator = PchipInterpolator(
            source_x,
            values,
            axis=0,
            extrapolate=True,
        )
    elif method == "natural_cubic":
        interpolator = CubicSpline(
            source_x,
            values,
            axis=0,
            bc_type="natural",
            extrapolate=True,
        )
    else:
        raise ValueError(f"unknown WP10c7m oracle method {method!r}")
    result = np.asarray(interpolator(target_x), dtype=float)
    if np.any(~np.isfinite(result)):
        raise RuntimeError("common-state interpolation is non-finite")
    return result


def _common_state(
    source_context,
    source_vector: np.ndarray,
    target_context,
    method: str,
) -> np.ndarray:
    source = unpack_causal_five_field_state(
        source_vector,
        int(source_context.grid.centers.size),
    )
    primitives = _interpolate(
        source_context.grid.centers,
        source.primitives,
        target_context.grid.centers,
        method,
    )
    state = causal_five_field_state_from_primitives(
        target_context,
        primitives,
    )
    vector = pack_causal_five_field_state(state)
    if not audit_causal_five_field_state_gates(
        target_context,
        vector,
    )["passed"]:
        raise RuntimeError(
            f"N{state.n_cells} {method} common state failed gates"
        )
    return vector


def _shared_direction(
    source_context,
    source_initial_vector: np.ndarray,
    source_endpoint_vector: np.ndarray,
    target_context,
    method: str,
) -> np.ndarray:
    n_source = int(source_context.grid.centers.size)
    initial = unpack_causal_five_field_state(
        source_initial_vector,
        n_source,
    )
    endpoint = unpack_causal_five_field_state(
        source_endpoint_vector,
        n_source,
    )
    direction = (
        endpoint.primitives - initial.primitives
    ) / TARGET_TIME_SECONDS
    return _interpolate(
        source_context.grid.centers,
        direction,
        target_context.grid.centers,
        method,
    )


def _scaled_algebraic_residual(
    vector: np.ndarray,
    evaluation,
    n_cells: int,
) -> float:
    state = unpack_causal_five_field_state(vector, n_cells)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    return float(
        np.max(
            np.abs(
                evaluation.residual[5 * n_cells :]
                / scaling.row_scales[5 * n_cells :]
            )
        )
    )


def _artifact(
    context,
    vector: np.ndarray,
    direction: np.ndarray,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    n_cells = int(context.grid.centers.size)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    terms = causal_five_field_residual_terms(
        context,
        vector,
        evaluation,
    )
    tangent_started = time.perf_counter()
    tangent = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
        rank_relative_threshold=1.0e-11 if n_cells == 32 else None,
        linear_solver="sparse" if n_cells == 128 else "dense",
    )
    tangent_seconds = time.perf_counter() - tangent_started
    jvp = causal_five_field_constraint_manifold_jvp(
        context,
        vector,
        direction,
        finite_difference_step=JVP_DIRECTIONAL_SECONDS,
    )
    profiles = causal_five_field_profile_fields(
        context,
        vector,
        evaluation,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    state = unpack_causal_five_field_state(vector, n_cells)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    conserved_scales = np.asarray(
        scaling.column_scales[: 5 * n_cells],
        dtype=float,
    ).reshape(n_cells, 5)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    primitive_tangent = np.asarray(
        tangent["full"]["primitive_tangent_per_s"],
        dtype=float,
    )
    conserved_tangent = np.asarray(
        tangent["full"]["conserved_tangent_per_s"],
        dtype=float,
    )
    tangent_fields = {
        "full_log_h_over_r_tangent": np.asarray(
            tangent["full"]["log_h_over_r_tangent_per_s"],
            dtype=float,
        ),
        "full_log_temperature_tangent": primitive_tangent[:, 3],
        "full_raw_killing_energy_tangent": conserved_tangent[:, 3],
        "full_scaled_killing_energy_tangent": (
            conserved_tangent[:, 3] / conserved_scales[:, 3]
        ),
    }
    for name, component in tangent["components"].items():
        tangent_fields[
            f"component_{name}_log_h_over_r_tangent"
        ] = np.asarray(
            component["log_h_over_r_tangent_per_s"],
            dtype=float,
        )
        tangent_fields[
            f"component_{name}_raw_killing_energy_tangent"
        ] = np.asarray(
            component["conserved_tangent_per_s"],
            dtype=float,
        )[:, 3]
        tangent_fields[
            f"component_{name}_scaled_killing_energy_tangent"
        ] = (
            np.asarray(
                component["conserved_tangent_per_s"],
                dtype=float,
            )[:, 3]
            / conserved_scales[:, 3]
        )
    term_fields = {
        "complete_killing_energy_residual_density": (
            np.asarray(evaluation.conservation_rows, dtype=float)[:, 3]
            / measures
        )
    }
    term_fields.update(
        {
            f"{name}_killing_energy_residual_density": (
                np.asarray(values, dtype=float)[:, 3] / measures
            )
            for name, values in terms.items()
        }
    )
    jvp_fields = {
        "complete_killing_energy_jvp_density": (
            np.asarray(jvp["conservation_jvp"], dtype=float)[:, 3]
            / measures
        )
    }
    jvp_fields.update(
        {
            f"{name}_killing_energy_jvp_density": (
                np.asarray(values, dtype=float)[:, 3] / measures
            )
            for name, values in jvp["term_jvps"].items()
        }
    )
    profile_fields = {
        name: np.asarray(values, dtype=float)
        for name, values in profiles.items()
        if name
        in (
            "log_surface_density",
            "log_temperature",
            "log_h_over_r",
            "log_integrated_pressure",
            "log_specific_internal_energy",
            "radial_velocity_over_c",
            "specific_stress",
        )
    }
    summary = {
        "n_cells": n_cells,
        "state_vector_sha256": _array_sha256(vector),
        "state_gates": audit_causal_five_field_state_gates(
            context,
            vector,
        ),
        "maximum_scaled_algebraic_residual": _scaled_algebraic_residual(
            vector,
            evaluation,
            n_cells,
        ),
        "tangent_solver": tangent["linear_solver"],
        "tangent_consistency_dimensions": list(
            tangent["consistency_dimensions"]
        ),
        "tangent_consistency_numerical_rank": tangent[
            "consistency_numerical_rank"
        ],
        "tangent_consistency_condition_estimate": tangent[
            "consistency_condition_estimate"
        ],
        "tangent_consistency_nonzeros": tangent[
            "consistency_nonzeros"
        ],
        "maximum_scaled_tangent_consistency_defect": tangent[
            "maximum_scaled_consistency_defect"
        ],
        "maximum_tangent_reconstruction_relative_defect": tangent[
            "maximum_tangent_reconstruction_relative_defect"
        ],
        "maximum_jvp_reconstruction_relative_defect": jvp[
            "maximum_reconstruction_relative_defect"
        ],
        "wall_seconds": float(time.perf_counter() - started),
        "tangent_wall_seconds": float(tangent_seconds),
    }
    arrays = {
        "radius_rg": radius_rg,
        "cell_measures": measures,
        "primitive_direction_per_s": np.asarray(direction, dtype=float),
        "profiles": profile_fields,
        "tangents": tangent_fields,
        "terms": term_fields,
        "jvp": jvp_fields,
    }
    return summary, arrays


def _selected_metrics(
    difference: np.ndarray,
    measures: np.ndarray,
    radius_rg: np.ndarray,
    selection: np.ndarray,
) -> dict:
    values = np.asarray(difference, dtype=float)[selection]
    selected_measures = np.asarray(measures, dtype=float)[selection]
    selected_radius = np.asarray(radius_rg, dtype=float)[selection]
    absolute = np.abs(values)
    peak = int(np.argmax(absolute))
    return {
        "maximum_absolute_difference": float(absolute[peak]),
        "measure_weighted_l1_difference": float(
            np.sum(selected_measures * absolute)
            / np.sum(selected_measures)
        ),
        "measure_weighted_l2_difference": float(
            np.sqrt(
                np.sum(selected_measures * values**2)
                / np.sum(selected_measures)
            )
        ),
        "maximum_difference_radius_rg": float(selected_radius[peak]),
    }


def _pair_metrics(
    coarse_context,
    fine_context,
    coarse: np.ndarray,
    fine: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    restricted = causal_restrict_cell_averages(
        coarse_context.grid,
        fine_context.grid,
        fine,
    )
    difference = np.asarray(coarse, dtype=float) - restricted
    radius_rg = (
        np.asarray(coarse_context.grid.centers, dtype=float)
        / coarse_context.grid.gravitational_radius
    )
    regions = {
        "full_domain": np.ones(radius_rg.size, dtype=bool),
        "boundary_excluded_two_cells": (
            (np.arange(radius_rg.size) >= 2)
            & (np.arange(radius_rg.size) < radius_rg.size - 2)
        ),
        "diagnosed_15_60_rg": (
            (radius_rg >= DIAGNOSED_INNER_RADIUS_RG)
            & (radius_rg <= DIAGNOSED_OUTER_RADIUS_RG)
        ),
    }
    return (
        {
            name: _selected_metrics(
                difference,
                coarse_context.grid.cell_measures,
                radius_rg,
                selection,
            )
            for name, selection in regions.items()
        },
        restricted,
        difference,
    )


def _pair_artifact(
    coarse_n: int,
    fine_n: int,
    contexts: dict,
    artifacts: dict,
) -> tuple[dict, dict]:
    summary = {
        "coarse_cells": coarse_n,
        "fine_cells": fine_n,
    }
    arrays = {}
    for category in ("profiles", "tangents", "terms", "jvp"):
        summary[category] = {}
        for name, coarse_values in artifacts[coarse_n][category].items():
            metrics, restricted, difference = _pair_metrics(
                contexts[coarse_n],
                contexts[fine_n],
                coarse_values,
                artifacts[fine_n][category][name],
            )
            summary[category][name] = metrics
            arrays[f"{category}_{name}_restricted_fine"] = restricted
            arrays[f"{category}_{name}_difference"] = difference
    return summary, arrays


def _orders(pair_rows: dict) -> dict:
    coarse = pair_rows["n32_n64"]
    fine = pair_rows["n64_n128"]
    result = {}
    for category in ("profiles", "tangents", "terms", "jvp"):
        result[category] = {}
        for name in coarse[category]:
            result[category][name] = {}
            for region in coarse[category][name]:
                coarse_error = coarse[category][name][region][
                    "maximum_absolute_difference"
                ]
                fine_error = fine[category][name][region][
                    "maximum_absolute_difference"
                ]
                if coarse_error > 0.0 and fine_error > 0.0:
                    order = causal_spatial_contraction_order(
                        coarse_error,
                        fine_error,
                    )
                else:
                    order = None
                result[category][name][region] = order
    return result


def _common_oracle_audit(
    method: str,
    contexts: dict,
    source_context,
    source_initial_vector: np.ndarray,
    source_endpoint_vector: np.ndarray,
) -> tuple[dict, dict]:
    vectors = {}
    artifact_summaries = {}
    artifacts = {}
    flat_arrays = {}
    for n_cells in RESOLUTIONS:
        vectors[n_cells] = _common_state(
            source_context,
            source_endpoint_vector,
            contexts[n_cells],
            method,
        )
        direction = _shared_direction(
            source_context,
            source_initial_vector,
            source_endpoint_vector,
            contexts[n_cells],
            method,
        )
        summary, arrays = _artifact(
            contexts[n_cells],
            vectors[n_cells],
            direction,
        )
        artifact_summaries[str(n_cells)] = summary
        artifacts[n_cells] = arrays
        for category in ("profiles", "tangents", "terms", "jvp"):
            for name, values in arrays[category].items():
                flat_arrays[
                    f"{method}_n{n_cells}_{category}_{name}"
                ] = values
        flat_arrays[f"{method}_n{n_cells}_direction"] = arrays[
            "primitive_direction_per_s"
        ]
    pair_rows = {}
    for coarse_n, fine_n, label in (
        (32, 64, "n32_n64"),
        (64, 128, "n64_n128"),
    ):
        pair, pair_arrays = _pair_artifact(
            coarse_n,
            fine_n,
            contexts,
            artifacts,
        )
        pair_rows[label] = pair
        for name, values in pair_arrays.items():
            flat_arrays[f"{method}_{label}_{name}"] = values
    return (
        {
            "method": method,
            "state_construction": (
                "single N64 evolved primitive profile interpolated in "
                "log radius and remapped exactly onto each DAE constraint "
                "manifold"
            ),
            "artifacts": artifact_summaries,
            "pair_audits": pair_rows,
            "observed_orders": _orders(pair_rows),
        },
        flat_arrays,
    )


def _native_audit(
    contexts: dict,
    initial: dict,
    endpoints: dict,
) -> tuple[dict, dict]:
    artifacts = {}
    summaries = {}
    flat_arrays = {}
    for n_cells in (32, 64):
        endpoint = endpoints[n_cells]
        endpoint_state = unpack_causal_five_field_state(
            endpoint.state_vector,
            n_cells,
        )
        initial_state = unpack_causal_five_field_state(
            initial[n_cells]["vector"],
            n_cells,
        )
        direction = (
            endpoint_state.primitives - initial_state.primitives
        ) / TARGET_TIME_SECONDS
        summary, arrays = _artifact(
            contexts[n_cells],
            endpoint.state_vector,
            direction,
        )
        summaries[str(n_cells)] = summary
        artifacts[n_cells] = arrays
        for category in ("profiles", "tangents", "terms", "jvp"):
            for name, values in arrays[category].items():
                flat_arrays[
                    f"native_n{n_cells}_{category}_{name}"
                ] = values
    pair, pair_arrays = _pair_artifact(
        32,
        64,
        contexts,
        artifacts,
    )
    for name, values in pair_arrays.items():
        flat_arrays[f"native_n32_n64_{name}"] = values
    return (
        {
            "state_construction": (
                "each mesh's own WP10c7l production endpoint and native "
                "evolved direction"
            ),
            "artifacts": summaries,
            "n32_n64_pair_audit": pair,
        },
        flat_arrays,
    )


def _authorization(
    evidence: dict,
    common_audits: dict,
) -> dict:
    raw_endpoint = float(
        evidence["common_time_contract"]["t_0p05"][
            "raw_n32_n64_log_h_over_r_difference"
        ]
    )
    method_rows = {}
    for method, audit in common_audits.items():
        coarse = audit["pair_audits"]["n32_n64"]["tangents"][
            "full_log_h_over_r_tangent"
        ]["full_domain"]["maximum_absolute_difference"]
        fine = audit["pair_audits"]["n64_n128"]["tangents"][
            "full_log_h_over_r_tangent"
        ]["full_domain"]["maximum_absolute_difference"]
        measured_ratio_projection = raw_endpoint * fine / coarse
        local_tangent_projection = TARGET_TIME_SECONDS * fine
        projected = max(
            measured_ratio_projection,
            local_tangent_projection,
        )
        required_orders = {
            name: {
                "region": region,
                "order": audit["observed_orders"]["tangents"][name][
                    region
                ],
            }
            for name, region in REQUIRED_ORDER_FIELDS.items()
        }
        method_rows[method] = {
            "n32_n64_log_h_tangent_difference_per_s": coarse,
            "n64_n128_log_h_tangent_difference_per_s": fine,
            "measured_endpoint_ratio_projection": (
                measured_ratio_projection
            ),
            "local_tangent_projection": local_tangent_projection,
            "conservative_projected_n64_n128_endpoint_difference": (
                projected
            ),
            "required_orders": required_orders,
            "minimum_required_order": min(
                float(value["order"])
                for value in required_orders.values()
            ),
        }
    projected_values = [
        row["conservative_projected_n64_n128_endpoint_difference"]
        for row in method_rows.values()
    ]
    oracle_uncertainty = float(
        max(projected_values) - min(projected_values)
    )
    conservative_total = float(
        max(projected_values)
        + PLANNED_COMBINED_TEMPORAL_RESERVE
        + oracle_uncertainty
    )
    minimum_order = min(
        row["minimum_required_order"] for row in method_rows.values()
    )
    authorized = bool(
        minimum_order >= MINIMUM_SPATIAL_ORDER
        and conservative_total <= MAXIMUM_AUTHORIZATION_BUDGET
    )
    return {
        "raw_wp10c7l_n32_n64_endpoint_difference": raw_endpoint,
        "oracle_methods": method_rows,
        "minimum_required_field_order_across_oracles": minimum_order,
        "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
        "maximum_projected_endpoint_difference": max(projected_values),
        "oracle_projection_uncertainty": oracle_uncertainty,
        "planned_combined_n64_n128_temporal_reserve": (
            PLANNED_COMBINED_TEMPORAL_RESERVE
        ),
        "conservative_authorization_total": conservative_total,
        "maximum_authorization_budget": MAXIMUM_AUTHORIZATION_BUDGET,
        "n128_campaign_authorized": authorized,
    }


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    evidence, evidence_sha256, initial, loaded = _load_inputs()
    contexts = {
        32: initial[32]["context"],
        64: initial[64]["context"],
        128: _context(128),
    }
    source_context = contexts[64]
    source_endpoint = loaded["endpoints"][64].state_vector
    transfers = {}
    for method in ORACLE_METHODS:
        transfers[method] = {
            str(n_cells): {
                "state_vector_sha256": _array_sha256(
                    _common_state(
                        source_context,
                        source_endpoint,
                        contexts[n_cells],
                        method,
                    )
                )
            }
            for n_cells in RESOLUTIONS
        }
    if args.preflight:
        print(
            json.dumps(
                {
                    "work_package": "WP10c7m",
                    "preflight_passed": True,
                    "wp10c7l_evidence_sha256": evidence_sha256,
                    "endpoint_provenance": loaded["provenance"],
                    "common_state_transfers": transfers,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    common_audits = {}
    flat_arrays = {}
    for method in ORACLE_METHODS:
        audit, arrays = _common_oracle_audit(
            method,
            contexts,
            source_context,
            initial[64]["vector"],
            source_endpoint,
        )
        common_audits[method] = audit
        flat_arrays.update(arrays)
        print(
            json.dumps(
                {
                    "stage": "common_state_oracle",
                    "method": method,
                    "status": "complete",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    native, native_arrays = _native_audit(
        contexts,
        initial,
        loaded["endpoints"],
    )
    flat_arrays.update(native_arrays)
    authorization = _authorization(evidence, common_audits)
    invariants_passed = bool(
        all(
            row["state_gates"]["passed"]
            and row["maximum_scaled_algebraic_residual"] <= 1.0e-10
            and row["maximum_scaled_tangent_consistency_defect"]
            <= 5.0e-9
            and row["maximum_tangent_reconstruction_relative_defect"]
            <= 5.0e-7
            and row["maximum_jvp_reconstruction_relative_defect"]
            <= 5.0e-7
            for audit in common_audits.values()
            for row in audit["artifacts"].values()
        )
        and all(
            row["state_gates"]["passed"]
            and row["maximum_scaled_algebraic_residual"] <= 1.0e-10
            for row in native["artifacts"].values()
        )
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **flat_arrays)
    authorized = bool(
        invariants_passed
        and authorization["n128_campaign_authorized"]
    )
    result = {
        "work_package": "WP10c7m",
        "base_commit": BASE_COMMIT,
        "scope": (
            "evolved-state common-profile and native-endpoint "
            "N32/N64/operator-only-N128 spatial-order audit"
        ),
        "trajectory_run": False,
        "target_elapsed_time_seconds": TARGET_TIME_SECONDS,
        "spatial_options": dict(wp10c7l.SPATIAL_OPTIONS),
        "wp10c7l_evidence": {
            "path": _relative(WP10C7L_OUTPUT),
            "sha256": evidence_sha256,
            "decision": evidence["decision"],
        },
        "endpoint_provenance": {
            str(name): row
            for name, row in loaded["provenance"].items()
        },
        "common_state_transfers": transfers,
        "common_state_operator_audits": common_audits,
        "native_evolved_state_audit": native,
        "authorization": authorization,
        "invariants": {
            "passed": invariants_passed,
        },
        "gates": {
            "invariants_passed": invariants_passed,
            "minimum_spatial_order_passed": bool(
                authorization[
                    "minimum_required_field_order_across_oracles"
                ]
                >= MINIMUM_SPATIAL_ORDER
            ),
            "projected_total_below_half_gate": bool(
                authorization["conservative_authorization_total"]
                <= MAXIMUM_AUTHORIZATION_BUDGET
            ),
            "wp10c7m_passed": authorized,
        },
        "decision": (
            "wp10c7m_n128_campaign_authorized"
            if authorized
            else "wp10c7m_stop_before_n128_campaign"
        ),
        "next_authorization": (
            "one_fresh_n128_0p05_campaign_with_temporal_control"
            if authorized
            else "repair_or_locally_refine_evolved_spatial_operator"
        ),
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    _write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
                "invariants_passed": invariants_passed,
                "authorization": authorization,
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
