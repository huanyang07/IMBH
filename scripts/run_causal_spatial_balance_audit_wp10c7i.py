"""Isolate the remaining causal transport-source balance defect."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_constraint_manifold_jvp,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_profile_fields,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_regression_seed_parameters,
    causal_five_field_residual_terms,
    causal_five_field_term_reconstruction_defect,
    causal_restrict_cell_averages,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ac05f352380616f2ec0e346adaf3613b054ee3e2"
RESOLUTIONS = (16, 32, 64)
ORACLE_RESOLUTIONS = (16, 32, 64, 128)
TARGET_EXTENSION_SECONDS = 1.53746e-2
SPATIAL_GATE = 5.0e-3
PRETRAJECTORY_GATE_FRACTION = 0.5
MINIMUM_ORDER = 1.8
MINIMUM_FULL_REDUCTION = 20.0
MINIMUM_DIAGNOSED_REDUCTION = 10.0
JVP_FINITE_DIFFERENCE_STEP = 2.0e-4
RANK_RELATIVE_THRESHOLD = 1.0e-11
DIAGNOSED_INNER_RADIUS_RG = 15.0
DIAGNOSED_OUTER_RADIUS_RG = 60.0
INTERIOR_PEAK_OUTER_RADIUS_RG = 30.0
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_balance_wp10c7i.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_spatial_balance_wp10c7i_arrays.npz"
)

VARIANTS = {
    "current_plm": {},
    "boundary_trace": {
        "boundary_trace_reconstruction": "plm_one_sided",
    },
    "cell_rates": {
        "cell_rate_scheme": "quadratic_log_radius",
    },
    "source_quadrature": {
        "cell_source_quadrature": "gauss_legendre_4",
    },
    "storage_quadrature": {
        "cell_storage_quadrature": "gauss_legendre_4",
    },
    "quadratic_face_high_order": {
        "spatial_reconstruction": "quadratic_admissible",
        "boundary_trace_reconstruction": "plm_one_sided",
        "cell_source_quadrature": "gauss_legendre_4",
        "cell_storage_quadrature": "gauss_legendre_4",
    },
    "smooth_plm_local_rate_high_order": {
        "boundary_trace_reconstruction": "plm_one_sided",
        "cell_source_quadrature": "gauss_legendre_4_local_rates",
        "cell_storage_quadrature": "gauss_legendre_4",
    },
    "quadratic_face_local_rate_high_order": {
        "spatial_reconstruction": "quadratic_admissible",
        "boundary_trace_reconstruction": "plm_one_sided",
        "cell_source_quadrature": "gauss_legendre_4_local_rates",
        "cell_storage_quadrature": "gauss_legendre_4",
    },
    "combined_high_order": {
        "boundary_trace_reconstruction": "plm_one_sided",
        "cell_rate_scheme": "quadratic_log_radius",
        "cell_source_quadrature": "gauss_legendre_4",
        "cell_storage_quadrature": "gauss_legendre_4",
    },
}

TANGENT_COMPONENTS = (
    "central_face_transport",
    "rusanov_face_transport",
    "flux_primary_closure",
    "perfect_fluid_geometry",
    "stress_geometry",
    "radiative_cooling",
    "vertical_work",
    "stress_relaxation",
    "stream",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--skip-n128",
        action="store_true",
        help="Skip the inexpensive residual/JVP N128 oracle row.",
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


def _context(variant: str, n_cells: int):
    options = dict(VARIANTS[variant])
    spatial_reconstruction = options.pop(
        "spatial_reconstruction",
        "plm_smooth",
    )
    return make_causal_five_field_regression_context(
        n_cells,
        spatial_reconstruction=spatial_reconstruction,
        **options,
    )


def _directions(radius_rg: np.ndarray) -> dict[str, np.ndarray]:
    radius = np.asarray(radius_rg, dtype=float)
    inner = np.exp(
        -0.5 * (np.log(radius / 19.0) / 0.30) ** 2
    )
    broad = np.exp(
        -0.5 * (np.log(radius / 40.0) / 0.75) ** 2
    )
    density = np.zeros((radius.size, 5), dtype=float)
    density[:, 0] = inner
    temperature = np.zeros_like(density)
    temperature[:, 3] = inner
    broad_temperature = np.zeros_like(density)
    broad_temperature[:, 3] = broad
    return {
        "inner_log_surface_density": density,
        "inner_log_temperature": temperature,
        "broad_log_temperature": broad_temperature,
    }


def _region_masks(radius_rg: np.ndarray) -> dict[str, np.ndarray]:
    radius = np.asarray(radius_rg, dtype=float)
    first_four = np.arange(radius.size) < min(4, radius.size)
    diagnosed = (
        (radius >= DIAGNOSED_INNER_RADIUS_RG)
        & (radius <= DIAGNOSED_OUTER_RADIUS_RG)
    )
    interior_peak = (
        (radius >= DIAGNOSED_INNER_RADIUS_RG)
        & (radius <= INTERIOR_PEAK_OUTER_RADIUS_RG)
    )
    return {
        "full_domain": np.ones(radius.size, dtype=bool),
        "first_four_inner_cells": first_four,
        "diagnosed_15_60_rg": diagnosed,
        "interior_peak_15_30_rg": interior_peak,
    }


def _selected_metrics(
    difference: np.ndarray,
    measures: np.ndarray,
    radius_rg: np.ndarray,
    selection: np.ndarray,
) -> dict:
    selected = np.asarray(difference, dtype=float)[selection]
    selected_measures = np.asarray(measures, dtype=float)[selection]
    selected_radius = np.asarray(radius_rg, dtype=float)[selection]
    absolute = np.abs(selected)
    peak = int(np.argmax(absolute))
    return {
        "maximum_absolute_difference": float(absolute[peak]),
        "measure_weighted_l2_difference": float(
            np.sqrt(
                np.sum(selected_measures * selected**2)
                / np.sum(selected_measures)
            )
        ),
        "maximum_difference_radius_rg": float(selected_radius[peak]),
    }


def _pair_metrics(
    coarse_context,
    fine_context,
    coarse_values: np.ndarray,
    fine_values: np.ndarray,
) -> dict:
    restricted = causal_restrict_cell_averages(
        coarse_context.grid,
        fine_context.grid,
        fine_values,
    )
    difference = np.asarray(coarse_values, dtype=float) - restricted
    radius_rg = (
        coarse_context.grid.centers
        / coarse_context.grid.gravitational_radius
    )
    masks = _region_masks(radius_rg)
    return {
        "regions": {
            name: _selected_metrics(
                difference,
                coarse_context.grid.cell_measures,
                radius_rg,
                selection,
            )
            for name, selection in masks.items()
        },
        "difference": difference,
        "restricted_fine": restricted,
    }


def _source_recovery_defect(context, evaluation) -> float:
    expected = np.asarray(
        context.stream_sources.weighted_killing_source_per_ct,
        dtype=float,
    )
    recovered = np.asarray(
        evaluation.integrated_source_components_per_ct["stream"],
        dtype=float,
    )[:, :4]
    scale = np.maximum(np.abs(expected), 1.0)
    return float(np.max(np.abs(recovered - expected) / scale))


def _operator_artifact(
    variant: str,
    n_cells: int,
    seed_parameters: dict,
) -> tuple[dict, dict]:
    context = _context(variant, n_cells)
    state = make_causal_five_field_seed(context, **seed_parameters)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    terms = causal_five_field_residual_terms(
        context,
        vector,
        evaluation,
    )
    reconstruction_defect = causal_five_field_term_reconstruction_defect(
        evaluation,
        terms,
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        state.primitives,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    radius_rg = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    profiles = causal_five_field_profile_fields(
        context,
        vector,
        evaluation,
    )
    term_densities = {
        name: np.asarray(values, dtype=float) / measures[:, None]
        for name, values in terms.items()
    }
    jvp_arrays = {}
    jvp_summary = {}
    for name, direction in _directions(radius_rg).items():
        audit = causal_five_field_constraint_manifold_jvp(
            context,
            vector,
            direction,
            finite_difference_step=JVP_FINITE_DIFFERENCE_STEP,
        )
        jvp_arrays[name] = {
            "conservation_density": (
                np.asarray(audit["conservation_jvp"], dtype=float)
                / measures[:, None]
            ),
            "term_densities": {
                term_name: np.asarray(values, dtype=float)
                / measures[:, None]
                for term_name, values in audit["term_jvps"].items()
            },
        }
        jvp_summary[name] = {
            "maximum_reconstruction_relative_defect": audit[
                "maximum_reconstruction_relative_defect"
            ],
            "maximum_entrywise_reconstruction_relative_defect": audit[
                "maximum_entrywise_reconstruction_relative_defect"
            ],
            "maximum_reconstruction_absolute_defect": audit[
                "maximum_reconstruction_absolute_defect"
            ],
        }
    algebraic_residual = max(
        float(np.max(np.abs(evaluation.primitive_map_rows))),
        float(np.max(np.abs(evaluation.interior_flux_rows))),
        float(np.max(np.abs(evaluation.inner_flux_rows))),
        float(np.max(np.abs(evaluation.outer_flux_rows))),
    )
    summary = {
        "variant": variant,
        "n_cells": n_cells,
        "spatial_options": {
            "spatial_reconstruction": context.spatial_reconstruction,
            "boundary_trace_reconstruction": (
                context.boundary_trace_reconstruction
            ),
            "cell_rate_scheme": context.cell_rate_scheme,
            "cell_source_quadrature": context.cell_source_quadrature,
            "cell_storage_quadrature": context.cell_storage_quadrature,
        },
        "maximum_algebraic_residual": algebraic_residual,
        "maximum_residual_reconstruction_relative_defect": (
            reconstruction_defect["maximum_relative_defect"]
        ),
        "maximum_stream_recovery_relative_defect": (
            _source_recovery_defect(context, evaluation)
        ),
        "minimum_admissibility_factor": float(
            np.min(reconstruction.admissibility_factors)
        ),
        "admissibility_limited_cell_count": int(
            np.count_nonzero(
                reconstruction.admissibility_factors < 1.0 - 1.0e-12
            )
        ),
        "outer_incoming_characteristics": int(
            evaluation.outer_incoming_characteristics
        ),
        "outer_boundary_choked": bool(evaluation.outer_boundary_choked),
        "jvp": jvp_summary,
    }
    arrays = {
        "radius_rg": radius_rg,
        "cell_measures": measures,
        "profiles": profiles,
        "term_densities": term_densities,
        "jvp": jvp_arrays,
    }
    return summary, arrays


def _tangent_artifact(
    variant: str,
    n_cells: int,
    seed_parameters: dict,
) -> tuple[dict, dict]:
    context = _context(variant, n_cells)
    state = make_causal_five_field_seed(context, **seed_parameters)
    vector = pack_causal_five_field_state(state)
    decomposition = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
        rank_relative_threshold=(
            RANK_RELATIVE_THRESHOLD if n_cells in (16, 32) else None
        ),
    )
    components = {
        name: np.asarray(
            decomposition["components"][name][
                "log_h_over_r_tangent_per_s"
            ],
            dtype=float,
        )
        for name in TANGENT_COMPONENTS
    }
    summary = {
        "variant": variant,
        "n_cells": n_cells,
        "consistency_dimensions": list(
            decomposition["consistency_dimensions"]
        ),
        "consistency_numerical_rank": decomposition[
            "consistency_numerical_rank"
        ],
        "consistency_condition_estimate": decomposition[
            "consistency_condition_estimate"
        ],
        "maximum_scaled_consistency_defect": decomposition[
            "maximum_scaled_consistency_defect"
        ],
        "maximum_residual_reconstruction_relative_defect": decomposition[
            "maximum_residual_reconstruction_relative_defect"
        ],
        "maximum_tangent_reconstruction_relative_defect": decomposition[
            "maximum_tangent_reconstruction_relative_defect"
        ],
    }
    arrays = {
        "radius_rg": np.asarray(decomposition["radius_rg"], dtype=float),
        "cell_measures": np.asarray(
            decomposition["cell_measures"],
            dtype=float,
        ),
        "full_log_h_over_r": np.asarray(
            decomposition["full"]["log_h_over_r_tangent_per_s"],
            dtype=float,
        ),
        "full_log_temperature": np.asarray(
            decomposition["full"]["primitive_tangent_per_s"][:, 3],
            dtype=float,
        ),
        "components": components,
    }
    return summary, arrays


def _compact_pair(pair: dict) -> dict:
    return {"regions": pair["regions"]}


def _variant_pair_audit(
    variant: str,
    operator_arrays: dict[int, dict],
    tangent_arrays: dict[int, dict],
) -> tuple[dict, dict]:
    pair_summaries = []
    pair_arrays = []
    for coarse_n, fine_n in ((16, 32), (32, 64)):
        coarse_context = _context(variant, coarse_n)
        fine_context = _context(variant, fine_n)
        tangent_fields = {
            "full_log_h_over_r": (
                tangent_arrays[coarse_n]["full_log_h_over_r"],
                tangent_arrays[fine_n]["full_log_h_over_r"],
            ),
            "full_log_temperature": (
                tangent_arrays[coarse_n]["full_log_temperature"],
                tangent_arrays[fine_n]["full_log_temperature"],
            ),
        }
        tangent_fields.update(
            {
                f"component_{name}": (
                    tangent_arrays[coarse_n]["components"][name],
                    tangent_arrays[fine_n]["components"][name],
                )
                for name in TANGENT_COMPONENTS
            }
        )
        tangent_pairs = {
            name: _pair_metrics(
                coarse_context,
                fine_context,
                coarse,
                fine,
            )
            for name, (coarse, fine) in tangent_fields.items()
        }
        residual_pairs = {
            name: _pair_metrics(
                coarse_context,
                fine_context,
                operator_arrays[coarse_n]["term_densities"][name][:, 3],
                operator_arrays[fine_n]["term_densities"][name][:, 3],
            )
            for name in operator_arrays[coarse_n]["term_densities"]
        }
        jvp_pairs = {}
        for direction in operator_arrays[coarse_n]["jvp"]:
            coarse_jvp = operator_arrays[coarse_n]["jvp"][direction]
            fine_jvp = operator_arrays[fine_n]["jvp"][direction]
            jvp_pairs[direction] = {
                "complete_energy": _pair_metrics(
                    coarse_context,
                    fine_context,
                    coarse_jvp["conservation_density"][:, 3],
                    fine_jvp["conservation_density"][:, 3],
                ),
                "term_energy": {
                    name: _pair_metrics(
                        coarse_context,
                        fine_context,
                        coarse_jvp["term_densities"][name][:, 3],
                        fine_jvp["term_densities"][name][:, 3],
                    )
                    for name in coarse_jvp["term_densities"]
                },
            }
        pair_summaries.append(
            {
                "coarse_cells": coarse_n,
                "fine_cells": fine_n,
                "tangent": {
                    name: _compact_pair(pair)
                    for name, pair in tangent_pairs.items()
                },
                "baseline_energy_residual": {
                    name: _compact_pair(pair)
                    for name, pair in residual_pairs.items()
                },
                "jvp_energy": {
                    direction: {
                        "complete": _compact_pair(
                            values["complete_energy"]
                        ),
                        "terms": {
                            name: _compact_pair(pair)
                            for name, pair in values[
                                "term_energy"
                            ].items()
                        },
                    }
                    for direction, values in jvp_pairs.items()
                },
            }
        )
        pair_arrays.append(
            {
                "tangent": tangent_pairs,
                "baseline_energy_residual": residual_pairs,
                "jvp_energy": jvp_pairs,
            }
        )
    observed_orders = {}
    for region in _region_masks(
        tangent_arrays[16]["radius_rg"]
    ):
        observed_orders[region] = {}
        for field in (
            "full_log_h_over_r",
            "full_log_temperature",
        ):
            coarse_error = pair_summaries[0]["tangent"][field][
                "regions"
            ][region]["maximum_absolute_difference"]
            fine_error = pair_summaries[1]["tangent"][field][
                "regions"
            ][region]["maximum_absolute_difference"]
            observed_orders[region][field] = float(
                np.log2(coarse_error / fine_error)
            )
    return {
        "pairs": pair_summaries,
        "observed_orders": observed_orders,
    }, {"pairs": pair_arrays}


def _decision(variant_audits: dict) -> dict:
    current = variant_audits["current_plm"]
    current_fine = current["pairs"][1]["tangent"][
        "full_log_h_over_r"
    ]["regions"]
    rows = {}
    for variant, audit in variant_audits.items():
        fine = audit["pairs"][1]["tangent"]["full_log_h_over_r"][
            "regions"
        ]
        full = fine["full_domain"]["maximum_absolute_difference"]
        band = fine["diagnosed_15_60_rg"][
            "maximum_absolute_difference"
        ]
        full_projected = TARGET_EXTENSION_SECONDS * full
        band_projected = TARGET_EXTENSION_SECONDS * band
        full_order = audit["observed_orders"]["full_domain"][
            "full_log_h_over_r"
        ]
        band_order = audit["observed_orders"]["diagnosed_15_60_rg"][
            "full_log_h_over_r"
        ]
        rows[variant] = {
            "n32_n64_full_tangent_difference_per_s": full,
            "n32_n64_diagnosed_tangent_difference_per_s": band,
            "projected_full_endpoint_difference": full_projected,
            "projected_diagnosed_endpoint_difference": band_projected,
            "full_tangent_order": full_order,
            "diagnosed_tangent_order": band_order,
            "full_reduction_from_current": (
                current_fine["full_domain"][
                    "maximum_absolute_difference"
                ]
                / full
            ),
            "diagnosed_reduction_from_current": (
                current_fine["diagnosed_15_60_rg"][
                    "maximum_absolute_difference"
                ]
                / band
            ),
            "passes_absolute_pretrajectory_gate": bool(
                full_projected
                <= PRETRAJECTORY_GATE_FRACTION * SPATIAL_GATE
                and band_projected
                <= PRETRAJECTORY_GATE_FRACTION * SPATIAL_GATE
            ),
            "passes_order_gate": bool(
                full_order >= MINIMUM_ORDER
                and band_order >= MINIMUM_ORDER
            ),
            "passes_reduction_gate": bool(
                current_fine["full_domain"][
                    "maximum_absolute_difference"
                ]
                / full
                >= MINIMUM_FULL_REDUCTION
                and current_fine["diagnosed_15_60_rg"][
                    "maximum_absolute_difference"
                ]
                / band
                >= MINIMUM_DIAGNOSED_REDUCTION
            ),
        }
    passing = [
        variant
        for variant, row in rows.items()
        if row["passes_absolute_pretrajectory_gate"]
        and row["passes_order_gate"]
        and row["passes_reduction_gate"]
    ]
    return {
        "variant_rows": rows,
        "passing_general_high_order_variants": passing,
        "general_high_order_repair_sufficient": bool(passing),
        "reference_state_fluctuation_operator_required": not bool(passing),
        "spatial_gate": SPATIAL_GATE,
        "pretrajectory_gate_fraction": PRETRAJECTORY_GATE_FRACTION,
        "maximum_pretrajectory_projected_error": (
            PRETRAJECTORY_GATE_FRACTION * SPATIAL_GATE
        ),
        "minimum_required_order": MINIMUM_ORDER,
        "minimum_required_full_reduction": MINIMUM_FULL_REDUCTION,
        "minimum_required_diagnosed_reduction": (
            MINIMUM_DIAGNOSED_REDUCTION
        ),
    }


def main() -> None:
    arguments = _arguments()
    output_path = _absolute(arguments.output)
    arrays_path = _absolute(arguments.arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_context = _context("current_plm", 16)
    seed_parameters = causal_five_field_regression_seed_parameters(
        baseline_context
    )
    oracle_resolutions = (
        RESOLUTIONS
        if arguments.skip_n128
        else ORACLE_RESOLUTIONS
    )
    operator_summaries = {}
    operator_arrays = {}
    tangent_summaries = {}
    tangent_arrays = {}
    for variant in VARIANTS:
        operator_summaries[variant] = {}
        operator_arrays[variant] = {}
        for n_cells in oracle_resolutions:
            summary, arrays = _operator_artifact(
                variant,
                n_cells,
                seed_parameters,
            )
            operator_summaries[variant][n_cells] = summary
            operator_arrays[variant][n_cells] = arrays
            print(
                json.dumps(
                    {
                        "stage": "operator_and_jvp",
                        "variant": variant,
                        "n_cells": n_cells,
                        "status": "complete",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        tangent_summaries[variant] = {}
        tangent_arrays[variant] = {}
        for n_cells in RESOLUTIONS:
            summary, arrays = _tangent_artifact(
                variant,
                n_cells,
                seed_parameters,
            )
            tangent_summaries[variant][n_cells] = summary
            tangent_arrays[variant][n_cells] = arrays
            print(
                json.dumps(
                    {
                        "stage": "consistent_tangent",
                        "variant": variant,
                        "n_cells": n_cells,
                        "status": "complete",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    variant_audits = {}
    pair_arrays = {}
    for variant in VARIANTS:
        summary, arrays = _variant_pair_audit(
            variant,
            operator_arrays[variant],
            tangent_arrays[variant],
        )
        variant_audits[variant] = summary
        pair_arrays[variant] = arrays
    decision = _decision(variant_audits)

    maximum_algebraic = max(
        summary["maximum_algebraic_residual"]
        for rows in operator_summaries.values()
        for summary in rows.values()
    )
    maximum_stream_defect = max(
        summary["maximum_stream_recovery_relative_defect"]
        for rows in operator_summaries.values()
        for summary in rows.values()
    )
    maximum_jvp_defect = max(
        direction["maximum_reconstruction_relative_defect"]
        for rows in operator_summaries.values()
        for summary in rows.values()
        for direction in summary["jvp"].values()
    )
    ranks_pass = all(
        summary["consistency_numerical_rank"]
        == summary["consistency_dimensions"][0]
        for rows in tangent_summaries.values()
        for n_cells, summary in rows.items()
        if n_cells in (16, 32)
    )
    invariants = {
        "maximum_algebraic_residual": maximum_algebraic,
        "maximum_stream_recovery_relative_defect": maximum_stream_defect,
        "maximum_jvp_reconstruction_relative_defect": maximum_jvp_defect,
        "all_n16_n32_consistency_systems_full_rank": ranks_pass,
        "passed": bool(
            maximum_algebraic <= 1.0e-10
            and maximum_stream_defect <= 2.0e-15
            and maximum_jvp_defect <= 2.0e-7
            and ranks_pass
        ),
    }

    flat_arrays = {}
    for variant, rows in operator_arrays.items():
        for n_cells, arrays in rows.items():
            prefix = f"{variant}_n{n_cells}_operator"
            flat_arrays[f"{prefix}_radius_rg"] = arrays["radius_rg"]
            for name, values in arrays["term_densities"].items():
                flat_arrays[f"{prefix}_term_{name}"] = values
            for direction, values in arrays["jvp"].items():
                flat_arrays[
                    f"{prefix}_jvp_{direction}_complete"
                ] = values["conservation_density"]
    for variant, rows in tangent_arrays.items():
        for n_cells, arrays in rows.items():
            prefix = f"{variant}_n{n_cells}_tangent"
            flat_arrays[f"{prefix}_radius_rg"] = arrays["radius_rg"]
            flat_arrays[f"{prefix}_full_log_h_over_r"] = arrays[
                "full_log_h_over_r"
            ]
            flat_arrays[f"{prefix}_full_log_temperature"] = arrays[
                "full_log_temperature"
            ]
            for name, values in arrays["components"].items():
                flat_arrays[f"{prefix}_component_{name}"] = values
    np.savez_compressed(arrays_path, **flat_arrays)

    result = {
        "work_package": "WP10c7i",
        "base_commit": BASE_COMMIT,
        "purpose": (
            "ablate boundary, cell-rate, and endogenous-source spatial "
            "consistency before any reference-state fluctuation correction"
        ),
        "trajectory_run": False,
        "variants": VARIANTS,
        "resolutions": list(RESOLUTIONS),
        "oracle_resolutions": list(oracle_resolutions),
        "target_extension_seconds": TARGET_EXTENSION_SECONDS,
        "jvp_finite_difference_step": JVP_FINITE_DIFFERENCE_STEP,
        "operator_summaries": {
            variant: {str(n): row for n, row in rows.items()}
            for variant, rows in operator_summaries.items()
        },
        "tangent_summaries": {
            variant: {str(n): row for n, row in rows.items()}
            for variant, rows in tangent_summaries.items()
        },
        "variant_audits": variant_audits,
        "invariants": invariants,
        "decision": decision,
        "arrays": {
            "path": _relative(arrays_path),
            "sha256": _sha256(arrays_path),
        },
        "passed": bool(
            invariants["passed"]
            and decision["general_high_order_repair_sufficient"]
        ),
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
                "invariants_passed": invariants["passed"],
                "general_high_order_repair_sufficient": decision[
                    "general_high_order_repair_sufficient"
                ],
                "passed": result["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
