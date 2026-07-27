"""Run the WP10c9c0c characteristic-pair and local-transfer audit.

The package reuses only committed frozen generators and histories:

* the WP10c8y common continuum perturbation on N64/N128/N256 local grids;
* the WP10c9a pure inward-shear packet on the ratio-1/2/4 hybrid grids;
* the exact WP10c9c0b physical generator-block decompositions.

No production operator, nonlinear trajectory, boundary treatment, or
scientific gate is changed.
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
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y
import run_causal_inner_characteristic_phase_audit_wp10c9a as wp10c9a
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v
import run_causal_inner_shear_energy_ledger_wp10c9c0b as wp10c9c0b

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    causal_five_field_characteristic_family_decomposition,
    causal_five_field_characteristic_family_projectors,
    causal_five_field_scaled_shear_energy_operators,
    causal_five_field_shear_energy_projectors,
    causal_local_quadratic_energy_work_ledger,
    causal_pairwise_family_cross_work,
    causal_pairwise_weighted_gram_ledger,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9c0c"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_family_transfer_audit_wp10c9c0c.py"
)
CORE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_family_transfer.py"
)
WP10C8Y_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_common_mode_audit_wp10c8y.json"
)
WP10C8Y_ARRAYS = (
    ROOT
    / "outputs/tables/causal_inner_common_mode_audit_wp10c8y_arrays.npz"
)
WP10C9A_OUTPUT = (
    ROOT
    / "outputs/tables/causal_inner_characteristic_phase_audit_wp10c9a.json"
)
WP10C9A_ARRAYS = (
    ROOT
    / "outputs/tables/causal_inner_characteristic_phase_audit_wp10c9a_arrays.npz"
)
WP10C9C0B_OUTPUT = (
    ROOT
    / "outputs/tables/causal_inner_shear_energy_ledger_wp10c9c0b.json"
)
WP10C9C0B_CACHE = (
    ROOT / "outputs/checkpoints/causal_inner_shear_energy_ledger_wp10c9c0b"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_family_transfer_audit_wp10c9c0c.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_family_transfer_audit_wp10c9c0c_arrays.npz"
)

MAXIMUM_PROJECTOR_DEFECT = 1.0e-10
MAXIMUM_DECOMPOSITION_DEFECT = 1.0e-12
MAXIMUM_PAIRWISE_LEDGER_DEFECT = 1.0e-12
MAXIMUM_LOCAL_BLOCK_CLOSURE_DEFECT = 1.0e-10
MINIMUM_ABSOLUTE_FAMILY_SIGNIFICANCE = 1.0e-4
MINIMUM_CONTROLLING_PAIR_FRACTION = 0.25
MINIMUM_CONTROLLING_BLOCK_FRACTION = 0.50
MINIMUM_RADIAL_PROFILE_COSINE = 0.90


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _relative_maximum_defect(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(first_values))),
        float(np.max(np.abs(second_values))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first_values - second_values)) / scale)


def _observed_order(coarse: float, fine: float) -> float | None:
    if not (
        np.isfinite(coarse)
        and np.isfinite(fine)
        and coarse > 0.0
        and fine > 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _common_contexts() -> tuple[dict[int, object], dict[int, dict]]:
    parents = {
        mesh: wp10c8v._parent_bundle(mesh)
        for mesh in wp10c8v.PARENT_MESHES
    }
    profiles = {
        mesh: wp10c8v._base_profiles(mesh, parents)
        for mesh in wp10c8y.MESHES
    }
    contexts = {
        mesh: wp10c8v._local_context(
            parents[128]["context"],
            profiles[mesh],
        )
        for mesh in wp10c8y.MESHES
    }
    return contexts, profiles


def _weighted_norm(
    values: np.ndarray,
    cell_measures: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    weights = np.asarray(cell_measures, dtype=float)
    weights = weights / np.sum(weights)
    return np.sqrt(
        np.einsum(
            "...ci,...ci,c->...",
            array,
            array,
            weights,
            optimize=True,
        )
    )


def _propagate_components(
    generator: np.ndarray,
    initial_components: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    rates = []
    for initial in initial_components:
        state = np.asarray(
            expm_multiply(
                generator,
                initial.ravel(),
                start=float(times[0]),
                stop=float(times[-1]),
                num=times.size,
                endpoint=True,
            ),
            dtype=float,
        ).reshape(times.size, initial.shape[0], 5)
        rate = (state.reshape(times.size, -1) @ generator.T).reshape(
            state.shape
        )
        states.append(state)
        rates.append(rate)
    return np.asarray(states), np.asarray(rates)


def _cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5
        * (values[:-1] + values[1:])
        * np.diff(times)[:, None, None],
        axis=0,
    )
    return result


def _unique_pair_rows(values: np.ndarray) -> list[tuple[float, int, int]]:
    matrix = np.asarray(values, dtype=float)
    rows = []
    for first in range(5):
        rows.append((float(matrix[first, first]), first, first))
        for second in range(first + 1, 5):
            rows.append(
                (
                    float(
                        matrix[first, second]
                        + matrix[second, first]
                    ),
                    first,
                    second,
                )
            )
    return rows


def _family_interaction_summary(
    by_mesh: dict,
    saved: dict[str, np.ndarray],
) -> dict:
    labels = CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    rows = {}
    cumulative = {}
    initial_composition = {}
    for mesh, data in by_mesh.items():
        cross_work = saved[f"common_N{mesh}_family_cross_work"]
        cumulative[mesh] = _cumulative_trapezoid(
            data["times"],
            cross_work,
        )
        initial = saved[f"common_N{mesh}_family_initial"]
        weights = data["measures"] / np.sum(data["measures"])
        squared = np.einsum(
            "fci,fci,c->f",
            initial,
            initial,
            weights,
            optimize=True,
        )
        initial_composition[f"N{mesh}"] = {
            labels[index]: float(value / np.sum(squared))
            for index, value in enumerate(squared)
        }
        final_rows = _unique_pair_rows(cumulative[mesh][-1])
        absolute_total = max(
            float(sum(abs(value) for value, _, _ in final_rows)),
            np.finfo(float).tiny,
        )
        final_rows.sort(key=lambda item: abs(item[0]), reverse=True)
        rows[f"N{mesh}"] = [
            {
                "pair": [labels[first], labels[second]],
                "signed_cumulative_work": value,
                "absolute_fraction": float(
                    abs(value) / absolute_total
                ),
            }
            for value, first, second in final_rows[:5]
        ]

    difference = cumulative[256] - cumulative[128]
    candidates = []
    for first in range(5):
        history = difference[:, first, first]
        index = int(np.argmax(np.abs(history)))
        candidates.append(
            (
                float(abs(history[index])),
                first,
                first,
                index,
                float(history[-1]),
            )
        )
        for second in range(first + 1, 5):
            history = (
                difference[:, first, second]
                + difference[:, second, first]
            )
            index = int(np.argmax(np.abs(history)))
            candidates.append(
                (
                    float(abs(history[index])),
                    first,
                    second,
                    index,
                    float(history[-1]),
                )
            )
    candidates.sort(reverse=True)
    maximum, first, second, index, final = candidates[0]
    saved["common_N64_family_cross_work_cumulative"] = cumulative[64]
    saved["common_N128_family_cross_work_cumulative"] = cumulative[128]
    saved["common_N256_family_cross_work_cumulative"] = cumulative[256]
    saved["common_N128_N256_family_cross_work_difference"] = difference
    return {
        "initial_family_energy_composition": initial_composition,
        "leading_final_cumulative_cross_work": rows,
        "fine_pair_controlling_cross_work_difference": {
            "pair": [labels[first], labels[second]],
            "maximum_absolute_difference": maximum,
            "controlling_time_seconds": float(
                by_mesh[128]["times"][index]
            ),
            "final_signed_difference": final,
        },
    }


def _pairwise_cross_mesh_error(
    coarse_components: np.ndarray,
    fine_components: np.ndarray,
    fine_measures: np.ndarray,
    coarse_measures: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    restricted = wp10c8v._restrict_pairwise(
        np.moveaxis(fine_components, 0, 1),
        fine_measures,
    )
    restricted = np.moveaxis(restricted, 1, 0)
    errors = restricted - coarse_components
    ledger = causal_pairwise_weighted_gram_ledger(
        errors,
        coarse_measures,
    )
    total_error = np.sum(errors, axis=0)
    total_reference = np.sum(coarse_components, axis=0)
    error_norm = _weighted_norm(total_error, coarse_measures)
    reference_norm = _weighted_norm(total_reference, coarse_measures)
    relative = error_norm / np.maximum(
        reference_norm,
        np.finfo(float).tiny,
    )
    controlling_time = int(np.argmax(relative))
    matrix = np.asarray(ledger.pairwise_gram[controlling_time])
    diagonal = np.diag(matrix)
    symmetric = np.array(matrix, copy=True)
    for first in range(5):
        for second in range(first + 1, 5):
            symmetric[first, second] += matrix[second, first]
            symmetric[second, first] = symmetric[first, second]
    candidates = []
    for first in range(5):
        candidates.append((float(diagonal[first]), first, first))
        for second in range(first + 1, 5):
            candidates.append(
                (float(symmetric[first, second]), first, second)
            )
    candidates.sort(reverse=True)
    contribution, first, second = candidates[0]
    total_squared = max(
        float(ledger.total_squared_norm[controlling_time]),
        np.finfo(float).tiny,
    )
    significance = np.max(
        ledger.component_relative_amplitudes,
        axis=0,
    )
    return (
        {
            "maximum_relative_l2_difference": float(np.max(relative)),
            "controlling_time_index": controlling_time,
            "controlling_pair": [
                CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES[first],
                CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES[second],
            ],
            "controlling_pair_signed_fraction": float(
                contribution / total_squared
            ),
            "component_maximum_relative_amplitudes": {
                family: float(significance[index])
                for index, family in enumerate(
                    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
                )
            },
            "significant_families": [
                family
                for index, family in enumerate(
                    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
                )
                if significance[index]
                >= MINIMUM_ABSOLUTE_FAMILY_SIGNIFICANCE
            ],
            "maximum_pairwise_ledger_closure_defect": (
                ledger.maximum_closure_defect
            ),
        },
        {
            "errors": errors,
            "pairwise_gram": ledger.pairwise_gram,
            "relative_l2_difference": relative,
            "component_relative_amplitudes": (
                ledger.component_relative_amplitudes
            ),
        },
    )


def _common_mode_audit(
    contexts: dict[int, object],
) -> tuple[dict, dict[str, np.ndarray], dict]:
    operators = wp10c8y._load_family_operators()["production"]
    saved = {}
    by_mesh = {}
    with np.load(WP10C8Y_ARRAYS, allow_pickle=False) as source:
        for mesh in wp10c8y.MESHES:
            print(
                f"WP10c9c0c: decomposing common N{mesh}",
                flush=True,
            )
            context = contexts[mesh]
            initial = np.asarray(
                source[f"N{mesh}_normalized_initial_state"],
                dtype=float,
            )
            amplitudes = np.asarray(
                source[f"N{mesh}_common_amplitudes"],
                dtype=float,
            )
            base = np.asarray(
                operators[mesh]["base_primitives"],
                dtype=float,
            )
            generator = wp10c8v._similarity_rescale_generator(
                np.asarray(operators[mesh]["generator"], dtype=float),
                np.asarray(
                    operators[mesh]["primitive_column_scales"],
                    dtype=float,
                ),
                amplitudes,
            )
            projectors, _bases = (
                causal_five_field_characteristic_family_projectors(
                    context,
                    base,
                    amplitudes,
                )
            )
            initial_components = (
                causal_five_field_characteristic_family_decomposition(
                    initial,
                    projectors,
                )
            )
            times = np.asarray(
                source[f"production_N{mesh}_times"],
                dtype=float,
            )
            family_state, family_rate = _propagate_components(
                generator,
                initial_components,
                times,
            )
            reference_state = np.asarray(
                source[f"production_N{mesh}_state"],
                dtype=float,
            )
            reference_rate = np.asarray(
                source[f"production_N{mesh}_rate"],
                dtype=float,
            )
            state_sum = np.sum(family_state, axis=0)
            rate_sum = np.sum(family_rate, axis=0)
            measures = np.asarray(
                source[f"N{mesh}_cell_measures"],
                dtype=float,
            )
            state_ledger = causal_pairwise_weighted_gram_ledger(
                family_state,
                measures,
            )
            cross_work = causal_pairwise_family_cross_work(
                family_state,
                generator,
                measures,
            )
            total_state = np.sum(family_state, axis=0)
            total_rate = np.sum(family_rate, axis=0)
            weights = measures / np.sum(measures)
            total_energy_rate = np.einsum(
                "tci,tci,c->t",
                total_state,
                total_rate,
                weights,
                optimize=True,
            )
            by_mesh[mesh] = {
                "state": family_state,
                "rate": family_rate,
                "measures": measures,
                "radius_rg": np.asarray(
                    source[f"N{mesh}_radius_rg"],
                    dtype=float,
                ),
                "times": times,
                "projector": projectors,
                "state_sum_defect": _relative_maximum_defect(
                    state_sum,
                    reference_state,
                ),
                "rate_sum_defect": _relative_maximum_defect(
                    rate_sum,
                    reference_rate,
                ),
                "initial_decomposition_defect": _relative_maximum_defect(
                    np.sum(initial_components, axis=0),
                    initial,
                ),
                "state_ledger_defect": (
                    state_ledger.maximum_closure_defect
                ),
                "cross_work_defect": _relative_maximum_defect(
                    np.sum(cross_work, axis=(1, 2)),
                    total_energy_rate,
                ),
            }
            saved[f"common_N{mesh}_family_initial"] = initial_components
            saved[f"common_N{mesh}_family_state"] = family_state
            saved[f"common_N{mesh}_family_rate"] = family_rate
            saved[f"common_N{mesh}_state_pairwise_gram"] = (
                state_ledger.pairwise_gram
            )
            saved[f"common_N{mesh}_family_cross_work"] = cross_work

    pair_reports = {}
    pair_arrays = {}
    for coarse, fine in ((64, 128), (128, 256)):
        label = f"N{coarse}_N{fine}"
        pair_reports[label] = {}
        for quantity in ("state", "rate"):
            report, arrays = _pairwise_cross_mesh_error(
                by_mesh[coarse][quantity],
                by_mesh[fine][quantity],
                by_mesh[fine]["measures"],
                by_mesh[coarse]["measures"],
            )
            report["controlling_time_seconds"] = float(
                by_mesh[coarse]["times"][
                    report["controlling_time_index"]
                ]
            )
            pair_reports[label][quantity] = report
            for name, values in arrays.items():
                pair_arrays[
                    f"common_{label}_{quantity}_{name}"
                ] = values
    saved.update(pair_arrays)
    spatial_order = {}
    for quantity in ("state", "rate"):
        coarse = pair_reports["N64_N128"][quantity][
            "maximum_relative_l2_difference"
        ]
        fine = pair_reports["N128_N256"][quantity][
            "maximum_relative_l2_difference"
        ]
        spatial_order[quantity] = _observed_order(coarse, fine)

    method = {
        "maximum_projector_identity_defect": max(
            row["projector"].maximum_identity_closure_defect
            for row in by_mesh.values()
        ),
        "maximum_projector_idempotence_defect": max(
            row["projector"].maximum_idempotence_defect
            for row in by_mesh.values()
        ),
        "maximum_projector_cross_defect": max(
            row["projector"].maximum_cross_projector_defect
            for row in by_mesh.values()
        ),
        "maximum_basis_condition_number": max(
            row["projector"].maximum_basis_condition_number
            for row in by_mesh.values()
        ),
        "maximum_eigenpair_defect": max(
            row["projector"].maximum_eigenpair_defect
            for row in by_mesh.values()
        ),
        "maximum_initial_decomposition_defect": max(
            row["initial_decomposition_defect"]
            for row in by_mesh.values()
        ),
        "maximum_history_state_sum_defect": max(
            row["state_sum_defect"] for row in by_mesh.values()
        ),
        "maximum_history_rate_sum_defect": max(
            row["rate_sum_defect"] for row in by_mesh.values()
        ),
        "maximum_state_pairwise_ledger_defect": max(
            row["state_ledger_defect"] for row in by_mesh.values()
        ),
        "maximum_cross_work_defect": max(
            row["cross_work_defect"] for row in by_mesh.values()
        ),
    }
    method["passed"] = bool(
        method["maximum_projector_identity_defect"]
        <= MAXIMUM_PROJECTOR_DEFECT
        and method["maximum_projector_idempotence_defect"]
        <= MAXIMUM_PROJECTOR_DEFECT
        and method["maximum_projector_cross_defect"]
        <= MAXIMUM_PROJECTOR_DEFECT
        and method["maximum_initial_decomposition_defect"]
        <= MAXIMUM_DECOMPOSITION_DEFECT
        and method["maximum_history_state_sum_defect"]
        <= MAXIMUM_DECOMPOSITION_DEFECT
        and method["maximum_history_rate_sum_defect"]
        <= MAXIMUM_DECOMPOSITION_DEFECT
        and method["maximum_state_pairwise_ledger_defect"]
        <= MAXIMUM_PAIRWISE_LEDGER_DEFECT
        and method["maximum_cross_work_defect"]
        <= MAXIMUM_PAIRWISE_LEDGER_DEFECT
    )
    interaction = _family_interaction_summary(by_mesh, saved)
    return (
        {
            "method_contract": method,
            "pairwise_cross_mesh_error": pair_reports,
            "observed_orders": spatial_order,
            "family_interaction_ledger": interaction,
        },
        saved,
        by_mesh,
    )


def _hybrid_integrated_restriction(
    values: np.ndarray,
    configuration: dict,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    layout = configuration["layout"]
    ratio = int(layout.refinement_ratio)
    parent_face = int(layout.parent_coupling_face_index)
    if array.shape[-1] != layout.n_cells:
        raise ValueError("hybrid integrated restriction is invalid")
    refined = array[..., : layout.n_refined_cells].reshape(
        array.shape[:-1] + (parent_face, ratio)
    ).sum(axis=-1)
    return np.concatenate(
        (refined, array[..., layout.n_refined_cells :]),
        axis=-1,
    )


def _cell_grams(
    configuration: dict,
) -> tuple[np.ndarray, dict]:
    context = configuration["context"]
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    projectors = causal_five_field_shear_energy_projectors(
        context,
        configuration["base_primitives"],
    )
    selected_index = 0
    physical = projectors.primitive_family_energy_grams[selected_index]
    grams = np.einsum(
        "ci,cij,cj,c->cij",
        amplitudes,
        physical,
        amplitudes,
        context.grid.cell_measures,
        optimize=True,
    )
    operators = causal_five_field_scaled_shear_energy_operators(
        projectors,
        amplitudes,
        context.grid.cell_measures,
        family="inward_shear",
    )
    reconstructed = np.zeros_like(operators["selected_energy_gram"])
    for cell, gram in enumerate(grams):
        reconstructed[
            5 * cell : 5 * (cell + 1),
            5 * cell : 5 * (cell + 1),
        ] = gram
    defect = _relative_maximum_defect(
        reconstructed,
        operators["selected_energy_gram"],
    )
    return grams, {
        "selected_cell_gram_reconstruction_defect": defect,
        "maximum_shear_projector_defect": (
            projectors.maximum_shear_projector_defect
        ),
        "maximum_family_projector_defect": (
            projectors.maximum_family_projector_defect
        ),
        "maximum_energy_partition_defect": (
            projectors.maximum_energy_partition_defect
        ),
    }


def _local_pure_shear_audit() -> tuple[dict, dict[str, np.ndarray], dict]:
    print("WP10c9c0c: loading frozen hybrid configurations", flush=True)
    _parent, by_label, labels = wp10c9a._configurations()
    configurations = {
        ratio: by_label[label] for ratio, label in labels.items()
    }
    saved = {}
    ledgers = {}
    method_rows = {}
    with np.load(WP10C9A_ARRAYS, allow_pickle=False) as histories:
        for ratio, configuration in configurations.items():
            print(
                f"WP10c9c0c: mapping local inward work ratio {ratio}",
                flush=True,
            )
            cache_path = WP10C9C0B_CACHE / f"ratio{ratio}.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                blocks = wp10c9c0b._amplitude_scaled_blocks(
                    configuration,
                    {name: np.asarray(cache[name]) for name in cache.files},
                )
            grams, gram_contract = _cell_grams(configuration)
            state = np.asarray(
                histories[f"inward_shear_ratio{ratio}_state_history"],
                dtype=float,
            )
            times = np.asarray(histories["times"], dtype=float)
            ledger = causal_local_quadratic_energy_work_ledger(
                state,
                times,
                grams,
                blocks,
            )
            initial_energy = max(
                float(np.sum(ledger.energy_by_cell[0])),
                np.finfo(float).tiny,
            )
            ledgers[ratio] = {
                "ledger": ledger,
                "initial_energy": initial_energy,
                "configuration": configuration,
            }
            method_rows[ratio] = {
                **gram_contract,
                "maximum_instantaneous_block_closure_defect": (
                    ledger.maximum_instantaneous_block_closure_defect
                ),
                "maximum_integrated_energy_closure_defect": (
                    ledger.maximum_integrated_energy_closure_defect
                ),
            }
            saved[f"local_ratio{ratio}_radius_rg"] = (
                configuration["context"].grid.centers
                / configuration["context"].grid.gravitational_radius
            )
            saved[f"local_ratio{ratio}_selected_energy_by_cell"] = (
                ledger.energy_by_cell / initial_energy
            )
            saved[f"local_ratio{ratio}_selected_rate_by_cell_per_s"] = (
                ledger.rate_by_cell_per_s / initial_energy
            )
            for name, values in ledger.rate_by_block_and_cell_per_s.items():
                saved[
                    f"local_ratio{ratio}_rate_{name}"
                ] = values / initial_energy
                saved[
                    f"local_ratio{ratio}_work_{name}"
                ] = (
                    ledger.cumulative_work_by_block_and_cell[name]
                    / initial_energy
                )

    restricted = {}
    for ratio, row in ledgers.items():
        ledger = row["ledger"]
        initial = row["initial_energy"]
        restricted[ratio] = {
            name: _hybrid_integrated_restriction(
                values / initial,
                row["configuration"],
            )
            for name, values in ledger.cumulative_work_by_block_and_cell.items()
        }
    fine_differences = {
        name: restricted[4][name] - restricted[2][name]
        for name in restricted[2]
    }
    total_difference = np.sum(
        np.asarray(list(fine_differences.values())),
        axis=0,
    )
    total_activity = np.sum(np.abs(total_difference), axis=1)
    controlling_time = int(np.argmax(total_activity))
    block_activity = {
        name: float(np.sum(np.abs(values[controlling_time])))
        for name, values in fine_differences.items()
    }
    ordered = sorted(
        block_activity.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    controlling_block, controlling_activity = ordered[0]
    all_block_activity = max(
        float(sum(block_activity.values())),
        np.finfo(float).tiny,
    )
    controlling_profile = np.abs(
        fine_differences[controlling_block][controlling_time]
    )
    total_profile = np.abs(total_difference[controlling_time])
    radius = (
        configurations[1]["context"].grid.centers
        / configurations[1]["context"].grid.gravitational_radius
    )
    peak_cell = int(np.argmax(controlling_profile))
    cumulative = np.cumsum(controlling_profile)
    if cumulative[-1] > np.finfo(float).tiny:
        lower = int(np.searchsorted(cumulative, 0.10 * cumulative[-1]))
        upper = int(np.searchsorted(cumulative, 0.90 * cumulative[-1]))
    else:
        lower = upper = peak_cell
    report = {
        "method_contract_by_ratio": method_rows,
        "fine_pair": "ratio2_ratio4",
        "controlling_time_index": controlling_time,
        "controlling_time_seconds": float(
            ledgers[2]["ledger"].times_seconds[controlling_time]
        ),
        "controlling_block": controlling_block,
        "controlling_block_absolute_fraction": float(
            controlling_activity / all_block_activity
        ),
        "block_absolute_activity": block_activity,
        "controlling_peak_radius_rg": float(radius[peak_cell]),
        "controlling_80_percent_interval_rg": [
            float(radius[lower]),
            float(radius[upper]),
        ],
        "maximum_instantaneous_block_closure_defect": max(
            row["maximum_instantaneous_block_closure_defect"]
            for row in method_rows.values()
        ),
        "maximum_integrated_energy_closure_defect_at_201_samples": max(
            row["maximum_integrated_energy_closure_defect"]
            for row in method_rows.values()
        ),
    }
    saved["local_fine_total_work_difference"] = total_difference
    saved["local_fine_controlling_block_work_difference"] = (
        fine_differences[controlling_block]
    )
    saved["local_parent_radius_rg"] = radius
    saved["local_controlling_total_profile"] = total_profile
    saved["local_controlling_block_profile"] = controlling_profile
    return report, saved, {
        "configurations": configurations,
        "ledgers": ledgers,
        "fine_differences": fine_differences,
        "controlling_profile": controlling_profile,
        "radius_rg": radius,
    }


def _cross_audit_correlation(
    common: dict,
    local: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    fine = common["pairwise_cross_mesh_error"]["N128_N256"]["rate"]
    pair = fine["controlling_pair"]
    time_index = int(fine["controlling_time_index"])
    # Recover the pairwise spatial contribution from the saved component
    # errors.  A signed pair is converted to absolute cell activity only for
    # localization; the signed global Gram fraction remains the binding
    # family-pair diagnostic.
    family_indices = [
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES.index(name)
        for name in pair
    ]
    common_errors = common["_working_errors"]
    first = common_errors[family_indices[0], time_index]
    second = common_errors[family_indices[1], time_index]
    if family_indices[0] == family_indices[1]:
        common_profile = np.sum(first**2, axis=1)
    else:
        common_profile = np.abs(
            2.0 * np.sum(first * second, axis=1)
        )
    common_radius = common["_working_radius"]
    local_profile = np.asarray(local["controlling_profile"], dtype=float)
    local_radius = np.asarray(local["radius_rg"], dtype=float)
    lower = max(float(common_radius[0]), float(local_radius[0]))
    upper = min(float(common_radius[-1]), float(local_radius[-1]))
    mask = (common_radius >= lower) & (common_radius <= upper)
    interpolated = np.interp(
        common_radius[mask],
        local_radius,
        local_profile,
    )
    first_profile = common_profile[mask]
    denominator = max(
        float(np.linalg.norm(first_profile) * np.linalg.norm(interpolated)),
        np.finfo(float).tiny,
    )
    cosine = float(np.dot(first_profile, interpolated) / denominator)

    def centroid(radius_values, profile):
        weight = np.asarray(profile, dtype=float)
        total = max(float(np.sum(weight)), np.finfo(float).tiny)
        return float(np.sum(radius_values * weight) / total)

    return (
        {
            "common_controlling_pair": pair,
            "common_controlling_time_seconds": (
                fine["controlling_time_seconds"]
            ),
            "pure_shear_controlling_block": local["report"][
                "controlling_block"
            ],
            "pure_shear_controlling_time_seconds": local["report"][
                "controlling_time_seconds"
            ],
            "radial_profile_cosine": cosine,
            "common_profile_centroid_rg": centroid(
                common_radius,
                common_profile,
            ),
            "pure_shear_profile_centroid_rg": centroid(
                local_radius,
                local_profile,
            ),
            "passed_radial_correlation_gate": bool(
                cosine >= MINIMUM_RADIAL_PROFILE_COSINE
            ),
        },
        {
            "correlation_common_radius_rg": common_radius,
            "correlation_common_pair_profile": common_profile,
            "correlation_local_radius_rg": local_radius,
            "correlation_local_block_profile": local_profile,
        },
    )


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C8Y_OUTPUT,
        WP10C8Y_ARRAYS,
        WP10C9A_OUTPUT,
        WP10C9A_ARRAYS,
        WP10C9C0B_OUTPUT,
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError("WP10c9c0c requires WP10c8y/c9a/c9c0b")
    c8y = json.loads(WP10C8Y_OUTPUT.read_text(encoding="utf-8"))
    c9a = json.loads(WP10C9A_OUTPUT.read_text(encoding="utf-8"))
    c9c0b = json.loads(WP10C9C0B_OUTPUT.read_text(encoding="utf-8"))
    if c8y["classification"] != (
        "common_mode_passed_boundary_insensitive_underresolution"
    ):
        raise RuntimeError("WP10c8y classification changed")
    if c9a["classification"] != (
        "characteristic_rate_phase_unresolved_operator_redesign_required"
    ):
        raise RuntimeError("WP10c9a classification changed")
    if c9c0b["classification"] != (
        "selected_shear_energy_defect_is_transport_window_or_family_transfer_sensitive"
    ):
        raise RuntimeError("WP10c9c0b classification changed")

    contexts, _profiles = _common_contexts()
    common_report, common_arrays, common_work = _common_mode_audit(
        contexts
    )
    local_report, local_arrays, local_work = _local_pure_shear_audit()
    common_report["_working_errors"] = common_arrays[
        "common_N128_N256_rate_errors"
    ]
    common_report["_working_radius"] = common_work[128]["radius_rg"]
    local_work["report"] = local_report
    correlation, correlation_arrays = _cross_audit_correlation(
        common_report,
        local_work,
    )
    del common_report["_working_errors"]
    del common_report["_working_radius"]

    fine_rate = common_report["pairwise_cross_mesh_error"][
        "N128_N256"
    ]["rate"]
    method_passed = bool(
        common_report["method_contract"]["passed"]
        and local_report["maximum_instantaneous_block_closure_defect"]
        <= MAXIMUM_LOCAL_BLOCK_CLOSURE_DEFECT
    )
    pair_localized = bool(
        abs(fine_rate["controlling_pair_signed_fraction"])
        >= MINIMUM_CONTROLLING_PAIR_FRACTION
    )
    block_localized = bool(
        local_report["controlling_block_absolute_fraction"]
        >= MINIMUM_CONTROLLING_BLOCK_FRACTION
    )
    mechanism_identified = bool(
        method_passed
        and pair_localized
        and block_localized
        and correlation["passed_radial_correlation_gate"]
    )
    if not method_passed:
        classification = "family_transfer_method_contract_failed"
    elif mechanism_identified:
        classification = (
            "common_mode_failure_localized_to_family_pair_and_"
            "pure_shear_transfer_block"
        )
    else:
        classification = (
            "common_mode_failure_remains_multifamily_or_nonlocal"
        )

    arrays = {}
    arrays.update(common_arrays)
    arrays.update(local_arrays)
    arrays.update(correlation_arrays)
    arrays["times_seconds"] = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        wp10c9a.TIME_SAMPLES,
    )
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "method_contract_passed": method_passed,
        "localized_mechanism_gate_passed": mechanism_identified,
        "common_mode_family_decomposition": common_report,
        "pure_inward_shear_local_work": local_report,
        "cross_audit_correlation": correlation,
        "decision": {
            "controlling_pair_gate_passed": pair_localized,
            "controlling_block_gate_passed": block_localized,
            "radial_correlation_gate_passed": correlation[
                "passed_radial_correlation_gate"
            ],
            "wp10c9c1_path_candidate_authorized": False,
            "production_operator_change_authorized": False,
            "new_truth_trajectory_authorized": False,
            "fixed_q_or_reduction_authorized": False,
        },
        "gates": {
            "maximum_projector_defect": MAXIMUM_PROJECTOR_DEFECT,
            "maximum_decomposition_defect": (
                MAXIMUM_DECOMPOSITION_DEFECT
            ),
            "maximum_pairwise_ledger_defect": (
                MAXIMUM_PAIRWISE_LEDGER_DEFECT
            ),
            "maximum_local_block_closure_defect": (
                MAXIMUM_LOCAL_BLOCK_CLOSURE_DEFECT
            ),
            "minimum_absolute_family_significance": (
                MINIMUM_ABSOLUTE_FAMILY_SIGNIFICANCE
            ),
            "minimum_controlling_pair_fraction": (
                MINIMUM_CONTROLLING_PAIR_FRACTION
            ),
            "minimum_controlling_block_fraction": (
                MINIMUM_CONTROLLING_BLOCK_FRACTION
            ),
            "minimum_radial_profile_cosine": (
                MINIMUM_RADIAL_PROFILE_COSINE
            ),
        },
        "provenance": {
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "core_file": CORE_FILE,
            "core_sha256": _sha256(ROOT / CORE_FILE),
            "wp10c8y_output_sha256": _sha256(WP10C8Y_OUTPUT),
            "wp10c8y_arrays_sha256": _sha256(WP10C8Y_ARRAYS),
            "wp10c9a_output_sha256": _sha256(WP10C9A_OUTPUT),
            "wp10c9a_arrays_sha256": _sha256(WP10C9A_ARRAYS),
            "wp10c9c0b_output_sha256": _sha256(WP10C9C0B_OUTPUT),
            "arrays_path": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
            "array_hashes": {
                name: _array_sha256(values)
                for name, values in arrays.items()
            },
            "python": sys.version,
            "platform": platform.platform(),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    DEFAULT_OUTPUT.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    payload, _arrays = run()
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "method_contract_passed": (
                    payload["method_contract_passed"]
                ),
                "localized_mechanism_gate_passed": (
                    payload["localized_mechanism_gate_passed"]
                ),
                "controlling_pair": payload[
                    "common_mode_family_decomposition"
                ]["pairwise_cross_mesh_error"]["N128_N256"]["rate"][
                    "controlling_pair"
                ],
                "controlling_block": payload[
                    "pure_inward_shear_local_work"
                ]["controlling_block"],
                "radial_profile_cosine": payload[
                    "cross_audit_correlation"
                ]["radial_profile_cosine"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
