"""Run the WP10c9c0b full shear-energy ledger and block attribution.

This package preserves the production operator and the frozen WP10c9a packet
definitions.  It replaces the non-orthogonal branch self-energy diagnostic by
an energy-orthogonal selected-family/complement partition, reconstructs the
cached evolving generator from physical residual and descriptor-rate blocks,
and attributes total/selected shear-energy rates and bounded ablations.
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
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_characteristic_phase_audit_wp10c9a as wp10c9a
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalShearEnergyLedger,
    causal_five_field_generator_block_decomposition,
    causal_five_field_scaled_shear_energy_operators,
    causal_five_field_shear_energy_projectors,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9c0b"
SCHEMA_VERSION = 1
THIS_RUNNER = (
    "scripts/run_causal_inner_shear_energy_ledger_wp10c9c0b.py"
)
CORE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_shear_energy_ledger.py"
)
WP10C9A_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a.json"
)
WP10C9A_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_characteristic_phase_audit_wp10c9a_arrays.npz"
)
WP10C9C0_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_shear_root_cause_audit_wp10c9c0.json"
)
WP10C9C0_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_shear_root_cause_audit_wp10c9c0_arrays.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_shear_energy_ledger_wp10c9c0b.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_shear_energy_ledger_wp10c9c0b_arrays.npz"
)
CACHE_ROOT = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_shear_energy_ledger_wp10c9c0b"
)

FAMILIES = ("inward_shear", "outward_shear")
FINITE_DIFFERENCE_STEP = 2.0e-6
STORAGE_RATE_DERIVATIVE_STEP = 2.0e-6
STORAGE_DIFFERENCE_STEP = 1.0e-4
STORAGE_QUADRATURE_ORDER = 4
STORAGE_DIRECTIONAL_STEP = 1.0e-3
COMPONENT_STENCIL_RADIUS = 4

MAXIMUM_PROJECTOR_DEFECT = 2.0e-6
MAXIMUM_ENERGY_PARTITION_DEFECT = 1.0e-10
MAXIMUM_BASE_RESIDUAL_RECONSTRUCTION_DEFECT = 1.0e-10
MAXIMUM_STATIONARY_JACOBIAN_RECONSTRUCTION_DEFECT = 1.0e-8
MAXIMUM_GENERATOR_RECONSTRUCTION_DEFECT = 1.0e-11
MAXIMUM_MASS_SOLVE_DEFECT = 1.0e-10
MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION = 1.0e-7
MAXIMUM_INSTANTANEOUS_LEDGER_DEFECT = 1.0e-10
MAXIMUM_FINE_INTEGRATED_LEDGER_DEFECT = 1.0e-6
MINIMUM_INTEGRATED_LEDGER_ORDER = 1.8
MINIMUM_SPATIAL_ENERGY_ORDER = 0.75
REFINED_LEDGER_TIME_SAMPLES = 801
MINIMUM_BLOCK_RATE_SIGNIFICANCE = 1.0e-4

ABLATION_GROUPS = {
    "numerical_dissipation": ("transport_rusanov",),
    "stress_principal_and_relaxation": (
        "source_stress_relaxation",
    ),
    "responsive_height": (
        "source_vertical_work",
        "descriptor_vertical_rate_dependence",
    ),
    "geometry_and_cooling": (
        "source_perfect_fluid_geometry",
        "source_stress_geometry",
        "source_radiative_cooling",
        "source_stream",
    ),
    "boundaries": (
        "transport_inner_boundary",
        "transport_outer_boundary",
    ),
    "mapped_descriptor": ("descriptor_mapped_rate_dependence",),
}

CUMULATIVE_GROUP_ORDER = (
    "conservative_transport",
    "boundaries",
    "numerical_dissipation",
    "stress_principal_and_relaxation",
    "responsive_height",
    "geometry_and_cooling",
    "mapped_descriptor",
    "unattributed",
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
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8))
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


def _observed_order(coarse: float, fine: float) -> float | None:
    if not (
        np.isfinite(coarse)
        and np.isfinite(fine)
        and coarse > 0.0
        and fine > 0.0
    ):
        return None
    return float(np.log2(coarse / fine))


def _energy_pair(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.maximum(
        np.asarray(first, dtype=float),
        np.finfo(float).tiny,
    )
    second_values = np.maximum(
        np.asarray(second, dtype=float),
        np.finfo(float).tiny,
    )
    return float(
        np.max(np.abs(np.log(second_values / first_values)))
    )


def _absolute_history_pair(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(first, dtype=float)
                - np.asarray(second, dtype=float)
            )
        )
    )


def _cache_paths(ratio: int) -> tuple[Path, Path]:
    stem = f"ratio{int(ratio)}"
    return CACHE_ROOT / f"{stem}.json", CACHE_ROOT / f"{stem}.npz"


def _decomposition_contract(
    ratio: int,
    configuration: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "ratio": int(ratio),
        "configuration_label": configuration["label"],
        "core_file": CORE_FILE,
        "core_sha256": _sha256(ROOT / CORE_FILE),
        "base_primitives_sha256": _array_sha256(
            configuration["base_primitives"]
        ),
        "primitive_column_scales_sha256": _array_sha256(
            configuration["operator"]["primitive_column_scales"]
        ),
        "full_generator_sha256": _array_sha256(
            configuration["operator"]["generator"]
        ),
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "storage_rate_derivative_step": (
            STORAGE_RATE_DERIVATIVE_STEP
        ),
        "storage_difference_step": STORAGE_DIFFERENCE_STEP,
        "storage_quadrature_order": STORAGE_QUADRATURE_ORDER,
        "storage_directional_step": STORAGE_DIRECTIONAL_STEP,
        "component_stencil_radius": COMPONENT_STENCIL_RADIUS,
    }


def _build_or_load_decomposition(
    ratio: int,
    configuration: dict,
    *,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _cache_paths(ratio)
    contract = _decomposition_contract(ratio, configuration)
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            all(payload.get(key) == value for key, value in contract.items())
            and payload.get("arrays_sha256") == _sha256(arrays_path)
        ):
            with np.load(arrays_path, allow_pickle=False) as source:
                arrays = {
                    name: np.asarray(source[name])
                    for name in source.files
                }
            return payload, arrays

    print(
        f"WP10c9c0b: building exact ratio-{ratio} generator blocks",
        flush=True,
    )
    started = time.perf_counter()
    decomposition = causal_five_field_generator_block_decomposition(
        configuration["context"],
        configuration["base_primitives"],
        primitive_column_scales=configuration["operator"][
            "primitive_column_scales"
        ],
        full_generator_per_s=configuration["operator"]["generator"],
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        storage_rate_derivative_step=STORAGE_RATE_DERIVATIVE_STEP,
        storage_difference_step=STORAGE_DIFFERENCE_STEP,
        storage_quadrature_order=STORAGE_QUADRATURE_ORDER,
        storage_directional_step=STORAGE_DIRECTIONAL_STEP,
        stencil_radius=COMPONENT_STENCIL_RADIUS,
    )
    arrays = {
        "descriptor_matrix": decomposition.descriptor_matrix,
        "scaled_primitive_rate_per_s": (
            decomposition.scaled_primitive_rate_per_s
        ),
        "physical_primitive_rate_per_s": (
            decomposition.physical_primitive_rate_per_s
        ),
    }
    for name, matrix in (
        decomposition.generator_blocks_per_s.items()
    ):
        arrays[f"block_{name}"] = matrix
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **contract,
        "component_names": decomposition.component_names,
        "reduced_pattern_colors": decomposition.reduced_pattern_colors,
        "maximum_base_residual_reconstruction_defect": (
            decomposition.maximum_base_residual_reconstruction_defect
        ),
        "maximum_stationary_jacobian_reconstruction_defect": (
            decomposition.maximum_stationary_jacobian_reconstruction_defect
        ),
        "maximum_generator_reconstruction_defect_before_remainder": (
            decomposition
            .maximum_generator_reconstruction_defect_before_remainder
        ),
        "maximum_generator_reconstruction_defect_after_remainder": (
            decomposition
            .maximum_generator_reconstruction_defect_after_remainder
        ),
        "maximum_mass_solve_relative_defect": (
            decomposition.maximum_mass_solve_relative_defect
        ),
        "residual_unattributed_relative_frobenius_norm": (
            decomposition.residual_unattributed_relative_frobenius_norm
        ),
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "wall_seconds": time.perf_counter() - started,
    }
    json_path.write_text(
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


def _amplitude_scaled_blocks(
    configuration: dict,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    source_scales = np.asarray(
        configuration["operator"]["primitive_column_scales"],
        dtype=float,
    )
    target_scales = np.asarray(
        configuration["amplitudes"],
        dtype=float,
    )
    return {
        name.removeprefix("block_"): (
            wp10c8v._similarity_rescale_generator(
                matrix,
                source_scales,
                target_scales,
            )
        )
        for name, matrix in arrays.items()
        if name.startswith("block_")
    }


def _quadratic_energy_history(
    history: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    state = np.asarray(history, dtype=float).reshape(
        history.shape[0],
        -1,
    )
    return 0.5 * np.einsum(
        "ti,ij,tj->t",
        state,
        np.asarray(gram, dtype=float),
        state,
        optimize=True,
    )


def _comoving_energy_history(
    configuration: dict,
    state_history: np.ndarray,
    projectors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = configuration["context"]
    amplitudes = np.asarray(configuration["amplitudes"], dtype=float)
    physical = np.asarray(state_history, dtype=float) * amplitudes[None]
    density = 0.5 * np.einsum(
        "tci,cij,tcj->tc",
        physical,
        projectors.primitive_energy_grams,
        physical,
        optimize=True,
    )
    weighted = density * context.grid.cell_measures[None, :]
    total = np.sum(weighted, axis=1)
    log_radius = np.log(
        context.grid.centers / context.grid.gravitational_radius
    )
    centroid = np.sum(weighted * log_radius[None, :], axis=1) / np.maximum(
        total,
        np.finfo(float).tiny,
    )
    half_width = 0.5 * np.log(
        wp10c9a.SUPPORT_OUTER_RG / wp10c9a.SUPPORT_INNER_RG
    )
    comoving = np.asarray(
        [
            np.sum(
                weighted[index][
                    np.abs(log_radius - center) <= half_width
                ]
            )
            for index, center in enumerate(centroid)
        ],
        dtype=float,
    )
    return total, comoving, centroid


def _preflight_energy_histories(
    configurations: dict[int, dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    reports = {}
    arrays = {}
    with np.load(WP10C9A_ARRAYS, allow_pickle=False) as source:
        for family in FAMILIES:
            by_ratio = {}
            for ratio, configuration in configurations.items():
                history = np.asarray(
                    source[f"{family}_ratio{ratio}_state_history"],
                    dtype=float,
                )
                context = configuration["context"]
                projectors = causal_five_field_shear_energy_projectors(
                    context,
                    configuration["base_primitives"],
                )
                full_operators = (
                    causal_five_field_scaled_shear_energy_operators(
                        projectors,
                        configuration["amplitudes"],
                        context.grid.cell_measures,
                        family=family,
                    )
                )
                support = (
                    context.grid.centers
                    >= (
                        wp10c9a.SUPPORT_INNER_RG
                        * context.grid.gravitational_radius
                    )
                ) & (
                    context.grid.centers
                    <= (
                        wp10c9a.SUPPORT_OUTER_RG
                        * context.grid.gravitational_radius
                    )
                )
                fixed_operators = (
                    causal_five_field_scaled_shear_energy_operators(
                        projectors,
                        configuration["amplitudes"],
                        context.grid.cell_measures,
                        family=family,
                        cell_mask=support,
                    )
                )
                active = (
                    context.grid.centers
                    <= (
                        configuration["active_outer_rg"]
                        * context.grid.gravitational_radius
                        * (1.0 + 2.0e-14)
                    )
                )
                active_operators = (
                    causal_five_field_scaled_shear_energy_operators(
                        projectors,
                        configuration["amplitudes"],
                        context.grid.cell_measures,
                        family=family,
                        cell_mask=active,
                    )
                )
                total, comoving, centroid = _comoving_energy_history(
                    configuration,
                    history,
                    projectors,
                )
                comoving_fraction = comoving / np.maximum(
                    total,
                    np.finfo(float).tiny,
                )
                values = {
                    "full_total": _quadratic_energy_history(
                        history,
                        full_operators["total_energy_gram"],
                    ),
                    "full_selected": _quadratic_energy_history(
                        history,
                        full_operators["selected_energy_gram"],
                    ),
                    "full_complement": _quadratic_energy_history(
                        history,
                        full_operators["complement_energy_gram"],
                    ),
                    "fixed_total": _quadratic_energy_history(
                        history,
                        fixed_operators["total_energy_gram"],
                    ),
                    "fixed_selected": _quadratic_energy_history(
                        history,
                        fixed_operators["selected_energy_gram"],
                    ),
                    "active_total": _quadratic_energy_history(
                        history,
                        active_operators["total_energy_gram"],
                    ),
                    "active_selected": _quadratic_energy_history(
                        history,
                        active_operators["selected_energy_gram"],
                    ),
                    "comoving_total": comoving,
                    "comoving_centroid_log_rg": centroid,
                }
                for name in tuple(values):
                    if name == "comoving_centroid_log_rg":
                        continue
                    values[name] = values[name] / max(
                        float(values[name][0]),
                        np.finfo(float).tiny,
                    )
                by_ratio[ratio] = {
                    "values": values,
                    "comoving_fraction": comoving_fraction,
                    "projector_contract": {
                        "minimum_positive_energy_eigenvalue": (
                            projectors.minimum_positive_energy_eigenvalue
                        ),
                        "maximum_shear_projector_defect": (
                            projectors.maximum_shear_projector_defect
                        ),
                        "maximum_family_projector_defect": (
                            projectors.maximum_family_projector_defect
                        ),
                        "maximum_partition_defect": (
                            projectors.maximum_partition_defect
                        ),
                        "maximum_energy_self_adjoint_defect": (
                            projectors
                            .maximum_energy_self_adjoint_defect
                        ),
                        "maximum_energy_partition_defect": (
                            projectors.maximum_energy_partition_defect
                        ),
                    },
                }
                for name, value in values.items():
                    arrays[
                        f"{family}_ratio{ratio}_preflight_{name}"
                    ] = value
                arrays[
                    f"{family}_ratio{ratio}_preflight_"
                    "comoving_fraction"
                ] = comoving_fraction
            metrics = {}
            names = tuple(
                name
                for name in by_ratio[1]["values"]
                if name != "comoving_centroid_log_rg"
            )
            for name in names:
                coarse = _energy_pair(
                    by_ratio[1]["values"][name],
                    by_ratio[2]["values"][name],
                )
                fine = _energy_pair(
                    by_ratio[2]["values"][name],
                    by_ratio[4]["values"][name],
                )
                metrics[name] = {
                    "ratio1_ratio2_defect": coarse,
                    "ratio2_ratio4_defect": fine,
                    "observed_order": _observed_order(coarse, fine),
                }
            reports[family] = {
                "by_ratio": {
                    ratio: {
                        "projector_contract": row[
                            "projector_contract"
                        ],
                        "minimum_comoving_fraction": float(
                            np.min(row["comoving_fraction"])
                        ),
                    }
                    for ratio, row in by_ratio.items()
                },
                "spatial_metrics": metrics,
            }
    return reports, arrays


def _group_blocks(
    blocks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = {
        name
        for names in ABLATION_GROUPS.values()
        for name in names
        if name not in blocks
    }
    if missing:
        raise RuntimeError(
            f"WP10c9c0b generator schema is missing {sorted(missing)}"
        )
    groups = {
        name: np.sum(
            np.asarray([blocks[item] for item in names]),
            axis=0,
        )
        for name, names in ABLATION_GROUPS.items()
    }
    groups["conservative_transport"] = (
        blocks["transport_central_perfect"]
        + blocks["transport_central_stress"]
    )
    groups["unattributed"] = blocks["residual_unattributed"]
    reconstructed = np.sum(
        np.asarray(list(groups.values())),
        axis=0,
    )
    boundary_and_source_names = {
        name
        for names in ABLATION_GROUPS.values()
        for name in names
    }
    boundary_and_source_names.update(
        {
            "transport_central_perfect",
            "transport_central_stress",
            "residual_unattributed",
        }
    )
    extra = set(blocks) - boundary_and_source_names
    if extra:
        groups["other_exact_components"] = np.sum(
            np.asarray([blocks[name] for name in sorted(extra)]),
            axis=0,
        )
        reconstructed += groups["other_exact_components"]
    full = np.sum(np.asarray(list(blocks.values())), axis=0)
    if not np.allclose(reconstructed, full, rtol=0.0, atol=1.0e-11):
        raise RuntimeError("WP10c9c0b grouped blocks do not close")
    return groups


def _cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    time_values = np.asarray(times, dtype=float)
    rate_values = np.asarray(values, dtype=float)
    result = np.zeros_like(rate_values)
    result[1:] = np.cumsum(
        0.5
        * (rate_values[:-1] + rate_values[1:])
        * np.diff(time_values)
    )
    return result


def _local_block_action(
    state: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply one exactly block-diagonal five-field operator."""

    values = np.asarray(state, dtype=float)
    operator = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] % 5:
        raise ValueError("local block action state is invalid")
    n_cells = values.shape[1] // 5
    if operator.shape != (5 * n_cells, 5 * n_cells):
        raise ValueError("local block action operator is invalid")
    local = np.asarray(
        [
            operator[
                5 * cell : 5 * (cell + 1),
                5 * cell : 5 * (cell + 1),
            ]
            for cell in range(n_cells)
        ],
        dtype=float,
    )
    reconstructed = np.zeros_like(operator)
    for cell, block in enumerate(local):
        reconstructed[
            5 * cell : 5 * (cell + 1),
            5 * cell : 5 * (cell + 1),
        ] = block
    scale = max(
        float(np.max(np.abs(operator))),
        np.finfo(float).tiny,
    )
    if float(np.max(np.abs(operator - reconstructed)) / scale) > 1.0e-12:
        raise ValueError("declared local operator is not block diagonal")
    shaped = values.reshape(values.shape[0], n_cells, 5)
    return np.einsum(
        "cij,tcj->tci",
        local,
        shaped,
        optimize=True,
    ).reshape(values.shape)


def _prepare_energy_actions(
    full_generator: np.ndarray,
    blocks: dict[str, np.ndarray],
    state: np.ndarray,
    operators: dict[str, np.ndarray],
) -> dict:
    """Cache generator actions shared by all energy-window ledgers."""

    values = np.asarray(state, dtype=float)
    full = np.asarray(full_generator, dtype=float)
    if values.ndim == 3 and values.shape[2] == 5:
        values = values.reshape(values.shape[0], -1)
    if (
        values.ndim != 2
        or full.shape != (values.shape[1], values.shape[1])
    ):
        raise ValueError("energy action inputs are invalid")

    projectors = {
        "selected": np.asarray(
            operators["selected_projector"],
            dtype=float,
        ),
        "orthogonal_shear_complement": np.asarray(
            operators["complement_projector"],
            dtype=float,
        ),
        "non_shear": np.asarray(
            operators["non_shear_projector"],
            dtype=float,
        ),
    }
    source_states = {
        name: _local_block_action(values, projector)
        for name, projector in projectors.items()
    }
    full_action = values @ full.T
    block_actions = {
        name: values @ np.asarray(matrix, dtype=float).T
        for name, matrix in blocks.items()
    }
    source_actions = {
        name: source @ full.T
        for name, source in source_states.items()
    }
    preserving_action = np.zeros_like(values)
    for name, source_action in source_actions.items():
        preserving_action += _local_block_action(
            source_action,
            projectors[name],
        )
    return {
        "state": values,
        "full": full_action,
        "blocks": block_actions,
        "sources": source_actions,
        "preserving": preserving_action,
        "transfer": full_action - preserving_action,
    }


def _fast_shear_energy_ledger(
    full_generator: np.ndarray,
    blocks: dict[str, np.ndarray],
    state: np.ndarray,
    times: np.ndarray,
    operators: dict[str, np.ndarray],
    *,
    family: str,
    action_cache: dict,
) -> CausalShearEnergyLedger:
    """Evaluate the public ledger contract with shared dense actions."""

    values = np.asarray(state, dtype=float)
    if values.ndim == 3 and values.shape[2] == 5:
        values = values.reshape(values.shape[0], -1)
    if not np.array_equal(values, action_cache["state"]):
        raise ValueError("energy action cache belongs to another history")
    full = np.asarray(full_generator, dtype=float)
    reconstructed = np.sum(
        np.asarray(list(blocks.values()), dtype=float),
        axis=0,
    )
    generator_scale = max(
        float(np.max(np.abs(full))),
        np.finfo(float).tiny,
    )
    if (
        full.shape != reconstructed.shape
        or float(np.max(np.abs(full - reconstructed)) / generator_scale)
        > 1.0e-10
    ):
        raise ValueError("fast ledger generator blocks do not close")

    metric_actions = {
        name: _local_block_action(
            values,
            np.asarray(operators[f"{name}_energy_gram"], dtype=float),
        )
        for name in ("total", "selected", "complement")
    }

    def energy(name: str) -> np.ndarray:
        return 0.5 * np.einsum(
            "ti,ti->t",
            metric_actions[name],
            values,
            optimize=True,
        )

    def rate(name: str, derivative: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ti,ti->t",
            metric_actions[name],
            derivative,
            optimize=True,
        )

    total_energy = energy("total")
    selected_energy = energy("selected")
    complement_energy = energy("complement")
    total_rate = rate("total", action_cache["full"])
    selected_rate = rate("selected", action_cache["full"])
    complement_rate = rate("complement", action_cache["full"])
    total_by_block = {
        name: rate("total", derivative)
        for name, derivative in action_cache["blocks"].items()
    }
    selected_by_block = {
        name: rate("selected", derivative)
        for name, derivative in action_cache["blocks"].items()
    }
    complement_by_block = {
        name: rate("complement", derivative)
        for name, derivative in action_cache["blocks"].items()
    }
    total_by_source = {
        name: rate("total", derivative)
        for name, derivative in action_cache["sources"].items()
    }
    selected_by_source = {
        name: rate("selected", derivative)
        for name, derivative in action_cache["sources"].items()
    }
    preserving_total = rate("total", action_cache["preserving"])
    transfer_total = rate("total", action_cache["transfer"])
    preserving_selected = rate(
        "selected",
        action_cache["preserving"],
    )
    transfer_selected = rate("selected", action_cache["transfer"])
    time_values = np.asarray(times, dtype=float)
    cumulative_total = _cumulative_trapezoid(
        time_values,
        total_rate,
    )
    cumulative_selected = _cumulative_trapezoid(
        time_values,
        selected_rate,
    )
    cumulative_complement = _cumulative_trapezoid(
        time_values,
        complement_rate,
    )

    def relative_defect(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        scale = max(
            float(np.max(np.abs(first))),
            float(np.max(np.abs(second))),
            np.finfo(float).tiny,
        )
        return float(np.max(np.abs(first - second)) / scale)

    def integrated_defect(
        values_: np.ndarray,
        integral: np.ndarray,
    ) -> float:
        change = values_ - float(values_[0])
        scale = max(
            float(np.max(np.abs(change))),
            float(np.max(np.abs(integral))),
            abs(float(values_[0])),
            np.finfo(float).tiny,
        )
        return float(np.max(np.abs(change - integral)) / scale)

    return CausalShearEnergyLedger(
        family=str(family),
        times_seconds=np.array(time_values, copy=True),
        total_energy=total_energy,
        selected_energy=selected_energy,
        complement_energy=complement_energy,
        total_energy_rate_per_s=total_rate,
        selected_energy_rate_per_s=selected_rate,
        complement_energy_rate_per_s=complement_rate,
        total_rate_by_block_per_s=total_by_block,
        selected_rate_by_block_per_s=selected_by_block,
        complement_rate_by_block_per_s=complement_by_block,
        selected_rate_by_source_partition_per_s=selected_by_source,
        total_rate_by_source_partition_per_s=total_by_source,
        preserving_total_rate_per_s=preserving_total,
        transfer_total_rate_per_s=transfer_total,
        preserving_selected_rate_per_s=preserving_selected,
        transfer_selected_rate_per_s=transfer_selected,
        cumulative_total_rate_integral=cumulative_total,
        cumulative_selected_rate_integral=cumulative_selected,
        cumulative_complement_rate_integral=cumulative_complement,
        maximum_instantaneous_energy_partition_defect=relative_defect(
            total_energy,
            selected_energy + complement_energy,
        ),
        maximum_instantaneous_block_ledger_defect=relative_defect(
            total_rate,
            np.sum(
                np.asarray(list(total_by_block.values()), dtype=float),
                axis=0,
            ),
        ),
        maximum_instantaneous_source_partition_defect=relative_defect(
            selected_rate,
            np.sum(
                np.asarray(list(selected_by_source.values()), dtype=float),
                axis=0,
            ),
        ),
        maximum_integrated_total_ledger_defect=integrated_defect(
            total_energy,
            cumulative_total,
        ),
        maximum_integrated_selected_ledger_defect=integrated_defect(
            selected_energy,
            cumulative_selected,
        ),
        maximum_integrated_complement_ledger_defect=integrated_defect(
            complement_energy,
            cumulative_complement,
        ),
    )


def _integrated_defect(
    times: np.ndarray,
    energy: np.ndarray,
    rate: np.ndarray,
    *,
    stride: int,
) -> float:
    indices = np.arange(0, times.size, int(stride), dtype=int)
    if indices[-1] != times.size - 1:
        indices = np.append(indices, times.size - 1)
    selected_times = np.asarray(times, dtype=float)[indices]
    selected_energy = np.asarray(energy, dtype=float)[indices]
    selected_rate = np.asarray(rate, dtype=float)[indices]
    integral = _cumulative_trapezoid(selected_times, selected_rate)
    change = selected_energy - float(selected_energy[0])
    scale = max(
        float(np.max(np.abs(change))),
        float(np.max(np.abs(integral))),
        abs(float(selected_energy[0])),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(change - integral)) / scale)


def _temporal_ledger_convergence(
    times: np.ndarray,
    energy: np.ndarray,
    rate: np.ndarray,
) -> dict:
    defects = {
        "51_samples": _integrated_defect(
            times,
            energy,
            rate,
            stride=4,
        ),
        "101_samples": _integrated_defect(
            times,
            energy,
            rate,
            stride=2,
        ),
        "201_samples": _integrated_defect(
            times,
            energy,
            rate,
            stride=1,
        ),
    }
    return {
        "defects": defects,
        "orders": {
            "51_to_101": _observed_order(
                defects["51_samples"],
                defects["101_samples"],
            ),
            "101_to_201": _observed_order(
                defects["101_samples"],
                defects["201_samples"],
            ),
        },
    }


def _ledger_report(
    ledger,
    *,
    prefix: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    initial_total = max(
        float(ledger.total_energy[0]),
        np.finfo(float).tiny,
    )
    initial_selected = max(
        float(ledger.selected_energy[0]),
        np.finfo(float).tiny,
    )
    initial_complement = max(
        float(ledger.complement_energy[0]),
        np.finfo(float).tiny,
    )
    arrays = {
        f"{prefix}_total_energy_normalized": (
            ledger.total_energy / initial_total
        ),
        f"{prefix}_selected_energy_normalized": (
            ledger.selected_energy / initial_selected
        ),
        f"{prefix}_complement_energy_normalized": (
            ledger.complement_energy / initial_complement
        ),
        f"{prefix}_total_rate_over_initial_energy_per_s": (
            ledger.total_energy_rate_per_s / initial_total
        ),
        f"{prefix}_selected_rate_over_initial_energy_per_s": (
            ledger.selected_energy_rate_per_s / initial_selected
        ),
        f"{prefix}_complement_rate_over_initial_energy_per_s": (
            ledger.complement_energy_rate_per_s / initial_complement
        ),
        f"{prefix}_preserving_total_rate_over_initial_energy_per_s": (
            ledger.preserving_total_rate_per_s / initial_total
        ),
        f"{prefix}_transfer_total_rate_over_initial_energy_per_s": (
            ledger.transfer_total_rate_per_s / initial_total
        ),
        f"{prefix}_preserving_selected_rate_over_initial_energy_per_s": (
            ledger.preserving_selected_rate_per_s / initial_selected
        ),
        f"{prefix}_transfer_selected_rate_over_initial_energy_per_s": (
            ledger.transfer_selected_rate_per_s / initial_selected
        ),
    }
    total_block_work = {}
    selected_block_work = {}
    maximum_total_block_rate = {}
    maximum_selected_block_rate = {}
    for name, values in ledger.total_rate_by_block_per_s.items():
        normalized = np.asarray(values, dtype=float) / initial_total
        arrays[f"{prefix}_total_block_rate_{name}"] = normalized
        integral = _cumulative_trapezoid(
            ledger.times_seconds,
            normalized,
        )
        arrays[f"{prefix}_total_block_work_{name}"] = integral
        total_block_work[name] = float(integral[-1])
        maximum_total_block_rate[name] = float(
            np.max(np.abs(normalized))
        )
    for name, values in ledger.selected_rate_by_block_per_s.items():
        normalized = np.asarray(values, dtype=float) / initial_selected
        arrays[f"{prefix}_selected_block_rate_{name}"] = normalized
        integral = _cumulative_trapezoid(
            ledger.times_seconds,
            normalized,
        )
        arrays[f"{prefix}_selected_block_work_{name}"] = integral
        selected_block_work[name] = float(integral[-1])
        maximum_selected_block_rate[name] = float(
            np.max(np.abs(normalized))
        )
    source_partition_work = {}
    for name, values in (
        ledger.selected_rate_by_source_partition_per_s.items()
    ):
        normalized = np.asarray(values, dtype=float) / initial_selected
        arrays[
            f"{prefix}_selected_source_partition_rate_{name}"
        ] = normalized
        integral = _cumulative_trapezoid(
            ledger.times_seconds,
            normalized,
        )
        arrays[
            f"{prefix}_selected_source_partition_work_{name}"
        ] = integral
        source_partition_work[name] = float(integral[-1])
    transfer_selected = _cumulative_trapezoid(
        ledger.times_seconds,
        ledger.transfer_selected_rate_per_s / initial_selected,
    )
    preserving_selected = _cumulative_trapezoid(
        ledger.times_seconds,
        ledger.preserving_selected_rate_per_s / initial_selected,
    )
    arrays[f"{prefix}_transfer_selected_work"] = transfer_selected
    arrays[f"{prefix}_preserving_selected_work"] = preserving_selected
    temporal = {
        "total": _temporal_ledger_convergence(
            ledger.times_seconds,
            ledger.total_energy,
            ledger.total_energy_rate_per_s,
        ),
        "selected": _temporal_ledger_convergence(
            ledger.times_seconds,
            ledger.selected_energy,
            ledger.selected_energy_rate_per_s,
        ),
        "complement": _temporal_ledger_convergence(
            ledger.times_seconds,
            ledger.complement_energy,
            ledger.complement_energy_rate_per_s,
        ),
    }
    return {
        "initial_total_energy": initial_total,
        "initial_selected_energy": initial_selected,
        "initial_complement_energy": initial_complement,
        "maximum_instantaneous_energy_partition_defect": (
            ledger.maximum_instantaneous_energy_partition_defect
        ),
        "maximum_instantaneous_block_ledger_defect": (
            ledger.maximum_instantaneous_block_ledger_defect
        ),
        "maximum_instantaneous_source_partition_defect": (
            ledger.maximum_instantaneous_source_partition_defect
        ),
        "integrated_defects_reported_by_core": {
            "total": ledger.maximum_integrated_total_ledger_defect,
            "selected": ledger.maximum_integrated_selected_ledger_defect,
            "complement": (
                ledger.maximum_integrated_complement_ledger_defect
            ),
        },
        "temporal_ledger_convergence": temporal,
        "final_total_block_work_over_initial_energy": total_block_work,
        "final_selected_block_work_over_initial_energy": (
            selected_block_work
        ),
        "maximum_total_block_rate_over_initial_energy_per_s": (
            maximum_total_block_rate
        ),
        "maximum_selected_block_rate_over_initial_energy_per_s": (
            maximum_selected_block_rate
        ),
        "final_selected_source_partition_work_over_initial_energy": (
            source_partition_work
        ),
        "final_selected_transfer_work_over_initial_energy": float(
            transfer_selected[-1]
        ),
        "final_selected_preserving_work_over_initial_energy": float(
            preserving_selected[-1]
        ),
    }, arrays


def _full_energy_ledgers(
    configurations: dict[int, dict],
    decompositions: dict[int, dict],
) -> tuple[dict, dict[str, np.ndarray], dict]:
    reports = {}
    arrays = {}
    ledger_data = {}
    times = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        wp10c9a.TIME_SAMPLES,
    )
    with np.load(WP10C9A_ARRAYS, allow_pickle=False) as source:
        for family in FAMILIES:
            family_reports = {}
            family_data = {}
            for ratio, configuration in configurations.items():
                history = np.asarray(
                    source[f"{family}_ratio{ratio}_state_history"],
                    dtype=float,
                )
                projectors = causal_five_field_shear_energy_projectors(
                    configuration["context"],
                    configuration["base_primitives"],
                )
                blocks = decompositions[ratio]["grouped_blocks"]
                full_operators = (
                    causal_five_field_scaled_shear_energy_operators(
                        projectors,
                        configuration["amplitudes"],
                        configuration["context"].grid.cell_measures,
                        family=family,
                    )
                )
                support = (
                    configuration["context"].grid.centers
                    >= (
                        wp10c9a.SUPPORT_INNER_RG
                        * configuration["context"].grid.gravitational_radius
                    )
                ) & (
                    configuration["context"].grid.centers
                    <= (
                        wp10c9a.SUPPORT_OUTER_RG
                        * configuration["context"].grid.gravitational_radius
                    )
                )
                fixed_operators = (
                    causal_five_field_scaled_shear_energy_operators(
                        projectors,
                        configuration["amplitudes"],
                        configuration["context"].grid.cell_measures,
                        family=family,
                        cell_mask=support,
                    )
                )
                action_cache = _prepare_energy_actions(
                    configuration["generator"],
                    blocks,
                    history,
                    full_operators,
                )
                full_ledger = _fast_shear_energy_ledger(
                    configuration["generator"],
                    blocks,
                    history,
                    times,
                    full_operators,
                    family=family,
                    action_cache=action_cache,
                )
                fixed_ledger = _fast_shear_energy_ledger(
                    configuration["generator"],
                    blocks,
                    history,
                    times,
                    fixed_operators,
                    family=family,
                    action_cache=action_cache,
                )
                full_report, full_arrays = _ledger_report(
                    full_ledger,
                    prefix=f"{family}_ratio{ratio}_full",
                )
                fixed_report, fixed_arrays = _ledger_report(
                    fixed_ledger,
                    prefix=f"{family}_ratio{ratio}_fixed",
                )
                arrays.update(full_arrays)
                arrays.update(fixed_arrays)
                family_reports[ratio] = {
                    "full_domain": full_report,
                    "fixed_packet_window": fixed_report,
                }
                family_data[ratio] = {
                    "full": full_ledger,
                    "fixed": fixed_ledger,
                }
            reports[family] = family_reports
            ledger_data[family] = family_data
    return reports, arrays, ledger_data


def _ledger_spatial_summary(ledger_data: dict) -> dict:
    reports = {}
    for family, by_ratio in ledger_data.items():
        metrics = {}
        energy_fields = {
            "full_total": lambda row: row["full"].total_energy,
            "full_selected": lambda row: row["full"].selected_energy,
            "full_complement": lambda row: row["full"].complement_energy,
            "fixed_total": lambda row: row["fixed"].total_energy,
            "fixed_selected": lambda row: row["fixed"].selected_energy,
        }
        for name, accessor in energy_fields.items():
            normalized = {
                ratio: accessor(row)
                / max(float(accessor(row)[0]), np.finfo(float).tiny)
                for ratio, row in by_ratio.items()
            }
            coarse = _energy_pair(normalized[1], normalized[2])
            fine = _energy_pair(normalized[2], normalized[4])
            metrics[name] = {
                "ratio1_ratio2_defect": coarse,
                "ratio2_ratio4_defect": fine,
                "observed_order": _observed_order(coarse, fine),
            }

        full_selected_rates = {
            ratio: (
                row["full"].selected_energy_rate_per_s
                / max(
                    float(row["full"].selected_energy[0]),
                    np.finfo(float).tiny,
                )
            )
            for ratio, row in by_ratio.items()
        }
        common_rate_scale = max(
            float(np.max(np.abs(values)))
            for values in full_selected_rates.values()
        )
        common_rate_scale = max(
            common_rate_scale,
            np.finfo(float).tiny,
        )
        block_metrics = {}
        names = tuple(
            by_ratio[1]["full"].selected_rate_by_block_per_s
        )
        for name in names:
            normalized = {
                ratio: (
                    row["full"].selected_rate_by_block_per_s[name]
                    / max(
                        float(row["full"].selected_energy[0]),
                        np.finfo(float).tiny,
                    )
                )
                for ratio, row in by_ratio.items()
            }
            coarse = _absolute_history_pair(
                normalized[1],
                normalized[2],
            )
            fine = _absolute_history_pair(
                normalized[2],
                normalized[4],
            )
            amplitude = max(
                float(np.max(np.abs(values)))
                for values in normalized.values()
            )
            block_metrics[name] = {
                "ratio1_ratio2_absolute_defect_per_s": coarse,
                "ratio2_ratio4_absolute_defect_per_s": fine,
                "ratio1_ratio2_defect_over_full_rate_scale": (
                    coarse / common_rate_scale
                ),
                "ratio2_ratio4_defect_over_full_rate_scale": (
                    fine / common_rate_scale
                ),
                "maximum_amplitude_over_full_rate_scale": (
                    amplitude / common_rate_scale
                ),
                "absolutely_significant": bool(
                    amplitude / common_rate_scale
                    >= MINIMUM_BLOCK_RATE_SIGNIFICANCE
                ),
                "observed_order": _observed_order(coarse, fine),
            }
        significant_names = [
            name
            for name, row in block_metrics.items()
            if row["absolutely_significant"]
        ]
        controlling = max(
            significant_names or list(block_metrics),
            key=lambda name: block_metrics[name][
                "ratio2_ratio4_defect_over_full_rate_scale"
            ],
        )
        partition_metrics = {}
        for name, accessor in {
            "preserving": (
                lambda row: row[
                    "full"
                ].preserving_selected_rate_per_s
            ),
            "transfer": (
                lambda row: row["full"].transfer_selected_rate_per_s
            ),
        }.items():
            normalized = {
                ratio: (
                    accessor(row)
                    / max(
                        float(row["full"].selected_energy[0]),
                        np.finfo(float).tiny,
                    )
                )
                for ratio, row in by_ratio.items()
            }
            cumulative = {
                ratio: _cumulative_trapezoid(
                    row["full"].times_seconds,
                    normalized[ratio],
                )
                for ratio, row in by_ratio.items()
            }
            rate_coarse = _absolute_history_pair(
                normalized[1],
                normalized[2],
            )
            rate_fine = _absolute_history_pair(
                normalized[2],
                normalized[4],
            )
            work_coarse = _absolute_history_pair(
                cumulative[1],
                cumulative[2],
            )
            work_fine = _absolute_history_pair(
                cumulative[2],
                cumulative[4],
            )
            partition_metrics[name] = {
                "rate": {
                    "ratio1_ratio2_absolute_defect_per_s": (
                        rate_coarse
                    ),
                    "ratio2_ratio4_absolute_defect_per_s": rate_fine,
                    "ratio1_ratio2_defect_over_full_rate_scale": (
                        rate_coarse / common_rate_scale
                    ),
                    "ratio2_ratio4_defect_over_full_rate_scale": (
                        rate_fine / common_rate_scale
                    ),
                    "observed_order": _observed_order(
                        rate_coarse,
                        rate_fine,
                    ),
                },
                "cumulative_work": {
                    "ratio1_ratio2_defect": work_coarse,
                    "ratio2_ratio4_defect": work_fine,
                    "observed_order": _observed_order(
                        work_coarse,
                        work_fine,
                    ),
                    "final_by_ratio": {
                        ratio: float(values[-1])
                        for ratio, values in cumulative.items()
                    },
                },
            }
        reports[family] = {
            "energy_metrics": metrics,
            "selected_rate_block_metrics": block_metrics,
            "selected_rate_common_scale_per_s": common_rate_scale,
            "minimum_block_rate_significance": (
                MINIMUM_BLOCK_RATE_SIGNIFICANCE
            ),
            "controlling_selected_rate_block": controlling,
            "selected_preserving_and_transfer_metrics": (
                partition_metrics
            ),
        }
    return reports


def _refined_integrated_ledger_contract(
    configurations: dict[int, dict],
    ledger_reports: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Refine only the quadrature sampling behind integrated ledgers."""

    times = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        REFINED_LEDGER_TIME_SAMPLES,
    )
    reports = {}
    arrays = {"refined_ledger_times_seconds": times}
    for family in FAMILIES:
        family_reports = {}
        for ratio, configuration in configurations.items():
            initial, _bases, _projection = wp10c9a._project_packet(
                configuration,
                family,
            )
            state = np.asarray(
                expm_multiply(
                    configuration["generator"],
                    np.asarray(initial, dtype=float).ravel(),
                    start=0.0,
                    stop=wp10c9a.TARGET_SECONDS,
                    num=REFINED_LEDGER_TIME_SAMPLES,
                    endpoint=True,
                ),
                dtype=float,
            )
            derivative = state @ configuration["generator"].T
            projectors = causal_five_field_shear_energy_projectors(
                configuration["context"],
                configuration["base_primitives"],
            )
            support = (
                configuration["context"].grid.centers
                >= (
                    wp10c9a.SUPPORT_INNER_RG
                    * configuration["context"].grid.gravitational_radius
                )
            ) & (
                configuration["context"].grid.centers
                <= (
                    wp10c9a.SUPPORT_OUTER_RG
                    * configuration["context"].grid.gravitational_radius
                )
            )
            domains = {
                "full_domain": None,
                "fixed_packet_window": support,
            }
            ratio_reports = {}
            for domain, mask in domains.items():
                operators = (
                    causal_five_field_scaled_shear_energy_operators(
                        projectors,
                        configuration["amplitudes"],
                        configuration["context"].grid.cell_measures,
                        family=family,
                        cell_mask=mask,
                    )
                )
                field_reports = {}
                for field in ("total", "selected", "complement"):
                    metric_state = _local_block_action(
                        state,
                        operators[f"{field}_energy_gram"],
                    )
                    energy = 0.5 * np.einsum(
                        "ti,ti->t",
                        metric_state,
                        state,
                        optimize=True,
                    )
                    rate = np.einsum(
                        "ti,ti->t",
                        metric_state,
                        derivative,
                        optimize=True,
                    )
                    defect = _integrated_defect(
                        times,
                        energy,
                        rate,
                        stride=1,
                    )
                    prior = ledger_reports[family][ratio][domain][
                        "temporal_ledger_convergence"
                    ][field]["defects"]["201_samples"]
                    field_reports[field] = {
                        "201_sample_defect": prior,
                        "801_sample_defect": defect,
                        "observed_order_over_factor_four": (
                            None
                            if prior <= 0.0 or defect <= 0.0
                            else float(np.log(prior / defect) / np.log(4.0))
                        ),
                    }
                    prefix = f"{family}_ratio{ratio}_{domain}_{field}"
                    initial_energy = max(
                        float(energy[0]),
                        np.finfo(float).tiny,
                    )
                    arrays[f"{prefix}_refined_energy_normalized"] = (
                        energy / initial_energy
                    )
                    arrays[
                        f"{prefix}_refined_rate_over_initial_energy_per_s"
                    ] = rate / initial_energy
                ratio_reports[domain] = field_reports
            family_reports[ratio] = ratio_reports
        reports[family] = family_reports
    maximum_defect = max(
        field["801_sample_defect"]
        for family in reports.values()
        for ratio in family.values()
        for domain in ratio.values()
        for field in domain.values()
    )
    minimum_order = min(
        field["observed_order_over_factor_four"]
        for family in reports.values()
        for ratio in family.values()
        for domain in ratio.values()
        for field in domain.values()
        if field["observed_order_over_factor_four"] is not None
    )
    return {
        "time_samples": REFINED_LEDGER_TIME_SAMPLES,
        "by_family": reports,
        "maximum_801_sample_defect": maximum_defect,
        "minimum_observed_order_over_factor_four": minimum_order,
        "passed": bool(
            maximum_defect <= MAXIMUM_FINE_INTEGRATED_LEDGER_DEFECT
            and minimum_order >= MINIMUM_INTEGRATED_LEDGER_ORDER
        ),
    }, arrays


def _propagate_energy_variant(
    generator: np.ndarray,
    initial: np.ndarray,
    operators: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    times = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        wp10c9a.TIME_SAMPLES,
    )
    state = np.asarray(
        expm_multiply(
            generator,
            np.asarray(initial, dtype=float).ravel(),
            start=0.0,
            stop=wp10c9a.TARGET_SECONDS,
            num=wp10c9a.TIME_SAMPLES,
            endpoint=True,
        ),
        dtype=float,
    )
    total = _quadratic_energy_history(
        state,
        operators["total_energy_gram"],
    )
    selected = _quadratic_energy_history(
        state,
        operators["selected_energy_gram"],
    )
    return {
        "times": times,
        "total": total / max(float(total[0]), np.finfo(float).tiny),
        "selected": selected
        / max(float(selected[0]), np.finfo(float).tiny),
    }


def _ablation_spatial_metric(
    by_ratio: dict[int, dict[str, np.ndarray]],
) -> dict:
    result = {}
    for field in ("total", "selected"):
        coarse = _energy_pair(
            by_ratio[1][field],
            by_ratio[2][field],
        )
        fine = _energy_pair(
            by_ratio[2][field],
            by_ratio[4][field],
        )
        result[field] = {
            "ratio1_ratio2_defect": coarse,
            "ratio2_ratio4_defect": fine,
            "observed_order": _observed_order(coarse, fine),
        }
    return result


def _dynamic_ablation_audit(
    configurations: dict[int, dict],
    decompositions: dict[int, dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run predeclared one-at-a-time and cumulative inward-shear controls."""

    initials = {}
    operators = {}
    for ratio, configuration in configurations.items():
        initial, _bases, _projection = wp10c9a._project_packet(
            configuration,
            "inward_shear",
        )
        initials[ratio] = initial
        projectors = causal_five_field_shear_energy_projectors(
            configuration["context"],
            configuration["base_primitives"],
        )
        operators[ratio] = (
            causal_five_field_scaled_shear_energy_operators(
                projectors,
                configuration["amplitudes"],
                configuration["context"].grid.cell_measures,
                family="inward_shear",
            )
        )

    group_names = tuple(CUMULATIVE_GROUP_ORDER)
    reports = {}
    arrays = {}

    def audit_variant(
        variant: str,
        generators: dict[int, np.ndarray],
    ) -> None:
        print(f"WP10c9c0b: propagating {variant}", flush=True)
        by_ratio = {}
        failed = None
        for ratio, generator in generators.items():
            try:
                values = _propagate_energy_variant(
                    generator,
                    initials[ratio],
                    operators[ratio],
                )
            except Exception as exc:  # pragma: no cover - diagnostic stop
                failed = f"{type(exc).__name__}: {exc}"
                break
            if any(np.any(~np.isfinite(value)) for value in values.values()):
                failed = "non-finite ablation history"
                break
            by_ratio[ratio] = values
            for name, value in values.items():
                arrays[
                    f"ablation_{variant}_ratio{ratio}_{name}"
                ] = value
        reports[variant] = {
            "failed": failed,
            "spatial_metrics": (
                None
                if failed is not None
                else _ablation_spatial_metric(by_ratio)
            ),
            "passed_selected_energy_gate": bool(
                failed is None
                and _ablation_spatial_metric(by_ratio)["selected"][
                    "observed_order"
                ]
                is not None
                and _ablation_spatial_metric(by_ratio)["selected"][
                    "observed_order"
                ]
                >= MINIMUM_SPATIAL_ENERGY_ORDER
            ),
        }

    for name in group_names:
        audit_variant(
            f"without_{name}",
            {
                ratio: (
                    configuration["generator"]
                    - decompositions[ratio]["grouped_blocks"][name]
                )
                for ratio, configuration in configurations.items()
            },
        )

    cumulative = {
        ratio: np.zeros_like(configuration["generator"])
        for ratio, configuration in configurations.items()
    }
    for name in group_names:
        for ratio in cumulative:
            cumulative[ratio] += decompositions[ratio][
                "grouped_blocks"
            ][name]
        audit_variant(
            f"cumulative_through_{name}",
            cumulative,
        )
    return reports, arrays


def _ledger_method_contract(
    preflight: dict,
    decomposition: dict,
    ledgers: dict,
    refined_integrated: dict,
) -> dict:
    projector_rows = [
        row
        for family in preflight.values()
        for row in family["by_ratio"].values()
    ]
    maximum_projector_defect = max(
        max(
            row["projector_contract"][
                "maximum_shear_projector_defect"
            ],
            row["projector_contract"][
                "maximum_family_projector_defect"
            ],
            row["projector_contract"]["maximum_partition_defect"],
            row["projector_contract"][
                "maximum_energy_self_adjoint_defect"
            ],
        )
        for row in projector_rows
    )
    maximum_energy_partition_defect = max(
        row["projector_contract"][
            "maximum_energy_partition_defect"
        ]
        for row in projector_rows
    )

    maximum_base_residual_defect = max(
        row["maximum_base_residual_reconstruction_defect"]
        for row in decomposition.values()
    )
    maximum_stationary_jacobian_defect = max(
        row["maximum_stationary_jacobian_reconstruction_defect"]
        for row in decomposition.values()
    )
    maximum_generator_defect = max(
        row["maximum_generator_reconstruction_defect_after_remainder"]
        for row in decomposition.values()
    )
    maximum_mass_solve_defect = max(
        row["maximum_mass_solve_relative_defect"]
        for row in decomposition.values()
    )
    maximum_unattributed_fraction = max(
        row["residual_unattributed_relative_frobenius_norm"]
        for row in decomposition.values()
    )

    ledger_rows = [
        domain
        for family in ledgers.values()
        for ratio in family.values()
        for domain in ratio.values()
    ]
    maximum_instantaneous_partition_defect = max(
        row["maximum_instantaneous_energy_partition_defect"]
        for row in ledger_rows
    )
    maximum_instantaneous_block_defect = max(
        row["maximum_instantaneous_block_ledger_defect"]
        for row in ledger_rows
    )
    maximum_instantaneous_source_defect = max(
        row["maximum_instantaneous_source_partition_defect"]
        for row in ledger_rows
    )
    fine_integrated_defects = []
    fine_integrated_orders = []
    for row in ledger_rows:
        for temporal in row["temporal_ledger_convergence"].values():
            defect = temporal["defects"]["201_samples"]
            order = temporal["orders"]["101_to_201"]
            fine_integrated_defects.append(defect)
            if (
                order is not None
                and temporal["defects"]["101_samples"] > 1.0e-14
            ):
                fine_integrated_orders.append(order)
    maximum_201_sample_integrated_defect = max(
        fine_integrated_defects
    )
    minimum_201_sample_integrated_order = min(
        fine_integrated_orders
    )
    maximum_fine_integrated_defect = refined_integrated[
        "maximum_801_sample_defect"
    ]
    minimum_integrated_order = refined_integrated[
        "minimum_observed_order_over_factor_four"
    ]

    checks = {
        "projector_contract": (
            maximum_projector_defect <= MAXIMUM_PROJECTOR_DEFECT
        ),
        "energy_partition_contract": (
            maximum_energy_partition_defect
            <= MAXIMUM_ENERGY_PARTITION_DEFECT
        ),
        "base_residual_reconstruction": (
            maximum_base_residual_defect
            <= MAXIMUM_BASE_RESIDUAL_RECONSTRUCTION_DEFECT
        ),
        "stationary_jacobian_reconstruction": (
            maximum_stationary_jacobian_defect
            <= MAXIMUM_STATIONARY_JACOBIAN_RECONSTRUCTION_DEFECT
        ),
        "generator_reconstruction": (
            maximum_generator_defect
            <= MAXIMUM_GENERATOR_RECONSTRUCTION_DEFECT
        ),
        "mass_solve": (
            maximum_mass_solve_defect <= MAXIMUM_MASS_SOLVE_DEFECT
        ),
        "unattributed_generator": (
            maximum_unattributed_fraction
            <= MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION
        ),
        "instantaneous_energy_partition": (
            maximum_instantaneous_partition_defect
            <= MAXIMUM_INSTANTANEOUS_LEDGER_DEFECT
        ),
        "instantaneous_block_ledger": (
            maximum_instantaneous_block_defect
            <= MAXIMUM_INSTANTANEOUS_LEDGER_DEFECT
        ),
        "instantaneous_source_partition": (
            maximum_instantaneous_source_defect
            <= MAXIMUM_INSTANTANEOUS_LEDGER_DEFECT
        ),
        "integrated_ledger_fine_defect": (
            maximum_fine_integrated_defect
            <= MAXIMUM_FINE_INTEGRATED_LEDGER_DEFECT
        ),
        "integrated_ledger_order": (
            minimum_integrated_order >= MINIMUM_INTEGRATED_LEDGER_ORDER
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "measurements": {
            "maximum_projector_defect": maximum_projector_defect,
            "maximum_energy_partition_defect": (
                maximum_energy_partition_defect
            ),
            "maximum_base_residual_reconstruction_defect": (
                maximum_base_residual_defect
            ),
            "maximum_stationary_jacobian_reconstruction_defect": (
                maximum_stationary_jacobian_defect
            ),
            "maximum_generator_reconstruction_defect": (
                maximum_generator_defect
            ),
            "maximum_mass_solve_defect": maximum_mass_solve_defect,
            "maximum_unattributed_generator_fraction": (
                maximum_unattributed_fraction
            ),
            "maximum_instantaneous_energy_partition_defect": (
                maximum_instantaneous_partition_defect
            ),
            "maximum_instantaneous_block_ledger_defect": (
                maximum_instantaneous_block_defect
            ),
            "maximum_instantaneous_source_partition_defect": (
                maximum_instantaneous_source_defect
            ),
            "maximum_fine_integrated_ledger_defect": (
                maximum_fine_integrated_defect
            ),
            "minimum_integrated_ledger_order": (
                minimum_integrated_order
            ),
            "maximum_201_sample_integrated_ledger_defect": (
                maximum_201_sample_integrated_defect
            ),
            "minimum_201_sample_integrated_ledger_order": (
                minimum_201_sample_integrated_order
            ),
        },
    }


def _scientific_decision(
    method_contract: dict,
    spatial: dict,
    ablations: dict,
) -> dict:
    inward = spatial["inward_shear"]
    energy = inward["energy_metrics"]
    total_order = energy["full_total"]["observed_order"]
    selected_order = energy["full_selected"]["observed_order"]
    fixed_selected_order = energy["fixed_selected"]["observed_order"]
    complement_order = energy["full_complement"]["observed_order"]
    partition_metrics = inward[
        "selected_preserving_and_transfer_metrics"
    ]
    preserving_rate_order = partition_metrics["preserving"]["rate"][
        "observed_order"
    ]
    preserving_work_order = partition_metrics["preserving"][
        "cumulative_work"
    ]["observed_order"]
    transfer_order = partition_metrics["transfer"]["rate"][
        "observed_order"
    ]
    transfer_work_order = partition_metrics["transfer"][
        "cumulative_work"
    ]["observed_order"]
    baseline_passed = bool(
        selected_order is not None
        and selected_order >= MINIMUM_SPATIAL_ENERGY_ORDER
    )
    improving = []
    for name, row in ablations.items():
        if not name.startswith("without_") or row["failed"] is not None:
            continue
        order = row["spatial_metrics"]["selected"]["observed_order"]
        group = name.removeprefix("without_")
        block_metric = inward["selected_rate_block_metrics"].get(group)
        significant = bool(
            block_metric is not None
            and block_metric["absolutely_significant"]
        )
        if order is not None and selected_order is not None:
            improving.append(
                {
                    "variant": name,
                    "observed_order": order,
                    "order_improvement": order - selected_order,
                    "block_absolutely_significant": significant,
                    "passed_gate": (
                        significant
                        and order >= MINIMUM_SPATIAL_ENERGY_ORDER
                    ),
                }
            )
    improving.sort(
        key=lambda row: row["order_improvement"],
        reverse=True,
    )
    passing_removals = [
        row for row in improving if row["passed_gate"]
    ]

    if not method_contract["passed"]:
        classification = "shear_energy_ledger_method_contract_failed"
        next_goal = (
            "repair the ledger method contract before interpreting "
            "operator attribution"
        )
    elif baseline_passed:
        classification = (
            "physical_selected_shear_energy_converges_"
            "legacy_damping_metric_reclassified"
        )
        next_goal = (
            "repeat the held-out shear packets with physical energy "
            "as the binding damping observable"
        )
    elif (
        fixed_selected_order is not None
        and fixed_selected_order >= MINIMUM_SPATIAL_ENERGY_ORDER
    ):
        classification = (
            "selected_shear_energy_defect_is_transport_window_"
            "or_family_transfer_sensitive"
        )
        next_goal = (
            "derive a face-resolved local shear-energy flux and decompose "
            "the frozen common mode into pairwise family transfer before "
            "changing an operator"
        )
    elif len(passing_removals) == 1:
        group = passing_removals[0]["variant"].removeprefix("without_")
        classification = (
            "selected_shear_energy_defect_is_uniquely_"
            f"sensitive_to_{group}"
        )
        next_goal = (
            f"audit the continuum and discrete {group} block before "
            "changing the nonlinear operator"
        )
    else:
        classification = (
            "selected_shear_energy_defect_persists_without_"
            "unique_operator_block"
        )
        next_goal = (
            "construct a face- and radius-resolved energy balance before "
            "selecting another operator redesign"
        )
    return {
        "classification": classification,
        "method_contract_passed": method_contract["passed"],
        "full_total_energy_order": total_order,
        "full_selected_energy_order": selected_order,
        "fixed_window_selected_energy_order": fixed_selected_order,
        "full_complement_energy_order": complement_order,
        "selected_preserving_rate_order": preserving_rate_order,
        "selected_preserving_cumulative_work_order": (
            preserving_work_order
        ),
        "selected_transfer_rate_order": transfer_order,
        "selected_transfer_cumulative_work_order": transfer_work_order,
        "controlling_selected_rate_block": inward[
            "controlling_selected_rate_block"
        ],
        "one_at_a_time_ablation_ranking": improving,
        "passing_one_at_a_time_removals": passing_removals,
        "unique_operator_block_identified": bool(
            len(passing_removals) == 1
        ),
        "multiple_interacting_blocks_implicated": bool(
            len(passing_removals) > 1
        ),
        "legacy_basis_normalization_is_sufficient_explanation": False,
        "path_inconsistency_proved": False,
        "wp10c9c1_path_candidate_authorized": False,
        "fixed_q_reduction_authorized": False,
        "recommended_next_goal": next_goal,
    }


def run(*, force: bool, preflight_only: bool) -> tuple[dict, dict]:
    started = time.perf_counter()
    required = (
        WP10C9A_OUTPUT,
        WP10C9A_ARRAYS,
        WP10C9C0_OUTPUT,
        WP10C9C0_ARRAYS,
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError("WP10c9c0b requires WP10c9a/c9c0 evidence")
    c9a = json.loads(WP10C9A_OUTPUT.read_text(encoding="utf-8"))
    c9c0 = json.loads(WP10C9C0_OUTPUT.read_text(encoding="utf-8"))
    if c9a["classification"] != (
        "characteristic_rate_phase_unresolved_operator_redesign_required"
    ):
        raise RuntimeError("WP10c9a classification changed")
    if c9c0["classification"] != (
        "path_inconsistency_not_proved_selected_shear_damping_persists"
    ):
        raise RuntimeError("WP10c9c0 classification changed")
    if c9c0["root_cause_decision"][
        "wp10c9c1_path_candidate_authorized"
    ]:
        raise RuntimeError("WP10c9c1 unexpectedly became authorized")

    print("WP10c9c0b: loading frozen packet configurations", flush=True)
    parent, by_label, labels = wp10c9a._configurations()
    configurations = {
        ratio: by_label[label] for ratio, label in labels.items()
    }
    arrays = {}
    print("WP10c9c0b: running energy-orthogonal preflight", flush=True)
    preflight, preflight_arrays = _preflight_energy_histories(
        configurations
    )
    arrays.update(preflight_arrays)
    if preflight_only:
        return {
            "classification": "preflight_only",
            "work_package": WORK_PACKAGE,
            "preflight": preflight,
        }, arrays

    decomposition_reports = {}
    decompositions = {}
    for ratio, configuration in configurations.items():
        report, cached = _build_or_load_decomposition(
            ratio,
            configuration,
            force=force,
        )
        decomposition_reports[ratio] = report
        blocks = _amplitude_scaled_blocks(configuration, cached)
        decompositions[ratio] = {
            "grouped_blocks": _group_blocks(blocks),
        }
        del blocks, cached

    print("WP10c9c0b: evaluating exact energy ledgers", flush=True)
    ledger_reports, ledger_arrays, ledger_data = (
        _full_energy_ledgers(configurations, decompositions)
    )
    arrays.update(ledger_arrays)
    spatial = _ledger_spatial_summary(ledger_data)
    print(
        "WP10c9c0b: refining integrated-ledger quadrature",
        flush=True,
    )
    refined_integrated, refined_arrays = (
        _refined_integrated_ledger_contract(
            configurations,
            ledger_reports,
        )
    )
    arrays.update(refined_arrays)
    print("WP10c9c0b: running bounded generator ablations", flush=True)
    ablations, ablation_arrays = _dynamic_ablation_audit(
        configurations,
        decompositions,
    )
    arrays.update(ablation_arrays)
    arrays["times_seconds"] = np.linspace(
        0.0,
        wp10c9a.TARGET_SECONDS,
        wp10c9a.TIME_SAMPLES,
    )
    method_contract = _ledger_method_contract(
        preflight,
        decomposition_reports,
        ledger_reports,
        refined_integrated,
    )
    decision = _scientific_decision(
        method_contract,
        spatial,
        ablations,
    )
    classification = decision["classification"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "passed": bool(
            method_contract["passed"]
            and (
                decision["full_selected_energy_order"] is not None
                and decision["full_selected_energy_order"]
                >= MINIMUM_SPATIAL_ENERGY_ORDER
            )
        ),
        "audit_completed": True,
        "production_changed": False,
        "wp10c9c1_path_candidate_authorized": False,
        "scope": {
            "families": FAMILIES,
            "ratios": tuple(configurations),
            "target_seconds": wp10c9a.TARGET_SECONDS,
            "time_samples": wp10c9a.TIME_SAMPLES,
            "production_operator_unchanged": True,
            "nonlinear_path_flux_implemented": False,
            "stress_source_block_semantics": (
                "combined resolved-shear principal contribution and "
                "local Maxwell-Cattaneo relaxation"
            ),
            "family_transfer_semantics": (
                "energy-orthogonal cellwise selected/complement transfer; "
                "not a parallel-transported connection law"
            ),
        },
        "gates": {
            "maximum_projector_defect": MAXIMUM_PROJECTOR_DEFECT,
            "maximum_energy_partition_defect": (
                MAXIMUM_ENERGY_PARTITION_DEFECT
            ),
            "maximum_base_residual_reconstruction_defect": (
                MAXIMUM_BASE_RESIDUAL_RECONSTRUCTION_DEFECT
            ),
            "maximum_stationary_jacobian_reconstruction_defect": (
                MAXIMUM_STATIONARY_JACOBIAN_RECONSTRUCTION_DEFECT
            ),
            "maximum_generator_reconstruction_defect": (
                MAXIMUM_GENERATOR_RECONSTRUCTION_DEFECT
            ),
            "maximum_mass_solve_defect": MAXIMUM_MASS_SOLVE_DEFECT,
            "maximum_unattributed_generator_fraction": (
                MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION
            ),
            "maximum_instantaneous_ledger_defect": (
                MAXIMUM_INSTANTANEOUS_LEDGER_DEFECT
            ),
            "maximum_fine_integrated_ledger_defect": (
                MAXIMUM_FINE_INTEGRATED_LEDGER_DEFECT
            ),
            "minimum_integrated_ledger_order": (
                MINIMUM_INTEGRATED_LEDGER_ORDER
            ),
            "minimum_spatial_energy_order": (
                MINIMUM_SPATIAL_ENERGY_ORDER
            ),
            "minimum_block_rate_significance": (
                MINIMUM_BLOCK_RATE_SIGNIFICANCE
            ),
        },
        "energy_orthogonal_preflight": preflight,
        "generator_block_decomposition": decomposition_reports,
        "energy_ledgers": ledger_reports,
        "refined_integrated_ledger_contract": refined_integrated,
        "spatial_energy_and_rate_attribution": spatial,
        "bounded_dynamic_ablations": ablations,
        "method_contract": method_contract,
        "scientific_decision": decision,
        "frozen_evidence": {
            "wp10c9a_json_sha256": _sha256(WP10C9A_OUTPUT),
            "wp10c9a_arrays_sha256": _sha256(WP10C9A_ARRAYS),
            "wp10c9c0_json_sha256": _sha256(WP10C9C0_OUTPUT),
            "wp10c9c0_arrays_sha256": _sha256(WP10C9C0_ARRAYS),
        },
        "provenance": {
            "runner": THIS_RUNNER,
            "core_file": CORE_FILE,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "core_sha256": _sha256(ROOT / CORE_FILE),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    return payload, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    payload, arrays = run(
        force=arguments.force,
        preflight_only=arguments.preflight_only,
    )
    if arguments.preflight_only:
        print(
            json.dumps(
                {
                    "classification": payload["classification"],
                    "inward_spatial_metrics": payload["preflight"][
                        "inward_shear"
                    ]["spatial_metrics"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload["machine_evidence"] = {
        "arrays_path": _relative(DEFAULT_ARRAYS),
        "arrays_sha256": _sha256(DEFAULT_ARRAYS),
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
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "passed": payload["passed"],
                "output": str(DEFAULT_OUTPUT),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
