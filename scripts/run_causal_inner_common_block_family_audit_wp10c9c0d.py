"""Run the WP10c9c0d block-by-family common-mode attribution audit.

This package is production-neutral and trajectory-neutral.  It combines the
already committed WP10c8y common-mode family histories with exact physical
generator-block decompositions on the same N64/N128/N256 local grids.  The
binding object is the directed four-index ledger

    W[block, receiver, source] = <x_receiver, G_block x_source>.

The companion receiver-projected vector actions are used to attribute the
cross-mesh rate defect without reducing it to a basis-dependent scalar work
alone.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y
import run_causal_inner_family_transfer_audit_wp10c9c0c as wp10c9c0c
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    causal_block_family_receiver_action,
    causal_block_family_transfer_ledger,
    causal_five_field_characteristic_family_projectors,
    causal_five_field_generator_block_decomposition,
)


BASE_COMMIT = "90f82c238e802abe22aa15b42f62b7d929048a60"
WORK_PACKAGE = "WP10c9c0d"
SCHEMA_VERSION = 2
THIS_RUNNER = (
    "scripts/run_causal_inner_common_block_family_audit_wp10c9c0d.py"
)
CORE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_family_transfer.py"
)
DECOMPOSITION_CORE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_shear_energy_ledger.py"
)
WP10C8Y_OUTPUT = (
    ROOT / "outputs/tables/causal_inner_common_mode_audit_wp10c8y.json"
)
WP10C8Y_ARRAYS = (
    ROOT
    / "outputs/tables/causal_inner_common_mode_audit_wp10c8y_arrays.npz"
)
WP10C9C0C_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_family_transfer_audit_wp10c9c0c.json"
)
WP10C9C0C_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_family_transfer_audit_wp10c9c0c_arrays.npz"
)
CACHE_ROOT = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_common_block_family_wp10c9c0d"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_block_family_audit_wp10c9c0d.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_block_family_audit_wp10c9c0d_arrays.npz"
)

MAXIMUM_BASE_RESIDUAL_RECONSTRUCTION_DEFECT = 1.0e-10
MAXIMUM_STATIONARY_JACOBIAN_RECONSTRUCTION_DEFECT = 2.0e-10
MAXIMUM_FINAL_GENERATOR_RECONSTRUCTION_DEFECT = 1.0e-12
MAXIMUM_MASS_SOLVE_DEFECT = 1.0e-10
MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION = 1.0e-7
MAXIMUM_RATE_ACTION_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_CROSS_WORK_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_PARENT_CROSS_WORK_REPRODUCTION_DEFECT = 1.0e-10
MINIMUM_ABSOLUTE_SIGNIFICANCE = 1.0e-4
MINIMUM_COMPONENT_ACTIVITY_FRACTION = 0.50
MINIMUM_COMPONENT_PERSISTENCE_FRACTION = 0.50
MINIMUM_RADIAL_PROFILE_COSINE = 0.90
MAXIMUM_RADIAL_CENTROID_RELATIVE_DEFECT = 0.10
MINIMUM_CONTRACTION_DEFECT = 1.0e-4


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


def _cache_paths(mesh: int) -> tuple[Path, Path]:
    stem = f"N{int(mesh)}"
    return CACHE_ROOT / f"{stem}.json", CACHE_ROOT / f"{stem}.npz"


def _decomposition_contract(
    mesh: int,
    base_primitives: np.ndarray,
    native_scales: np.ndarray,
    target_scales: np.ndarray,
    base_physical_rate_per_s: np.ndarray,
    native_generator: np.ndarray,
    target_generator: np.ndarray,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "mesh": int(mesh),
        "decomposition_core_file": DECOMPOSITION_CORE_FILE,
        "decomposition_core_sha256": _sha256(
            ROOT / DECOMPOSITION_CORE_FILE
        ),
        "base_primitives_sha256": _array_sha256(base_primitives),
        "native_scales_sha256": _array_sha256(native_scales),
        "common_amplitudes_sha256": _array_sha256(target_scales),
        "base_physical_rate_sha256": _array_sha256(
            base_physical_rate_per_s
        ),
        "native_generator_sha256": _array_sha256(native_generator),
        "common_generator_sha256": _array_sha256(target_generator),
        "wp10c8y_arrays_sha256": _sha256(WP10C8Y_ARRAYS),
    }


def _similarity_rescale(
    matrix: np.ndarray,
    source_scales: np.ndarray,
    target_scales: np.ndarray,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    source = np.asarray(source_scales, dtype=float).ravel()
    target = np.asarray(target_scales, dtype=float).ravel()
    if (
        values.shape != (source.size, source.size)
        or target.shape != source.shape
        or np.any(source <= 0.0)
        or np.any(target <= 0.0)
    ):
        raise ValueError("invalid c9c0d similarity transformation")
    return (
        (source / target)[:, None]
        * values
        * (target / source)[None, :]
    )


def _build_or_load_decomposition(
    *,
    mesh: int,
    context,
    base_primitives: np.ndarray,
    native_scales: np.ndarray,
    target_scales: np.ndarray,
    base_physical_rate_per_s: np.ndarray,
    native_generator: np.ndarray,
    target_generator: np.ndarray,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    json_path, arrays_path = _cache_paths(mesh)
    contract = _decomposition_contract(
        mesh,
        base_primitives,
        native_scales,
        target_scales,
        base_physical_rate_per_s,
        native_generator,
        target_generator,
    )
    if not force and json_path.exists() and arrays_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        expected = {key: payload.get(key) for key in contract}
        if expected == contract and payload.get("arrays_sha256") == _sha256(
            arrays_path
        ):
            with np.load(arrays_path, allow_pickle=False) as source:
                return payload, {
                    name: np.asarray(source[name])
                    for name in source.files
                }

    print(
        f"WP10c9c0d: constructing exact N{mesh} generator blocks",
        flush=True,
    )
    started = time.perf_counter()
    decomposition = causal_five_field_generator_block_decomposition(
        context,
        base_primitives,
        primitive_column_scales=native_scales.ravel(),
        full_generator_per_s=native_generator,
        primitive_rate_per_s=base_physical_rate_per_s,
    )
    transformed_blocks = {
        name: _similarity_rescale(
            matrix,
            native_scales,
            target_scales,
        )
        for name, matrix in decomposition.generator_blocks_per_s.items()
    }
    transformed_sum = np.sum(
        np.asarray(list(transformed_blocks.values()), dtype=float),
        axis=0,
    )
    transformed_reconstruction_defect = _relative_maximum_defect(
        transformed_sum,
        target_generator,
    )
    transformed_remainder_fraction = float(
        np.linalg.norm(transformed_blocks["residual_unattributed"])
        / max(
            np.linalg.norm(target_generator),
            np.finfo(float).tiny,
        )
    )
    arrays = {
        "descriptor_matrix": (
            decomposition.descriptor_matrix
            * (
                np.asarray(target_scales, dtype=float).ravel()
                / np.asarray(native_scales, dtype=float).ravel()
            )[None, :]
        ),
        "scaled_primitive_rate_per_s": (
            np.asarray(
                decomposition.physical_primitive_rate_per_s,
                dtype=float,
            )
            / np.asarray(target_scales, dtype=float)
        ),
        "physical_primitive_rate_per_s": (
            decomposition.physical_primitive_rate_per_s
        ),
        "native_primitive_column_scales": np.asarray(
            native_scales,
            dtype=float,
        ),
        "target_primitive_column_scales": np.asarray(
            target_scales,
            dtype=float,
        ),
    }
    arrays.update(
        {
            f"block_{name}": matrix
            for name, matrix in transformed_blocks.items()
        }
    )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        **contract,
        "coordinate_construction": (
            "native_wp10c8x_decomposition_then_exact_similarity_rescale"
        ),
        "component_names": list(decomposition.component_names),
        "reduced_pattern_colors": (
            decomposition.reduced_pattern_colors
        ),
        "maximum_base_residual_reconstruction_defect": (
            decomposition.maximum_base_residual_reconstruction_defect
        ),
        "maximum_stationary_jacobian_reconstruction_defect": (
            decomposition
            .maximum_stationary_jacobian_reconstruction_defect
        ),
        "maximum_generator_reconstruction_defect_before_remainder": (
            decomposition
            .maximum_generator_reconstruction_defect_before_remainder
        ),
        "native_maximum_generator_reconstruction_defect_after_remainder": (
            decomposition
            .maximum_generator_reconstruction_defect_after_remainder
        ),
        "maximum_generator_reconstruction_defect_after_remainder": (
            transformed_reconstruction_defect
        ),
        "maximum_mass_solve_relative_defect": (
            decomposition.maximum_mass_solve_relative_defect
        ),
        "native_residual_unattributed_relative_frobenius_norm": (
            decomposition.residual_unattributed_relative_frobenius_norm
        ),
        "residual_unattributed_relative_frobenius_norm": (
            transformed_remainder_fraction
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


def _common_inputs() -> dict[int, dict]:
    contexts, _profiles = wp10c9c0c._common_contexts()
    operators = wp10c8y._load_family_operators()["production"]
    result = {}
    with np.load(WP10C8Y_ARRAYS, allow_pickle=False) as c8y, np.load(
        WP10C9C0C_ARRAYS,
        allow_pickle=False,
    ) as c0c:
        for mesh in wp10c8y.MESHES:
            amplitudes = np.asarray(
                c8y[f"N{mesh}_common_amplitudes"],
                dtype=float,
            )
            native_scales = np.asarray(
                operators[mesh]["primitive_column_scales"],
                dtype=float,
            )
            native_generator = np.asarray(
                operators[mesh]["generator"],
                dtype=float,
            )
            generator = wp10c8v._similarity_rescale_generator(
                native_generator,
                native_scales,
                amplitudes,
            )
            result[mesh] = {
                "context": contexts[mesh],
                "base_primitives": np.asarray(
                    operators[mesh]["base_primitives"],
                    dtype=float,
                ),
                "amplitudes": amplitudes,
                "native_scales": native_scales,
                "native_generator": native_generator,
                "base_physical_rate_per_s": np.asarray(
                    operators[mesh]["base_physical_rate_per_s"],
                    dtype=float,
                ),
                "generator": generator,
                "family_state": np.asarray(
                    c0c[f"common_N{mesh}_family_state"],
                    dtype=float,
                ),
                "family_rate": np.asarray(
                    c0c[f"common_N{mesh}_family_rate"],
                    dtype=float,
                ),
                "times": np.asarray(
                    c8y[f"production_N{mesh}_times"],
                    dtype=float,
                ),
                "radius_rg": np.asarray(
                    c8y[f"N{mesh}_radius_rg"],
                    dtype=float,
                ),
                "cell_measures": np.asarray(
                    c8y[f"N{mesh}_cell_measures"],
                    dtype=float,
                ),
                "parent_cross_work": np.asarray(
                    c0c[f"common_N{mesh}_family_cross_work"],
                    dtype=float,
                ),
            }
    return result


def _blocks_from_cache(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name.removeprefix("block_"): np.asarray(matrix, dtype=float)
        for name, matrix in arrays.items()
        if name.startswith("block_")
    }


def _decomposition_method_contract(reports: dict[int, dict]) -> dict:
    measurements = {
        "maximum_base_residual_reconstruction_defect": max(
            row["maximum_base_residual_reconstruction_defect"]
            for row in reports.values()
        ),
        "maximum_stationary_jacobian_reconstruction_defect": max(
            row["maximum_stationary_jacobian_reconstruction_defect"]
            for row in reports.values()
        ),
        "maximum_final_generator_reconstruction_defect": max(
            row[
                "maximum_generator_reconstruction_defect_after_remainder"
            ]
            for row in reports.values()
        ),
        "maximum_mass_solve_relative_defect": max(
            row["maximum_mass_solve_relative_defect"]
            for row in reports.values()
        ),
        "maximum_unattributed_generator_fraction": max(
            row["residual_unattributed_relative_frobenius_norm"]
            for row in reports.values()
        ),
    }
    checks = {
        "base_residual_reconstruction": (
            measurements[
                "maximum_base_residual_reconstruction_defect"
            ]
            <= MAXIMUM_BASE_RESIDUAL_RECONSTRUCTION_DEFECT
        ),
        "stationary_jacobian_reconstruction": (
            measurements[
                "maximum_stationary_jacobian_reconstruction_defect"
            ]
            <= MAXIMUM_STATIONARY_JACOBIAN_RECONSTRUCTION_DEFECT
        ),
        "final_generator_reconstruction": (
            measurements[
                "maximum_final_generator_reconstruction_defect"
            ]
            <= MAXIMUM_FINAL_GENERATOR_RECONSTRUCTION_DEFECT
        ),
        "mass_solve": (
            measurements["maximum_mass_solve_relative_defect"]
            <= MAXIMUM_MASS_SOLVE_DEFECT
        ),
        "unattributed_generator": (
            measurements["maximum_unattributed_generator_fraction"]
            <= MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION
        ),
    }
    return {
        "measurements": measurements,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _restrict_family_action(
    fine: np.ndarray,
    fine_measures: np.ndarray,
) -> np.ndarray:
    # Input is receiver,time,cell,field.  The existing conservative
    # restriction supports arbitrary leading dimensions.
    return wp10c8v._restrict_pairwise(fine, fine_measures)


def _weighted_history_norm(
    values: np.ndarray,
    measures: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(measures, dtype=float)
    weights = weights / np.sum(weights)
    array = np.asarray(values, dtype=float)
    return np.sqrt(
        np.einsum(
            "...ci,...ci,c->...",
            array,
            array,
            weights,
            optimize=True,
        )
    )


def _component_label(
    block_names: tuple[str, ...],
    index: tuple[int, int, int],
) -> dict:
    block, receiver, source = index
    return {
        "block": block_names[block],
        "receiver": CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES[
            receiver
        ],
        "source": CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES[source],
    }


def _component_action(
    *,
    data: dict,
    blocks: dict[str, np.ndarray],
    projectors,
    block_name: str,
    source: int,
) -> np.ndarray:
    return causal_block_family_receiver_action(
        data["family_state"][source],
        blocks[block_name],
        projectors,
    )


def _pair_action_attribution(
    *,
    coarse_mesh: int,
    fine_mesh: int,
    inputs: dict[int, dict],
    blocks: dict[int, dict[str, np.ndarray]],
    projectors: dict,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    coarse = inputs[coarse_mesh]
    fine = inputs[fine_mesh]
    block_names = tuple(blocks[coarse_mesh])
    if tuple(blocks[fine_mesh]) != block_names:
        raise RuntimeError("common block schemas differ across meshes")
    coarse_rate = np.sum(coarse["family_rate"], axis=0)
    fine_rate = np.sum(fine["family_rate"], axis=0)
    restricted_rate = wp10c8v._restrict_pairwise(
        fine_rate,
        fine["cell_measures"],
    )
    total_error = restricted_rate - coarse_rate
    error_norm = _weighted_history_norm(
        total_error,
        coarse["cell_measures"],
    )
    reference_norm = _weighted_history_norm(
        coarse_rate,
        coarse["cell_measures"],
    )
    relative_error = error_norm / np.maximum(
        reference_norm,
        np.finfo(float).tiny,
    )
    restricted_source_rate = wp10c8v._restrict_pairwise(
        fine["family_rate"],
        fine["cell_measures"],
    )
    source_error = restricted_source_rate - coarse["family_rate"]
    source_error_norm = _weighted_history_norm(
        source_error,
        coarse["cell_measures"],
    )
    source_reference_norm = _weighted_history_norm(
        coarse["family_rate"],
        coarse["cell_measures"],
    )
    source_relative_error = source_error_norm / np.maximum(
        source_reference_norm,
        np.finfo(float).tiny,
    )
    n_blocks = len(block_names)
    component_norm = np.empty(
        (
            coarse["times"].size,
            n_blocks,
            5,
            5,
        ),
        dtype=float,
    )
    component_signed_projection = np.empty_like(component_norm)
    component_significance = np.empty_like(component_norm)
    weights = coarse["cell_measures"] / np.sum(
        coarse["cell_measures"]
    )
    total_squared = np.einsum(
        "tci,tci,c->t",
        total_error,
        total_error,
        weights,
        optimize=True,
    )
    reconstructed_error = np.zeros_like(total_error)
    reconstructed_source_error = np.zeros_like(source_error)
    for block_index, name in enumerate(block_names):
        for source in range(5):
            coarse_action = _component_action(
                data=coarse,
                blocks=blocks[coarse_mesh],
                projectors=projectors[coarse_mesh],
                block_name=name,
                source=source,
            )
            fine_action = _component_action(
                data=fine,
                blocks=blocks[fine_mesh],
                projectors=projectors[fine_mesh],
                block_name=name,
                source=source,
            )
            restricted = _restrict_family_action(
                fine_action,
                fine["cell_measures"],
            )
            difference = restricted - coarse_action
            reconstructed_error += np.sum(difference, axis=0)
            reconstructed_source_error[source] += np.sum(
                difference,
                axis=0,
            )
            norms = _weighted_history_norm(
                difference,
                coarse["cell_measures"],
            ).T
            component_norm[:, block_index, :, source] = norms
            component_significance[
                :, block_index, :, source
            ] = norms / np.maximum(
                reference_norm[:, None],
                np.finfo(float).tiny,
            )
            component_signed_projection[
                :, block_index, :, source
            ] = np.einsum(
                "rtci,tci,c->tr",
                difference,
                total_error,
                weights,
                optimize=True,
            ) / np.maximum(
                total_squared[:, None],
                np.finfo(float).tiny,
            )
    absolute_total = np.sum(component_norm, axis=(1, 2, 3))
    component_activity_fraction = component_norm / np.maximum(
        absolute_total[:, None, None, None],
        np.finfo(float).tiny,
    )
    source_absolute_total = np.sum(component_norm, axis=(1, 2))
    source_component_activity_fraction = component_norm / np.maximum(
        source_absolute_total[:, None, None, :],
        np.finfo(float).tiny,
    )
    source_conditioned = {}
    source_top_components = np.empty((5, 3), dtype=int)
    for source, family in enumerate(
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    ):
        source_time = int(np.argmax(source_relative_error[source]))
        source_flat = int(
            np.argmax(
                source_component_activity_fraction[
                    source_time,
                    :,
                    :,
                    source,
                ]
            )
        )
        block, receiver = np.unravel_index(
            source_flat,
            source_component_activity_fraction.shape[1:3],
        )
        source_top_components[source] = (block, receiver, source)
        source_conditioned[family] = {
            "maximum_relative_rate_error": float(
                np.max(source_relative_error[source])
            ),
            "controlling_time_index": source_time,
            "controlling_time_seconds": float(
                coarse["times"][source_time]
            ),
            "controlling_component": _component_label(
                block_names,
                (block, receiver, source),
            ),
            "controlling_component_activity_fraction": float(
                source_component_activity_fraction[
                    source_time,
                    block,
                    receiver,
                    source,
                ]
            ),
            "controlling_component_significance": float(
                component_norm[
                    source_time,
                    block,
                    receiver,
                    source,
                ]
                / max(
                    float(source_reference_norm[source, source_time]),
                    np.finfo(float).tiny,
                )
            ),
            "maximum_rate_error_reconstruction_defect": (
                _relative_maximum_defect(
                    reconstructed_source_error[source],
                    source_error[source],
                )
            ),
        }
    controlling_time = int(np.argmax(relative_error))
    flat_index = int(
        np.argmax(component_activity_fraction[controlling_time])
    )
    controlling_component = np.unravel_index(
        flat_index,
        component_activity_fraction.shape[1:],
    )
    flat_by_time = component_activity_fraction.reshape(
        component_activity_fraction.shape[0],
        -1,
    )
    top_indices = np.argmax(flat_by_time, axis=1)
    top_components = np.asarray(
        [
            np.unravel_index(index, component_activity_fraction.shape[1:])
            for index in top_indices
        ],
        dtype=int,
    )
    report = {
        "pair": f"N{coarse_mesh}_N{fine_mesh}",
        "maximum_relative_rate_error": float(np.max(relative_error)),
        "controlling_time_index": controlling_time,
        "controlling_time_seconds": float(
            coarse["times"][controlling_time]
        ),
        "controlling_component": _component_label(
            block_names,
            controlling_component,
        ),
        "controlling_component_activity_fraction": float(
            component_activity_fraction[
                (controlling_time,) + controlling_component
            ]
        ),
        "controlling_component_signed_projection": float(
            component_signed_projection[
                (controlling_time,) + controlling_component
            ]
        ),
        "controlling_component_significance": float(
            component_significance[
                (controlling_time,) + controlling_component
            ]
        ),
        "maximum_rate_error_reconstruction_defect": (
            _relative_maximum_defect(
                reconstructed_error,
                total_error,
            )
        ),
        "source_conditioned_rate_error": source_conditioned,
    }
    arrays = {
        "relative_rate_error": relative_error,
        "component_norm": component_norm,
        "component_activity_fraction": component_activity_fraction,
        "component_signed_projection": component_signed_projection,
        "component_significance": component_significance,
        "top_components": top_components,
        "source_relative_rate_error": source_relative_error,
        "source_component_activity_fraction": (
            source_component_activity_fraction
        ),
        "source_top_components": source_top_components,
    }
    working = {
        "total_error": total_error,
        "block_names": block_names,
        "controlling_component": controlling_component,
        "coarse_mesh": coarse_mesh,
        "fine_mesh": fine_mesh,
    }
    return report, arrays, working


def _component_error_profile(
    *,
    index: tuple[int, int, int],
    time_index: int,
    working: dict,
    inputs: dict[int, dict],
    blocks: dict[int, dict[str, np.ndarray]],
    projectors: dict,
) -> dict[str, np.ndarray]:
    block, receiver, source = index
    coarse_mesh = working["coarse_mesh"]
    fine_mesh = working["fine_mesh"]
    coarse = inputs[coarse_mesh]
    fine = inputs[fine_mesh]
    name = working["block_names"][block]
    coarse_action = _component_action(
        data=coarse,
        blocks=blocks[coarse_mesh],
        projectors=projectors[coarse_mesh],
        block_name=name,
        source=source,
    )[receiver]
    fine_action = _component_action(
        data=fine,
        blocks=blocks[fine_mesh],
        projectors=projectors[fine_mesh],
        block_name=name,
        source=source,
    )[receiver]
    error = wp10c8v._restrict_pairwise(
        fine_action,
        fine["cell_measures"],
    ) - coarse_action
    total = working["total_error"]
    weights = coarse["cell_measures"] / np.sum(
        coarse["cell_measures"]
    )
    signed = (
        np.sum(error[time_index] * total[time_index], axis=1)
        * weights
    )
    activity = np.linalg.norm(error[time_index], axis=1) * np.sqrt(
        weights
    )
    return {
        "signed_projection_profile": signed,
        "absolute_activity_profile": activity,
        "radius_rg": coarse["radius_rg"],
    }


def _restrict_scalar_profile(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size % 2:
        raise ValueError("scalar profile is not nested")
    return np.sum(array.reshape(-1, 2), axis=1)


def _profile_cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    denominator = max(
        float(np.linalg.norm(a) * np.linalg.norm(b)),
        np.finfo(float).tiny,
    )
    return float(np.dot(a, b) / denominator)


def _profile_centroid(radius: np.ndarray, profile: np.ndarray) -> float:
    weights = np.abs(np.asarray(profile, dtype=float))
    total = max(float(np.sum(weights)), np.finfo(float).tiny)
    return float(np.sum(np.asarray(radius) * weights) / total)


def _mesh_stability_and_onset(
    *,
    reports: dict[str, dict],
    arrays: dict[str, dict[str, np.ndarray]],
    workings: dict[str, dict],
    inputs: dict[int, dict],
    blocks: dict[int, dict[str, np.ndarray]],
    projectors: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    coarse_key = "N64_N128"
    fine_key = "N128_N256"
    coarse_relative = arrays[coarse_key]["relative_rate_error"]
    fine_relative = arrays[fine_key]["relative_rate_error"]
    mask = (
        (fine_relative > coarse_relative)
        & (fine_relative >= MINIMUM_CONTRACTION_DEFECT)
    )
    onset_candidates = np.flatnonzero(mask)
    onset = (
        int(onset_candidates[0])
        if onset_candidates.size
        else int(np.argmax(fine_relative / np.maximum(
            coarse_relative,
            np.finfo(float).tiny,
        )))
    )
    controlling = int(reports[fine_key]["controlling_time_index"])
    fine_component = tuple(
        workings[fine_key]["controlling_component"]
    )
    coarse_at_fine_time_flat = int(
        np.argmax(
            arrays[coarse_key]["component_activity_fraction"][
                controlling
            ]
        )
    )
    coarse_component = np.unravel_index(
        coarse_at_fine_time_flat,
        arrays[coarse_key]["component_activity_fraction"].shape[1:],
    )
    same_component = coarse_component == fine_component

    fine_profile = _component_error_profile(
        index=fine_component,
        time_index=controlling,
        working=workings[fine_key],
        inputs=inputs,
        blocks=blocks,
        projectors=projectors,
    )
    coarse_profile = _component_error_profile(
        index=fine_component,
        time_index=controlling,
        working=workings[coarse_key],
        inputs=inputs,
        blocks=blocks,
        projectors=projectors,
    )
    restricted_fine = _restrict_scalar_profile(
        fine_profile["absolute_activity_profile"]
    )
    cosine = _profile_cosine(
        coarse_profile["absolute_activity_profile"],
        restricted_fine,
    )
    coarse_centroid = _profile_centroid(
        coarse_profile["radius_rg"],
        coarse_profile["absolute_activity_profile"],
    )
    fine_centroid = _profile_centroid(
        fine_profile["radius_rg"],
        fine_profile["absolute_activity_profile"],
    )
    centroid_defect = abs(fine_centroid - coarse_centroid) / max(
        abs(coarse_centroid),
        np.finfo(float).tiny,
    )
    fine_fraction = arrays[fine_key][
        "component_activity_fraction"
    ][(slice(None),) + fine_component]
    fine_significance = arrays[fine_key][
        "component_significance"
    ][(slice(None),) + fine_component]
    active_window = np.arange(fine_fraction.size) >= onset
    significant_window = (
        active_window
        & (fine_significance >= MINIMUM_ABSOLUTE_SIGNIFICANCE)
    )
    denominator = max(
        int(np.count_nonzero(significant_window)),
        1,
    )
    persistence = float(
        np.count_nonzero(
            significant_window
            & (
                fine_fraction
                >= MINIMUM_COMPONENT_ACTIVITY_FRACTION
            )
        )
        / denominator
    )
    onset_flat = int(
        np.argmax(
            arrays[fine_key]["component_activity_fraction"][onset]
        )
    )
    onset_component = np.unravel_index(
        onset_flat,
        arrays[fine_key]["component_activity_fraction"].shape[1:],
    )
    gates = {
        "same_component_between_refinement_pairs": same_component,
        "controlling_activity_fraction": (
            reports[fine_key][
                "controlling_component_activity_fraction"
            ]
            >= MINIMUM_COMPONENT_ACTIVITY_FRACTION
        ),
        "absolute_significance": (
            reports[fine_key]["controlling_component_significance"]
            >= MINIMUM_ABSOLUTE_SIGNIFICANCE
        ),
        "persistence": (
            persistence >= MINIMUM_COMPONENT_PERSISTENCE_FRACTION
        ),
        "radial_profile_cosine": (
            cosine >= MINIMUM_RADIAL_PROFILE_COSINE
        ),
        "radial_centroid": (
            centroid_defect <= MAXIMUM_RADIAL_CENTROID_RELATIVE_DEFECT
        ),
    }
    return (
        {
            "first_loss_of_contraction_index": onset,
            "first_loss_of_contraction_seconds": float(
                inputs[128]["times"][onset]
            ),
            "fine_controlling_component": _component_label(
                workings[fine_key]["block_names"],
                fine_component,
            ),
            "fine_onset_component": _component_label(
                workings[fine_key]["block_names"],
                onset_component,
            ),
            "coarse_component_at_fine_controlling_time": (
                _component_label(
                    workings[coarse_key]["block_names"],
                    coarse_component,
                )
            ),
            "fine_component_persistence_fraction": persistence,
            "radial_profile_cosine": cosine,
            "coarse_profile_centroid_rg": coarse_centroid,
            "fine_profile_centroid_rg": fine_centroid,
            "radial_centroid_relative_defect": centroid_defect,
            "gates": gates,
            "passed": bool(all(gates.values())),
        },
        {
            "fine_controlling_radius_rg": fine_profile["radius_rg"],
            "fine_controlling_absolute_profile": (
                fine_profile["absolute_activity_profile"]
            ),
            "fine_controlling_signed_profile": (
                fine_profile["signed_projection_profile"]
            ),
            "coarse_controlling_radius_rg": (
                coarse_profile["radius_rg"]
            ),
            "coarse_controlling_absolute_profile": (
                coarse_profile["absolute_activity_profile"]
            ),
            "coarse_controlling_signed_profile": (
                coarse_profile["signed_projection_profile"]
            ),
            "fine_relative_rate_error": fine_relative,
            "coarse_relative_rate_error": coarse_relative,
        },
    )


def _cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5
        * (values[:-1] + values[1:])
        * np.diff(times).reshape(
            (times.size - 1,) + (1,) * (values.ndim - 1)
        ),
        axis=0,
    )
    return result


def _cross_work_summary(
    *,
    inputs: dict[int, dict],
    ledgers: dict[int, object],
) -> tuple[dict, dict[str, np.ndarray]]:
    labels = CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    arrays = {}
    by_mesh = {}
    cumulative = {}
    shear_mediation = {}
    inward_shear = (
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES.index(
            "inward_shear"
        )
    )
    outward_shear = (
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES.index(
            "outward_shear"
        )
    )
    for mesh, ledger in ledgers.items():
        # Public ledger order is block,time,receiver,source.
        work = np.moveaxis(ledger.global_cross_work_per_s, 0, 1)
        cumulative[mesh] = _cumulative_trapezoid(
            inputs[mesh]["times"],
            work,
        )
        arrays[f"N{mesh}_block_family_work_per_s"] = work
        arrays[f"N{mesh}_block_family_cumulative_work"] = (
            cumulative[mesh]
        )
        final = cumulative[mesh][-1]
        entries = []
        for block, name in enumerate(ledger.block_names):
            for receiver in range(5):
                for source in range(5):
                    entries.append(
                        (
                            float(abs(final[block, receiver, source])),
                            float(final[block, receiver, source]),
                            name,
                            labels[receiver],
                            labels[source],
                        )
                    )
        entries.sort(reverse=True)
        absolute_total = max(
            float(sum(row[0] for row in entries)),
            np.finfo(float).tiny,
        )
        by_mesh[f"N{mesh}"] = [
            {
                "block": name,
                "receiver": receiver,
                "source": source,
                "signed_cumulative_work": signed,
                "absolute_fraction": absolute / absolute_total,
            }
            for absolute, signed, name, receiver, source in entries[:8]
        ]
        directed_in_out = final[
            :,
            inward_shear,
            outward_shear,
        ]
        directed_out_in = final[
            :,
            outward_shear,
            inward_shear,
        ]
        mediation_activity = (
            np.abs(directed_in_out) + np.abs(directed_out_in)
        )
        mediation_total = max(
            float(np.sum(mediation_activity)),
            np.finfo(float).tiny,
        )
        mediation_block = int(np.argmax(mediation_activity))
        shear_mediation[f"N{mesh}"] = {
            "controlling_block": ledger.block_names[mediation_block],
            "controlling_block_activity_fraction": float(
                mediation_activity[mediation_block] / mediation_total
            ),
            "inward_receiver_outward_source_cumulative_work": float(
                directed_in_out[mediation_block]
            ),
            "outward_receiver_inward_source_cumulative_work": float(
                directed_out_in[mediation_block]
            ),
            "dominance_gate_passed": bool(
                mediation_activity[mediation_block] / mediation_total
                >= MINIMUM_COMPONENT_ACTIVITY_FRACTION
            ),
        }
    difference = cumulative[256] - cumulative[128]
    arrays["N128_N256_block_family_cumulative_work_difference"] = (
        difference
    )
    maximum_index = np.unravel_index(
        int(np.argmax(np.abs(difference))),
        difference.shape,
    )
    time_index, block, receiver, source = maximum_index
    ledger = ledgers[128]
    final_difference = difference[-1]
    fine_directed_in_out = final_difference[
        :,
        inward_shear,
        outward_shear,
    ]
    fine_directed_out_in = final_difference[
        :,
        outward_shear,
        inward_shear,
    ]
    fine_mediation_activity = (
        np.abs(fine_directed_in_out) + np.abs(fine_directed_out_in)
    )
    fine_mediation_total = max(
        float(np.sum(fine_mediation_activity)),
        np.finfo(float).tiny,
    )
    fine_mediation_block = int(np.argmax(fine_mediation_activity))
    stable_mediation_block = len(
        {
            row["controlling_block"]
            for row in shear_mediation.values()
        }
    ) == 1
    return (
        {
            "leading_final_terms": by_mesh,
            "fine_pair_maximum_cumulative_difference": {
                "time_seconds": float(inputs[128]["times"][time_index]),
                "block": ledger.block_names[block],
                "receiver": labels[receiver],
                "source": labels[source],
                "signed_difference": float(
                    difference[maximum_index]
                ),
            },
            "inward_outward_shear_mediation": {
                "by_mesh": shear_mediation,
                "same_controlling_block_on_all_meshes": (
                    stable_mediation_block
                ),
                "fine_pair_cumulative_difference": {
                    "controlling_block": (
                        ledger.block_names[fine_mediation_block]
                    ),
                    "controlling_block_activity_fraction": float(
                        fine_mediation_activity[fine_mediation_block]
                        / fine_mediation_total
                    ),
                    "inward_receiver_outward_source_difference": (
                        float(
                            fine_directed_in_out[fine_mediation_block]
                        )
                    ),
                    "outward_receiver_inward_source_difference": (
                        float(
                            fine_directed_out_in[fine_mediation_block]
                        )
                    ),
                    "dominance_gate_passed": bool(
                        fine_mediation_activity[fine_mediation_block]
                        / fine_mediation_total
                        >= MINIMUM_COMPONENT_ACTIVITY_FRACTION
                    ),
                },
            },
        },
        arrays,
    )


def run(*, force: bool) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    required = (
        WP10C8Y_OUTPUT,
        WP10C8Y_ARRAYS,
        WP10C9C0C_OUTPUT,
        WP10C9C0C_ARRAYS,
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError("WP10c9c0d requires WP10c8y/c9c0c")
    c8y = json.loads(WP10C8Y_OUTPUT.read_text(encoding="utf-8"))
    c0c = json.loads(WP10C9C0C_OUTPUT.read_text(encoding="utf-8"))
    if c8y["classification"] != (
        "common_mode_passed_boundary_insensitive_underresolution"
    ):
        raise RuntimeError("WP10c8y classification changed")
    if c0c["classification"] != (
        "common_mode_failure_remains_multifamily_or_nonlocal"
    ):
        raise RuntimeError("WP10c9c0c classification changed")
    if c0c["decision"]["wp10c9c1_path_candidate_authorized"]:
        raise RuntimeError("WP10c9c1 unexpectedly became authorized")

    inputs = _common_inputs()
    decomposition_reports = {}
    blocks = {}
    projectors = {}
    ledgers = {}
    method_rows = {}
    for mesh, data in inputs.items():
        report, cached = _build_or_load_decomposition(
            mesh=mesh,
            context=data["context"],
            base_primitives=data["base_primitives"],
            native_scales=data["native_scales"],
            target_scales=data["amplitudes"],
            base_physical_rate_per_s=(
                data["base_physical_rate_per_s"]
            ),
            native_generator=data["native_generator"],
            target_generator=data["generator"],
            force=force,
        )
        decomposition_reports[mesh] = report
        blocks[mesh] = _blocks_from_cache(cached)
        projectors[mesh], _bases = (
            causal_five_field_characteristic_family_projectors(
                data["context"],
                data["base_primitives"],
                data["amplitudes"],
            )
        )
        ledger = causal_block_family_transfer_ledger(
            data["family_state"],
            blocks[mesh],
            projectors[mesh],
            data["cell_measures"],
        )
        ledgers[mesh] = ledger
        reconstructed_cross_work = np.sum(
            ledger.global_cross_work_per_s,
            axis=0,
        )
        parent_cross_work_defect = _relative_maximum_defect(
            reconstructed_cross_work,
            data["parent_cross_work"],
        )
        method_rows[mesh] = {
            "maximum_rate_action_closure_defect": (
                ledger.maximum_rate_action_closure_defect
            ),
            "maximum_cross_work_closure_defect": (
                ledger.maximum_cross_work_closure_defect
            ),
            "maximum_parent_cross_work_reproduction_defect": (
                parent_cross_work_defect
            ),
            "maximum_projector_identity_defect": (
                projectors[mesh].maximum_identity_closure_defect
            ),
        }

    decomposition_method = _decomposition_method_contract(
        decomposition_reports
    )
    transfer_measurements = {
        "maximum_rate_action_closure_defect": max(
            row["maximum_rate_action_closure_defect"]
            for row in method_rows.values()
        ),
        "maximum_cross_work_closure_defect": max(
            row["maximum_cross_work_closure_defect"]
            for row in method_rows.values()
        ),
        "maximum_parent_cross_work_reproduction_defect": max(
            row["maximum_parent_cross_work_reproduction_defect"]
            for row in method_rows.values()
        ),
        "maximum_projector_identity_defect": max(
            row["maximum_projector_identity_defect"]
            for row in method_rows.values()
        ),
    }
    transfer_checks = {
        "rate_action_closure": (
            transfer_measurements["maximum_rate_action_closure_defect"]
            <= MAXIMUM_RATE_ACTION_CLOSURE_DEFECT
        ),
        "cross_work_closure": (
            transfer_measurements["maximum_cross_work_closure_defect"]
            <= MAXIMUM_CROSS_WORK_CLOSURE_DEFECT
        ),
        "parent_cross_work_reproduction": (
            transfer_measurements[
                "maximum_parent_cross_work_reproduction_defect"
            ]
            <= MAXIMUM_PARENT_CROSS_WORK_REPRODUCTION_DEFECT
        ),
        "projector_identity": (
            transfer_measurements["maximum_projector_identity_defect"]
            <= MAXIMUM_RATE_ACTION_CLOSURE_DEFECT
        ),
    }
    transfer_method = {
        "by_mesh": method_rows,
        "measurements": transfer_measurements,
        "checks": transfer_checks,
        "passed": bool(all(transfer_checks.values())),
    }

    pair_reports = {}
    pair_arrays = {}
    workings = {}
    for coarse, fine in ((64, 128), (128, 256)):
        key = f"N{coarse}_N{fine}"
        print(f"WP10c9c0d: attributing {key}", flush=True)
        report, arrays, working = _pair_action_attribution(
            coarse_mesh=coarse,
            fine_mesh=fine,
            inputs=inputs,
            blocks=blocks,
            projectors=projectors,
        )
        pair_reports[key] = report
        pair_arrays[key] = arrays
        workings[key] = working

    stability, stability_arrays = _mesh_stability_and_onset(
        reports=pair_reports,
        arrays=pair_arrays,
        workings=workings,
        inputs=inputs,
        blocks=blocks,
        projectors=projectors,
    )
    cross_work, cross_work_arrays = _cross_work_summary(
        inputs=inputs,
        ledgers=ledgers,
    )
    mediation = cross_work["inward_outward_shear_mediation"]
    fine_mediation = mediation["by_mesh"]["N256"]
    outward_source = pair_reports["N128_N256"][
        "source_conditioned_rate_error"
    ]["outward_shear"]
    mediation_block = fine_mediation["controlling_block"]
    outward_error_block = outward_source["controlling_component"][
        "block"
    ]
    targeted_checks = {
        "mesh_stable_interaction_block": (
            mediation["same_controlling_block_on_all_meshes"]
        ),
        "interaction_block_dominance": (
            fine_mediation["dominance_gate_passed"]
        ),
        "fine_interaction_defect_block_dominance": (
            mediation["fine_pair_cumulative_difference"][
                "dominance_gate_passed"
            ]
        ),
        "same_block_controls_outward_shear_rate_error": (
            mediation_block == outward_error_block
        ),
        "outward_shear_rate_error_block_dominance": (
            outward_source[
                "controlling_component_activity_fraction"
            ]
            >= MINIMUM_COMPONENT_ACTIVITY_FRACTION
        ),
    }
    targeted_interaction = {
        "interaction": "inward_outward_shear",
        "fine_mesh_mediating_block": mediation_block,
        "fine_mesh_mediating_block_activity_fraction": (
            fine_mediation["controlling_block_activity_fraction"]
        ),
        "fine_pair_interaction_defect_block": (
            mediation["fine_pair_cumulative_difference"][
                "controlling_block"
            ]
        ),
        "fine_pair_interaction_defect_block_activity_fraction": (
            mediation["fine_pair_cumulative_difference"][
                "controlling_block_activity_fraction"
            ]
        ),
        "outward_shear_rate_error_block": outward_error_block,
        "outward_shear_rate_error_receiver": (
            outward_source["controlling_component"]["receiver"]
        ),
        "outward_shear_rate_error_block_activity_fraction": (
            outward_source[
                "controlling_component_activity_fraction"
            ]
        ),
        "checks": targeted_checks,
        "passed": bool(all(targeted_checks.values())),
    }
    method_passed = bool(
        decomposition_method["passed"] and transfer_method["passed"]
    )
    mechanism_identified = bool(
        method_passed
        and stability["passed"]
        and targeted_interaction["passed"]
    )
    if not method_passed:
        classification = "common_block_family_method_contract_failed"
    elif mechanism_identified:
        classification = (
            "mesh_stable_common_block_family_mechanism_identified"
        )
    else:
        classification = (
            "common_mode_defect_remains_multiblock_after_direct_ledger"
        )

    arrays = {}
    for key, values in pair_arrays.items():
        for name, value in values.items():
            arrays[f"{key}_{name}"] = value
    arrays.update(stability_arrays)
    arrays.update(cross_work_arrays)
    arrays["times_seconds"] = inputs[64]["times"]
    arrays["block_names"] = np.asarray(
        ledgers[64].block_names,
        dtype="U64",
    )
    arrays["family_names"] = np.asarray(
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        dtype="U32",
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
        "decomposition_method_contract": decomposition_method,
        "transfer_method_contract": transfer_method,
        "decomposition_by_mesh": decomposition_reports,
        "rate_error_attribution": pair_reports,
        "mesh_stability_and_onset": stability,
        "block_family_cross_work": cross_work,
        "targeted_common_shear_interaction": targeted_interaction,
        "decision": {
            "one_mesh_stable_localized_interaction_identified": (
                mechanism_identified
            ),
            "wp10c9c1_path_candidate_authorized": False,
            "production_operator_change_authorized": False,
            "new_truth_trajectory_authorized": False,
            "fixed_q_or_reduction_authorized": False,
        },
        "gates": {
            "maximum_base_residual_reconstruction_defect": (
                MAXIMUM_BASE_RESIDUAL_RECONSTRUCTION_DEFECT
            ),
            "maximum_stationary_jacobian_reconstruction_defect": (
                MAXIMUM_STATIONARY_JACOBIAN_RECONSTRUCTION_DEFECT
            ),
            "maximum_final_generator_reconstruction_defect": (
                MAXIMUM_FINAL_GENERATOR_RECONSTRUCTION_DEFECT
            ),
            "maximum_mass_solve_defect": MAXIMUM_MASS_SOLVE_DEFECT,
            "maximum_unattributed_generator_fraction": (
                MAXIMUM_UNATTRIBUTED_GENERATOR_FRACTION
            ),
            "maximum_rate_action_closure_defect": (
                MAXIMUM_RATE_ACTION_CLOSURE_DEFECT
            ),
            "maximum_cross_work_closure_defect": (
                MAXIMUM_CROSS_WORK_CLOSURE_DEFECT
            ),
            "maximum_parent_cross_work_reproduction_defect": (
                MAXIMUM_PARENT_CROSS_WORK_REPRODUCTION_DEFECT
            ),
            "minimum_absolute_significance": (
                MINIMUM_ABSOLUTE_SIGNIFICANCE
            ),
            "minimum_component_activity_fraction": (
                MINIMUM_COMPONENT_ACTIVITY_FRACTION
            ),
            "minimum_component_persistence_fraction": (
                MINIMUM_COMPONENT_PERSISTENCE_FRACTION
            ),
            "minimum_radial_profile_cosine": (
                MINIMUM_RADIAL_PROFILE_COSINE
            ),
            "maximum_radial_centroid_relative_defect": (
                MAXIMUM_RADIAL_CENTROID_RELATIVE_DEFECT
            ),
        },
        "provenance": {
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "core_file": CORE_FILE,
            "core_sha256": _sha256(ROOT / CORE_FILE),
            "decomposition_core_file": DECOMPOSITION_CORE_FILE,
            "decomposition_core_sha256": _sha256(
                ROOT / DECOMPOSITION_CORE_FILE
            ),
            "wp10c8y_output_sha256": _sha256(WP10C8Y_OUTPUT),
            "wp10c8y_arrays_sha256": _sha256(WP10C8Y_ARRAYS),
            "wp10c9c0c_output_sha256": _sha256(WP10C9C0C_OUTPUT),
            "wp10c9c0c_arrays_sha256": _sha256(WP10C9C0C_ARRAYS),
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload, _arrays = run(force=args.force)
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
                "first_loss_seconds": payload[
                    "mesh_stability_and_onset"
                ]["first_loss_of_contraction_seconds"],
                "fine_component": payload[
                    "mesh_stability_and_onset"
                ]["fine_controlling_component"],
                "fine_activity_fraction": payload[
                    "rate_error_attribution"
                ]["N128_N256"][
                    "controlling_component_activity_fraction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
