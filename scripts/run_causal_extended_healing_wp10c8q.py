"""Run the WP10c8q slow-rate divergence and extended-healing audit.

The package has three deliberately separated tracks:

* a no-new-evolution incidence/ledger audit of the committed WP10c8p data;
* exact-history N64 continuation of the decisive pair through 0.125 s;
* a finite-amplitude equal-coordinate audit seeded by the worst tangent
  ambiguity in the complete 34-coordinate slow-time rate vector.

No production residual, source, spatial flux, coordinate definition, or BDF
formula is changed by this runner.
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

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_natural_healing_wp10c8p as wp10c8p
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_PRIMITIVE_NAMES,
    CausalFiveFieldBDFRestart,
    causal_cumulative_trapezoid,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_loading_time,
    causal_five_field_observable_snapshot,
    causal_five_field_reconstruct_face_charts,
    causal_gate_normalized_finite_time_null_gain,
    causal_internal_face_boundary_rates,
    causal_mesh_coincident_moment_shells,
    causal_path_integrated_component_decomposition,
    causal_refined_spread_upper_bound,
    causal_transport_rank_audit,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_bdf_restart,
    save_causal_five_field_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    _central_perfect_flux_from_validated_face_charts,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "1e42839c094d3e7c2dc89e963a681e0004afa556"
WORK_PACKAGE = "WP10c8q"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_extended_healing_wp10c8q.py"
PARENT_JSON = ROOT / "outputs/tables/causal_natural_healing_wp10c8p.json"
PARENT_ARRAYS = (
    ROOT / "outputs/tables/causal_natural_healing_wp10c8p_arrays.npz"
)
PARENT_FIBER_JSON = (
    ROOT / "outputs/tables/causal_nonlinear_fiber_audit_wp10c8o.json"
)
PARENT_FIBER_ARRAYS = (
    ROOT / "outputs/tables/causal_nonlinear_fiber_audit_wp10c8o_arrays.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs/tables/causal_extended_healing_wp10c8q.json"
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_extended_healing_wp10c8q_arrays.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8q"
)
RATE_FIBER_CACHE_JSON = CHECKPOINT_DIRECTORY / "slow_rate_fiber_audit.json"
RATE_FIBER_CACHE_ARRAYS = CHECKPOINT_DIRECTORY / "slow_rate_fiber_audit_arrays.npz"

PARENT_DURATION_SECONDS = 2.5e-2
TARGET_DURATION_SECONDS = 1.25e-1
CONTINUATION_DURATION_SECONDS = TARGET_DURATION_SECONDS - PARENT_DURATION_SECONDS
OUTPUT_OFFSETS_SECONDS = (
    0.0,
    0.0025,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.125,
)
COARSE_SUBDIVISIONS = 100
FINE_SUBDIVISIONS = 200
CONTINUATION_COARSE_SUBDIVISIONS = 80
CONTINUATION_FINE_SUBDIVISIONS = 160
INTERFACE_INDEX = 4
MJE_COMPONENTS = (0, 2, 3)
INTERFACE_FLUX_RELATIVE_GATE = 1.0e-3
MAXIMUM_LEDGER_RELATIVE_DEFECT = 1.0e-3
MAXIMUM_TRACE_RECONSTRUCTION_DEFECT = 1.0e-9
TEMPORAL_UNCERTAINTY_GATE = 2.5e-2
TEMPORAL_RELATIVE_UNCERTAINTY = 0.10
TEMPORAL_RELATIVE_FLOOR = 0.10
HEALING_SPREAD_GATE = 0.10
MINIMUM_AUXILIARY_EFOLDS = 2.0
MAXIMUM_RANK_ONE_SECONDARY_RATIO = 0.10
MAXIMUM_SECONDARY_INTERFACE_RATIO = 0.25
MINIMUM_SIGNIFICANT_SLOW_RATE_SPREAD = 0.10
RATE_FIBER_MULTIPLIERS = (5.0e-4, 1.0e-3, 2.0e-3)
RATE_FIBER_CONFIRMATION_MULTIPLIER = 1.0e-3


def _normalized_mje_interface_half_difference(
    plus_fluxes: np.ndarray,
    minus_fluxes: np.ndarray,
    interface_flux_scales: np.ndarray,
) -> np.ndarray:
    plus = np.asarray(plus_fluxes, dtype=float)
    minus = np.asarray(minus_fluxes, dtype=float)
    if plus.shape != minus.shape or plus.ndim < 2 or plus.shape[-1] < 4:
        raise ValueError(
            "interface flux histories must have matching "
            "(..., face, field) shapes"
        )
    scales = np.asarray(interface_flux_scales, dtype=float).reshape(
        plus.shape[-2],
        len(MJE_COMPONENTS),
    )
    return (
        0.5 * (plus - minus)[..., MJE_COMPONENTS]
        / (INTERFACE_FLUX_RELATIVE_GATE * scales)
    )


def _flat_interface_mje_half_difference(
    plus_fluxes: np.ndarray,
    minus_fluxes: np.ndarray,
) -> np.ndarray:
    plus = np.asarray(plus_fluxes, dtype=float)
    minus = np.asarray(minus_fluxes, dtype=float)
    if (
        plus.shape != minus.shape
        or plus.ndim != 1
        or plus.size % len(MJE_COMPONENTS)
        or np.any(~np.isfinite(plus))
        or np.any(~np.isfinite(minus))
    ):
        raise ValueError("flat interface M/J/E flux outputs are invalid")
    return (0.5 * (plus - minus)).reshape(-1, len(MJE_COMPONENTS))


def _continuation_reference_window(
    reference_states: np.ndarray,
    *,
    restart_elapsed_time: float,
    timestep: float,
    replay_steps: int,
) -> np.ndarray:
    reference = np.asarray(reference_states, dtype=float)
    reference_start = int(round(restart_elapsed_time / timestep))
    reference_stop = reference_start + replay_steps + 1
    if reference_start < 0 or reference_stop > reference.shape[0]:
        raise ValueError("replay reference window falls outside the saved trajectory")
    return reference[reference_start:reference_stop]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Assemble only the committed-evidence divergence/trace audit.",
    )
    parser.add_argument(
        "--skip-fresh-rates",
        action="store_true",
        help="Development-only: omit expensive continued fresh-rate rows.",
    )
    parser.add_argument(
        "--skip-rate-fiber",
        action="store_true",
        help="Development-only: omit the new slow-rate fiber search.",
    )
    parser.add_argument(
        "--skip-continuation-replay",
        action="store_true",
        help="Development-only: omit split replay of N64 coarse continuation.",
    )
    parser.add_argument(
        "--include-n128-rate-fiber",
        action="store_true",
        help="Confirm the decisive N64 slow-rate direction at N128.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute restart and trajectory caches.",
    )
    parser.add_argument(
        "--trajectory-only",
        choices=(
            "n64-coarse-minus",
            "n64-coarse-plus",
            "n64-fine-minus",
            "n64-fine-plus",
            "n64-replay-minus",
            "n64-replay-plus",
        ),
        default=None,
        help="Populate exactly one continued-trajectory cache and exit.",
    )
    parser.add_argument(
        "--rate-fiber-only",
        action="store_true",
        help="Populate the restartable slow-rate fiber cache and exit.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


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


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_parent_evidence() -> tuple[dict, dict, dict, dict]:
    for path in (
        PARENT_JSON,
        PARENT_ARRAYS,
        PARENT_FIBER_JSON,
        PARENT_FIBER_ARRAYS,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required WP10c8q parent is missing: {path}")
    healing = json.loads(PARENT_JSON.read_text(encoding="utf-8"))
    fiber = json.loads(PARENT_FIBER_JSON.read_text(encoding="utf-8"))
    if not (
        healing.get("work_package") == "WP10c8p"
        and healing.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and fiber.get("work_package") == "WP10c8o"
        and fiber.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_FIBER_ARRAYS)
    ):
        raise RuntimeError("WP10c8q parent evidence failed provenance checks")
    healing_arrays = dict(np.load(PARENT_ARRAYS, allow_pickle=False))
    fiber_arrays = dict(np.load(PARENT_FIBER_ARRAYS, allow_pickle=False))
    return healing, healing_arrays, fiber, fiber_arrays


def _runtime_contracts():
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )
    fiber = json.loads(PARENT_FIBER_JSON.read_text(encoding="utf-8"))
    with np.load(PARENT_FIBER_ARRAYS, allow_pickle=False) as arrays:
        contracts = {
            n_cells: wp10c8p._mesh_contract(
                n_cells=n_cells,
                parent=fiber,
                parent_arrays=arrays,
                initial_by_mesh=initial_by_mesh,
                vectors_by_mesh=vectors_by_mesh,
                shell_edges_rg=shell_edges_rg,
            )
            for n_cells in (64, 128)
        }
    return initial_by_mesh, vectors_by_mesh, state_provenance, contracts


def _signed_relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    one = np.asarray(left, dtype=float)
    two = np.asarray(right, dtype=float)
    scale = np.maximum(np.maximum(np.abs(one), np.abs(two)), 1.0)
    return float(np.max(np.abs(one - two) / scale))


def _coordinate_index(
    names: tuple[str, ...],
    shell: int,
    suffix: str,
) -> int:
    return names.index(f"shell_{shell}_{suffix}")


def _existing_divergence_audit(
    *,
    contract: dict,
    arrays: dict[str, np.ndarray],
    n_cells: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    prefix = f"n{n_cells}_fine_"
    minus_prefix = prefix + "minus_"
    plus_prefix = prefix + "plus_"
    times = np.asarray(arrays[minus_prefix + "times"], dtype=float)
    if not np.array_equal(times, arrays[plus_prefix + "times"]):
        raise RuntimeError("committed plus/minus time grids differ")
    minus_flux = np.asarray(arrays[minus_prefix + "macro_fluxes"], dtype=float)
    plus_flux = np.asarray(arrays[plus_prefix + "macro_fluxes"], dtype=float)
    face_half = 0.5 * (plus_flux - minus_flux)
    internal_boundary_rate = causal_internal_face_boundary_rates(face_half)
    internal_boundary_transport = causal_cumulative_trapezoid(
        times,
        internal_boundary_rate,
    )
    shell_actual = 0.5 * (
        np.asarray(arrays[plus_prefix + "shell_actual_storage"], dtype=float)
        - np.asarray(arrays[minus_prefix + "shell_actual_storage"], dtype=float)
    )
    shell_vertical = 0.5 * (
        np.asarray(arrays[plus_prefix + "shell_vertical_storage"], dtype=float)
        - np.asarray(arrays[minus_prefix + "shell_vertical_storage"], dtype=float)
    )
    shell_boundary = 0.5 * (
        np.asarray(arrays[plus_prefix + "shell_boundary_transport"], dtype=float)
        - np.asarray(arrays[minus_prefix + "shell_boundary_transport"], dtype=float)
    )
    shell_source = 0.5 * (
        np.asarray(arrays[plus_prefix + "shell_integrated_sources"], dtype=float)
        - np.asarray(arrays[minus_prefix + "shell_integrated_sources"], dtype=float)
    )
    shell_defect = shell_actual + shell_vertical + shell_boundary - shell_source
    shell_scale = (
        np.abs(shell_actual)
        + np.abs(shell_vertical)
        + np.abs(shell_boundary)
        + np.abs(shell_source)
    )
    shell_relative_defect = np.abs(shell_defect) / np.maximum(
        shell_scale,
        np.finfo(float).tiny,
    )
    physical_boundary_transport = shell_boundary - internal_boundary_transport

    coordinate_names = tuple(
        str(value)
        for value in np.asarray(arrays[minus_prefix + "coordinate_names"])
    )
    coordinate_scales = np.asarray(
        arrays[minus_prefix + "coordinate_scales"], dtype=float
    )
    shell_scales = wp10c8p._shell_scale_matrix(
        coordinate_names,
        coordinate_scales,
        shell_actual.shape[1],
    )
    loading_time = causal_five_field_loading_time(
        contract["context"],
        contract["anchor_vector"],
    )
    final_slow_storage_change = (
        loading_time
        * shell_actual[-1]
        / times[-1]
        / shell_scales
    )
    final_slow_boundary_change = (
        loading_time
        * shell_boundary[-1]
        / times[-1]
        / shell_scales
    )
    minus_rates = np.asarray(
        arrays[minus_prefix + "normalized_coordinate_rates"], dtype=float
    )
    plus_rates = np.asarray(
        arrays[plus_prefix + "normalized_coordinate_rates"], dtype=float
    )
    windowed_rate_half = 0.5 * (plus_rates - minus_rates)
    slow_rate_half = (
        loading_time / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
    ) * windowed_rate_half

    interface_scales = np.asarray(
        contract["interface_flux_scales"], dtype=float
    ).reshape(face_half.shape[1], len(MJE_COMPONENTS))
    signed_gate_normalized_faces = (
        np.take(face_half, MJE_COMPONENTS, axis=-1)
        / (INTERFACE_FLUX_RELATIVE_GATE * interface_scales[None, :, :])
    )
    incidence = np.zeros((face_half.shape[1] + 1, face_half.shape[1]))
    for face in range(face_half.shape[1]):
        incidence[face, face] = 1.0
        incidence[face + 1, face] = -1.0
    singular = np.linalg.svd(incidence, compute_uv=False)
    normalized_incidence_ratio = np.asarray(
        [
            np.linalg.norm(incidence @ signed_gate_normalized_faces[0, :, j])
            / max(
                np.linalg.norm(signed_gate_normalized_faces[0, :, j]),
                np.finfo(float).tiny,
            )
            for j in range(len(MJE_COMPONENTS))
        ]
    )

    shell_mje_slow_rates = np.zeros((5, 3), dtype=float)
    for shell in range(5):
        for column, suffix in enumerate(
            ("rest_mass", "angular_momentum", "killing_energy")
        ):
            shell_mje_slow_rates[shell, column] = slow_rate_half[
                0,
                _coordinate_index(coordinate_names, shell, suffix),
            ]
    maximum_ledger = float(
        np.max(np.take(shell_relative_defect[1:], MJE_COMPONENTS, axis=-1))
    )
    shell_frozen_scale_defect = np.abs(shell_defect) / shell_scales[None, :, :]
    maximum_frozen_scale_ledger = float(
        np.max(
            np.take(
                shell_frozen_scale_defect[1:],
                MJE_COMPONENTS,
                axis=-1,
            )
        )
    )
    maximum_slow_rate = float(np.max(np.abs(slow_rate_half[0])))
    summary = {
        "n_cells": n_cells,
        "loading_time_seconds": loading_time,
        "incidence_shape": incidence.shape,
        "incidence_rank": int(np.linalg.matrix_rank(incidence)),
        "incidence_singular_values": singular,
        "initial_normalized_incidence_ratios_mje": normalized_incidence_ratio,
        "initial_signed_gate_normalized_internal_faces_mje": (
            signed_gate_normalized_faces[0]
        ),
        "final_shell_actual_change_over_scale_mje": (
            np.take(shell_actual[-1], MJE_COMPONENTS, axis=-1)
            / np.take(shell_scales, MJE_COMPONENTS, axis=-1)
        ),
        "final_shell_boundary_change_over_scale_mje": (
            np.take(shell_boundary[-1], MJE_COMPONENTS, axis=-1)
            / np.take(shell_scales, MJE_COMPONENTS, axis=-1)
        ),
        "final_slow_storage_change_mje": final_slow_storage_change[
            :, MJE_COMPONENTS
        ],
        "final_slow_boundary_change_mje": final_slow_boundary_change[
            :, MJE_COMPONENTS
        ],
        "initial_shell_mje_slow_rate_half_difference": shell_mje_slow_rates,
        "initial_maximum_34_coordinate_slow_rate_half_difference": (
            maximum_slow_rate
        ),
        "initial_controlling_slow_rate_coordinate": coordinate_names[
            int(np.argmax(np.abs(slow_rate_half[0])))
        ],
        "maximum_physical_mje_shell_ledger_relative_defect": maximum_ledger,
        "maximum_physical_mje_shell_frozen_scale_defect": (
            maximum_frozen_scale_ledger
        ),
        "maximum_external_physical_boundary_transport_fraction": float(
            np.max(
                np.abs(
                    np.take(
                        physical_boundary_transport,
                        MJE_COMPONENTS,
                        axis=-1,
                    )
                )
                / np.maximum(
                    np.abs(
                        np.take(shell_boundary, MJE_COMPONENTS, axis=-1)
                    ),
                    1.0,
                )
            )
        ),
        "flux_gauge_hypothesis_rejected_for_decisive_pair": bool(
            np.linalg.matrix_rank(incidence) == face_half.shape[1]
            and np.min(normalized_incidence_ratio) > 0.1
            and maximum_frozen_scale_ledger <= MAXIMUM_LEDGER_RELATIVE_DEFECT
        ),
        "passed": bool(
            maximum_frozen_scale_ledger <= MAXIMUM_LEDGER_RELATIVE_DEFECT
        ),
    }
    audit_arrays = {
        "times": times,
        "coordinate_names": np.asarray(coordinate_names, dtype="U"),
        "coordinate_scales": coordinate_scales,
        "shell_scales": shell_scales,
        "internal_face_flux_half_difference": face_half,
        "internal_boundary_rate_half_difference": internal_boundary_rate,
        "internal_boundary_transport_half_difference": (
            internal_boundary_transport
        ),
        "physical_boundary_transport_half_difference": (
            physical_boundary_transport
        ),
        "shell_actual_half_difference": shell_actual,
        "shell_vertical_half_difference": shell_vertical,
        "shell_boundary_half_difference": shell_boundary,
        "shell_source_half_difference": shell_source,
        "shell_ledger_defect": shell_defect,
        "shell_ledger_relative_defect": shell_relative_defect,
        "shell_ledger_frozen_scale_defect": shell_frozen_scale_defect,
        "signed_gate_normalized_internal_faces_mje": (
            signed_gate_normalized_faces
        ),
        "windowed_coordinate_rate_half_difference": windowed_rate_half,
        "slow_coordinate_rate_half_difference": slow_rate_half,
        "incidence_matrix": incidence,
    }
    return summary, audit_arrays


def _trace_attribution(
    *,
    contract: dict,
    arrays: dict[str, np.ndarray],
    n_cells: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    prefix = f"n{n_cells}_fine_"
    minus_primitives = np.asarray(
        arrays[prefix + "minus_output_primitives"], dtype=float
    )[0]
    plus_primitives = np.asarray(
        arrays[prefix + "plus_output_primitives"], dtype=float
    )[0]
    reconstruction_minus = causal_five_field_reconstruct_face_charts(
        contract["context"], minus_primitives
    )
    reconstruction_plus = causal_five_field_reconstruct_face_charts(
        contract["context"], plus_primitives
    )
    geometry = causal_mesh_coincident_moment_shells(
        contract["context"], contract["shell_edges_rg"]
    )
    face = int(geometry.edge_indices[INTERFACE_INDEX])
    minus_trace = np.concatenate(
        (
            reconstruction_minus.left_face_charts[face],
            reconstruction_minus.right_face_charts[face],
        )
    )
    plus_trace = np.concatenate(
        (
            reconstruction_plus.left_face_charts[face],
            reconstruction_plus.right_face_charts[face],
        )
    )

    def perfect_flux(trace: np.ndarray) -> np.ndarray:
        values = np.asarray(trace, dtype=float)
        return C * _central_perfect_flux_from_validated_face_charts(
            contract["context"], face, values[:5], values[5:]
        )

    decomposition = causal_path_integrated_component_decomposition(
        perfect_flux,
        minus_trace,
        plus_trace,
        quadrature_order=16,
        finite_difference_relative_step=2.0e-6,
    )
    committed_difference = np.asarray(
        arrays[prefix + "plus_interface4_perfect_flux"], dtype=float
    )[0] - np.asarray(
        arrays[prefix + "minus_interface4_perfect_flux"], dtype=float
    )[0]
    endpoint_defect = _signed_relative_defect(
        decomposition.endpoint_difference,
        committed_difference,
    )
    trace_names = tuple(
        f"{side}_{name}"
        for side in ("left", "right")
        for name in CAUSAL_FIVE_FIELD_PRIMITIVE_NAMES
    )
    component_norms = np.linalg.norm(
        np.take(
            decomposition.component_contributions,
            MJE_COMPONENTS,
            axis=-1,
        ),
        axis=1,
    )
    control = int(np.argmax(component_norms))
    grouped_by_primitive = (
        decomposition.component_contributions[:5]
        + decomposition.component_contributions[5:]
    )
    summary = {
        "n_cells": n_cells,
        "face_index": face,
        "trace_component_names": trace_names,
        "endpoint_production_perfect_flux_defect": endpoint_defect,
        "maximum_path_reconstruction_relative_defect": (
            decomposition.maximum_reconstruction_relative_defect
        ),
        "maximum_path_quadrature_relative_defect": (
            decomposition.maximum_quadrature_relative_defect
        ),
        "controlling_trace_component": trace_names[control],
        "left_group_mje_contribution": np.sum(
            np.take(
                decomposition.component_contributions[:5],
                MJE_COMPONENTS,
                axis=-1,
            ),
            axis=0,
        ),
        "right_group_mje_contribution": np.sum(
            np.take(
                decomposition.component_contributions[5:],
                MJE_COMPONENTS,
                axis=-1,
            ),
            axis=0,
        ),
        "primitive_group_mje_contributions": np.take(
            grouped_by_primitive, MJE_COMPONENTS, axis=-1
        ),
        "passed": bool(
            endpoint_defect <= MAXIMUM_TRACE_RECONSTRUCTION_DEFECT
            and decomposition.maximum_reconstruction_relative_defect
            <= MAXIMUM_TRACE_RECONSTRUCTION_DEFECT
        ),
    }
    attribution_arrays = {
        "minus_trace": minus_trace,
        "plus_trace": plus_trace,
        "trace_component_names": np.asarray(trace_names, dtype="U"),
        "endpoint_difference": decomposition.endpoint_difference,
        "committed_endpoint_difference": committed_difference,
        "component_contributions": decomposition.component_contributions,
        "reconstructed_difference": decomposition.reconstructed_difference,
        "primitive_group_contributions": grouped_by_primitive,
    }
    return summary, attribution_arrays


def _cached_trace_attribution(
    n_cells: int,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    """Reuse an exact trace audit when its production dependencies agree."""

    if not DEFAULT_OUTPUT.exists() or not DEFAULT_ARRAYS.exists():
        return None
    evidence = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    dae_path = ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py"
    healing_path = ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_healing.py"
    source_hashes = evidence.get("source_hashes", {})
    if not (
        evidence.get("work_package") == WORK_PACKAGE
        and evidence.get("authorization", {}).get("wp10c8p_arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and source_hashes.get(_relative(dae_path)) == _sha256(dae_path)
        and source_hashes.get(_relative(healing_path)) == _sha256(healing_path)
        and str(n_cells) in evidence.get("existing_evidence", {})
    ):
        return None
    prefix = f"n{n_cells}_trace_attribution_"
    with np.load(DEFAULT_ARRAYS, allow_pickle=False) as arrays:
        values = {
            name.removeprefix(prefix): np.asarray(arrays[name])
            for name in arrays.files
            if name.startswith(prefix)
        }
    if not values:
        return None
    return (
        evidence["existing_evidence"][str(n_cells)]["trace_attribution"],
        values,
    )


def _parent_trajectory(
    contract: dict,
    *,
    resolution: str,
    side: str,
) -> dict:
    subdivisions = (
        wp10c8p.COARSE_SUBDIVISIONS
        if resolution == "coarse"
        else wp10c8p.FINE_SUBDIVISIONS
    )
    return wp10c8p._run_or_load_trajectory(
        context=contract["context"],
        initial_vector=contract[f"{side}_vector"],
        n_cells=64,
        resolution=resolution,
        side=side,
        subdivisions=subdivisions,
        force=False,
    )


def _restart_path(resolution: str, side: str, label: str) -> Path:
    return CHECKPOINT_DIRECTORY / f"N064_{resolution}_{side}_{label}_restart.npz"


def _capture_parent_history(
    *,
    contract: dict,
    resolution: str,
    side: str,
    force: bool,
) -> tuple[CausalFiveFieldBDFRestart, dict]:
    parent = _parent_trajectory(contract, resolution=resolution, side=side)
    subdivisions = (
        wp10c8p.COARSE_SUBDIVISIONS
        if resolution == "coarse"
        else wp10c8p.FINE_SUBDIVISIONS
    )
    timestep = PARENT_DURATION_SECONDS / subdivisions
    path = _restart_path(resolution, side, "t0p025")
    provenance = {
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "purpose": "exact_wp10c8p_history_replay",
        "resolution": resolution,
        "side": side,
        "parent_trajectory_sha256": parent["sha256"],
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
        "initial_state_sha256": _array_sha256(contract[f"{side}_vector"]),
    }
    if path.exists() and not force:
        restart = load_causal_five_field_bdf_restart(
            path, contract["context"]
        )
        if not (
            restart.provenance == provenance
            and restart.elapsed_time == PARENT_DURATION_SECONDS
            and restart.history.previous_timestep_seconds == timestep
            and np.array_equal(restart.state_vector, parent["states"][-1])
        ):
            raise RuntimeError(f"stale WP10c8q restart: {path}")
        return restart, {
            "path": _relative(path),
            "sha256": _sha256(path),
            "cached": True,
            "bitwise_parent_replay": True,
        }

    snapshots = [np.asarray(contract[f"{side}_vector"], dtype=float).copy()]

    def progress(_completed, _total, state, _history) -> None:
        snapshots.append(np.asarray(state, dtype=float).copy())
        print(
            f"WP10c8q history N64 {resolution} {side}: "
            f"step {_completed}/{_total}",
            flush=True,
        )

    result = evolve_causal_five_field_fixed_bdf2(
        contract["context"],
        contract[f"{side}_vector"],
        np.zeros_like(contract[f"{side}_vector"]),
        timestep,
        PARENT_DURATION_SECONDS,
        subdivisions,
        wp10c8p._step_config(),
        startup_with_bdf1=True,
        progress=progress,
    )
    states = np.asarray(snapshots, dtype=float)
    equal = bool(
        result.passed
        and result.history is not None
        and states.shape == parent["states"].shape
        and np.array_equal(states, parent["states"])
    )
    if not equal or result.history is None:
        raise RuntimeError("WP10c8q replay did not reproduce WP10c8p bitwise")
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
    save_causal_five_field_bdf_restart(path, contract["context"], restart)
    loaded = load_causal_five_field_bdf_restart(path, contract["context"])
    if not causal_five_field_bdf_restarts_equal(restart, loaded):
        raise RuntimeError("saved WP10c8q restart does not round-trip")
    return loaded, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "cached": False,
        "bitwise_parent_replay": equal,
    }


def _continued_trajectory_path(resolution: str, side: str) -> Path:
    return CHECKPOINT_DIRECTORY / (
        f"N064_{resolution}_{side}_continued_t0p125.npz"
    )


def _continuation_segment_path(
    resolution: str,
    side: str,
    target_time_seconds: float,
) -> Path:
    label = f"t{target_time_seconds:.3f}".replace(".", "p")
    return CHECKPOINT_DIRECTORY / (
        f"N064_{resolution}_{side}_continued_{label}.npz"
    )


def _merge_fixed_result_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one fixed-trajectory result row is required")
    maximum_keys = (
        "maximum_scaled_residual",
        "maximum_scaled_algebraic_residual",
        "maximum_scaled_primitive_change",
        "maximum_scaled_total_change",
        "maximum_discrete_ledger_relative_defect",
        "maximum_linear_residual",
        "maximum_newton_iterations",
    )
    sum_keys = (
        "subdivisions",
        "completed_steps",
        "bdf1_steps",
        "bdf2_steps",
        "function_evaluations",
        "jacobian_evaluations",
        "newton_iterations",
        "wall_seconds",
    )
    ledger_keys = (
        "actual_conserved_storage",
        "actual_vertical_storage",
        "trapezoidal_boundary_transport",
        "trapezoidal_endogenous_source",
        "exact_prescribed_stream_source",
        "closure_defect",
    )
    ledger = {
        key: np.sum(
            np.asarray(
                [row["cumulative_physical_ledger"][key] for row in rows],
                dtype=float,
            ),
            axis=0,
        )
        for key in ledger_keys
    }
    denominator = np.sum(
        np.asarray(
            [
                np.abs(ledger["actual_conserved_storage"]),
                np.abs(ledger["actual_vertical_storage"]),
                np.abs(ledger["trapezoidal_boundary_transport"]),
                np.abs(ledger["trapezoidal_endogenous_source"]),
                np.abs(ledger["exact_prescribed_stream_source"]),
            ]
        ),
        axis=0,
    )
    component_defects = np.abs(ledger["closure_defect"]) / np.maximum(
        denominator,
        np.finfo(float).tiny,
    )
    merged = {
        "passed": bool(all(row["passed"] for row in rows)),
        "message": "segmented fixed BDF2 continuation completed",
        "timestep_seconds": rows[0]["timestep_seconds"],
        "state_gates": rows[-1]["state_gates"],
        "cumulative_physical_ledger": ledger,
        "cumulative_physical_ledger_relative_defect": float(
            np.max(component_defects)
        ),
        "cumulative_physical_component_relative_defects": component_defects,
    }
    merged.update(
        {key: max(row[key] for row in rows) for key in maximum_keys}
    )
    merged.update({key: sum(row[key] for row in rows) for key in sum_keys})
    return merged


def _run_or_load_continued_trajectory(
    *,
    contract: dict,
    resolution: str,
    side: str,
    force: bool,
) -> dict:
    parent = _parent_trajectory(contract, resolution=resolution, side=side)
    total_subdivisions = (
        COARSE_SUBDIVISIONS if resolution == "coarse" else FINE_SUBDIVISIONS
    )
    timestep = TARGET_DURATION_SECONDS / total_subdivisions
    path = _continued_trajectory_path(resolution, side)
    final_restart_path = _restart_path(resolution, side, "t0p125")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "resolution": resolution,
        "side": side,
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "total_subdivisions": total_subdivisions,
        "timestep_seconds": timestep,
        "parent_trajectory_sha256": parent["sha256"],
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
        "continuation": "exact_history_bdf2_without_new_bdf1_startup",
    }
    if path.exists() and final_restart_path.exists() and not force:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            states = np.asarray(source["states"], dtype=float)
        if not all(metadata.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale WP10c8q trajectory: {path}")
        restart = load_causal_five_field_bdf_restart(
            final_restart_path, contract["context"]
        )
        if not np.array_equal(restart.state_vector, states[-1]):
            raise RuntimeError("WP10c8q cached final history/state differ")
        return {
            "path": path,
            "sha256": _sha256(path),
            "restart_path": final_restart_path,
            "restart_sha256": _sha256(final_restart_path),
            "states": states,
            "summary": metadata["summary"],
            "history_replay": metadata["history_replay"],
            "cached": True,
        }

    restart, history_replay = _capture_parent_history(
        contract=contract,
        resolution=resolution,
        side=side,
        force=force,
    )
    states_parts = [np.asarray(parent["states"], dtype=float)]
    segment_rows: list[dict] = []
    segment_provenance: list[dict] = []
    current_restart = restart
    current_time = PARENT_DURATION_SECONDS
    for target_time in (0.05, 0.10, 0.125):
        segment_steps = int(round((target_time - current_time) / timestep))
        if segment_steps < 1:
            raise RuntimeError("WP10c8q continuation segment is empty")
        segment_duration = segment_steps * timestep
        segment_path = _continuation_segment_path(
            resolution,
            side,
            target_time,
        )
        segment_restart_path = _restart_path(
            resolution,
            side,
            f"segment_t{target_time:.3f}".replace(".", "p"),
        )
        segment_expected = {
            **expected,
            "segment_start_seconds": current_time,
            "segment_target_seconds": target_time,
            "segment_subdivisions": segment_steps,
        }
        if segment_path.exists() and segment_restart_path.exists() and not force:
            with np.load(segment_path, allow_pickle=False) as source:
                segment_metadata = json.loads(
                    str(source["metadata_json"].item())
                )
                segment_states = np.asarray(source["states"], dtype=float)
            if not all(
                segment_metadata.get(key) == value
                for key, value in segment_expected.items()
            ):
                raise RuntimeError(
                    f"stale WP10c8q continuation segment: {segment_path}"
                )
            current_restart = load_causal_five_field_bdf_restart(
                segment_restart_path,
                contract["context"],
            )
            if not np.array_equal(
                current_restart.state_vector,
                segment_states[-1],
            ):
                raise RuntimeError("cached segment restart/state differ")
            segment_row = segment_metadata["summary"]
            cached = True
        else:
            segment_snapshots: list[np.ndarray] = []

            def progress(_completed, _total, state, _history) -> None:
                segment_snapshots.append(np.asarray(state, dtype=float).copy())
                print(
                    f"WP10c8q N64 {resolution} {side}: "
                    f"segment {target_time:.3f} step {_completed}/{_total}",
                    flush=True,
                )

            started = time.perf_counter()
            result = evolve_causal_five_field_fixed_bdf2(
                contract["context"],
                current_restart.state_vector,
                current_restart.history.previous_physical_increment,
                timestep,
                segment_duration,
                segment_steps,
                wp10c8p._step_config(),
                startup_with_bdf1=False,
                initial_history=current_restart.history,
                progress=progress,
            )
            wall_seconds = time.perf_counter() - started
            if not result.passed or result.history is None:
                raise RuntimeError(
                    f"WP10c8q {resolution} {side} continuation failed"
                )
            segment_states = np.asarray(segment_snapshots, dtype=float)
            segment_row = wp10c8p._result_row(result, wall_seconds)
            segment_restart = CausalFiveFieldBDFRestart(
                state_vector=result.state_vector,
                history=result.history,
                elapsed_time=target_time,
                dt_next=timestep,
                next_order=2,
                accepted_steps=int(round(target_time / timestep)),
                rejected_attempts=0,
                provenance={
                    **segment_expected,
                    "purpose": "exact_wp10c8q_segment_history",
                    "segment_state_sha256": _array_sha256(segment_states),
                },
            )
            save_causal_five_field_bdf_restart(
                segment_restart_path,
                contract["context"],
                segment_restart,
            )
            current_restart = load_causal_five_field_bdf_restart(
                segment_restart_path,
                contract["context"],
            )
            segment_metadata = {
                **segment_expected,
                "summary": _plain(segment_row),
            }
            np.savez_compressed(
                segment_path,
                states=segment_states,
                metadata_json=np.asarray(
                    json.dumps(
                        segment_metadata,
                        sort_keys=True,
                        allow_nan=False,
                    )
                ),
            )
            cached = False
        states_parts.append(segment_states)
        segment_rows.append(segment_row)
        segment_provenance.append(
            {
                "target_time_seconds": target_time,
                "path": _relative(segment_path),
                "sha256": _sha256(segment_path),
                "restart_path": _relative(segment_restart_path),
                "restart_sha256": _sha256(segment_restart_path),
                "cached": cached,
            }
        )
        current_time = target_time

    states = np.concatenate(states_parts, axis=0)
    if states.shape[0] != total_subdivisions + 1:
        raise RuntimeError("WP10c8q continued state count is inconsistent")
    summary = _merge_fixed_result_rows(segment_rows)
    final_provenance = {
        **expected,
        "purpose": "exact_wp10c8q_final_history",
        "trajectory_state_sha256": _array_sha256(states),
        "segments": segment_provenance,
    }
    final_restart = CausalFiveFieldBDFRestart(
        state_vector=current_restart.state_vector,
        history=current_restart.history,
        elapsed_time=TARGET_DURATION_SECONDS,
        dt_next=timestep,
        next_order=2,
        accepted_steps=total_subdivisions,
        rejected_attempts=0,
        provenance=final_provenance,
    )
    save_causal_five_field_bdf_restart(
        final_restart_path, contract["context"], final_restart
    )
    metadata = {
        **expected,
        "summary": _plain(summary),
        "history_replay": history_replay,
        "segments": segment_provenance,
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
        "path": path,
        "sha256": _sha256(path),
        "restart_path": final_restart_path,
        "restart_sha256": _sha256(final_restart_path),
        "states": states,
        "summary": summary,
        "history_replay": history_replay,
        "cached": False,
    }


def _split_continuation_replay(
    *,
    contract: dict,
    side: str,
    reference_states: np.ndarray,
    force: bool,
) -> dict:
    path = CHECKPOINT_DIRECTORY / f"N064_coarse_{side}_continuation_replay.json"
    segment_restart_path = _restart_path(
        "coarse",
        side,
        "segment_t0p100",
    )
    if not segment_restart_path.exists():
        raise RuntimeError(
            "WP10c8q continuation replay requires the exact t=0.10 restart"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "side": side,
        "reference_states_sha256": _array_sha256(reference_states),
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
        "replay_start_seconds": 0.10,
        "replay_restart_sha256": _sha256(segment_restart_path),
    }
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            return {**cached["result"], "path": _relative(path), "cached": True}
        raise RuntimeError(f"stale WP10c8q continuation replay: {path}")
    restart = load_causal_five_field_bdf_restart(
        segment_restart_path,
        contract["context"],
    )
    timestep = float(restart.history.previous_timestep_seconds)
    replay_duration = TARGET_DURATION_SECONDS - float(restart.elapsed_time)
    replay_steps = int(round(replay_duration / timestep))
    if not np.isclose(
        replay_steps * timestep,
        replay_duration,
        rtol=0.0,
        atol=16.0 * np.finfo(float).eps * TARGET_DURATION_SECONDS,
    ):
        raise RuntimeError(
            "WP10c8q replay duration is not commensurate with saved history"
        )
    exact_replay_duration = replay_steps * timestep
    replay_states = [np.asarray(restart.state_vector, dtype=float).copy()]

    def progress(_completed, _total, state, _history) -> None:
        replay_states.append(np.asarray(state, dtype=float).copy())
        print(
            f"WP10c8q N64 coarse replay {side}: "
            f"step {_completed}/{_total}",
            flush=True,
        )

    replay_result = evolve_causal_five_field_fixed_bdf2(
        contract["context"],
        restart.state_vector,
        restart.history.previous_physical_increment,
        timestep,
        exact_replay_duration,
        replay_steps,
        wp10c8p._step_config(),
        startup_with_bdf1=False,
        initial_history=restart.history,
        progress=progress,
    )
    reference_continuation = _continuation_reference_window(
        reference_states,
        restart_elapsed_time=float(restart.elapsed_time),
        timestep=timestep,
        replay_steps=replay_steps,
    )
    replay = np.asarray(replay_states, dtype=float)
    equal = bool(
        replay_result.passed
        and replay.shape == reference_continuation.shape
        and np.array_equal(replay, reference_continuation)
    )
    result = {
        "passed": equal,
        "maximum_absolute_state_difference": (
            float(np.max(np.abs(replay - reference_continuation)))
            if replay.shape == reference_continuation.shape
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {**expected, "result": result},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**result, "path": _relative(path), "cached": False}


def _run_continuation_campaign(
    *,
    contract: dict,
    force: bool,
    compute_fresh_rates: bool,
    run_replay: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = contract["context"]
    baseline = causal_five_field_observable_snapshot(
        context,
        contract["anchor_vector"],
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    radius_rg = context.grid.centers / context.grid.gravitational_radius
    edges_rg = context.grid.edges / context.grid.gravitational_radius
    _common, interpolation = wp10c8i._common_log_h_interpolation(
        radius_rg, edges_rg
    )
    rate_cache = (
        wp10c8p._persistent_fresh_rate_cache(64)
        if compute_fresh_rates
        else {}
    )
    trajectories = {}
    diagnostics = {}
    all_arrays: dict[str, np.ndarray] = {}
    for resolution, subdivisions in (
        ("coarse", COARSE_SUBDIVISIONS),
        ("fine", FINE_SUBDIVISIONS),
    ):
        for side in ("minus", "plus"):
            trajectory = _run_or_load_continued_trajectory(
                contract=contract,
                resolution=resolution,
                side=side,
                force=force,
            )
            trajectories[f"{resolution}_{side}"] = trajectory
            summary, values = wp10c8p._trajectory_diagnostics(
                context=context,
                states=trajectory["states"],
                subdivisions=subdivisions,
                shell_edges_rg=contract["shell_edges_rg"],
                baseline_snapshot=baseline,
                anchor_interface_scales=contract["interface_flux_scales"],
                coordinate_names=contract["coordinate_names"],
                coordinate_scales=contract["coordinate_scales"],
                primitive_scales=contract["primitive_scales"],
                conservation_scales=contract["conservation_scales"],
                common_interpolation=interpolation,
                compute_fresh_rates=compute_fresh_rates,
                rate_cache=rate_cache,
                duration_seconds=TARGET_DURATION_SECONDS,
                output_offsets_seconds=OUTPUT_OFFSETS_SECONDS,
            )
            diagnostics[f"{resolution}_{side}"] = summary
            all_arrays.update(
                {
                    f"{resolution}_{side}_{name}": value
                    for name, value in values.items()
                }
            )

    pair_rows = {}
    pair_arrays = {}
    for resolution in ("coarse", "fine"):
        summary, values = wp10c8p._pair_diagnostics(
            minus={
                name.removeprefix(f"{resolution}_minus_"): value
                for name, value in all_arrays.items()
                if name.startswith(f"{resolution}_minus_")
            },
            plus={
                name.removeprefix(f"{resolution}_plus_"): value
                for name, value in all_arrays.items()
                if name.startswith(f"{resolution}_plus_")
            },
            coordinate_scales=contract["coordinate_scales"],
            coordinate_names=contract["coordinate_names"],
        )
        pair_rows[resolution] = summary
        pair_arrays[resolution] = values
        all_arrays.update(
            {f"{resolution}_pair_{name}": value for name, value in values.items()}
        )

    uncertainty, upper = causal_refined_spread_upper_bound(
        pair_arrays["coarse"]["full_spreads"],
        pair_arrays["fine"]["full_spreads"],
    )
    relative_mask = pair_arrays["fine"]["full_spreads"] >= TEMPORAL_RELATIVE_FLOOR
    temporal_passed = bool(
        np.max(uncertainty) <= TEMPORAL_UNCERTAINTY_GATE
        and np.all(
            uncertainty[relative_mask]
            <= TEMPORAL_RELATIVE_UNCERTAINTY
            * pair_arrays["fine"]["full_spreads"][relative_mask]
        )
    )
    initial = upper[0]
    final = upper[-1]
    significant = initial > HEALING_SPREAD_GATE
    late_growth = np.diff(upper[-3:, significant], axis=0)
    late_growth_uncertainty = (
        uncertainty[-2:, significant]
        + uncertainty[-3:-1, significant]
    )
    no_late_regrowth = bool(
        np.all(late_growth <= late_growth_uncertainty)
    )
    healing_passed = bool(
        np.any(significant)
        and np.all(final[significant] <= HEALING_SPREAD_GATE)
        and np.all(final[significant] <= 0.5 * initial[significant])
        and no_late_regrowth
    )
    normalized_transport = (
        pair_arrays["fine"]["interface4_transport_half_difference"][
            :, MJE_COMPONENTS
        ]
        / (
            INTERFACE_FLUX_RELATIVE_GATE
            * np.asarray(contract["interface_flux_scales"], dtype=float)
            .reshape(4, 3)[INTERFACE_INDEX - 1]
        )
    )
    normalized_all_interface_transport = (
        _normalized_mje_interface_half_difference(
            all_arrays["fine_plus_macro_fluxes"],
            all_arrays["fine_minus_macro_fluxes"],
            contract["interface_flux_scales"],
        )
    )
    per_interface_maximum = np.max(
        np.abs(normalized_all_interface_transport),
        axis=(0, 2),
    )
    controlling_interface = int(np.argmax(per_interface_maximum)) + 1
    secondary_interface_ratio = float(
        np.max(
            np.delete(per_interface_maximum, INTERFACE_INDEX - 1)
        )
        / max(
            per_interface_maximum[INTERFACE_INDEX - 1],
            np.finfo(float).tiny,
        )
    )
    interface_localized = bool(
        controlling_interface == INTERFACE_INDEX
        and secondary_interface_ratio <= MAXIMUM_SECONDARY_INTERFACE_RATIO
    )
    transport_rank = causal_transport_rank_audit(
        normalized_transport,
        maximum_secondary_ratio=MAXIMUM_RANK_ONE_SECONDARY_RATIO,
    )
    controlling_transport = np.max(np.abs(normalized_transport), axis=1)
    e_folds = float(
        np.log(
            max(controlling_transport[0], np.finfo(float).tiny)
            / max(controlling_transport[-1], np.finfo(float).tiny)
        )
    )
    replay = {"evaluated": False, "passed": True}
    if run_replay:
        replay_rows = {
            side: _split_continuation_replay(
                contract=contract,
                side=side,
                reference_states=trajectories[f"coarse_{side}"]["states"],
                force=force,
            )
            for side in ("minus", "plus")
        }
        replay = {
            "evaluated": True,
            "sides": replay_rows,
            "passed": bool(all(row["passed"] for row in replay_rows.values())),
        }
    trajectory_passed = bool(
        all(row["summary"]["passed"] for row in trajectories.values())
    )
    diagnostic_passed = bool(
        all(
            row["maximum_physical_mje_shell_ledger_relative_defect"]
            <= MAXIMUM_LEDGER_RELATIVE_DEFECT
            and row["maximum_flux_reconstruction_defect"]
            <= wp10c8p.MAXIMUM_FLUX_RECONSTRUCTION_DEFECT
            and row["all_output_state_gates_passed"]
            and row["all_fresh_rate_audits_passed"]
            for row in diagnostics.values()
        )
    )
    binding = bool(
        trajectory_passed
        and diagnostic_passed
        and temporal_passed
        and compute_fresh_rates
        and run_replay
        and replay["passed"]
    )
    if not binding:
        classification = "development_or_numerically_inconclusive"
    elif healing_passed:
        classification = "healing_supported_through_0p125s"
    else:
        classification = "healing_not_observed_through_0p125s_only"
    row = {
        "trajectory_provenance": {
            key: {
                "path": _relative(value["path"]),
                "sha256": value["sha256"],
                "restart_path": _relative(value["restart_path"]),
                "restart_sha256": value["restart_sha256"],
                "cached": value["cached"],
                "summary": value["summary"],
                "history_replay": value["history_replay"],
            }
            for key, value in trajectories.items()
        },
        "trajectory_diagnostics": diagnostics,
        "pair_diagnostics": pair_rows,
        "maximum_temporal_uncertainty": float(np.max(uncertainty)),
        "temporal_uncertainty_passed": temporal_passed,
        "initial_maximum_uncertainty_inclusive_spread": float(np.max(initial)),
        "final_maximum_uncertainty_inclusive_spread": float(np.max(final)),
        "no_late_regrowth_within_temporal_uncertainty": no_late_regrowth,
        "measured_controlling_transport_e_folds": e_folds,
        "transport_rank": {
            "singular_values": transport_rank.singular_values,
            "second_to_first_ratio": transport_rank.second_to_first_ratio,
            "third_to_first_ratio": transport_rank.third_to_first_ratio,
            "dominant_direction": transport_rank.dominant_direction,
            "passed": transport_rank.passed,
        },
        "interface_localization": {
            "per_interface_maximum_gate_normalized_half_spread": (
                per_interface_maximum
            ),
            "controlling_interface": controlling_interface,
            "maximum_secondary_to_interface4_ratio": (
                secondary_interface_ratio
            ),
            "maximum_allowed_secondary_ratio": (
                MAXIMUM_SECONDARY_INTERFACE_RATIO
            ),
            "localized_at_interface4": interface_localized,
        },
        "deterministic_continuation_replay": replay,
        "binding_diagnostics_complete": binding,
        "classification": classification,
        "one_auxiliary_authorized": bool(
            binding
            and e_folds >= MINIMUM_AUXILIARY_EFOLDS
            and transport_rank.passed
            and interface_localized
            and False  # multiple amplitudes/fibers/anchors remain required
        ),
    }
    all_arrays.update(
        {
            "decision_temporal_uncertainty": uncertainty,
            "decision_upper_spreads": upper,
            "decision_normalized_interface4_transport": normalized_transport,
            "decision_normalized_all_interface_transport": (
                normalized_all_interface_transport
            ),
        }
    )
    return row, all_arrays


def _slow_rate_tangent_seed(
    cache: dict[str, np.ndarray],
    metadata: dict,
    loading_time_seconds: float,
):
    rate_rows, _gates, diagnostics = wp10c8i._rate_output_rows(
        cache, metadata, wp10c8o.LEVEL_INDEX
    )
    slow_rows = (
        float(loading_time_seconds)
        / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
        * rate_rows
    )
    scaled_amplitudes = (
        np.asarray(cache["physical_input_amplitudes"], dtype=float)
        / np.asarray(cache["primitive_column_scales"], dtype=float)
    )
    audit = causal_gate_normalized_finite_time_null_gain(
        slow_rows,
        np.asarray(cache[f"level_{wp10c8o.LEVEL_INDEX}_constraints"], dtype=float),
        np.ones(slow_rows.shape[0], dtype=float),
        state_weights=np.asarray(cache["state_weights"], dtype=float),
        state_amplitudes_scaled=scaled_amplitudes,
    )
    return audit.controlling_admissible_state_direction, audit, slow_rows, diagnostics


def _actual_slow_rate_row(row: dict, runtime: dict, loading_time: float) -> None:
    values = runtime["arrays"]
    minus = np.asarray(values["minus_coordinate_rate_output"], dtype=float)
    plus = np.asarray(values["plus_coordinate_rate_output"], dtype=float)
    slow_half = (
        0.5
        * (plus - minus)
        * float(loading_time)
        / wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
    )
    names = tuple(
        str(value) for value in np.asarray(values["coordinate_names"])
    )
    control = int(np.argmax(np.abs(slow_half)))
    row["slow_rate_audit"] = {
        "maximum_absolute_half_difference_per_unit_slow_time": float(
            np.max(np.abs(slow_half))
        ),
        "controlling_coordinate_index": control,
        "controlling_coordinate": names[control],
        "signed_half_differences_per_unit_slow_time": slow_half,
    }
    runtime["arrays"]["slow_rate_half_difference_per_unit_slow_time"] = slow_half


def _held_out_slow_rate_direction(
    audit,
    primary_direction: np.ndarray,
    scaled_amplitudes: np.ndarray,
) -> tuple[np.ndarray, dict]:
    basis = np.asarray(audit.null_basis_audit.basis, dtype=float)
    weights = np.asarray(
        audit.null_basis_audit.state_weights,
        dtype=float,
    )
    primary = np.asarray(primary_direction, dtype=float)
    amplitudes = np.asarray(scaled_amplitudes, dtype=float)
    if (
        basis.ndim != 2
        or weights.shape != (basis.shape[0],)
        or primary.shape != (basis.shape[0],)
        or amplitudes.shape != primary.shape
        or np.any(amplitudes <= 0.0)
    ):
        raise ValueError("held-out slow-rate direction inputs are invalid")
    primary_null = basis.T @ (weights * primary)
    primary_norm = float(np.linalg.norm(primary_null))
    if primary_norm <= np.finfo(float).tiny:
        raise ValueError("primary slow-rate direction vanished in the null basis")
    primary_unit = primary_null / primary_norm
    projector = np.eye(primary_unit.size) - np.outer(
        primary_unit,
        primary_unit,
    )
    projected_operator = (
        np.asarray(audit.gate_normalized_null_operator, dtype=float)
        @ projector
    )
    _left, singular, right_h = np.linalg.svd(
        projected_operator,
        full_matrices=False,
    )
    if singular.size == 0 or singular[0] <= np.finfo(float).tiny:
        raise ValueError("no independent slow-rate response direction exists")
    held_out_null = projector @ right_h[0]
    held_out_null /= np.linalg.norm(held_out_null)
    held_out = basis @ held_out_null
    maximum_amplitude_ratio = float(
        np.max(np.abs(held_out) / amplitudes)
    )
    held_out /= max(1.0, maximum_amplitude_ratio)
    pivot = int(np.argmax(np.abs(held_out)))
    if held_out[pivot] < 0.0:
        held_out *= -1.0
        held_out_null *= -1.0
    weighted_overlap = float(
        primary @ (weights * held_out)
        / max(
            np.sqrt(primary @ (weights * primary))
            * np.sqrt(held_out @ (weights * held_out)),
            np.finfo(float).tiny,
        )
    )
    return held_out, {
        "construction": (
            "maximum complete slow-rate response in the weighted constraint "
            "null space orthogonal to the primary admissible direction"
        ),
        "projected_response_singular_value": float(singular[0]),
        "weighted_orthogonality_defect": abs(weighted_overlap),
        "unscaled_maximum_amplitude_ratio": maximum_amplitude_ratio,
    }


def _run_rate_fiber_audit(
    *,
    initial_by_mesh: dict,
    vectors_by_mesh: dict,
    contracts: dict,
    include_n128: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    cache64, metadata64, cache64_path = wp10c8o._load_anchor_cache(
        64,
        wp10c8o.PRIMARY_ANCHOR,
        vectors_by_mesh[64][wp10c8o.PRIMARY_ANCHOR],
    )
    loading64 = causal_five_field_loading_time(
        contracts[64]["context"], contracts[64]["anchor_vector"]
    )
    seed, tangent, slow_rows, rate_diagnostics = _slow_rate_tangent_seed(
        cache64, metadata64, loading64
    )
    coordinate_names = tuple(
        metadata64["levels"][wp10c8o.LEVEL_INDEX]["coordinate_names"]
    )
    tangent_control = int(tangent.controlling_admissible_output_index)
    rows = {}
    runtimes = {}
    arrays: dict[str, np.ndarray] = {
        "n64_tangent_seed": seed,
        "n64_tangent_slow_rate_rows": slow_rows,
        "n64_tangent_coordinate_names": np.asarray(coordinate_names, dtype="U"),
    }
    for multiplier in RATE_FIBER_MULTIPLIERS:
        case_id = f"n64_slow_rate_alpha_{multiplier:.4e}"
        row, pair_arrays, runtime = wp10c8o._build_pair(
            case_id=case_id,
            seed_name="wp10c8q_worst_tangent_slow_rate_direction",
            seed_origin=(
                "gate-weighted complete 34-coordinate rate response per unit "
                "loading-time slow coordinate"
            ),
            seed_direction=seed,
            seed_multiplier=multiplier,
            initial=initial_by_mesh[64],
            vector=vectors_by_mesh[64][wp10c8o.PRIMARY_ANCHOR],
            cache=cache64,
            shell_edges_rg=contracts[64]["shell_edges_rg"],
            require_face58_switch=False,
        )
        wp10c8o._complete_pair_rates(
            row, runtime, binding_dae_storage_audit=True
        )
        _actual_slow_rate_row(row, runtime, loading64)
        rows[case_id] = row
        runtimes[case_id] = runtime
        arrays.update(
            {
                f"{case_id}_{name}": value
                for name, value in runtime["arrays"].items()
            }
        )
    decisive_id = f"n64_slow_rate_alpha_{RATE_FIBER_CONFIRMATION_MULTIPLIER:.4e}"
    held_out_id = None
    scaled_amplitudes = (
        np.asarray(cache64["physical_input_amplitudes"], dtype=float)
        / np.asarray(cache64["primitive_column_scales"], dtype=float)
    )
    held_out_seed, held_out_seed_diagnostics = (
        _held_out_slow_rate_direction(
            tangent,
            seed,
            scaled_amplitudes,
        )
    )
    held_out_id = "n64_slow_rate_held_out_direction"
    held_out_row, _held_out_pair_arrays, held_out_runtime = (
        wp10c8o._build_pair(
            case_id=held_out_id,
            seed_name="wp10c8q_held_out_complete_slow_rate_direction",
            seed_origin=held_out_seed_diagnostics["construction"],
            seed_direction=held_out_seed,
            seed_multiplier=RATE_FIBER_CONFIRMATION_MULTIPLIER,
            initial=initial_by_mesh[64],
            vector=vectors_by_mesh[64][wp10c8o.PRIMARY_ANCHOR],
            cache=cache64,
            shell_edges_rg=contracts[64]["shell_edges_rg"],
            require_face58_switch=False,
        )
    )
    held_out_row["seed_direction_diagnostics"] = held_out_seed_diagnostics
    wp10c8o._complete_pair_rates(
        held_out_row,
        held_out_runtime,
        binding_dae_storage_audit=True,
    )
    _actual_slow_rate_row(held_out_row, held_out_runtime, loading64)
    rows[held_out_id] = held_out_row
    runtimes[held_out_id] = held_out_runtime
    arrays.update(
        {
            f"{held_out_id}_{name}": value
            for name, value in held_out_runtime["arrays"].items()
        }
    )

    second_anchor_label = "t_0p10"
    cache_second, metadata_second, cache_second_path = (
        wp10c8o._load_anchor_cache(
            64,
            second_anchor_label,
            vectors_by_mesh[64][second_anchor_label],
        )
    )
    loading_second = causal_five_field_loading_time(
        contracts[64]["context"],
        vectors_by_mesh[64][second_anchor_label],
    )
    second_seed, second_tangent, second_rows, second_rate_diagnostics = (
        _slow_rate_tangent_seed(
            cache_second,
            metadata_second,
            loading_second,
        )
    )
    second_anchor_id = "n64_t_0p10_slow_rate_decisive"
    second_anchor_row, _second_pair_arrays, second_anchor_runtime = (
        wp10c8o._build_pair(
            case_id=second_anchor_id,
            seed_name="wp10c8q_held_out_anchor_worst_slow_rate_direction",
            seed_origin=(
                "gate-weighted complete slow-rate response at the held-out "
                "t=0.10 s truth anchor"
            ),
            seed_direction=second_seed,
            seed_multiplier=RATE_FIBER_CONFIRMATION_MULTIPLIER,
            initial=initial_by_mesh[64],
            vector=vectors_by_mesh[64][second_anchor_label],
            cache=cache_second,
            shell_edges_rg=contracts[64]["shell_edges_rg"],
            require_face58_switch=False,
        )
    )
    second_anchor_row["anchor_label"] = second_anchor_label
    second_anchor_row["anchor_role"] = "held_out"
    wp10c8o._complete_pair_rates(
        second_anchor_row,
        second_anchor_runtime,
        binding_dae_storage_audit=True,
    )
    _actual_slow_rate_row(
        second_anchor_row,
        second_anchor_runtime,
        loading_second,
    )
    rows[second_anchor_id] = second_anchor_row
    runtimes[second_anchor_id] = second_anchor_runtime
    arrays.update(
        {
            f"{second_anchor_id}_{name}": value
            for name, value in second_anchor_runtime["arrays"].items()
        }
    )
    n128_id = None
    if include_n128:
        cache128, _metadata128, cache128_path = wp10c8o._load_anchor_cache(
            128,
            wp10c8o.PRIMARY_ANCHOR,
            vectors_by_mesh[128][wp10c8o.PRIMARY_ANCHOR],
        )
        prolonged = wp10c8o._prolong_decisive_physical_direction(
            runtimes[decisive_id],
            RATE_FIBER_CONFIRMATION_MULTIPLIER,
            np.asarray(cache128["primitive_column_scales"], dtype=float),
        )
        n128_id = "n128_prolonged_slow_rate_decisive"
        row128, _pair_arrays128, runtime128 = wp10c8o._build_pair(
            case_id=n128_id,
            seed_name="prolonged_n64_slow_rate_direction",
            seed_origin=(
                "piecewise-constant prolongation of the exact N64 corrected "
                "pair half-difference; no N128 output optimization"
            ),
            seed_direction=prolonged,
            seed_multiplier=RATE_FIBER_CONFIRMATION_MULTIPLIER,
            initial=initial_by_mesh[128],
            vector=vectors_by_mesh[128][wp10c8o.PRIMARY_ANCHOR],
            cache=cache128,
            shell_edges_rg=contracts[128]["shell_edges_rg"],
            require_face58_switch=False,
        )
        wp10c8o._complete_pair_rates(
            row128, runtime128, binding_dae_storage_audit=True
        )
        loading128 = causal_five_field_loading_time(
            contracts[128]["context"], contracts[128]["anchor_vector"]
        )
        _actual_slow_rate_row(row128, runtime128, loading128)
        rows[n128_id] = row128
        runtimes[n128_id] = runtime128
        arrays.update(
            {f"{n128_id}_{name}": value for name, value in runtime128["arrays"].items()}
        )
    amplitudes = np.asarray(RATE_FIBER_MULTIPLIERS)
    maxima = np.asarray(
        [
            rows[f"n64_slow_rate_alpha_{value:.4e}"]["slow_rate_audit"]
            ["maximum_absolute_half_difference_per_unit_slow_time"]
            for value in amplitudes
        ]
    )
    linear_ratios = maxima / amplitudes
    amplitude_defect = float(
        np.max(np.abs(linear_ratios / linear_ratios[1] - 1.0))
    )
    robustness_case_ids = [
        *(f"n64_slow_rate_alpha_{value:.4e}" for value in amplitudes),
        *(() if held_out_id is None else (held_out_id,)),
        second_anchor_id,
        *(() if n128_id is None else (n128_id,)),
    ]
    transport_vectors = []
    transport_vectors_by_case = {}
    for case_id in robustness_case_ids:
        case_arrays = runtimes[case_id]["arrays"]
        complete_vector = _flat_interface_mje_half_difference(
            case_arrays["plus_interface_flux"],
            case_arrays["minus_interface_flux"],
        )
        vector = complete_vector[INTERFACE_INDEX - 1]
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise RuntimeError("slow-rate robustness transport vector vanished")
        unit_vector = vector / norm
        transport_vectors.append(unit_vector)
        transport_vectors_by_case[case_id] = unit_vector
    transport_vectors = np.asarray(transport_vectors, dtype=float)
    robustness_rank = causal_transport_rank_audit(
        transport_vectors,
        maximum_secondary_ratio=MAXIMUM_RANK_ONE_SECONDARY_RATIO,
    )
    same_anchor_case_ids = [
        *(f"n64_slow_rate_alpha_{value:.4e}" for value in amplitudes),
        held_out_id,
    ]
    same_anchor_rank = causal_transport_rank_audit(
        np.asarray(
            [transport_vectors_by_case[case_id] for case_id in same_anchor_case_ids]
        ),
        maximum_secondary_ratio=MAXIMUM_RANK_ONE_SECONDARY_RATIO,
    )
    cross_anchor_case_ids = [decisive_id, second_anchor_id]
    cross_anchor_rank = causal_transport_rank_audit(
        np.asarray(
            [transport_vectors_by_case[case_id] for case_id in cross_anchor_case_ids]
        ),
        maximum_secondary_ratio=MAXIMUM_RANK_ONE_SECONDARY_RATIO,
    )
    mesh_case_ids = (
        []
        if n128_id is None
        else [decisive_id, n128_id]
    )
    mesh_rank = (
        None
        if not mesh_case_ids
        else causal_transport_rank_audit(
            np.asarray(
                [transport_vectors_by_case[case_id] for case_id in mesh_case_ids]
            ),
            maximum_secondary_ratio=MAXIMUM_RANK_ONE_SECONDARY_RATIO,
        )
    )
    robustness_cases_passed = bool(
        all(
            rows[case_id]["lift_valid"]
            and rows[case_id]["fresh_rate_output_evaluated"]
            and rows[case_id]["full_output"][
                "all_fresh_rate_gates_passed"
            ]
            for case_id in robustness_case_ids
        )
    )
    robustness_spreads = {
        case_id: rows[case_id]["slow_rate_audit"][
            "maximum_absolute_half_difference_per_unit_slow_time"
        ]
        for case_id in robustness_case_ids
    }
    robustness_spreads_significant = bool(
        all(
            value >= MINIMUM_SIGNIFICANT_SLOW_RATE_SPREAD
            for value in robustness_spreads.values()
        )
    )
    arrays["robustness_case_ids"] = np.asarray(
        robustness_case_ids,
        dtype="U",
    )
    arrays["robustness_unit_interface4_transport_vectors"] = (
        transport_vectors
    )
    summary = {
        "tangent": {
            "loading_time_seconds": loading64,
            "controlling_coordinate_index": tangent_control,
            "controlling_coordinate": coordinate_names[tangent_control],
            "maximum_admissible_lower_gain_per_unit_slow_time": (
                tangent.maximum_admissible_lower_gain
            ),
            "maximum_admissible_upper_gain_per_unit_slow_time": (
                tangent.maximum_admissible_upper_gain
            ),
            "constraint_rank": tangent.null_basis_audit.constraint_rank,
            "nullity": tangent.null_basis_audit.nullity,
            "rate_row_diagnostics": rate_diagnostics,
            "operator_cache_path": _relative(cache64_path),
        },
        "cases": rows,
        "decisive_n64_pair": decisive_id,
        "held_out_n64_direction_pair": held_out_id,
        "held_out_anchor_pair": second_anchor_id,
        "n128_confirmation_pair": n128_id,
        "amplitude_linearity_defect": amplitude_defect,
        "multi_case_transport_rank": {
            "case_ids": robustness_case_ids,
            "singular_values": robustness_rank.singular_values,
            "second_to_first_ratio": robustness_rank.second_to_first_ratio,
            "third_to_first_ratio": robustness_rank.third_to_first_ratio,
            "dominant_direction": robustness_rank.dominant_direction,
            "passed": robustness_rank.passed,
        },
        "same_anchor_transport_rank": {
            "case_ids": same_anchor_case_ids,
            "singular_values": same_anchor_rank.singular_values,
            "second_to_first_ratio": same_anchor_rank.second_to_first_ratio,
            "third_to_first_ratio": same_anchor_rank.third_to_first_ratio,
            "dominant_direction": same_anchor_rank.dominant_direction,
            "passed": same_anchor_rank.passed,
        },
        "cross_anchor_transport_rank": {
            "case_ids": cross_anchor_case_ids,
            "singular_values": cross_anchor_rank.singular_values,
            "second_to_first_ratio": cross_anchor_rank.second_to_first_ratio,
            "third_to_first_ratio": cross_anchor_rank.third_to_first_ratio,
            "dominant_direction": cross_anchor_rank.dominant_direction,
            "passed": cross_anchor_rank.passed,
        },
        "mesh_confirmation_transport_rank": (
            None
            if mesh_rank is None
            else {
                "case_ids": mesh_case_ids,
                "singular_values": mesh_rank.singular_values,
                "second_to_first_ratio": mesh_rank.second_to_first_ratio,
                "third_to_first_ratio": mesh_rank.third_to_first_ratio,
                "dominant_direction": mesh_rank.dominant_direction,
                "passed": mesh_rank.passed,
            }
        ),
        "all_robustness_cases_passed": robustness_cases_passed,
        "slow_rate_spreads_per_unit_slow_time": robustness_spreads,
        "all_robustness_slow_rate_spreads_significant": (
            robustness_spreads_significant
        ),
        "multiple_amplitudes_evaluated": True,
        "held_out_direction_evaluated": held_out_id is not None,
        "second_anchor_evaluated": True,
        "held_out_anchor_tangent": {
            "anchor_label": second_anchor_label,
            "loading_time_seconds": loading_second,
            "controlling_coordinate_index": int(
                second_tangent.controlling_admissible_output_index
            ),
            "maximum_admissible_lower_gain_per_unit_slow_time": (
                second_tangent.maximum_admissible_lower_gain
            ),
            "maximum_admissible_upper_gain_per_unit_slow_time": (
                second_tangent.maximum_admissible_upper_gain
            ),
            "rate_row_diagnostics": second_rate_diagnostics,
            "operator_cache_path": _relative(cache_second_path),
            "slow_rate_rows_sha256": _array_sha256(second_rows),
        },
        "architecture_binding": bool(
            held_out_id is not None
            and include_n128
            and robustness_cases_passed
            and robustness_spreads_significant
            and same_anchor_rank.passed
            and cross_anchor_rank.passed
            and mesh_rank is not None
            and mesh_rank.passed
            and robustness_rank.passed
        ),
        "semantics": (
            "This audit targets instantaneous slow-rate sufficiency. A healed "
            "fast-averaged leading vector field remains a separate dynamical "
            "question tested by the history-preserving continuation."
        ),
    }
    return summary, arrays


def _run_or_load_rate_fiber_audit(
    *,
    initial_by_mesh: dict,
    vectors_by_mesh: dict,
    contracts: dict,
    include_n128: bool,
    force: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "include_n128": include_n128,
        "runner_sha256": _sha256(ROOT / THIS_RUNNER),
        "parent_fiber_arrays_sha256": _sha256(PARENT_FIBER_ARRAYS),
    }
    if (
        RATE_FIBER_CACHE_JSON.exists()
        and RATE_FIBER_CACHE_ARRAYS.exists()
        and not force
    ):
        cached = json.loads(RATE_FIBER_CACHE_JSON.read_text(encoding="utf-8"))
        if not all(cached.get(key) == value for key, value in expected.items()):
            raise RuntimeError("stale WP10c8q slow-rate fiber cache")
        if cached.get("arrays_sha256") != _sha256(RATE_FIBER_CACHE_ARRAYS):
            raise RuntimeError("WP10c8q slow-rate fiber array cache differs")
        with np.load(RATE_FIBER_CACHE_ARRAYS, allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
        return cached["summary"], arrays

    summary, arrays = _run_rate_fiber_audit(
        initial_by_mesh=initial_by_mesh,
        vectors_by_mesh=vectors_by_mesh,
        contracts=contracts,
        include_n128=include_n128,
    )
    RATE_FIBER_CACHE_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RATE_FIBER_CACHE_ARRAYS, **arrays)
    RATE_FIBER_CACHE_JSON.write_text(
        json.dumps(
            {
                **expected,
                "arrays_path": _relative(RATE_FIBER_CACHE_ARRAYS),
                "arrays_sha256": _sha256(RATE_FIBER_CACHE_ARRAYS),
                "summary": _plain(summary),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, arrays


def main() -> None:
    args = _arguments()
    started = time.perf_counter()
    healing, parent_arrays, _fiber, _fiber_arrays = _load_parent_evidence()
    initial_by_mesh, vectors_by_mesh, state_provenance, contracts = (
        _runtime_contracts()
    )
    if args.rate_fiber_only:
        summary, _arrays = _run_or_load_rate_fiber_audit(
            initial_by_mesh=initial_by_mesh,
            vectors_by_mesh=vectors_by_mesh,
            contracts=contracts,
            include_n128=args.include_n128_rate_fiber,
            force=args.force,
        )
        print(json.dumps(_plain(summary), indent=2, sort_keys=True))
        return
    if args.trajectory_only is not None:
        _mesh, resolution, side = args.trajectory_only.split("-")
        if resolution == "replay":
            reference = _run_or_load_continued_trajectory(
                contract=contracts[64],
                resolution="coarse",
                side=side,
                force=False,
            )
            row = _split_continuation_replay(
                contract=contracts[64],
                side=side,
                reference_states=reference["states"],
                force=args.force,
            )
        else:
            trajectory = _run_or_load_continued_trajectory(
                contract=contracts[64],
                resolution=resolution,
                side=side,
                force=args.force,
            )
            row = {
                "path": _relative(trajectory["path"]),
                "sha256": trajectory["sha256"],
                "restart_path": _relative(trajectory["restart_path"]),
                "restart_sha256": trajectory["restart_sha256"],
                "summary": trajectory["summary"],
                "history_replay": trajectory["history_replay"],
                "cached": trajectory["cached"],
            }
        print(json.dumps(_plain(row), indent=2, sort_keys=True))
        return

    existing = {}
    all_arrays: dict[str, np.ndarray] = {}
    for n_cells in (64, 128):
        divergence, divergence_arrays = _existing_divergence_audit(
            contract=contracts[n_cells],
            arrays=parent_arrays,
            n_cells=n_cells,
        )
        cached_trace = _cached_trace_attribution(n_cells)
        if cached_trace is None:
            trace, trace_arrays = _trace_attribution(
                contract=contracts[n_cells],
                arrays=parent_arrays,
                n_cells=n_cells,
            )
        else:
            trace, trace_arrays = cached_trace
        existing[str(n_cells)] = {
            "divergence": divergence,
            "trace_attribution": trace,
        }
        all_arrays.update(
            {
                f"n{n_cells}_existing_divergence_{name}": value
                for name, value in divergence_arrays.items()
            }
        )
        all_arrays.update(
            {
                f"n{n_cells}_trace_attribution_{name}": value
                for name, value in trace_arrays.items()
            }
        )

    continuation = None
    rate_fiber = None
    if not args.audit_only:
        continuation, continuation_arrays = _run_continuation_campaign(
            contract=contracts[64],
            force=args.force,
            compute_fresh_rates=not args.skip_fresh_rates,
            run_replay=not args.skip_continuation_replay,
        )
        all_arrays.update(
            {
                f"n64_continuation_{name}": value
                for name, value in continuation_arrays.items()
            }
        )
        if not args.skip_rate_fiber:
            rate_fiber, rate_arrays = _run_or_load_rate_fiber_audit(
                initial_by_mesh=initial_by_mesh,
                vectors_by_mesh=vectors_by_mesh,
                contracts=contracts,
                include_n128=args.include_n128_rate_fiber,
                force=args.force,
            )
            all_arrays.update(
                {f"rate_fiber_{name}": value for name, value in rate_arrays.items()}
            )

    existing_passed = bool(
        all(
            row["divergence"]["passed"]
            and row["divergence"]["flux_gauge_hypothesis_rejected_for_decisive_pair"]
            and row["trace_attribution"]["passed"]
            for row in existing.values()
        )
    )
    if args.audit_only:
        decision = (
            "wp10c8q_existing_evidence_confirms_conservative_slow_rate_redistribution"
            if existing_passed
            else "wp10c8q_existing_evidence_audit_failed"
        )
        next_action = "run_exact_history_n64_continuation_to_0p125s"
    elif continuation is None or not continuation["binding_diagnostics_complete"]:
        decision = "wp10c8q_development_diagnostics_nonbinding"
        next_action = "complete_history_continuation_rates_and_replay"
    elif continuation["classification"] == "healing_supported_through_0p125s":
        decision = "wp10c8q_healing_supported_through_0p125s"
        next_action = "complete_mesh_fiber_anchor_robustness_before_closure"
    elif (
        rate_fiber is not None
        and rate_fiber["held_out_direction_evaluated"]
        and rate_fiber["second_anchor_evaluated"]
        and rate_fiber["n128_confirmation_pair"] is not None
        and rate_fiber["all_robustness_cases_passed"]
        and rate_fiber["all_robustness_slow_rate_spreads_significant"]
    ):
        if (
            continuation["transport_rank"]["passed"]
            and continuation["interface_localization"][
                "localized_at_interface4"
            ]
        ):
            if not rate_fiber["same_anchor_transport_rank"]["passed"]:
                decision = (
                    "wp10c8q_persistent_localized_multimode_interface_state"
                )
                next_action = "design_conservative_interface_state_vector"
            elif not rate_fiber["mesh_confirmation_transport_rank"]["passed"]:
                decision = (
                    "wp10c8q_persistent_interface_state_not_mesh_supported"
                )
                next_action = "resolve_interface_state_cross_mesh_direction"
            elif not rate_fiber["cross_anchor_transport_rank"]["passed"]:
                decision = (
                    "wp10c8q_state_dependent_rank_one_interface_map_"
                    "not_yet_closed"
                )
                next_action = "map_interface_transport_direction_across_anchors"
            else:
                decision = (
                    "wp10c8q_persistent_localized_rank_one_interface_state_"
                    "supported"
                )
                next_action = (
                    "prototype_independent_interface_state_then_repeat_"
                    "worst_case_fiber_audit"
                )
        elif continuation["interface_localization"]["localized_at_interface4"]:
            decision = "wp10c8q_persistent_localized_multimode_interface_state"
            next_action = "design_conservative_interface_state_vector"
        else:
            decision = "wp10c8q_persistent_distributed_transport_state"
            next_action = "design_conservative_staggered_coarse_finite_volume_model"
    else:
        decision = "wp10c8q_healing_not_observed_through_0p125s_only"
        next_action = "complete_mesh_fiber_anchor_robustness"

    arrays_path = _absolute(args.arrays)
    output_path = _absolute(args.output)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **all_arrays)
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT / "scripts/run_causal_natural_healing_wp10c8p.py",
        ROOT / "scripts/run_causal_nonlinear_fiber_audit_wp10c8o.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_healing.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_bdf_evolution.py",
        PARENT_JSON,
        PARENT_ARRAYS,
        PARENT_FIBER_JSON,
        PARENT_FIBER_ARRAYS,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "existing_evidence_divergence_audit": True,
            "exact_history_n64_continuation": not args.audit_only,
            "slow_rate_fiber_audit": bool(rate_fiber is not None),
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "new_coordinate_added": False,
            "memory_model_fit": False,
            "macrostep_run": False,
            "fresh_coordinate_rates_evaluated": bool(
                not args.audit_only and not args.skip_fresh_rates
            ),
        },
        "authorization": {
            "wp10c8p_path": _relative(PARENT_JSON),
            "wp10c8p_sha256": _sha256(PARENT_JSON),
            "wp10c8p_arrays_path": _relative(PARENT_ARRAYS),
            "wp10c8p_arrays_sha256": _sha256(PARENT_ARRAYS),
        },
        "frozen_contract": {
            "coordinate_level": wp10c8o.LEVEL_NAME,
            "coordinate_count": 34,
            "duration_seconds": TARGET_DURATION_SECONDS,
            "output_offsets_seconds": OUTPUT_OFFSETS_SECONDS,
            "coarse_subdivisions": COARSE_SUBDIVISIONS,
            "fine_subdivisions": FINE_SUBDIVISIONS,
            "continuation_startup": "preserve_exact_history_no_new_bdf1",
        },
        "gates": {
            "maximum_ledger_relative_defect": MAXIMUM_LEDGER_RELATIVE_DEFECT,
            "maximum_trace_reconstruction_defect": (
                MAXIMUM_TRACE_RECONSTRUCTION_DEFECT
            ),
            "maximum_temporal_uncertainty_gate_units": TEMPORAL_UNCERTAINTY_GATE,
            "healing_spread_gate": HEALING_SPREAD_GATE,
            "minimum_auxiliary_e_folds": MINIMUM_AUXILIARY_EFOLDS,
            "maximum_rank_one_secondary_ratio": MAXIMUM_RANK_ONE_SECONDARY_RATIO,
            "maximum_secondary_interface_ratio": (
                MAXIMUM_SECONDARY_INTERFACE_RATIO
            ),
        },
        "existing_evidence": existing,
        "continuation": continuation,
        "slow_rate_fiber": rate_fiber,
        "state_provenance": {
            str(n_cells): state_provenance[str(n_cells)][
                wp10c8o.PRIMARY_ANCHOR
            ]
            for n_cells in (64, 128)
        },
        "decision": decision,
        "next_action": next_action,
        "semantics": (
            "The committed pair proves nonunique instantaneous slow-scaled "
            "rates and real conservative redistribution. Only the extended "
            "history-preserving dynamics can decide whether that ambiguity "
            "is an initial-layer effect or a persistent interface state."
        ),
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
        "source_hashes": {_relative(path): _sha256(path) for path in source_paths},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_plain(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
