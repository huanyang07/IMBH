"""Run the WP10c8n Rusanov candidate localization and screening audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_characteristic_extension_wp10c7l as wp10c7l
import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_spatial_balance_adaptive_wp10c7k as wp10c7k
import run_causal_tangent_descriptor_wp10c8l as wp10c8l_a
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_regression_seed_parameters,
    causal_five_field_reconstruct_face_charts,
    load_causal_five_field_adaptive_bdf2_restart,
    make_causal_five_field_regression_context,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    causal_five_field_rusanov_control_diagnostics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_mixed_reduction import (
    causal_weighted_constraint_null_basis,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_rusanov_certification import (
    rusanov_structured_possible_winner_closure,
    rusanov_structured_zero_remainder_decomposition,
    rusanov_structured_zero_remainder_preflight,
    rusanov_weighted_anchor_gap_radius_screen,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "4dc5cea0342d35135e31078669e7e71ba7d16cf9"
WORK_PACKAGE = "WP10c8n"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_rusanov_candidate_screen_wp10c8n.py"
PARENT_RESULT = (
    ROOT / "outputs/tables/causal_rusanov_all_face_preflight_wp10c8m.json"
)
PARENT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_rusanov_all_face_preflight_wp10c8m_arrays.npz"
)
CACHED_RESULT = (
    ROOT / "outputs/tables/causal_rusanov_structured_preflight_wp10c8m.json"
)
LOCKED_CASES = ((64, "t_0"), (64, "t_0p025"))
LEVEL_INDEX = 4
LOCKED_HORIZONS_SECONDS = (0.0, 1.0e-2, 2.5e-2)
TIME_PANELS = 128
RADIUS_LADDER = np.asarray(
    (0.0, 5.0e-3, 6.0e-3, 1.0e-2, 1.0e-1, 1.0, 2.05, 2.25),
    dtype=float,
)
MINIMUM_INITIAL_WEIGHTED_RADIUS = 1.0
PROPAGATED_RADIUS_TIME_PANELS = 32
PROPAGATED_RADIUS_MARGIN_FACTOR = 1.005
HEADROOM_MAXIMUM_GATE_FRACTION = 5.0e-3
REPORTING_MAXIMUM_GATE_FRACTION = 1.0e-2
TOP_FACE_COUNT = 5
TOP_CANDIDATE_FAMILY_COUNT = 3
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_rusanov_candidate_screen_wp10c8n.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_rusanov_candidate_screen_wp10c8n_arrays.npz"
)
CANDIDATE_LABELS = tuple(
    f"{side}:{family}"
    for side in ("left", "right")
    for family in (
        "inward_acoustic",
        "inward_shear",
        "material",
        "outward_shear",
        "outward_acoustic",
    )
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
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
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _load_n64_states() -> tuple[dict, dict[str, np.ndarray], dict]:
    evidence, evidence_sha256 = wp10c7l._validate_wp10c7k()
    baseline = make_causal_five_field_regression_context(
        16, spatial_reconstruction="plm_smooth"
    )
    seed_parameters = causal_five_field_regression_seed_parameters(baseline)
    initial = wp10c7k._initial_bundle(64, seed_parameters)
    expected = evidence["initialization"]["meshes"]["64"]
    if initial["vector_sha256"] != expected["state_vector_sha256"]:
        raise RuntimeError("the fresh N64 initial state differs from WP10c7k")
    path = wp10c8i._t_0p025_path(64)
    restart = load_causal_five_field_adaptive_bdf2_restart(
        path, initial["context"]
    )
    if not (
        restart.elapsed_time == 2.5e-2
        and restart.provenance.get("work_package") == "WP10c7l"
        and restart.provenance.get("trajectory_mode") == "production"
        and restart.provenance.get("n_cells") == 64
    ):
        raise RuntimeError("the N64 t=0.025 checkpoint provenance differs")
    vectors = {
        "t_0": np.asarray(initial["vector"], dtype=float),
        "t_0p025": np.asarray(restart.state_vector, dtype=float),
    }
    provenance = {
        "t_0": {
            "state_vector_sha256": _array_sha256(vectors["t_0"]),
            "elapsed_time_seconds": 0.0,
            "wp10c7k_evidence_sha256": evidence_sha256,
        },
        "t_0p025": {
            "path": _relative(path),
            "sha256": _sha256(path),
            "state_vector_sha256": _array_sha256(vectors["t_0p025"]),
            "elapsed_time_seconds": restart.elapsed_time,
        },
    }
    return initial, vectors, provenance


def _n64_five_shell_edges(initial: dict) -> np.ndarray:
    grid = initial["context"].grid
    edges = np.asarray(grid.edges, dtype=float) / grid.gravitational_radius
    targets = wp10c8h.SHELL_LAYOUT_TARGETS_RG["five_shell"]
    indices = np.asarray(
        [
            0,
            *(int(np.argmin(np.abs(edges - target))) for target in targets),
            edges.size - 1,
        ],
        dtype=int,
    )
    if np.unique(indices).size != indices.size or np.any(np.diff(indices) <= 0):
        raise RuntimeError("the N64 five-shell layout is invalid")
    return edges[indices]


def _parent_case_arrays(
    archive: np.lib.npyio.NpzFile,
    case_id: str,
) -> dict[str, np.ndarray]:
    prefix = f"{case_id}_"
    names = {
        "dynamic": "dynamic",
        "constraint_null_basis": "constraint_null_basis",
        "generator_left_factors": "generator_left_factors",
        "generator_right_factors": "generator_right_factors",
        "branch_face_indices": "branch_face_indices",
        "branch_candidate_indices": "branch_candidate_indices",
    }
    return {
        output: np.asarray(archive[prefix + suffix])
        for output, suffix in names.items()
    }


def _cached_seed_mask(
    cached: dict,
    case_id: str,
    faces: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    identities = cached["cases"][case_id]["factor_audit"][
        "candidate_identities"
    ]
    pairs = {
        (int(row["face_index"]), int(row["competitor_code"]))
        for row in identities
    }
    return np.asarray(
        [
            (int(face), int(candidate)) in pairs
            for face, candidate in zip(faces, candidates, strict=True)
        ],
        dtype=bool,
    )


def _case_inputs(
    *,
    case_id: str,
    initial: dict,
    vector: np.ndarray,
    shell_edges: np.ndarray,
    parent_case: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict, dict, dict]:
    parent_arrays, metadata, provenance = wp10c8l_a._load_parent_operator(
        initial,
        vector,
        shell_edges,
        n_cells=64,
        label=case_id.removeprefix("n64_"),
    )
    arrays = dict(parent_arrays)
    arrays["dynamic"] = np.asarray(parent_case["dynamic"], dtype=float)
    arrays["production_rusanov_kink_generator_left_factors"] = np.asarray(
        parent_case["generator_left_factors"], dtype=float
    )
    arrays["production_rusanov_kink_generator_right_factors"] = np.asarray(
        parent_case["generator_right_factors"], dtype=float
    )
    arrays["production_rusanov_kink_face_indices"] = np.asarray(
        parent_case["branch_face_indices"], dtype=int
    )
    arrays["production_rusanov_kink_competitor_codes"] = np.asarray(
        parent_case["branch_candidate_indices"], dtype=int
    )

    n_cells = initial["state"].n_cells
    state = unpack_causal_five_field_state(vector, n_cells)
    diagnostics = causal_five_field_rusanov_control_diagnostics(
        initial["context"], np.asarray(state.primitives, dtype=float)
    )
    diagnostic_faces = np.asarray(diagnostics["face_indices"], dtype=int)
    expected_faces = np.arange(1, n_cells, dtype=int)
    if not np.array_equal(diagnostic_faces, expected_faces):
        raise RuntimeError("Rusanov diagnostics do not cover every face")
    controls = np.asarray(diagnostics["control_codes"], dtype=int)
    speeds = np.asarray(
        diagnostics["candidate_absolute_speeds_over_c"], dtype=float
    )
    jumps = np.asarray(diagnostics["conserved_jumps"], dtype=float)
    faces = np.asarray(
        arrays["production_rusanov_kink_face_indices"], dtype=int
    )
    candidates = np.asarray(
        arrays["production_rusanov_kink_competitor_codes"], dtype=int
    )
    if faces.size != (n_cells - 1) * (speeds.shape[1] - 1):
        raise RuntimeError("the all-face branch count is not complete")
    pairs = np.column_stack((faces, candidates))
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise RuntimeError("the all-face branch identities are not unique")
    base_gaps = np.empty(faces.size, dtype=float)
    physical = np.empty((5, faces.size), dtype=float)
    for column, (face, candidate) in enumerate(
        zip(faces, candidates, strict=True)
    ):
        row = int(face) - 1
        control = int(controls[row])
        if int(candidate) == control:
            raise RuntimeError("the all-face set contains its nominal controller")
        base_gaps[column] = speeds[row, control] - speeds[row, int(candidate)]
        measure = float(initial["context"].grid.face_measures[int(face)])
        physical[:, column] = C * (-0.5 * measure * jumps[row])
    if np.any(base_gaps < 0.0):
        raise RuntimeError("the nominal Rusanov controller is not maximal")
    arrays["production_rusanov_kink_physical_flux_left_factors"] = physical
    response, gates, names, blocks = wp10c8i._response_stack(
        arrays, metadata, LEVEL_INDEX
    )
    constraints = np.asarray(arrays[f"level_{LEVEL_INDEX}_constraints"])
    generator_left = np.asarray(
        arrays["production_rusanov_kink_generator_left_factors"], dtype=float
    )
    direct_left = np.zeros((faces.size, response.shape[0]), dtype=float)
    interface_start, interface_end = blocks["macro_interface_flux"]
    interface_scales = np.asarray(arrays["interface_flux_scales"], dtype=float)
    shell_faces = np.asarray(arrays["shell_edge_indices"], dtype=int)
    component_by_name = {
        "rest_mass": 0,
        "angular_momentum": 2,
        "killing_energy": 3,
    }
    for local_row, name in enumerate(metadata["interface_flux_names"]):
        parts = str(name).split("_", 2)
        boundary_index = int(parts[1])
        component = component_by_name[parts[2]]
        physical_face = int(shell_faces[boundary_index])
        selected = faces == physical_face
        direct_left[selected, interface_start + local_row] = (
            physical[component, selected] / interface_scales[local_row]
        )
    if interface_end - interface_start != len(
        metadata["interface_flux_names"]
    ):
        raise RuntimeError("interface response metadata differ")
    rate_start, rate_end = blocks["coarse_coordinate_rate"]
    direct_left[:, rate_start:rate_end] = (
        wp10c8i.COORDINATE_RATE_WINDOW_SECONDS
        * (constraints @ generator_left).T
    )
    weights = np.asarray(arrays["state_weights"], dtype=float)
    fresh_basis = causal_weighted_constraint_null_basis(
        constraints, state_weights=weights
    )
    basis = np.asarray(parent_case["constraint_null_basis"], dtype=float)
    basis_difference = float(np.max(np.abs(basis - fresh_basis.basis)))
    if basis_difference > 5.0e-13:
        raise RuntimeError("the saved and fresh constraint-null bases differ")
    return arrays, metadata, provenance, {
        "response": np.asarray(response, dtype=float),
        "gates": np.asarray(gates, dtype=float),
        "names": tuple(names),
        "blocks": blocks,
        "direct_left": direct_left,
        "basis": basis,
        "constraints": constraints,
        "weights": weights,
        "base_gaps": base_gaps,
        "faces": faces,
        "candidates": candidates,
        "controls_by_face": controls,
        "candidate_speeds_by_face": speeds,
        "physical_left_factors": physical,
        "context": initial["context"],
        "base_primitives": np.asarray(state.primitives, dtype=float).ravel(),
        "primitive_scales": np.asarray(
            arrays["primitive_column_scales"], dtype=float
        ),
        "physical_input_amplitudes": np.asarray(
            arrays["physical_input_amplitudes"], dtype=float
        ),
        "basis_difference": basis_difference,
        "constraint_defect": float(np.max(np.abs(constraints @ basis))),
        "weighted_orthogonality_defect": float(
            np.max(
                np.abs(
                    basis.T @ (weights[:, None] * basis)
                    - np.eye(basis.shape[1])
                )
            )
        ),
    }


def _preflight(
    arrays: dict[str, np.ndarray],
    data: dict,
    mask: np.ndarray,
    horizon: float,
):
    selected = np.asarray(mask, dtype=bool)
    return rusanov_structured_zero_remainder_preflight(
        base_generator_per_s=np.asarray(arrays["dynamic"], dtype=float),
        output_operator=data["response"],
        generator_left_factors=np.asarray(
            arrays["production_rusanov_kink_generator_left_factors"],
            dtype=float,
        )[:, selected],
        generator_right_factors=np.asarray(
            arrays["production_rusanov_kink_generator_right_factors"],
            dtype=float,
        )[:, selected],
        branch_face_indices=data["faces"][selected],
        initial_basis=data["basis"],
        horizon_seconds=horizon,
        output_gates=data["gates"],
        direct_output_left_factors=data["direct_left"][selected],
        time_steps=TIME_PANELS,
        maximum_gate_fraction=REPORTING_MAXIMUM_GATE_FRACTION,
    )


def _bound_row(result, names: tuple[str, ...]) -> dict:
    fractions = np.asarray(result.per_output_gate_fractions, dtype=float)
    control = int(np.argmax(fractions))
    total = float(result.per_output_total_bounds[control])
    dynamic = float(result.per_output_dynamic_bounds[control])
    direct = float(result.per_output_direct_bounds[control])
    return {
        "branch_count": int(result.branch_count),
        "face_count": int(result.face_count),
        "used_face_compression": bool(result.used_face_compression),
        "left_compression_relative_defect": float(
            result.left_compression_relative_defect
        ),
        "direct_factor_relative_defect": float(
            result.direct_factor_relative_defect
        ),
        "maximum_gate_fraction": float(result.maximum_gate_fraction),
        "controlling_output_index": control,
        "controlling_output": names[control],
        "controlling_dynamic_share": (
            dynamic / float(result.per_output_total_bounds[control])
            if total > 0.0
            else 0.0
        ),
        "controlling_direct_share": (
            direct / float(result.per_output_total_bounds[control])
            if total > 0.0
            else 0.0
        ),
        "headroom_passed": bool(
            result.maximum_gate_fraction
            <= HEADROOM_MAXIMUM_GATE_FRACTION
        ),
    }


def _nominal_weighted_tube_radius(
    generator: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    horizon: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    from scipy.sparse.linalg import expm_multiply

    if horizon == 0.0:
        propagated = basis[None, :, :]
        times = np.asarray([0.0])
    else:
        propagated = np.asarray(
            expm_multiply(
                generator,
                basis,
                start=0.0,
                stop=horizon,
                num=PROPAGATED_RADIUS_TIME_PANELS + 1,
                endpoint=True,
                traceA=float(np.trace(generator)),
            ),
            dtype=float,
        )
        times = np.linspace(
            0.0, horizon, PROPAGATED_RADIUS_TIME_PANELS + 1
        )
    square_root_weights = np.sqrt(weights)
    radii = np.asarray(
        [
            np.linalg.svd(
                square_root_weights[:, None] * state_basis,
                compute_uv=False,
            )[0]
            for state_basis in propagated
        ],
        dtype=float,
    )
    return float(np.max(radii)), times, radii


def _decomposition(
    arrays: dict[str, np.ndarray],
    data: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    all_mask = np.ones(data["faces"].size, dtype=bool)
    result = rusanov_structured_zero_remainder_decomposition(
        base_generator_per_s=np.asarray(arrays["dynamic"], dtype=float),
        output_operator=data["response"],
        generator_left_factors=np.asarray(
            arrays["production_rusanov_kink_generator_left_factors"],
            dtype=float,
        ),
        generator_right_factors=np.asarray(
            arrays["production_rusanov_kink_generator_right_factors"],
            dtype=float,
        ),
        branch_face_indices=data["faces"],
        initial_basis=data["basis"],
        horizon_seconds=max(LOCKED_HORIZONS_SECONDS),
        output_gates=data["gates"],
        direct_output_left_factors=data["direct_left"],
        time_steps=TIME_PANELS,
        maximum_gate_fraction=REPORTING_MAXIMUM_GATE_FRACTION,
    )
    fractions = np.asarray(result.bound.per_output_gate_fractions)
    output_index = int(np.argmax(fractions))
    gate = float(data["gates"][output_index])
    face_contributions = result.per_face_total_bounds[:, output_index] / gate
    face_order = np.argsort(face_contributions)[::-1]
    branch_contributions = (
        result.per_branch_dynamic_bounds[:, output_index]
        + result.per_branch_direct_bounds[:, output_index]
    ) / gate
    family_contributions = np.asarray(
        [
            np.sum(branch_contributions[data["candidates"] == candidate])
            for candidate in range(len(CANDIDATE_LABELS))
        ],
        dtype=float,
    )
    family_order = np.argsort(family_contributions)[::-1]
    row = _bound_row(result.bound, data["names"])
    row.update(
        {
            "top_faces": [
                {
                    "face_index": int(result.face_indices[index]),
                    "gate_fraction_attribution": float(
                        face_contributions[index]
                    ),
                }
                for index in face_order[:TOP_FACE_COUNT]
            ],
            "top_candidate_families": [
                {
                    "candidate_code": int(index),
                    "candidate_label": CANDIDATE_LABELS[index],
                    "gate_fraction_attribution": float(
                        family_contributions[index]
                    ),
                }
                for index in family_order[:TOP_CANDIDATE_FAMILY_COUNT]
            ],
            "attribution_reconstruction_defect": float(
                np.max(
                    np.abs(
                        np.sum(result.per_face_total_bounds, axis=0)
                        - result.bound.per_output_total_bounds
                    )
                )
            ),
            "all_candidate_mask_count": int(np.count_nonzero(all_mask)),
        }
    )
    arrays_out = {
        "per_branch_dynamic_bounds": result.per_branch_dynamic_bounds,
        "per_branch_direct_bounds": result.per_branch_direct_bounds,
        "per_branch_dynamic_winner_counts": (
            result.per_branch_dynamic_winner_counts
        ),
        "per_face_dynamic_bounds": result.per_face_dynamic_bounds,
        "per_face_direct_bounds": result.per_face_direct_bounds,
        "per_face_total_bounds": result.per_face_total_bounds,
        "direct_winning_branch_indices": (
            result.direct_winning_branch_indices
        ),
        "all_candidate_gate_fractions": fractions,
    }
    return row, arrays_out


def _leave_one_diagnostics(
    arrays: dict[str, np.ndarray],
    data: dict,
    decomposition: dict,
) -> dict:
    baseline = float(decomposition["maximum_gate_fraction"])
    baseline_control = int(decomposition["controlling_output_index"])

    def counterfactual_row(result) -> dict:
        fractions = np.asarray(result.per_output_gate_fractions, dtype=float)
        control = int(np.argmax(fractions))
        fixed = float(fractions[baseline_control])
        return {
            "maximum_gate_fraction": float(result.maximum_gate_fraction),
            "post_removal_controlling_output_index": control,
            "post_removal_controlling_output": data["names"][control],
            "baseline_controlling_output_index": baseline_control,
            "baseline_controlling_output": data["names"][baseline_control],
            "baseline_controlling_output_gate_fraction_after_removal": fixed,
            "global_maximum_reduction": (
                baseline - float(result.maximum_gate_fraction)
            ),
            "baseline_controlling_output_reduction": baseline - fixed,
        }

    rows_by_face = []
    for top in decomposition["top_faces"]:
        face = int(top["face_index"])
        result = _preflight(
            arrays,
            data,
            data["faces"] != face,
            max(LOCKED_HORIZONS_SECONDS),
        )
        rows_by_face.append(
            {
                "removed_face": face,
                **counterfactual_row(result),
            }
        )
    rows_by_family = []
    for top in decomposition["top_candidate_families"]:
        candidate = int(top["candidate_code"])
        result = _preflight(
            arrays,
            data,
            data["candidates"] != candidate,
            max(LOCKED_HORIZONS_SECONDS),
        )
        rows_by_family.append(
            {
                "removed_candidate_code": candidate,
                "removed_candidate_label": CANDIDATE_LABELS[candidate],
                **counterfactual_row(result),
            }
        )
    return {
        "semantics": (
            "counterfactual recomputation of the zero-remainder comparison "
            "bound after removing one face or one candidate family; both "
            "the new global controller and the fixed original controlling "
            "row are reported because the output argmax may change"
        ),
        "by_face": rows_by_face,
        "by_candidate_family": rows_by_family,
    }


def _nonlinear_switch_witness(
    arrays: dict[str, np.ndarray],
    data: dict,
    decomposition: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    face = int(decomposition["top_faces"][0]["face_index"])
    face_columns = np.flatnonzero(data["faces"] == face)
    right = np.asarray(
        arrays["production_rusanov_kink_generator_right_factors"], dtype=float
    )
    projected_norms = np.linalg.norm(
        data["basis"].T @ right[:, face_columns], axis=0
    )
    thresholds = np.full(face_columns.size, np.inf, dtype=float)
    active = projected_norms > 0.0
    thresholds[active] = (
        data["base_gaps"][face_columns[active]]
        / projected_norms[active]
    )
    local = int(np.argmin(thresholds))
    branch = int(face_columns[local])
    threshold = float(thresholds[local])
    candidate = int(data["candidates"][branch])
    nominal_control = int(data["controls_by_face"][face - 1])
    null_coordinates = data["basis"].T @ right[:, branch]
    norm = float(np.linalg.norm(null_coordinates))
    if not np.isfinite(threshold) or norm <= 0.0:
        raise RuntimeError("the controlling face has no finite null switch")
    direction = -(data["basis"] @ null_coordinates) / norm
    weighted_norm = float(
        np.sqrt(np.sum(data["weights"] * direction * direction))
    )
    constraint_defect = float(
        np.max(np.abs(data["constraints"] @ direction))
    )
    radii = np.unique(
        np.asarray(
            (
                0.95 * threshold,
                0.999 * threshold,
                1.001 * threshold,
                1.05 * threshold,
            ),
            dtype=float,
        )
    )
    base_primitives = data["base_primitives"]
    scales = data["primitive_scales"]
    physical_scale_over_amplitude = (
        scales / data["physical_input_amplitudes"]
    )
    rows = []
    signed_gaps = []
    control_codes = []
    admissibility = []
    amplitude_ratios = []
    for radius in radii:
        primitives = base_primitives + radius * scales * direction
        audit = causal_five_field_rusanov_control_diagnostics(
            data["context"], primitives.reshape(-1, 5)
        )
        reconstruction = causal_five_field_reconstruct_face_charts(
            data["context"], primitives.reshape(-1, 5)
        )
        speeds = np.asarray(
            audit["candidate_absolute_speeds_over_c"], dtype=float
        )
        signed_gap = float(
            speeds[face - 1, nominal_control]
            - speeds[face - 1, candidate]
        )
        control = int(np.asarray(audit["control_codes"])[face - 1])
        minimum_admissibility = float(
            np.min(reconstruction.admissibility_factors)
        )
        maximum_amplitude_ratio = float(
            np.max(
                np.abs(
                    radius * physical_scale_over_amplitude * direction
                )
            )
        )
        signed_gaps.append(signed_gap)
        control_codes.append(control)
        admissibility.append(minimum_admissibility)
        amplitude_ratios.append(maximum_amplitude_ratio)
        rows.append(
            {
                "weighted_radius": float(radius),
                "nominal_control_minus_candidate_gap_over_c": signed_gap,
                "actual_control_code": control,
                "actual_control_label": CANDIDATE_LABELS[control],
                "minimum_reconstruction_admissibility_factor": (
                    minimum_admissibility
                ),
                "maximum_declared_pointwise_amplitude_ratio": (
                    maximum_amplitude_ratio
                ),
            }
        )
    before = np.asarray(signed_gaps) > 0.0
    after = np.asarray(signed_gaps) < 0.0
    single_mask = np.zeros(data["faces"].size, dtype=bool)
    single_mask[branch] = True
    instantaneous = _preflight(arrays, data, single_mask, 0.0)
    witness = bool(
        np.any(before)
        and np.any(after)
        and np.all(np.asarray(admissibility) == 1.0)
        and np.max(amplitude_ratios) < 1.0
        and threshold < MINIMUM_INITIAL_WEIGHTED_RADIUS
    )
    return {
        "face_index": face,
        "nominal_control_code": nominal_control,
        "nominal_control_label": CANDIDATE_LABELS[nominal_control],
        "challenger_code": candidate,
        "challenger_label": CANDIDATE_LABELS[candidate],
        "linear_null_switch_threshold_radius": threshold,
        "direction_weighted_norm": weighted_norm,
        "direction_constraint_defect": constraint_defect,
        "rows": rows,
        "nonlinear_switch_witnessed": witness,
        "single_challenger_instantaneous_bound": _bound_row(
            instantaneous, data["names"]
        ),
        "semantics": (
            "finite-amplitude evaluation of the unmodified production "
            "Rusanov candidate map along the exact anchor-null direction "
            "that decreases the controlling speed gap"
        ),
    }, {
        "nonlinear_witness_direction": direction,
        "nonlinear_witness_radii": radii,
        "nonlinear_witness_signed_gaps": np.asarray(signed_gaps),
        "nonlinear_witness_control_codes": np.asarray(control_codes, dtype=int),
        "nonlinear_witness_admissibility_factors": np.asarray(admissibility),
        "nonlinear_witness_amplitude_ratios": np.asarray(amplitude_ratios),
    }


def _case(
    *,
    case_id: str,
    arrays: dict[str, np.ndarray],
    data: dict,
    cached: dict,
    parent: dict,
    primary_leave_one: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    left = np.asarray(
        arrays["production_rusanov_kink_generator_left_factors"], dtype=float
    )
    right = np.asarray(
        arrays["production_rusanov_kink_generator_right_factors"], dtype=float
    )
    seed = _cached_seed_mask(
        cached, case_id, data["faces"], data["candidates"]
    )
    nominal_radius_rows = {}
    nominal_radius_arrays: dict[str, np.ndarray] = {}
    required_radius = MINIMUM_INITIAL_WEIGHTED_RADIUS
    for horizon in LOCKED_HORIZONS_SECONDS:
        maximum_radius, radius_times, radius_values = (
            _nominal_weighted_tube_radius(
                np.asarray(arrays["dynamic"], dtype=float),
                data["basis"],
                data["weights"],
                horizon,
            )
        )
        required = max(
            MINIMUM_INITIAL_WEIGHTED_RADIUS,
            PROPAGATED_RADIUS_MARGIN_FACTOR * maximum_radius,
        )
        required_radius = max(required_radius, required)
        key = f"{horizon:.3e}"
        nominal_radius_rows[key] = {
            "maximum_weighted_radius": maximum_radius,
            "required_radius_with_margin": required,
            "time_panel_count": PROPAGATED_RADIUS_TIME_PANELS,
        }
        nominal_radius_arrays[f"nominal_radius_{key}_times"] = radius_times
        nominal_radius_arrays[f"nominal_radius_{key}_values"] = radius_values
    decomposition, result_arrays = _decomposition(arrays, data)
    result_arrays.update(nominal_radius_arrays)
    parent_maximum = float(
        parent["cases"][case_id]["rows"]["2.500e-02"]["128"][
            "maximum_gate_fraction"
        ]
    )
    reproduction_defect = abs(
        float(decomposition["maximum_gate_fraction"]) - parent_maximum
    )
    if reproduction_defect > 5.0e-12:
        raise RuntimeError("the all-candidate parent bound was not reproduced")

    screen = rusanov_weighted_anchor_gap_radius_screen(
        base_speed_gaps=data["base_gaps"],
        gap_gradient_vectors=right,
        state_weights=data["weights"],
        neighborhood_radii=RADIUS_LADDER,
        branch_face_indices=data["faces"],
        forced_candidate_mask=seed,
    )
    radius_rows: list[dict] = []
    for radius_index, radius in enumerate(RADIUS_LADDER):
        mask = screen.possible_candidate_masks[radius_index]
        horizon_rows = {}
        for horizon in LOCKED_HORIZONS_SECONDS:
            result = _preflight(arrays, data, mask, horizon)
            horizon_rows[f"{horizon:.3e}"] = _bound_row(
                result, data["names"]
            )
        radius_rows.append(
            {
                "weighted_radius": float(radius),
                "possible_candidate_count": int(np.count_nonzero(mask)),
                "possible_face_count": int(np.unique(data["faces"][mask]).size),
                "covers_nominal_propagated_null_tube_with_margin": bool(
                    radius >= required_radius
                ),
                "uniform_gap_variation_certified": False,
                "trajectory_containment_certified": False,
                "horizons": horizon_rows,
            }
        )
        result_arrays[
            f"radius_{radius_index}_possible_candidate_mask"
        ] = mask

    closure_rows = {}
    closure_masks = []
    closure_headroom = True
    for horizon in LOCKED_HORIZONS_SECONDS:
        closure = rusanov_structured_possible_winner_closure(
            base_generator_per_s=np.asarray(arrays["dynamic"], dtype=float),
            generator_left_factors=left,
            generator_right_factors=right,
            branch_face_indices=data["faces"],
            base_speed_gaps=data["base_gaps"],
            initial_basis=data["basis"],
            horizon_seconds=horizon,
            state_metric_diagonal=data["weights"],
            seed_candidate_mask=seed,
            time_steps=TIME_PANELS,
        )
        mask = closure.possible_candidate_mask
        bound = _preflight(arrays, data, mask, horizon)
        bound_row = _bound_row(bound, data["names"])
        closure_headroom = bool(
            closure_headroom
            and closure.converged
            and bound_row["headroom_passed"]
        )
        key = f"{horizon:.3e}"
        closure_rows[key] = {
            "iteration_count": int(closure.iteration_count),
            "converged": bool(closure.converged),
            "used_face_compression": bool(closure.used_face_compression),
            "left_compression_relative_defect": float(
                closure.left_compression_relative_defect
            ),
            "possible_candidate_count": int(closure.possible_candidate_count),
            "possible_face_count": int(closure.possible_face_count),
            "maximum_positive_gap_closure_ratio": float(
                closure.maximum_gap_closure_ratio
            ),
            "maximum_nominal_weighted_state_radius": float(
                closure.maximum_nominal_state_radius
            ),
            "maximum_branch_weighted_state_deviation": float(
                closure.maximum_branch_state_deviation
            ),
            "maximum_total_weighted_state_radius": float(
                closure.maximum_total_state_radius
            ),
            "minimum_radius_with_margin_for_this_envelope": float(
                PROPAGATED_RADIUS_MARGIN_FACTOR
                * closure.maximum_total_state_radius
            ),
            "bound": bound_row,
            "binding": False,
        }
        closure_masks.append(mask)
        result_arrays[f"closure_{key}_possible_candidate_mask"] = mask
        result_arrays[f"closure_{key}_nominal_gap_variations"] = (
            closure.nominal_maximum_gap_variations
        )
        result_arrays[f"closure_{key}_closed_gap_variations"] = (
            closure.closed_maximum_gap_variations
        )
        result_arrays[f"closure_{key}_time_grid_seconds"] = (
            closure.time_grid_seconds
        )
        result_arrays[f"closure_{key}_nominal_state_radius_bounds"] = (
            closure.nominal_state_radius_bounds
        )
        result_arrays[f"closure_{key}_branch_state_deviation_bounds"] = (
            closure.branch_state_deviation_bounds
        )
        result_arrays[f"closure_{key}_total_state_radius_bounds"] = (
            closure.total_state_radius_bounds
        )

    row = {
        "all_candidate_parent_reproduction": {
            "parent_maximum_gate_fraction": parent_maximum,
            "recomputed_maximum_gate_fraction": float(
                decomposition["maximum_gate_fraction"]
            ),
            "absolute_defect": reproduction_defect,
            "passed": True,
        },
        "constraint_null_contract": {
            "coordinate_count": int(data["constraints"].shape[0]),
            "null_dimension": int(data["basis"].shape[1]),
            "basis_sha256": _array_sha256(data["basis"]),
            "state_weights_sha256": _array_sha256(data["weights"]),
            "maximum_constraint_defect": data["constraint_defect"],
            "weighted_orthogonality_defect": (
                data["weighted_orthogonality_defect"]
            ),
            "minimum_initial_weighted_radius": (
                MINIMUM_INITIAL_WEIGHTED_RADIUS
            ),
            "nominal_propagated_radius_by_horizon": nominal_radius_rows,
            "minimum_common_radius_for_this_anchor": required_radius,
        },
        "all_candidate_decomposition": decomposition,
        "weighted_anchor_radius_screen": {
            "binding": False,
            "semantics": screen.semantics,
            "rows": radius_rows,
            "common_contract_radius_selected": None,
            "rigorous_finite_neighborhood_authorized": False,
        },
        "structured_null_tube_closure": {
            "binding": False,
            "zero_nonlinear_remainder": True,
            "anchor_branch_factors_only": True,
            "rows": closure_rows,
            "headroom_passed_all_horizons": closure_headroom,
        },
        "cached_seed_candidate_count": int(np.count_nonzero(seed)),
    }
    if primary_leave_one:
        row["leave_one_diagnostics"] = _leave_one_diagnostics(
            arrays, data, decomposition
        )
        witness, witness_arrays = _nonlinear_switch_witness(
            arrays, data, decomposition
        )
        row["nonlinear_candidate_switch_witness"] = witness
        result_arrays.update(witness_arrays)
    result_arrays.update(
        {
            "dynamic": np.asarray(arrays["dynamic"], dtype=float),
            "constraint_null_basis": data["basis"],
            "state_weights": data["weights"],
            "constraints": data["constraints"],
            "generator_left_factors": left,
            "generator_right_factors": right,
            "physical_left_factors": data["physical_left_factors"],
            "direct_output_left_factors": data["direct_left"],
            "branch_face_indices": data["faces"],
            "branch_candidate_indices": data["candidates"],
            "control_codes_by_face": data["controls_by_face"],
            "candidate_absolute_speeds_by_face": (
                data["candidate_speeds_by_face"]
            ),
            "base_speed_gaps": data["base_gaps"],
            "weighted_dual_gap_gradient_norms": (
                screen.weighted_dual_gradient_norms
            ),
            "candidate_threshold_radii": screen.candidate_threshold_radii,
            "cached_seed_candidate_mask": seed,
            "response_operator": data["response"],
            "output_gates": data["gates"],
        }
    )
    return row, result_arrays


def main() -> None:
    arguments = _arguments()
    output_path = _absolute(arguments.output)
    arrays_path = _absolute(arguments.arrays)
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    cached = json.loads(CACHED_RESULT.read_text(encoding="utf-8"))
    if parent.get("decision") != "wp10c8m_b2_all_face_zero_remainder_infeasible":
        raise RuntimeError("WP10c8m all-face evidence has the wrong decision")
    if _sha256(PARENT_ARRAYS) != parent["artifacts"]["arrays_sha256"]:
        raise RuntimeError("WP10c8m all-face array hash does not match")

    archive = np.load(PARENT_ARRAYS)
    initial_n64, vectors_n64, state_provenance = _load_n64_states()
    shell_edges = _n64_five_shell_edges(initial_n64)
    cases = {}
    all_arrays: dict[str, np.ndarray] = {}
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_rusanov_certification.py",
        ROOT
        / "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py",
        PARENT_RESULT,
        PARENT_ARRAYS,
        CACHED_RESULT,
    )
    original_runner = wp10c8l_a.THIS_RUNNER
    original_children = wp10c8l_a.WP10C8M_RUNNERS
    try:
        wp10c8l_a.THIS_RUNNER = THIS_RUNNER
        wp10c8l_a.WP10C8M_RUNNERS = tuple(
            dict.fromkeys((*original_children, THIS_RUNNER))
        )
        for n_cells, label in LOCKED_CASES:
            case_id = f"n{n_cells}_{label}"
            initial = initial_n64
            vector = vectors_n64[label]
            arrays, metadata, provenance, data = _case_inputs(
                case_id=case_id,
                initial=initial,
                vector=vector,
                shell_edges=shell_edges,
                parent_case=_parent_case_arrays(archive, case_id),
            )
            row, arrays_out = _case(
                case_id=case_id,
                arrays=arrays,
                data=data,
                cached=cached,
                parent=parent,
                primary_leave_one=(case_id == "n64_t_0p025"),
            )
            row["operator_provenance"] = provenance
            row["state_provenance"] = state_provenance[label]
            row["state_vector_sha256"] = _array_sha256(vector)
            cases[case_id] = row
            all_arrays.update(
                {f"{case_id}_{name}": value for name, value in arrays_out.items()}
            )
    finally:
        wp10c8l_a.THIS_RUNNER = original_runner
        wp10c8l_a.WP10C8M_RUNNERS = original_children
        archive.close()

    null_tube_headroom = bool(
        all(
            row["structured_null_tube_closure"][
                "headroom_passed_all_horizons"
            ]
            for row in cases.values()
        )
    )
    witness_row = cases["n64_t_0p025"].get(
        "nonlinear_candidate_switch_witness", {}
    )
    nonlinear_obstruction = bool(
        witness_row.get("nonlinear_switch_witnessed", False)
        and witness_row.get(
            "single_challenger_instantaneous_bound", {}
        ).get("maximum_gate_fraction", 0.0)
        > HEADROOM_MAXIMUM_GATE_FRACTION
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "binding": False,
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "moment_ladder_changed": False,
            "uniform_gap_variation_certified": False,
            "nonlinear_remainders_certified": False,
            "trajectory_containment_certified": False,
            "n128_or_wp10c8i_authorized": False,
        },
        "gates": {
            "headroom_maximum_gate_fraction": (
                HEADROOM_MAXIMUM_GATE_FRACTION
            ),
            "reporting_maximum_gate_fraction": (
                REPORTING_MAXIMUM_GATE_FRACTION
            ),
            "minimum_initial_weighted_radius": (
                MINIMUM_INITIAL_WEIGHTED_RADIUS
            ),
            "propagated_radius_time_panels": (
                PROPAGATED_RADIUS_TIME_PANELS
            ),
            "propagated_radius_margin_factor": (
                PROPAGATED_RADIUS_MARGIN_FACTOR
            ),
            "locked_horizons_seconds": list(LOCKED_HORIZONS_SECONDS),
            "time_panels": TIME_PANELS,
            "radius_ladder": RADIUS_LADDER.tolist(),
        },
        "parents": {
            "all_face_result": {
                "path": _relative(PARENT_RESULT),
                "sha256": _sha256(PARENT_RESULT),
                "decision": parent["decision"],
            },
            "all_face_arrays": {
                "path": _relative(PARENT_ARRAYS),
                "sha256": _sha256(PARENT_ARRAYS),
            },
            "cached_result": {
                "path": _relative(CACHED_RESULT),
                "sha256": _sha256(CACHED_RESULT),
                "decision": cached["decision"],
            },
        },
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
        "candidate_labels": list(CANDIDATE_LABELS),
        "cases": cases,
        "decision": (
            "wp10c8n_possible_winner_screen_rejected_by_nonlinear_witness"
            if nonlinear_obstruction
            else (
                "wp10c8n_null_tube_has_zero_remainder_headroom"
                if null_tube_headroom
                else "wp10c8n_null_tube_zero_remainder_headroom_failed"
            )
        ),
        "next_action": (
            "close_uniform_exact_max_tangent_certificate_and_design_shifted_fiber_or_nonlinear_closure_test"
            if nonlinear_obstruction
            else (
                "certify_uniform_candidate_bounds_remainders_and_containment_at_n64"
                if null_tube_headroom
                else "do_not_build_finite_neighborhood_contract_try_branch_aware_finite_amplitude_localization"
            )
        ),
        "semantics": (
            "Nonbinding WP10c8n localization. Passing any anchor-gradient or "
            "zero-remainder row does not authorize N128, WP10c8i, a moment "
            "change, or reduced evolution. Every finite-neighborhood claim "
            "remains open until uniform candidate bounds, nonlinear state and "
            "output remainders, and one-radius trajectory containment pass."
        ),
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **all_arrays)
    output["artifacts"] = {
        "arrays_path": _relative(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
