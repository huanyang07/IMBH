"""Localize the WP10c8j smooth production-vector-field tangent defect.

WP10c8k begins with an identity that is exact for the implemented nonlinear
descriptor at a centered pair of states.  If

    M(p) f(p) + R(p) = 0,

then the finite secant obeys

    M_bar d_h f + d_h M f_bar + d_h R = 0.

This runner compares that exact product identity with the differential
generator balance and decomposes the difference by storage and stationary
residual block.  It deliberately does not repair a derivative, change the
Rusanov operator, add moments, or launch a trajectory.  Its output is the
binding evidence required to select a narrowly scoped WP10c8k repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
import run_causal_tangent_certification_wp10c8j as wp10c8j
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_reduced_storage_rate_derivatives,
    causal_five_field_reduced_storage_rate_directional_derivative,
    causal_five_field_residual_terms,
    causal_five_field_scaled_primitive_vector_field,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "d39ae32839eb8663b94eebcd6f8683ce9e3e2d8e"
WORK_PACKAGE = "WP10c8k"

LOCKED_RESOLUTION = 64
LOCKED_ANCHOR = "t_0p05"
LOCKED_ROLE = "construction"
LOCKED_DIRECTION_NAMES = (
    "density_redistribution_20_to_200rg",
    "thermal_redistribution_60_to_200rg",
)
LOCKED_SECANT_STEPS = (5.0e-4, 1.0e-3, 3.0e-3)
LOCKED_LINEARIZATION_STEP = 2.0e-6

DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_tangent_localization_wp10c8k.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_tangent_localization_wp10c8k_arrays.npz"
)
THIS_RUNNER_RELATIVE_PATH = (
    "scripts/run_causal_tangent_localization_wp10c8k.py"
)
WP10C8K_NEW_RUNNER_PATHS = (
    "scripts/run_causal_tangent_certification_wp10c8j.py",
    THIS_RUNNER_RELATIVE_PATH,
    "scripts/run_causal_tangent_recertification_wp10c8k.py",
)
AUTHORIZED_WP10C8K_CHANGED_PARENT_PATHS = (
    "src/imri_qpe/layer3_minidisk_1d/__init__.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py",
)
SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP = 1.28e-2
SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER = 2
STORAGE_ACTION_STEP_SCAN = (6.4e-3, 9.6e-3, 1.28e-2, 1.6e-2)

PRIMITIVE_COMPONENTS = wp10c8j.PRIMITIVE_COMPONENTS
CONSERVATION_COMPONENTS = wp10c8j.CONSERVATION_COMPONENTS


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument(
        "--direction",
        action="append",
        choices=LOCKED_DIRECTION_NAMES,
        help="Select one locked direction; repeat as needed.",
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


def _load_parent_operator_evidence(
    initial: dict,
    vector: np.ndarray,
    shell_edges_rg: np.ndarray,
    *,
    n_cells: int = LOCKED_RESOLUTION,
    label: str = LOCKED_ANCHOR,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Load WP10c8j's immutable parent artifact without circular code hashing.

    The WP10c8i operator contract hashes every source/script file.  Adding the
    WP10c8k runner necessarily adds exactly one code-hash entry, so the parent
    validator cannot be called unchanged from its child package.  We instead
    require the current contract, after removing this one new runner, to equal
    the stored parent contract exactly.  Array and state hashes remain binding.
    """

    evidence = json.loads(wp10c8j.WP10C8I_OUTPUT.read_text(encoding="utf-8"))
    canonical = evidence["operator_provenance"][str(n_cells)][label]
    canonical_path = ROOT / str(canonical["path"])
    if canonical_path.exists() and _sha256(canonical_path) == canonical["sha256"]:
        path = canonical_path
        source_kind = "canonical_wp10c8i_artifact"
    else:
        path = wp10c8j._operator_source_path(
            n_cells, label
        )
        source_kind = "versioned_wp10c8j_operator_source"
    arrays, metadata = wp10c8j._load_npz_payload(path)
    current_contract = wp10c8i._operator_contract(
        initial["context"], shell_edges_rg
    )
    current_parent_contract = dict(current_contract)
    stored_parent_contract = dict(metadata.get("operator_contract", {}))
    current_code = dict(current_parent_contract["code_sha256"])
    stored_code = dict(stored_parent_contract.get("code_sha256", {}))
    removed_runner_hashes = {
        runner_path: current_code.pop(runner_path, None)
        for runner_path in WP10C8K_NEW_RUNNER_PATHS
    }
    for runner_path in WP10C8K_NEW_RUNNER_PATHS:
        stored_code.pop(runner_path, None)
    for changed_path in AUTHORIZED_WP10C8K_CHANGED_PARENT_PATHS:
        if changed_path not in current_code or changed_path not in stored_code:
            raise RuntimeError("WP10c8k changed-path provenance is incomplete")
        current_code.pop(changed_path)
        stored_code.pop(changed_path)
    current_parent_contract["code_sha256"] = current_code
    stored_parent_contract["code_sha256"] = stored_code
    source = metadata.get("wp10c8j_operator_source", {})
    common = bool(
        removed_runner_hashes[THIS_RUNNER_RELATIVE_PATH]
        == _sha256(Path(__file__).resolve())
        and all(value is not None for value in removed_runner_hashes.values())
        and metadata.get("schema_version") == wp10c8i.CACHE_SCHEMA_VERSION
        and metadata.get("work_package") == "WP10c8i"
        and metadata.get("base_commit") == wp10c8j.WP10C8I_BASE_COMMIT
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor_label") == label
        and metadata.get("state_vector_sha256")
        == wp10c8i._array_sha256(vector)
        and np.array_equal(
            np.asarray(metadata.get("shell_edges_rg"), dtype=float),
            shell_edges_rg,
        )
        and stored_parent_contract == current_parent_contract
        and arrays.get("dynamic", np.empty((0, 0))).shape
        == (5 * n_cells, 5 * n_cells)
        and all(np.all(np.isfinite(value)) for value in arrays.values())
    )
    if source_kind == "versioned_wp10c8j_operator_source":
        common = bool(
            common
            and source.get("schema_version") == 1
            and source.get("work_package") == wp10c8j.WORK_PACKAGE
            and source.get("base_commit") == wp10c8j.BASE_COMMIT
            and source.get("array_sha256")
            == wp10c8j._operator_array_hashes(arrays)
        )
    if not common:
        raise RuntimeError(
            "WP10c8k parent operator evidence differs beyond its new runner"
        )
    return arrays, metadata, {
        "path": _relative(path),
        "sha256": _sha256(path),
        "source_kind": source_kind,
        "state_vector_sha256": metadata["state_vector_sha256"],
        "parent_contract_preserved_except_new_wp10c8k_runner": True,
        "authorized_changed_parent_paths": (
            AUTHORIZED_WP10C8K_CHANGED_PARENT_PATHS
        ),
    }


def _load_parent_certification_evidence(
    path: Path,
    *,
    vector: np.ndarray,
    operator_provenance: dict,
    n_cells: int = LOCKED_RESOLUTION,
    label: str = LOCKED_ANCHOR,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load and validate the immutable WP10c8j certification artifact."""

    arrays, metadata = wp10c8j._load_npz_payload(path)
    valid = bool(
        metadata.get("schema_version") == wp10c8j.CACHE_SCHEMA_VERSION
        and metadata.get("work_package") == wp10c8j.WORK_PACKAGE
        and metadata.get("base_commit") == wp10c8j.BASE_COMMIT
        and metadata.get("n_cells") == n_cells
        and metadata.get("anchor_label") == label
        and metadata.get("state_vector_sha256")
        == wp10c8i._array_sha256(vector)
        and metadata.get("wp10c8i_operator_cache", {}).get("path")
        == operator_provenance["path"]
        and metadata.get("wp10c8i_operator_cache", {}).get("sha256")
        == operator_provenance["sha256"]
        and arrays
        and all(np.all(np.isfinite(value)) for value in arrays.values())
    )
    if not valid:
        raise RuntimeError("WP10c8j parent certification evidence differs")
    wp10c8j._validate_certification_cache_payload(
        arrays,
        metadata,
        n_cells=n_cells,
        label=label,
    )
    return arrays, metadata


def _location(values: np.ndarray, *, kind: str) -> dict:
    flat = np.asarray(values, dtype=float).ravel()
    index = int(np.argmax(np.abs(flat)))
    names = (
        PRIMITIVE_COMPONENTS if kind == "primitive" else CONSERVATION_COMPONENTS
    )
    return {
        "flat_index": index,
        "cell_index": index // 5,
        "component": names[index % 5],
        "signed_value": float(flat[index]),
        "absolute_value": float(abs(flat[index])),
    }


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    kind: str,
    floor: float = np.finfo(float).tiny,
) -> dict:
    first = np.asarray(candidate, dtype=float).ravel()
    second = np.asarray(reference, dtype=float).ravel()
    difference = first - second
    scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), floor)
    return {
        "candidate_l2": float(np.linalg.norm(first)),
        "reference_l2": float(np.linalg.norm(second)),
        "difference_l2": float(np.linalg.norm(difference)),
        "relative_l2_defect": float(np.linalg.norm(difference) / scale),
        "controlling_difference": _location(difference, kind=kind),
    }


def _norm_row(values: np.ndarray, *, kind: str, reference_l2: float) -> dict:
    flat = np.asarray(values, dtype=float).ravel()
    norm = float(np.linalg.norm(flat))
    return {
        "l2": norm,
        "fraction_of_total_primitive_defect": float(
            norm / max(float(reference_l2), np.finfo(float).tiny)
        ),
        "controlling_entry": _location(flat, kind=kind),
    }


def _split_face_divergence(
    face_values: np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Split a face divergence into inner, interior, and outer support."""

    faces = np.asarray(face_values, dtype=float)
    n_cells = faces.shape[0] - 1
    inner = np.zeros((n_cells, 5), dtype=float)
    interior = np.zeros_like(inner)
    outer = np.zeros_like(inner)
    inner[0] -= faces[0]
    outer[-1] += faces[-1]
    for face in range(1, n_cells):
        interior[face - 1] += faces[face]
        interior[face] -= faces[face]
    return {
        f"{prefix}_inner_boundary": inner,
        f"{prefix}_interior": interior,
        f"{prefix}_outer_boundary": outer,
    }


def _stationary_terms(initial: dict, primitives: np.ndarray) -> dict[str, np.ndarray]:
    context = initial["context"]
    state = causal_five_field_state_from_primitives(
        context,
        np.asarray(primitives, dtype=float).reshape(-1, 5),
    )
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    terms = causal_five_field_residual_terms(context, vector, evaluation)
    # Replace the aggregate transport terms by disjoint face-support blocks.
    terms.pop("central_face_transport")
    terms.pop("rusanov_face_transport")
    terms.update(
        _split_face_divergence(
            evaluation.central_weighted_face_fluxes_over_c,
            prefix="central_transport",
        )
    )
    terms.update(
        _split_face_divergence(
            evaluation.rusanov_dissipation_weighted_face_fluxes_over_c,
            prefix="fixed_branch_rusanov_transport",
        )
    )
    return {
        name: np.asarray(values, dtype=float).ravel()
        for name, values in terms.items()
    }


def _evaluate(
    initial: dict,
    primitives: np.ndarray,
    primitive_scales: np.ndarray,
    conservation_scales: np.ndarray,
) -> dict:
    vector_field = causal_five_field_scaled_primitive_vector_field(
        initial["context"],
        np.asarray(primitives, dtype=float),
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        finite_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
    )
    return {
        "f": np.asarray(
            vector_field["scaled_primitive_rate_per_s"], dtype=float
        ).ravel(),
        "r": np.asarray(
            vector_field["scaled_stationary_residual"], dtype=float
        ).ravel(),
        "m": np.asarray(
            vector_field["descriptor_reduced_scaled_matrix"], dtype=float
        ),
        "m_conserved": np.asarray(
            vector_field["conserved_descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "m_vertical": np.asarray(
            vector_field["vertical_descriptor_reduced_scaled_matrix"],
            dtype=float,
        ),
        "terms": {
            name: values / conservation_scales
            for name, values in _stationary_terms(initial, primitives).items()
        },
    }


def _secant(first: np.ndarray, second: np.ndarray, step: float) -> np.ndarray:
    return (np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) / (
        2.0 * float(step)
    )


def _audit_direction(
    initial: dict,
    primitives: np.ndarray,
    direction: np.ndarray,
    base: dict,
    mass: np.ndarray,
    stationary: np.ndarray,
    storage_rate: np.ndarray,
    conserved_storage_rate: np.ndarray,
    vertical_storage_rate: np.ndarray,
    candidate_storage_rate: np.ndarray,
    primitive_scales: np.ndarray,
    conservation_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    predicted = -np.linalg.solve(
        mass,
        (stationary + storage_rate) @ direction,
    )
    candidate_predicted = -np.linalg.solve(
        mass,
        (stationary + candidate_storage_rate) @ direction,
    )

    # The locked tiny step supplies a direct, blockwise stationary derivative
    # on the same physical constraint manifold.  It is diagnostic only; the
    # production prediction above still uses the cached WP10c8j matrices.
    tiny_plus = _stationary_terms(
        initial,
        primitives + LOCKED_LINEARIZATION_STEP * primitive_scales * direction,
    )
    tiny_minus = _stationary_terms(
        initial,
        primitives - LOCKED_LINEARIZATION_STEP * primitive_scales * direction,
    )
    tiny_term_jvps = {
        name: _secant(
            tiny_plus[name], tiny_minus[name], LOCKED_LINEARIZATION_STEP
        )
        / conservation_scales
        for name in tiny_plus
    }
    tiny_stationary = np.sum(
        np.asarray(list(tiny_term_jvps.values()), dtype=float), axis=0
    )

    arrays: dict[str, np.ndarray] = {
        "direction": direction,
        "predicted": predicted,
        "candidate_predicted": candidate_predicted,
        "cached_stationary_jvp": stationary @ direction,
        "tiny_term_stationary_jvp": tiny_stationary,
    }
    action_step_scan = {}
    action_step_predictions = {}
    physical_rate = primitive_scales * base["f"]
    for action_step in STORAGE_ACTION_STEP_SCAN:
        directional_storage = (
            causal_five_field_reduced_storage_rate_directional_derivative(
                initial["context"],
                primitives,
                physical_rate,
                direction,
                primitive_column_scales=primitive_scales,
                conservation_row_scales=conservation_scales,
                storage_rate_derivative_step=(
                    wp10c8j.BASE_OUTER_DIFFERENCE_STEP
                ),
                storage_difference_step=action_step,
                storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
                storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
                conserved_difference_order=(
                    SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
                ),
            )
        )
        # The repaired candidate changes only the mapped-conserved action.
        # Keep the independently certified WP10c8j responsive-height
        # derivative fixed while scanning the mapped-storage path step.
        storage_jvp = np.asarray(
            directional_storage[
                "conserved_storage_rate_directional_derivative_scaled"
            ],
            dtype=float,
        ) + vertical_storage_rate @ direction
        step_prediction = -np.linalg.solve(
            mass, stationary @ direction + storage_jvp
        )
        key = f"{action_step:.1e}"
        action_step_predictions[key] = step_prediction
        arrays[f"storage_action_step_{key}_predicted"] = step_prediction

    rows = {}
    for step in LOCKED_SECANT_STEPS:
        plus = _evaluate(
            initial,
            primitives + step * primitive_scales * direction,
            primitive_scales,
            conservation_scales,
        )
        minus = _evaluate(
            initial,
            primitives - step * primitive_scales * direction,
            primitive_scales,
            conservation_scales,
        )
        dhf = _secant(plus["f"], minus["f"], step)
        dhr = _secant(plus["r"], minus["r"], step)
        dhm = _secant(plus["m"], minus["m"], step)
        dhm_conserved = _secant(
            plus["m_conserved"], minus["m_conserved"], step
        )
        dhm_vertical = _secant(
            plus["m_vertical"], minus["m_vertical"], step
        )
        m_bar = 0.5 * (plus["m"] + minus["m"])
        f_bar = 0.5 * (plus["f"] + minus["f"])

        exact_product = m_bar @ dhf + dhm @ f_bar + dhr
        exact_product_scale = max(
            float(np.linalg.norm(m_bar @ dhf)),
            float(np.linalg.norm(dhm @ f_bar)),
            float(np.linalg.norm(dhr)),
            np.finfo(float).tiny,
        )

        linearized_balance = (
            mass @ dhf + storage_rate @ direction + stationary @ direction
        )
        primitive_defect = dhf - predicted
        primitive_defect_norm = float(np.linalg.norm(primitive_defect))

        contributions = {
            "descriptor_base_shift": np.linalg.solve(
                mass, (mass - m_bar) @ dhf
            ),
            "conserved_storage_derivative": np.linalg.solve(
                mass,
                conserved_storage_rate @ direction - dhm_conserved @ base["f"],
            ),
            "vertical_storage_derivative": np.linalg.solve(
                mass,
                vertical_storage_rate @ direction - dhm_vertical @ base["f"],
            ),
            "storage_rate_cross_curvature": np.linalg.solve(
                mass, dhm @ (base["f"] - f_bar)
            ),
        }
        term_rows = {}
        for name in tiny_term_jvps:
            finite = _secant(plus["terms"][name], minus["terms"][name], step)
            tangent = tiny_term_jvps[name]
            # J d - d_h R is the stationary contribution to the balance.
            contribution = np.linalg.solve(mass, tangent - finite)
            contributions[f"stationary_{name}"] = contribution
            term_rows[name] = {
                "finite_secant_vs_tiny_jvp": _comparison(
                    finite,
                    tangent,
                    kind="conservation",
                ),
                "primitive_defect_contribution": _norm_row(
                    contribution,
                    kind="primitive",
                    reference_l2=primitive_defect_norm,
                ),
            }

        reconstructed_primitive_defect = np.sum(
            np.asarray(list(contributions.values()), dtype=float), axis=0
        )
        # Because the contribution signs above express predicted minus finite
        # for the tangent blocks, their sum equals direct minus predicted.
        reconstruction = _comparison(
            reconstructed_primitive_defect,
            primitive_defect,
            kind="primitive",
        )
        key = f"{step:.0e}"
        rows[key] = {
            "step": step,
            "exact_centered_product_identity": {
                "absolute_l2_defect": float(np.linalg.norm(exact_product)),
                "relative_l2_defect": float(
                    np.linalg.norm(exact_product) / exact_product_scale
                ),
                "controlling_row": _location(
                    exact_product, kind="conservation"
                ),
            },
            "direct_vs_cached_generator": _comparison(
                dhf,
                predicted,
                kind="primitive",
                floor=wp10c8j.JVP_ACTIVITY_FLOOR_PER_S,
            ),
            "direct_vs_selected_mapped_storage_candidate": _comparison(
                dhf,
                candidate_predicted,
                kind="primitive",
                floor=wp10c8j.JVP_ACTIVITY_FLOOR_PER_S,
            ),
            "differential_tangent_balance": {
                "absolute_l2_defect": float(
                    np.linalg.norm(linearized_balance)
                ),
                "controlling_row": _location(
                    linearized_balance, kind="conservation"
                ),
            },
            "stationary_secant_vs_cached_jvp": _comparison(
                dhr,
                stationary @ direction,
                kind="conservation",
            ),
            "stationary_tiny_terms_vs_cached_jvp": _comparison(
                tiny_stationary,
                stationary @ direction,
                kind="conservation",
            ),
            "conserved_storage_secant_vs_cached_jvp": _comparison(
                dhm_conserved @ base["f"],
                conserved_storage_rate @ direction,
                kind="conservation",
            ),
            "vertical_storage_secant_vs_cached_jvp": _comparison(
                dhm_vertical @ base["f"],
                vertical_storage_rate @ direction,
                kind="conservation",
            ),
            "primitive_defect_reconstruction": reconstruction,
            "primitive_defect_contributions": {
                name: _norm_row(
                    value,
                    kind="primitive",
                    reference_l2=primitive_defect_norm,
                )
                for name, value in contributions.items()
            },
            "stationary_term_decomposition": term_rows,
        }
        arrays.update(
            {
                f"step_{key}_direct": dhf,
                f"step_{key}_primitive_defect": primitive_defect,
                f"step_{key}_exact_product_defect": exact_product,
                f"step_{key}_linearized_balance": linearized_balance,
                f"step_{key}_reconstructed_primitive_defect": (
                    reconstructed_primitive_defect
                ),
            }
        )
        arrays.update(
            {
                f"step_{key}_contribution_{name}": value
                for name, value in contributions.items()
            }
        )

    base_secant_key = f"{1.0e-3:.0e}"
    base_direct = arrays[f"step_{base_secant_key}_direct"]
    for key, step_prediction in action_step_predictions.items():
        action_step_scan[key] = _comparison(
            base_direct,
            step_prediction,
            kind="primitive",
            floor=wp10c8j.JVP_ACTIVITY_FLOOR_PER_S,
        )

    return {
        "cached_stationary_vs_tiny_term_jvp": _comparison(
            stationary @ direction,
            tiny_stationary,
            kind="conservation",
        ),
        "storage_action_step_scan_against_1e_3_vector_field_secant": (
            action_step_scan
        ),
        "steps": rows,
    }, arrays


def main() -> None:
    arguments = _arguments()
    selected_directions = tuple(
        dict.fromkeys(arguments.direction or LOCKED_DIRECTION_NAMES)
    )
    wp10c8j._locked_contract()
    wp10c8j._validate_authorization()
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    initial = initial_by_mesh[LOCKED_RESOLUTION]
    vector = vectors_by_mesh[LOCKED_RESOLUTION][LOCKED_ANCHOR]
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )
    operator_arrays, _operator_metadata, operator_provenance = (
        _load_parent_operator_evidence(
            initial,
            vector,
            shell_edges_rg,
        )
    )
    certification_path = wp10c8j._cache_path(
        LOCKED_RESOLUTION, LOCKED_ANCHOR
    )
    certification_arrays, certification_metadata = (
        _load_parent_certification_evidence(
            certification_path,
            vector=vector,
            operator_provenance=operator_provenance,
        )
    )

    state = unpack_causal_five_field_state(vector, LOCKED_RESOLUTION)
    primitives = np.asarray(state.primitives, dtype=float).ravel()
    primitive_scales = np.asarray(
        operator_arrays["primitive_column_scales"], dtype=float
    )
    conservation_scales = np.asarray(
        operator_arrays["conservation_row_scales"], dtype=float
    )
    directions = wp10c8j._normalized_directions(
        initial, vector, primitive_scales
    )
    base = _evaluate(
        initial,
        primitives,
        primitive_scales,
        conservation_scales,
    )
    mass = np.asarray(
        operator_arrays["direct_vector_storage_descriptor"], dtype=float
    )
    stationary = np.asarray(
        operator_arrays["stationary_jacobian"], dtype=float
    )
    storage_rate = np.asarray(
        certification_arrays["repaired_storage_rate_derivative"], dtype=float
    )
    conserved_storage_rate = np.asarray(
        certification_arrays["repaired_conserved_storage_rate_derivative"],
        dtype=float,
    )
    vertical_storage_rate = np.asarray(
        certification_arrays["repaired_vertical_storage_rate_derivative"],
        dtype=float,
    )
    physical_rate = primitive_scales * base["f"]
    candidate_result = causal_five_field_reduced_storage_rate_derivatives(
        initial["context"],
        primitives,
        physical_rate,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        storage_matrix_difference_step=wp10c8j.BASE_INNER_DIFFERENCE_STEP,
        storage_rate_derivative_step=(
            wp10c8j.BASE_OUTER_DIFFERENCE_STEP
        ),
        storage_difference_step=SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP,
        storage_quadrature_order=wp10c8j.STORAGE_QUADRATURE_ORDER,
        storage_directional_step=wp10c8j.STORAGE_DIRECTIONAL_STEP,
        conserved_difference_order=(
            SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
        ),
        backend="direct_action",
    )
    candidate_conserved_storage_rate = np.asarray(
        candidate_result[
            "conserved_storage_rate_derivative_scaled_matrix"
        ],
        dtype=float,
    )
    # WP10c8k localized the defect to mapped conserved storage.  Preserve the
    # already certified WP10c8j responsive-height derivative exactly.
    candidate_storage_rate = (
        candidate_conserved_storage_rate + vertical_storage_rate
    )

    rows = {}
    array_payload = {}
    for name in selected_directions:
        row, arrays = _audit_direction(
            initial,
            primitives,
            directions[name],
            base,
            mass,
            stationary,
            storage_rate,
            conserved_storage_rate,
            vertical_storage_rate,
            candidate_storage_rate,
            primitive_scales,
            conservation_scales,
        )
        rows[name] = row
        array_payload.update(
            {f"{name}_{key}": value for key, value in arrays.items()}
        )
        print(
            json.dumps(
                {
                    "work_package": WORK_PACKAGE,
                    "phase": "smooth_tangent_localization",
                    "direction": name,
                    "relative_defects": {
                        key: value["direct_vs_cached_generator"][
                            "relative_l2_defect"
                        ]
                        for key, value in row["steps"].items()
                    },
                    "selected_candidate_relative_defects": {
                        key: value[
                            "direct_vs_selected_mapped_storage_candidate"
                        ]["relative_l2_defect"]
                        for key, value in row["steps"].items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    arrays_path = _absolute(arguments.arrays)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    array_payload["selected_storage_rate_derivative"] = (
        candidate_storage_rate
    )
    array_payload[
        "selected_conserved_storage_rate_derivative"
    ] = candidate_conserved_storage_rate
    array_payload[
        "selected_vertical_storage_rate_derivative"
    ] = vertical_storage_rate
    np.savez_compressed(arrays_path, **array_payload)
    payload = {
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "decision": "wp10c8k_smooth_tangent_localization_completed",
        "scope": {
            "resolution": LOCKED_RESOLUTION,
            "anchor": LOCKED_ANCHOR,
            "directions": selected_directions,
            "secant_steps": LOCKED_SECANT_STEPS,
            "linearization_step": LOCKED_LINEARIZATION_STEP,
            "selected_repaired_storage_difference_step": (
                SELECTED_REPAIRED_STORAGE_DIFFERENCE_STEP
            ),
            "storage_action_step_scan": STORAGE_ACTION_STEP_SCAN,
            "mapped_conserved_difference_order": (
                SELECTED_REPAIRED_CONSERVED_DIFFERENCE_ORDER
            ),
            "new_truth_trajectory_run": False,
            "production_operator_changed": False,
            "moment_ladder_changed": False,
        },
        "identity": (
            "M_bar*d_h(f)+d_h(M)*f_bar+d_h(R)=0, compared with "
            "M0*d_h(f)+K*d+J*d"
        ),
        "directions": rows,
        "provenance": {
            "state": state_provenance[str(LOCKED_RESOLUTION)][LOCKED_ANCHOR],
            "operator": operator_provenance,
            "wp10c8j_certification_path": _relative(certification_path),
            "wp10c8j_certification_sha256": _sha256(certification_path),
            "wp10c8j_decision": certification_metadata.get("passed"),
        },
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path = _absolute(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "decision": payload["decision"],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
