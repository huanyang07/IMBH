"""Run the WP10c8y continuum-matched inner-mode audit.

WP10c8x showed that the inherited N128/N256 equal-coordinate pairs were
already different spatial perturbations at the initial time.  This package
defines one smooth compact continuum chart, projects it jointly onto the
local equal-coordinate fibers, and requires the state and fresh-rate
profiles to agree before any phase history becomes binding.

The production boundary is not changed.  Only the production trace and the
already-certified audit-only outgoing-linear flux trace are propagated, and
only after the common-initial-mode precondition passes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.linalg import eigh, subspace_angles

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_anchor_excision_audit_wp10c8w as wp10c8w
import run_causal_inner_boundary_consistency_audit_wp10c8x as wp10c8x
import run_causal_inner_phase_spatial_preflight_wp10c8v as wp10c8v

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_exact_equal_coordinate_lift_pair,
    causal_five_field_state_from_primitives,
    causal_weighted_constraint_fiber_null_projection,
    causal_weighted_constraint_normal_basis,
    pack_causal_five_field_state,
)


BASE_COMMIT = "6764fc117ce453b4deb5c6b1c275a19c7352b4be"
WORK_PACKAGE = "WP10c8y"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_inner_common_mode_audit_wp10c8y.py"
CORE_DAE_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py"
)
CORE_FIBER_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_nonlinear_fiber.py"
)
CORE_SPATIAL_FILE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_spatial_audit.py"
)

MESHES = (64, 128, 256)
FAMILIES = {
    "production": ("inherit", "inherit"),
    "flux_linear": ("linear_outgoing", "inherit"),
}
PROFILE_FIELDS = (1, 4)
PROFILE_DEGREE = 4
PROFILE_COLUMNS = len(PROFILE_FIELDS) * (PROFILE_DEGREE + 1)
COMMON_AMPLITUDE_DEGREE = 5
SEED_MULTIPLIER = 1.0e-3
COMMON_EXTERIOR_INNER_RG = wp10c8w.COMMON_EXTERIOR_INNER_RG

MAXIMUM_PAIR_COORDINATE_DEFECT = 2.0e-10
MAXIMUM_CONSTRAINT_CONDITION = 1.0e10
MINIMUM_INITIAL_SIGNED_COSINE = 0.99
MAXIMUM_INITIAL_AMPLITUDE_DEFECT = 0.05
MAXIMUM_INITIAL_RELATIVE_L2_DEFECT = 0.10
MINIMUM_TEMPLATE_COSINE = 0.20
MINIMUM_HISTORY_SPATIAL_ORDER = 0.75
MINIMUM_HISTORY_SIGNED_COSINE = 0.90
MAXIMUM_ZERO_CROSSING_DEFECT = 0.10
MAXIMUM_FREQUENCY_DEFECT = 0.10
MAXIMUM_DAMPING_DEFECT = 0.25
MAXIMUM_BOUNDARY_FAMILY_HISTORY_DEFECT = 1.0e-3
MINIMUM_BOUNDARY_FAMILY_SIGNED_COSINE = 0.999
MODAL_ENERGY_RESERVE = 0.999

WP10C8X_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_boundary_consistency_audit_wp10c8x.json"
)
WP10C8W_CHECKPOINTS = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8w"
)
WP10C8X_CHECKPOINTS = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8x"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8y"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_mode_audit_wp10c8y.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_inner_common_mode_audit_wp10c8y_arrays.npz"
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _observed_order(coarse: float, fine: float) -> float | None:
    first = float(coarse)
    second = float(fine)
    if not (
        np.isfinite(first)
        and np.isfinite(second)
        and first > 0.0
        and second > 0.0
    ):
        return None
    return float(np.log2(first / second))


def _common_amplitude_function(anchor: dict[str, np.ndarray]):
    """Return a smooth positive primitive metric in the continuum chart."""

    radii = np.asarray(anchor["radius_rg"], dtype=float)
    amplitudes = np.asarray(
        anchor["physical_input_amplitudes"],
        dtype=float,
    )
    if np.any(amplitudes <= 0.0):
        raise ValueError("common-amplitude anchor must be positive")
    argument = np.log(radii)
    fits = tuple(
        Chebyshev.fit(
            argument,
            np.log(amplitudes[:, field]),
            deg=COMMON_AMPLITUDE_DEGREE,
        )
        for field in range(amplitudes.shape[1])
    )

    def evaluate(radius_rg: np.ndarray) -> np.ndarray:
        values = np.log(np.asarray(radius_rg, dtype=float))
        result = np.stack(
            [np.exp(fit(values)) for fit in fits],
            axis=-1,
        )
        if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
            raise RuntimeError("common primitive amplitude is invalid")
        return result

    return evaluate


def _continuum_profile_basis(
    radius_rg: np.ndarray,
    *,
    inner_rg: float,
    outer_rg: float,
) -> np.ndarray:
    """Sample the declared smooth compact stress/transport basis.

    The definition is analytic in ``ln R`` and independent of every mesh
    operator.  A squared-sine endpoint envelope makes the value and first
    derivative vanish at both ends.  The exponential factor concentrates the
    family toward the causal inner region without creating a grid-scale
    pulse.
    """

    radius = np.asarray(radius_rg, dtype=float)
    coordinate = (
        np.log(radius) - np.log(float(inner_rg))
    ) / (
        np.log(float(outer_rg)) - np.log(float(inner_rg))
    )
    active = (coordinate > 0.0) & (coordinate < 1.0)
    envelope = np.zeros_like(coordinate)
    envelope[active] = (
        np.sin(np.pi * coordinate[active]) ** 2
        * np.exp(-3.0 * coordinate[active])
    )
    basis = np.zeros(
        (radius.size, 5, PROFILE_COLUMNS),
        dtype=float,
    )
    column = 0
    chebyshev_argument = 2.0 * coordinate - 1.0
    for field in PROFILE_FIELDS:
        for degree in range(PROFILE_DEGREE + 1):
            coefficients = np.zeros(degree + 1, dtype=float)
            coefficients[-1] = 1.0
            shape = np.polynomial.chebyshev.chebval(
                chebyshev_argument,
                coefficients,
            )
            basis[:, field, column] = envelope * shape
            column += 1
    return basis


def _template_coefficients() -> np.ndarray:
    coefficients = np.zeros(PROFILE_COLUMNS, dtype=float)
    radial_offset = 0
    stress_offset = PROFILE_DEGREE + 1
    coefficients[radial_offset] = -0.35
    coefficients[stress_offset] = 1.0
    return coefficients


def _restrict_basis(
    fine: np.ndarray,
    fine_measures: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fine, dtype=float)
    moved = np.moveaxis(values, -1, 0)
    restricted = wp10c8v._restrict_pairwise(
        moved,
        fine_measures,
    )
    return np.moveaxis(restricted, 0, -1)


def _weighted_matrix(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    fields = np.asarray(values, dtype=float)[mask]
    normalized = np.asarray(weights, dtype=float)[mask]
    normalized = normalized / np.sum(normalized)
    return (
        fields
        * np.sqrt(normalized)[:, None, None]
        / np.sqrt(fields.shape[1])
    ).reshape(-1, fields.shape[-1])


def _profile_metrics(
    coarse: np.ndarray,
    fine_restricted: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict:
    first = np.asarray(coarse, dtype=float)[mask]
    second = np.asarray(fine_restricted, dtype=float)[mask]
    normalized = np.asarray(weights, dtype=float)[mask]
    normalized = normalized / np.sum(normalized)
    first_norm = float(
        np.sqrt(np.sum(normalized[:, None] * first**2))
    )
    second_norm = float(
        np.sqrt(np.sum(normalized[:, None] * second**2))
    )
    difference_norm = float(
        np.sqrt(np.sum(normalized[:, None] * (second - first) ** 2))
    )
    return {
        "signed_cosine": float(
            np.sum(normalized[:, None] * first * second)
            / max(
                first_norm * second_norm,
                np.finfo(float).tiny,
            )
        ),
        "amplitude_ratio": float(
            second_norm / max(first_norm, np.finfo(float).tiny)
        ),
        "amplitude_defect": float(
            abs(
                second_norm / max(first_norm, np.finfo(float).tiny)
                - 1.0
            )
        ),
        "relative_l2_difference": float(
            difference_norm
            / max(first_norm, np.finfo(float).tiny)
        ),
        "coarse_norm": first_norm,
        "fine_norm": second_norm,
    }


def _initial_metric_passed(metric: dict) -> bool:
    return bool(
        metric["signed_cosine"] >= MINIMUM_INITIAL_SIGNED_COSINE
        and metric["amplitude_defect"]
        <= MAXIMUM_INITIAL_AMPLITUDE_DEFECT
        and metric["relative_l2_difference"]
        <= MAXIMUM_INITIAL_RELATIVE_L2_DEFECT
    )


def _candidate_gate_score(metrics: dict[str, dict]) -> float:
    score = 0.0
    for metric in metrics.values():
        score = max(
            score,
            max(
                0.0,
                MINIMUM_INITIAL_SIGNED_COSINE
                - metric["signed_cosine"],
            )
            / max(1.0 - MINIMUM_INITIAL_SIGNED_COSINE, 1.0e-12),
            metric["amplitude_defect"]
            / MAXIMUM_INITIAL_AMPLITUDE_DEFECT,
            metric["relative_l2_difference"]
            / MAXIMUM_INITIAL_RELATIVE_L2_DEFECT,
        )
    return float(score)


def _normalize_coefficients(
    coefficients: np.ndarray,
    reference_matrix: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(coefficients, dtype=float)
    norm = float(np.linalg.norm(reference_matrix @ vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("common profile candidate has zero norm")
    return vector / norm


def _select_common_direction(
    *,
    state_maps: dict[int, np.ndarray],
    rate_maps: dict[str, dict[int, np.ndarray]],
    measures: dict[int, np.ndarray],
    radii: dict[int, np.ndarray],
    active_outer_rg: float,
) -> tuple[np.ndarray, dict]:
    """Choose a smooth jointly resolved direction in the declared basis."""

    coarse_mesh = 128
    fine_mesh = 256
    coarse_mask = np.asarray(
        radii[coarse_mesh] <= active_outer_rg * (1.0 + 2.0e-14),
        dtype=bool,
    )
    state_coarse = _weighted_matrix(
        state_maps[coarse_mesh],
        measures[coarse_mesh],
        coarse_mask,
    )
    state_fine = _weighted_matrix(
        _restrict_basis(
            state_maps[fine_mesh],
            measures[fine_mesh],
        ),
        measures[coarse_mesh],
        coarse_mask,
    )
    matrices = [(state_coarse, state_fine)]
    for family in sorted(rate_maps):
        matrices.append(
            (
                _weighted_matrix(
                    rate_maps[family][coarse_mesh],
                    measures[coarse_mesh],
                    coarse_mask,
                ),
                _weighted_matrix(
                    _restrict_basis(
                        rate_maps[family][fine_mesh],
                        measures[fine_mesh],
                    ),
                    measures[coarse_mesh],
                    coarse_mask,
                ),
            )
        )

    mismatch = np.zeros(
        (PROFILE_COLUMNS, PROFILE_COLUMNS),
        dtype=float,
    )
    reference = np.zeros_like(mismatch)
    for coarse, fine in matrices:
        scale = max(
            float(np.trace(coarse.T @ coarse)),
            np.finfo(float).tiny,
        )
        mismatch += (fine - coarse).T @ (fine - coarse) / scale
        reference += coarse.T @ coarse / scale
    regularization = 1.0e-12 * max(
        float(np.trace(reference)) / PROFILE_COLUMNS,
        1.0,
    )
    eigenvalues, eigenvectors = eigh(
        mismatch + regularization * np.eye(PROFILE_COLUMNS),
        reference + regularization * np.eye(PROFILE_COLUMNS),
    )
    target = _normalize_coefficients(
        _template_coefficients(),
        state_coarse,
    )
    candidates = [("template", target)]
    for index in range(PROFILE_COLUMNS):
        candidates.append(
            (
                f"generalized_mode_{index}",
                _normalize_coefficients(
                    eigenvectors[:, index],
                    state_coarse,
                ),
            )
        )
    identity = np.eye(PROFILE_COLUMNS)
    for exponent in np.linspace(-6.0, 3.0, 19):
        strength = 10.0**float(exponent)
        try:
            vector = np.linalg.solve(
                mismatch + strength * reference
                + regularization * identity,
                strength * (reference @ target),
            )
        except np.linalg.LinAlgError:
            continue
        candidates.append(
            (
                f"template_tradeoff_{exponent:+.1f}",
                _normalize_coefficients(vector, state_coarse),
            )
        )

    target_state = state_coarse @ target
    rows = []
    for label, coefficients in candidates:
        if float(np.dot(state_coarse @ coefficients, target_state)) < 0.0:
            coefficients = -coefficients
        candidate_metrics = {}
        state_by_mesh = {
            mesh: np.einsum(
                "nfk,k->nf",
                state_maps[mesh],
                coefficients,
            )
            for mesh in MESHES
        }
        restricted_state = _restrict_basis(
            state_maps[256],
            measures[256],
        )
        candidate_metrics["state"] = _profile_metrics(
            state_by_mesh[128],
            np.einsum("nfk,k->nf", restricted_state, coefficients),
            measures[128],
            coarse_mask,
        )
        for family in sorted(rate_maps):
            restricted_rate = _restrict_basis(
                rate_maps[family][256],
                measures[256],
            )
            candidate_metrics[f"rate_{family}"] = _profile_metrics(
                np.einsum(
                    "nfk,k->nf",
                    rate_maps[family][128],
                    coefficients,
                ),
                np.einsum(
                    "nfk,k->nf",
                    restricted_rate,
                    coefficients,
                ),
                measures[128],
                coarse_mask,
            )
        template_cosine = float(
            np.dot(state_coarse @ coefficients, target_state)
            / max(
                np.linalg.norm(state_coarse @ coefficients)
                * np.linalg.norm(target_state),
                np.finfo(float).tiny,
            )
        )
        rows.append(
            {
                "label": label,
                "coefficients": coefficients,
                "metrics": candidate_metrics,
                "template_cosine": template_cosine,
                "gate_score": _candidate_gate_score(candidate_metrics),
                "passed": bool(
                    template_cosine >= MINIMUM_TEMPLATE_COSINE
                    and all(
                        _initial_metric_passed(metric)
                        for metric in candidate_metrics.values()
                    )
                ),
            }
        )
    passing = [row for row in rows if row["passed"]]
    if passing:
        selected = sorted(
            passing,
            key=lambda row: (
                -row["template_cosine"],
                row["gate_score"],
                row["label"],
            ),
        )[0]
    else:
        eligible = [
            row
            for row in rows
            if row["template_cosine"] >= MINIMUM_TEMPLATE_COSINE
        ]
        selected = sorted(
            eligible if eligible else rows,
            key=lambda row: (
                row["gate_score"],
                -row["template_cosine"],
                row["label"],
            ),
        )[0]
    return np.asarray(selected["coefficients"], dtype=float), {
        "selected_label": selected["label"],
        "selected_template_cosine": selected["template_cosine"],
        "selected_linear_gate_score": selected["gate_score"],
        "selected_linear_metrics": selected["metrics"],
        "selected_linear_passed": selected["passed"],
        "generalized_eigenvalues": eigenvalues,
        "candidate_count": len(rows),
        "passing_candidate_labels": [
            row["label"] for row in rows if row["passed"]
        ],
        "candidate_summaries": [
            {
                key: value
                for key, value in row.items()
                if key != "coefficients"
            }
            for row in rows
        ],
    }


def _family_operator_path(mesh: int, family: str) -> Path:
    return (
        WP10C8X_CHECKPOINTS
        / f"N{mesh:03d}_{family}_arrays.npz"
    )


def _parent_anchor_path(mesh: int) -> Path:
    return (
        WP10C8W_CHECKPOINTS
        / f"N{mesh:03d}_anchor_inherit_arrays.npz"
    )


def _load_family_operators() -> dict[str, dict[int, dict[str, np.ndarray]]]:
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for mesh in MESHES:
            path = _family_operator_path(mesh, family)
            if not path.exists():
                raise FileNotFoundError(
                    f"WP10c8y requires WP10c8x operator {path}"
                )
            result[family][mesh] = _load_npz(path)
    return result


def _build_mesh_fiber_data(
    *,
    contexts: dict[int, object],
    anchors: dict[int, dict[str, np.ndarray]],
    shell_zero_outer_rg: float,
    active_outer_rg: float,
    amplitude_function,
    family_operators: dict[str, dict[int, dict[str, np.ndarray]]],
) -> tuple[dict[int, dict], dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    mesh_data = {}
    state_maps = {}
    rate_maps = {family: {} for family in FAMILIES}
    for mesh in MESHES:
        context = contexts[mesh]
        anchor = anchors[mesh]
        primitives = np.asarray(anchor["base_primitives"], dtype=float)
        shell_edges = wp10c8w._local_shell_edges_rg(
            context,
            shell_zero_outer_rg,
        )
        print(
            f"WP10c8y: building N{mesh} coordinate fiber",
            flush=True,
        )
        _state, _vector, reduced, level = wp10c8w._reduced_and_ladder(
            context,
            primitives,
            shell_edges,
        )
        scales = np.asarray(
            reduced["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
        weights = wp10c8w._continuum_weights(
            context,
            active_outer_rg=active_outer_rg,
        )
        normal = causal_weighted_constraint_normal_basis(
            np.asarray(level.constraint_matrix, dtype=float),
            weights,
        )
        radius = np.asarray(anchor["radius_rg"], dtype=float)
        common_amplitudes = amplitude_function(radius)
        continuum_basis = _continuum_profile_basis(
            radius,
            inner_rg=float(anchor["grid_edges_rg"][0]),
            outer_rg=active_outer_rg,
        )
        projected = np.empty_like(continuum_basis)
        for column in range(PROFILE_COLUMNS):
            physical = (
                common_amplitudes
                * continuum_basis[:, :, column]
            )
            scaled = physical.ravel() / scales.ravel()
            projected_scaled = (
                causal_weighted_constraint_fiber_null_projection(
                    scaled,
                    weights,
                    normal,
                )
            )
            projected[:, :, column] = (
                scales.ravel() * projected_scaled
            ).reshape(-1, 5) / common_amplitudes
        state_maps[mesh] = projected
        common_generators = {}
        for family in FAMILIES:
            operator = family_operators[family][mesh]
            if not np.array_equal(
                np.asarray(operator["base_primitives"], dtype=float),
                primitives,
            ):
                raise RuntimeError(
                    f"WP10c8y N{mesh} {family} base anchor changed"
                )
            generator = wp10c8v._similarity_rescale_generator(
                np.asarray(operator["generator"], dtype=float),
                np.asarray(
                    operator["primitive_column_scales"],
                    dtype=float,
                ),
                common_amplitudes,
            )
            common_generators[family] = generator
            rate_maps[family][mesh] = (
                generator @ projected.reshape(-1, PROFILE_COLUMNS)
            ).reshape(-1, 5, PROFILE_COLUMNS)
        mesh_data[mesh] = {
            "context": context,
            "primitives": primitives,
            "shell_edges_rg": shell_edges,
            "reduced": reduced,
            "level": level,
            "weights": weights,
            "normal": normal,
            "common_amplitudes": common_amplitudes,
            "continuum_basis": continuum_basis,
            "projected_basis": projected,
            "radius_rg": radius,
            "cell_measures": np.asarray(
                anchor["cell_measures"],
                dtype=float,
            ),
            "common_generators": common_generators,
        }
    return mesh_data, state_maps, rate_maps


def _exact_common_pair(
    *,
    data: dict,
    coefficients: np.ndarray,
    active_outer_rg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    context = data["context"]
    primitives = np.asarray(data["primitives"], dtype=float)
    amplitudes = np.asarray(data["common_amplitudes"], dtype=float)
    scales = np.asarray(
        data["reduced"]["primitive_column_scales"],
        dtype=float,
    ).reshape(-1, 5)
    seed_dimensionless = np.einsum(
        "nfk,k->nf",
        data["continuum_basis"],
        np.asarray(coefficients, dtype=float),
    )
    physical_seed = amplitudes * seed_dimensionless
    radius = np.asarray(data["radius_rg"], dtype=float)
    physical_seed[
        radius > active_outer_rg * (1.0 + 2.0e-14)
    ] = 0.0
    pair = causal_exact_equal_coordinate_lift_pair(
        base_primitive_vector=primitives.ravel(),
        primitive_column_scales=scales.ravel(),
        state_weights=np.asarray(data["weights"], dtype=float),
        physical_input_amplitudes=amplitudes.ravel(),
        target_coordinate_values=np.asarray(
            data["level"].coordinate_values,
            dtype=float,
        ),
        target_coordinate_scales=np.asarray(
            data["level"].coordinate_scales,
            dtype=float,
        ),
        constraint_matrix=np.asarray(
            data["level"].constraint_matrix,
            dtype=float,
        ),
        seed_direction=physical_seed.ravel() / scales.ravel(),
        seed_multiplier=SEED_MULTIPLIER,
        coordinate_evaluator=wp10c8w._moment_evaluator(
            context,
            np.asarray(data["shell_edges_rg"], dtype=float),
        ),
    )
    minus = np.asarray(pair.minus.primitive_vector).reshape(-1, 5)
    plus = np.asarray(pair.plus.primitive_vector).reshape(-1, 5)
    half = 0.5 * (plus - minus)
    minus_state = causal_five_field_state_from_primitives(context, minus)
    plus_state = causal_five_field_state_from_primitives(context, plus)
    minus_gates = wp10c8w._audit_local_state_gates(
        context,
        pack_causal_five_field_state(minus_state),
    )
    plus_gates = wp10c8w._audit_local_state_gates(
        context,
        pack_causal_five_field_state(plus_state),
    )
    buffer = radius > active_outer_rg * (1.0 + 2.0e-14)
    scaled_half = half / amplitudes
    buffer_maximum = (
        float(np.max(np.abs(scaled_half[buffer])))
        if np.any(buffer)
        else 0.0
    )
    report = {
        "maximum_pairwise_coordinate_defect": (
            pair.maximum_pairwise_coordinate_defect
        ),
        "constraint_rank": pair.normal_basis.numerical_rank,
        "constraint_condition": pair.normal_basis.condition_estimate,
        "maximum_buffer_scaled_half_difference": buffer_maximum,
        "minus": {
            "optimizer_success": pair.minus.optimizer_success,
            "maximum_coordinate_defect": (
                pair.minus.maximum_coordinate_defect
            ),
            "correction_fraction": pair.minus.correction_fraction,
            "weighted_direction_cosine": (
                pair.minus.weighted_direction_cosine
            ),
            "maximum_pointwise_amplitude_ratio": (
                pair.minus.maximum_pointwise_amplitude_ratio
            ),
            "state_gates_passed": minus_gates["passed"],
        },
        "plus": {
            "optimizer_success": pair.plus.optimizer_success,
            "maximum_coordinate_defect": (
                pair.plus.maximum_coordinate_defect
            ),
            "correction_fraction": pair.plus.correction_fraction,
            "weighted_direction_cosine": (
                pair.plus.weighted_direction_cosine
            ),
            "maximum_pointwise_amplitude_ratio": (
                pair.plus.maximum_pointwise_amplitude_ratio
            ),
            "state_gates_passed": plus_gates["passed"],
        },
    }
    report["passed"] = bool(
        report["maximum_pairwise_coordinate_defect"]
        <= MAXIMUM_PAIR_COORDINATE_DEFECT
        and report["constraint_condition"]
        <= MAXIMUM_CONSTRAINT_CONDITION
        and all(
            report[side]["optimizer_success"]
            and report[side]["maximum_coordinate_defect"]
            <= MAXIMUM_PAIR_COORDINATE_DEFECT
            and report[side]["state_gates_passed"]
            for side in ("minus", "plus")
        )
    )
    return minus, plus, half, report


def _normalized_initial(
    half_difference: np.ndarray,
    amplitudes: np.ndarray,
    cell_measures: np.ndarray,
) -> np.ndarray:
    state = (
        np.asarray(half_difference, dtype=float)
        / np.asarray(amplitudes, dtype=float)
    )
    norm = wp10c8v._continuum_norm(state, cell_measures)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("WP10c8y exact initial profile is zero")
    return state / norm


def _initial_pair_metrics(
    *,
    states: dict[int, np.ndarray],
    rates: dict[str, dict[int, np.ndarray]],
    mesh_data: dict[int, dict],
    active_outer_rg: float,
) -> dict:
    result = {}
    for coarse, fine in ((64, 128), (128, 256)):
        pair = f"N{coarse}_N{fine}"
        restricted_state = wp10c8v._restrict_pairwise(
            states[fine][None, ...],
            mesh_data[fine]["cell_measures"],
        )[0]
        radius = np.asarray(mesh_data[coarse]["radius_rg"], dtype=float)
        masks = {
            "full_active": (
                radius <= active_outer_rg * (1.0 + 2.0e-14)
            ),
            "common_exterior": (
                (radius >= COMMON_EXTERIOR_INNER_RG)
                & (
                    radius
                    <= active_outer_rg * (1.0 + 2.0e-14)
                )
            ),
        }
        result[pair] = {}
        for region, mask in masks.items():
            region_metrics = {
                "state": _profile_metrics(
                    states[coarse],
                    restricted_state,
                    mesh_data[coarse]["cell_measures"],
                    mask,
                )
            }
            for family in FAMILIES:
                restricted_rate = wp10c8v._restrict_pairwise(
                    rates[family][fine][None, ...],
                    mesh_data[fine]["cell_measures"],
                )[0]
                region_metrics[f"rate_{family}"] = _profile_metrics(
                    rates[family][coarse],
                    restricted_rate,
                    mesh_data[coarse]["cell_measures"],
                    mask,
                )
            result[pair][region] = region_metrics
    fine = result["N128_N256"]
    binding_regions = ("full_active", "common_exterior")
    passed = bool(
        all(
            _initial_metric_passed(metric)
            for region in binding_regions
            for metric in fine[region].values()
        )
    )
    return {
        "pairwise": result,
        "binding_regions": list(binding_regions),
        "passed": passed,
    }


def _history_family(
    family: str,
    *,
    family_operators: dict[str, dict[int, dict[str, np.ndarray]]],
    mesh_data: dict[int, dict],
    half_differences: dict[int, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray], dict[int, dict]]:
    histories = {}
    propagated_arrays = {}
    saved = {}
    safety = {}
    for mesh in MESHES:
        arrays = {
            key: np.asarray(value)
            for key, value in family_operators[family][mesh].items()
        }
        arrays["physical_input_amplitudes"] = np.asarray(
            mesh_data[mesh]["common_amplitudes"],
            dtype=float,
        )
        arrays["matched_half_difference"] = np.asarray(
            half_differences[mesh],
            dtype=float,
        )
        arrays["initial_normalization"] = np.asarray(
            wp10c8v._continuum_norm(
                arrays["matched_half_difference"]
                / arrays["physical_input_amplitudes"],
                arrays["cell_measures"],
            ),
            dtype=float,
        )
        safety[mesh] = wp10c8w._generator_propagation_safety(arrays)
        if not safety[mesh]["passed"]:
            return (
                {
                    "available": False,
                    "reason": "propagation_safety_failed",
                    "propagation_safety": safety,
                    "passed": False,
                },
                saved,
                histories,
            )
        histories[mesh] = wp10c8w._propagate(arrays)
        propagated_arrays[mesh] = arrays
        for field in (
            "times",
            "state",
            "rate",
            "stress_rate_signal",
        ):
            saved[f"{family}_N{mesh}_{field}"] = histories[mesh][field]

    pairwise = {
        "N64_N128": wp10c8w._pair_metrics(
            histories[64],
            histories[128],
            propagated_arrays[64],
            propagated_arrays[128],
            lower_rg=COMMON_EXTERIOR_INNER_RG,
        ),
        "N128_N256": wp10c8w._pair_metrics(
            histories[128],
            histories[256],
            propagated_arrays[128],
            propagated_arrays[256],
            lower_rg=COMMON_EXTERIOR_INNER_RG,
        ),
    }
    state_coarse = pairwise["N64_N128"]["state"][
        "maximum_relative_l2_difference"
    ]
    state_fine = pairwise["N128_N256"]["state"][
        "maximum_relative_l2_difference"
    ]
    rate_coarse = pairwise["N64_N128"]["rate"][
        "maximum_relative_l2_difference"
    ]
    rate_fine = pairwise["N128_N256"]["rate"][
        "maximum_relative_l2_difference"
    ]
    state_order = _observed_order(state_coarse, state_fine)
    rate_order = _observed_order(rate_coarse, rate_fine)
    signals = {
        mesh: wp10c8v._signal_diagnostics(
            histories[mesh]["times"],
            histories[mesh]["stress_rate_signal"],
        )
        for mesh in MESHES
    }
    signal_pairs = {
        "N64_N128": wp10c8v._signal_pair_metrics(
            signals[64],
            signals[128],
        ),
        "N128_N256": wp10c8v._signal_pair_metrics(
            signals[128],
            signals[256],
        ),
    }
    fine_signal = signal_pairs["N128_N256"]
    signal_passed = bool(
        fine_signal["maximum_zero_crossing_relative_defect"] is not None
        and fine_signal["frequency_relative_defect"] is not None
        and fine_signal["damping_relative_defect"] is not None
        and fine_signal["maximum_zero_crossing_relative_defect"]
        <= MAXIMUM_ZERO_CROSSING_DEFECT
        and fine_signal["frequency_relative_defect"]
        <= MAXIMUM_FREQUENCY_DEFECT
        and fine_signal["damping_relative_defect"]
        <= MAXIMUM_DAMPING_DEFECT
    )
    minimum_cosine = min(
        pairwise["N128_N256"]["state"]["minimum_signed_cosine"],
        pairwise["N128_N256"]["rate"]["minimum_signed_cosine"],
    )
    passed = bool(
        state_order is not None
        and rate_order is not None
        and state_order >= MINIMUM_HISTORY_SPATIAL_ORDER
        and rate_order >= MINIMUM_HISTORY_SPATIAL_ORDER
        and minimum_cosine >= MINIMUM_HISTORY_SIGNED_COSINE
        and signal_passed
    )
    return (
        {
            "available": True,
            "propagation_safety": safety,
            "pairwise_history": pairwise,
            "state_observed_order": state_order,
            "rate_observed_order": rate_order,
            "minimum_fine_signed_cosine": minimum_cosine,
            "signals": signals,
            "signal_pairs": signal_pairs,
            "signal_passed": signal_passed,
            "passed": passed,
        },
        saved,
        histories,
    )


def _weighted_snapshot_analysis(
    state: np.ndarray,
    rate: np.ndarray,
    measures: np.ndarray,
    radius_rg: np.ndarray,
    *,
    active_outer_rg: float,
) -> dict:
    radius = np.asarray(radius_rg, dtype=float)
    active = radius <= active_outer_rg * (1.0 + 2.0e-14)
    weights = np.asarray(measures, dtype=float)[active]
    weights = weights / np.sum(weights)
    root = np.repeat(np.sqrt(weights / 5.0), 5)
    state_matrix = np.asarray(state, dtype=float)[:, active].reshape(
        state.shape[0],
        -1,
    )
    rate_matrix = np.asarray(rate, dtype=float)[:, active].reshape(
        rate.shape[0],
        -1,
    )
    weighted_state = state_matrix * root[None, :]
    weighted_rate = rate_matrix * root[None, :]
    _time, singular_values, right_t = np.linalg.svd(
        weighted_state,
        full_matrices=False,
    )
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / max(
        float(np.sum(energy)),
        np.finfo(float).tiny,
    )
    rank = int(np.searchsorted(cumulative, MODAL_ENERGY_RESERVE) + 1)
    rank = min(max(rank, 1), right_t.shape[0])
    spatial = right_t[:rank].T
    coordinates = weighted_state @ spatial
    coordinate_rates = weighted_rate @ spatial
    reduced = np.linalg.lstsq(
        coordinates,
        coordinate_rates,
        rcond=None,
    )[0].T
    eigenvalues = np.linalg.eigvals(reduced)
    order = np.argsort(-np.abs(np.imag(eigenvalues)))
    eigenvalues = eigenvalues[order]
    leading = spatial[:, 0].reshape(-1, 5)
    stress = leading[:, 4]
    sign_changes = int(
        np.count_nonzero(stress[:-1] * stress[1:] < 0.0)
    )
    effective_wavelength = (
        float(
            2.0
            * (radius[active][-1] - radius[active][0])
            / max(sign_changes, 1)
        )
        if radius[active].size > 1
        else None
    )
    cells_per_wavelength = (
        float(
            effective_wavelength
            / np.mean(np.diff(radius[active]))
        )
        if effective_wavelength is not None
        else None
    )
    return {
        "rank_at_energy_reserve": rank,
        "singular_values": singular_values[: min(rank + 3, 12)],
        "energy_reserve": MODAL_ENERGY_RESERVE,
        "weighted_spatial_basis": spatial,
        "reduced_eigenvalues_real": np.real(eigenvalues[:8]),
        "reduced_eigenvalues_imag": np.imag(eigenvalues[:8]),
        "leading_stress_sign_changes": sign_changes,
        "leading_effective_radial_wavelength_rg": effective_wavelength,
        "leading_cells_per_wavelength": cells_per_wavelength,
    }


def _mode_diagnosis(
    histories: dict[str, dict[int, dict]],
    mesh_data: dict[int, dict],
    *,
    active_outer_rg: float,
) -> dict:
    result = {}
    for family, by_mesh in histories.items():
        if not by_mesh:
            continue
        analyses = {
            mesh: _weighted_snapshot_analysis(
                by_mesh[mesh]["state"],
                by_mesh[mesh]["rate"],
                mesh_data[mesh]["cell_measures"],
                mesh_data[mesh]["radius_rg"],
                active_outer_rg=active_outer_rg,
            )
            for mesh in (128, 256)
        }
        coarse_basis = analyses[128]["weighted_spatial_basis"]
        fine_state = wp10c8v._restrict_pairwise(
            by_mesh[256]["state"],
            mesh_data[256]["cell_measures"],
        )
        fine_rate = wp10c8v._restrict_pairwise(
            by_mesh[256]["rate"],
            mesh_data[256]["cell_measures"],
        )
        restricted = _weighted_snapshot_analysis(
            fine_state,
            fine_rate,
            mesh_data[128]["cell_measures"],
            mesh_data[128]["radius_rg"],
            active_outer_rg=active_outer_rg,
        )
        fine_basis = restricted["weighted_spatial_basis"]
        comparison_rank = min(
            coarse_basis.shape[1],
            fine_basis.shape[1],
        )
        angles = subspace_angles(
            coarse_basis[:, :comparison_rank],
            fine_basis[:, :comparison_rank],
        )
        result[family] = {
            "N128": {
                key: value
                for key, value in analyses[128].items()
                if key != "weighted_spatial_basis"
            },
            "N256": {
                key: value
                for key, value in analyses[256].items()
                if key != "weighted_spatial_basis"
            },
            "restricted_N256": {
                key: value
                for key, value in restricted.items()
                if key != "weighted_spatial_basis"
            },
            "comparison_rank": comparison_rank,
            "principal_angles_degrees": np.degrees(angles),
            "maximum_principal_angle_degrees": float(
                np.max(np.degrees(angles))
            ),
        }
    return result


def _boundary_family_comparison(
    histories: dict[str, dict[int, dict]],
    mesh_data: dict[int, dict],
    *,
    active_outer_rg: float,
) -> dict:
    first = histories["production"]
    second = histories["flux_linear"]
    result = {}
    for mesh in MESHES:
        radius = np.asarray(mesh_data[mesh]["radius_rg"], dtype=float)
        mask = (
            (radius >= COMMON_EXTERIOR_INNER_RG)
            & (radius <= active_outer_rg * (1.0 + 2.0e-14))
        )
        weights = np.asarray(
            mesh_data[mesh]["cell_measures"],
            dtype=float,
        )[mask]
        weights = weights / np.sum(weights)
        by_field = {}
        for field in ("state", "rate"):
            reference = np.asarray(first[mesh][field], dtype=float)[:, mask]
            candidate = np.asarray(second[mesh][field], dtype=float)[:, mask]
            reference_norm = np.sqrt(
                np.sum(
                    weights[None, :, None] * reference**2,
                    axis=(1, 2),
                )
            )
            candidate_norm = np.sqrt(
                np.sum(
                    weights[None, :, None] * candidate**2,
                    axis=(1, 2),
                )
            )
            difference_norm = np.sqrt(
                np.sum(
                    weights[None, :, None]
                    * (candidate - reference) ** 2,
                    axis=(1, 2),
                )
            )
            cosine = np.sum(
                weights[None, :, None] * reference * candidate,
                axis=(1, 2),
            ) / np.maximum(
                reference_norm * candidate_norm,
                np.finfo(float).tiny,
            )
            relative = difference_norm / np.maximum(
                reference_norm,
                np.finfo(float).tiny,
            )
            by_field[field] = {
                "maximum_relative_l2_difference": float(
                    np.max(relative)
                ),
                "final_relative_l2_difference": float(relative[-1]),
                "minimum_signed_cosine": float(np.min(cosine)),
                "final_signed_cosine": float(cosine[-1]),
            }
        result[f"N{mesh}"] = by_field
    fine = result["N256"]
    passed = bool(
        max(
            fine["state"]["maximum_relative_l2_difference"],
            fine["rate"]["maximum_relative_l2_difference"],
        )
        <= MAXIMUM_BOUNDARY_FAMILY_HISTORY_DEFECT
        and min(
            fine["state"]["minimum_signed_cosine"],
            fine["rate"]["minimum_signed_cosine"],
        )
        >= MINIMUM_BOUNDARY_FAMILY_SIGNED_COSINE
    )
    return {
        "by_mesh": result,
        "passed": passed,
    }


def run() -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    if not WP10C8X_OUTPUT.exists():
        raise FileNotFoundError("WP10c8y requires WP10c8x evidence")
    wp10c8x_payload = json.loads(
        WP10C8X_OUTPUT.read_text(encoding="utf-8")
    )
    if wp10c8x_payload.get("classification") != (
        "static_pass_but_common_initial_mode_unresolved"
    ):
        raise RuntimeError("WP10c8x stop classification changed")

    parents = {
        mesh: wp10c8v._parent_bundle(mesh)
        for mesh in wp10c8v.PARENT_MESHES
    }
    base_profiles = {
        mesh: wp10c8v._base_profiles(mesh, parents)
        for mesh in MESHES
    }
    contexts = {
        mesh: wp10c8v._local_context(
            parents[128]["context"],
            base_profiles[mesh],
        )
        for mesh in MESHES
    }
    active_outer_rg = float(
        base_profiles[64]["grid"].edges[
            wp10c8v._active_cell_count(64)
        ]
        / contexts[64].grid.gravitational_radius
    )
    wp10c8w._ACTIVE_OUTER_RG = active_outer_rg
    parent_operator = wp10c8v._load_npz(parents[128]["operator_path"])
    shell_zero_outer_rg = float(parent_operator["shell_edges_rg"][1])
    anchors = {
        mesh: _load_npz(_parent_anchor_path(mesh))
        for mesh in MESHES
    }
    family_operators = _load_family_operators()
    amplitude_function = _common_amplitude_function(anchors[256])
    (
        mesh_data,
        state_maps,
        rate_maps,
    ) = _build_mesh_fiber_data(
        contexts=contexts,
        anchors=anchors,
        shell_zero_outer_rg=shell_zero_outer_rg,
        active_outer_rg=active_outer_rg,
        amplitude_function=amplitude_function,
        family_operators=family_operators,
    )
    coefficients, selection = _select_common_direction(
        state_maps=state_maps,
        rate_maps=rate_maps,
        measures={
            mesh: mesh_data[mesh]["cell_measures"]
            for mesh in MESHES
        },
        radii={
            mesh: mesh_data[mesh]["radius_rg"]
            for mesh in MESHES
        },
        active_outer_rg=active_outer_rg,
    )

    pair_rows = {}
    pairs = {}
    half_differences = {}
    states = {}
    rates = {family: {} for family in FAMILIES}
    arrays = {
        "selected_coefficients": coefficients,
        "generalized_eigenvalues": np.asarray(
            selection["generalized_eigenvalues"],
            dtype=float,
        ),
    }
    for mesh in MESHES:
        print(f"WP10c8y: exact N{mesh} common lift", flush=True)
        minus, plus, half, report = _exact_common_pair(
            data=mesh_data[mesh],
            coefficients=coefficients,
            active_outer_rg=active_outer_rg,
        )
        pair_rows[mesh] = report
        pairs[mesh] = (minus, plus)
        half_differences[mesh] = half
        states[mesh] = _normalized_initial(
            half,
            mesh_data[mesh]["common_amplitudes"],
            mesh_data[mesh]["cell_measures"],
        )
        for family in FAMILIES:
            rates[family][mesh] = (
                mesh_data[mesh]["common_generators"][family]
                @ states[mesh].ravel()
            ).reshape(-1, 5)
        arrays[f"N{mesh}_radius_rg"] = mesh_data[mesh]["radius_rg"]
        arrays[f"N{mesh}_cell_measures"] = mesh_data[mesh][
            "cell_measures"
        ]
        arrays[f"N{mesh}_common_amplitudes"] = mesh_data[mesh][
            "common_amplitudes"
        ]
        arrays[f"N{mesh}_continuum_seed"] = np.einsum(
            "nfk,k->nf",
            mesh_data[mesh]["continuum_basis"],
            coefficients,
        )
        arrays[f"N{mesh}_pair_minus"] = minus
        arrays[f"N{mesh}_pair_plus"] = plus
        arrays[f"N{mesh}_half_difference"] = half
        arrays[f"N{mesh}_normalized_initial_state"] = states[mesh]
        for family in FAMILIES:
            arrays[f"N{mesh}_initial_rate_{family}"] = rates[family][
                mesh
            ]

    initial_metrics = _initial_pair_metrics(
        states=states,
        rates=rates,
        mesh_data=mesh_data,
        active_outer_rg=active_outer_rg,
    )
    pair_contract_passed = all(row["passed"] for row in pair_rows.values())
    common_initial_passed = bool(
        pair_contract_passed and initial_metrics["passed"]
    )

    history_results = {}
    history_cache = {}
    if common_initial_passed:
        for family in FAMILIES:
            print(f"WP10c8y: propagating {family}", flush=True)
            result, saved, histories = _history_family(
                family,
                family_operators=family_operators,
                mesh_data=mesh_data,
                half_differences=half_differences,
            )
            history_results[family] = result
            history_cache[family] = histories
            arrays.update(saved)

    passed_histories = [
        family
        for family, result in history_results.items()
        if result.get("available") and result.get("passed")
    ]
    boundary_family_comparison = {}
    if all(history_cache.get(family) for family in FAMILIES):
        boundary_family_comparison = _boundary_family_comparison(
            history_cache,
            mesh_data,
            active_outer_rg=active_outer_rg,
        )
    boundary_insensitive_underresolution = bool(
        common_initial_passed
        and history_results
        and not passed_histories
        and boundary_family_comparison.get("passed", False)
    )
    mode_diagnosis = {}
    if (
        common_initial_passed
        and history_results
        and not passed_histories
    ):
        mode_diagnosis = _mode_diagnosis(
            history_cache,
            mesh_data,
            active_outer_rg=active_outer_rg,
        )

    if not pair_contract_passed:
        classification = "common_equal_coordinate_lift_failed"
    elif not common_initial_passed:
        classification = "common_continuum_mode_not_resolved"
    elif boundary_insensitive_underresolution:
        classification = (
            "common_mode_passed_boundary_insensitive_underresolution"
        )
    elif not passed_histories:
        classification = "common_mode_passed_but_inner_phase_unresolved"
    else:
        classification = "bounded_common_mode_history_converged"

    DEFAULT_ARRAYS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEFAULT_ARRAYS, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "classification": classification,
        "scope": {
            "production_boundary_changed": False,
            "production_physics_changed": False,
            "new_nonlinear_truth_evolution": False,
            "n512_history_run": False,
            "fixed_q_averaging_run": False,
            "formal_fast_average_certified": False,
            "reduced_architecture_selected": False,
        },
        "profile_definition": {
            "coordinate": "ln(R/r_g)",
            "active_inner_rg": float(
                anchors[64]["grid_edges_rg"][0]
            ),
            "active_outer_rg": active_outer_rg,
            "fields": list(PROFILE_FIELDS),
            "field_names": ["beta_R", "chi_specific"],
            "polynomial_family": "Chebyshev",
            "degree": PROFILE_DEGREE,
            "endpoint_envelope": "sin(pi*x)^2*exp(-3*x)",
            "seed_multiplier": SEED_MULTIPLIER,
            "defined_from_mesh_eigenvector": False,
            "common_amplitude_degree": COMMON_AMPLITUDE_DEGREE,
        },
        "selection": selection,
        "pair_contracts": pair_rows,
        "initial_profile": initial_metrics,
        "common_initial_profile_passed": common_initial_passed,
        "history_families": history_results,
        "passed_history_families": passed_histories,
        "boundary_family_comparison": boundary_family_comparison,
        "boundary_insensitive_underresolution": (
            boundary_insensitive_underresolution
        ),
        "mode_diagnosis": mode_diagnosis,
        "gates": {
            "maximum_pair_coordinate_defect": (
                MAXIMUM_PAIR_COORDINATE_DEFECT
            ),
            "minimum_initial_signed_cosine": (
                MINIMUM_INITIAL_SIGNED_COSINE
            ),
            "maximum_initial_amplitude_defect": (
                MAXIMUM_INITIAL_AMPLITUDE_DEFECT
            ),
            "maximum_initial_relative_l2_defect": (
                MAXIMUM_INITIAL_RELATIVE_L2_DEFECT
            ),
            "minimum_template_cosine": MINIMUM_TEMPLATE_COSINE,
            "minimum_history_spatial_order": (
                MINIMUM_HISTORY_SPATIAL_ORDER
            ),
            "minimum_history_signed_cosine": (
                MINIMUM_HISTORY_SIGNED_COSINE
            ),
            "maximum_zero_crossing_defect": (
                MAXIMUM_ZERO_CROSSING_DEFECT
            ),
            "maximum_frequency_defect": MAXIMUM_FREQUENCY_DEFECT,
            "maximum_damping_defect": MAXIMUM_DAMPING_DEFECT,
            "maximum_boundary_family_history_defect": (
                MAXIMUM_BOUNDARY_FAMILY_HISTORY_DEFECT
            ),
            "minimum_boundary_family_signed_cosine": (
                MINIMUM_BOUNDARY_FAMILY_SIGNED_COSINE
            ),
        },
        "decision": {
            "common_equal_coordinate_lift_passed": (
                pair_contract_passed
            ),
            "common_initial_profile_passed": common_initial_passed,
            "bounded_history_run": bool(history_results),
            "bounded_history_passed": bool(passed_histories),
            "n512_local_confirmation_authorized": bool(
                passed_histories
            ),
            "production_boundary_replacement_authorized": False,
            "embedded_patch_preflight_authorized": (
                boundary_insensitive_underresolution
            ),
            "production_embedded_patch_authorized": False,
            "fixed_q_averaging_authorized": False,
        },
        "artifacts": {
            "runner": THIS_RUNNER,
            "runner_sha256": _sha256(ROOT / THIS_RUNNER),
            "core_dae_sha256": _sha256(ROOT / CORE_DAE_FILE),
            "core_fiber_sha256": _sha256(ROOT / CORE_FIBER_FILE),
            "core_spatial_sha256": _sha256(ROOT / CORE_SPATIAL_FILE),
            "wp10c8x_json": _relative(WP10C8X_OUTPUT),
            "wp10c8x_json_sha256": _sha256(WP10C8X_OUTPUT),
            "arrays": _relative(DEFAULT_ARRAYS),
            "arrays_sha256": _sha256(DEFAULT_ARRAYS),
            "parent_operator_arrays": {
                family: {
                    f"N{mesh}": {
                        "path": _relative(
                            _family_operator_path(mesh, family)
                        ),
                        "sha256": _sha256(
                            _family_operator_path(mesh, family)
                        ),
                    }
                    for mesh in MESHES
                }
                for family in FAMILIES
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
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
                "common_initial_profile_passed": payload[
                    "common_initial_profile_passed"
                ],
                "passed_history_families": payload[
                    "passed_history_families"
                ],
                "decision": payload["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
