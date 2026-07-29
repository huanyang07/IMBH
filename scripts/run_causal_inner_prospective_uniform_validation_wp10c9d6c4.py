#!/usr/bin/env python3
"""Run prospective held-out uniform validation without changing the operator.

WP10c9d6c3 certified the unchanged monolithic uniform operator for two smooth,
proper-measure continuum perturbations.  This package freezes that background,
projection, tangent, scaling, and convergence contract before testing:

1. a C4 proper-measure fit to the historical common perturbation, used only as
   a calibration profile; and
2. four prospectively declared analytic held-outs with distinct supports,
   widths, and five-field mixtures, including a broad outer-inner profile and
   a smooth first-cell-dominated mode that leaves through the excision boundary.

No physical or numerical operator is changed.  Every profile is projected
independently on N64/N128/N256/N512 at two quadrature orders.  The inherited
instantaneous and cumulative export gates remain binding, and explicit
sign/amplitude propagation checks certify that the declared tests remain in
the frozen linear regime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as wp10c9d6c3

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_reconstruct_face_charts,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_characteristic_phase import (
    causal_five_field_characteristic_basis,
)  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c4"
ANALYZED_BASE_COMMIT = "3d973aa28c68d242bd33c88efe2226e2e48eb281"
ANALYZED_BASE_PARENT = "da2d7612cc9a2fff7093bee705f3f5fbe2d2101d"
ANALYZED_BASE_TREE = "09cf98cbf29af9ad51e95be6c8591ff39ea73beb"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_prospective_uniform_validation_wp10c9d6c4.py"
)

MESHES = tuple(wp10c9d6c3.MESHES)
LABELS = tuple(wp10c9d6c3.LABELS)
ACTIVE_CELLS = tuple(wp10c9d6c3.ACTIVE_CELLS)
REFERENCE_LABEL = wp10c9d6c3.REFERENCE_LABEL
PRIMARY_PROJECTION_ORDER = wp10c9d6c3.PRIMARY_PROJECTION_ORDER
SECONDARY_PROJECTION_ORDER = wp10c9d6c3.SECONDARY_PROJECTION_ORDER
PRIMARY_STRIDE = wp10c9d6c3.PRIMARY_STRIDE
STRIDE_AUDITS = tuple(wp10c9d6c3.STRIDE_AUDITS)

CALIBRATION_PROFILES = ("historical_common_smooth_fit",)
HELDOUT_PROFILES = (
    "heldout_mid_inner",
    "heldout_broad_outer_inner",
    "heldout_first_cell_outgoing",
    "heldout_two_lobe_mixed",
)
PERTURBATIONS = CALIBRATION_PROFILES + HELDOUT_PROFILES

# These definitions are frozen before any c4 propagation is evaluated.
# Coefficients are dimensionless multipliers of the c3 physical field scales.
ANALYTIC_PROFILE_DEFINITIONS = {
    "heldout_mid_inner": {
        "role": "heldout",
        "description": "moderately narrow mixed profile in the middle inner domain",
        "components": (
            {
                "center_over_rg": 4.60,
                "log_width": 0.17,
                "coefficients": (
                    -0.010,
                    0.006,
                    -0.009,
                    0.0018,
                    -0.028,
                ),
            },
        ),
    },
    "heldout_broad_outer_inner": {
        "role": "heldout",
        "description": (
            "two-component broad profile extending toward the outer "
            "inner domain"
        ),
        "components": (
            {
                "center_over_rg": 5.80,
                "log_width": 0.48,
                "coefficients": (
                    0.004,
                    0.006,
                    -0.003,
                    0.0008,
                    0.018,
                ),
            },
            {
                "center_over_rg": 9.20,
                "log_width": 0.28,
                "coefficients": (
                    -0.002,
                    0.004,
                    0.006,
                    -0.0006,
                    -0.014,
                ),
            },
        ),
    },
    "heldout_first_cell_outgoing": {
        "role": "heldout",
        "description": (
            "smooth first-cell-dominated inward-acoustic characteristic "
            "that exits through the excision boundary"
        ),
        "center_over_rg": 1.84,
        "log_width": 0.065,
        "characteristic_family": "inward_acoustic",
        "amplitude": 0.015,
    },
    "heldout_two_lobe_mixed": {
        "role": "heldout",
        "description": "oppositely signed two-lobe mixed profile",
        "components": (
            {
                "center_over_rg": 2.75,
                "log_width": 0.11,
                "coefficients": (
                    0.007,
                    -0.010,
                    0.004,
                    0.0012,
                    0.024,
                ),
            },
            {
                "center_over_rg": 3.75,
                "log_width": 0.16,
                "coefficients": (
                    -0.005,
                    0.008,
                    0.007,
                    -0.0010,
                    -0.020,
                ),
            },
        ),
    },
}

HISTORICAL_PROFILE_DEFINITION = {
    "role": "calibration_only",
    "source": (
        "N128 historical common physical perturbation from the committed "
        "WP10c9d6c replay inputs"
    ),
    "primary_degree": 5,
    "independent_degree": 7,
    "coefficient_count": wp10c9d6c3.BACKGROUND_COEFFICIENT_COUNT,
    "inner_anchor": "first N128 target cell average",
    "outer_anchor": "zero perturbation",
}

AMPLITUDE_SIGN_FACTORS = (-1.0, 0.5)
AMPLITUDE_SIGN_LABELS = (REFERENCE_LABEL, LABELS[-1])

MINIMUM_EXPORT_ORDER = wp10c9d6c3.MINIMUM_EXPORT_ORDER
MAXIMUM_FINE_PHYSICAL_DIFFERENCE = (
    wp10c9d6c3.MAXIMUM_FINE_PHYSICAL_DIFFERENCE
)
MINIMUM_HISTORY_COSINE = wp10c9d6c3.MINIMUM_HISTORY_COSINE
MINIMUM_ERROR_COSINE = wp10c9d6c3.MINIMUM_ERROR_COSINE
MINIMUM_RELATIVE_ACTIVITY = wp10c9d6c3.MINIMUM_RELATIVE_ACTIVITY
MAXIMUM_RESTART_DEFECT = wp10c9d6c3.MAXIMUM_RESTART_DEFECT
MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE = (
    wp10c9d6c3.MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE
)
MAXIMUM_LIFT_RATE_RELATIVE_DIFFERENCE = (
    wp10c9d6c3.MAXIMUM_LIFT_RATE_RELATIVE_DIFFERENCE
)
MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO = (
    wp10c9d6c3.MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO
)
MAXIMUM_CONTINUUM_MODEL_DISAGREEMENT = (
    wp10c9d6c3.MAXIMUM_CONTINUUM_MODEL_DISAGREEMENT
)

MAXIMUM_FROZEN_BACKGROUND_DEFECT = 1.0e-14
MAXIMUM_HISTORICAL_FIT_RELATIVE_L2_DEFECT = 0.08
MINIMUM_HISTORICAL_FIT_COSINE = 0.995
MAXIMUM_HISTORICAL_REPRESENTATION_DIFFERENCE = 0.02
MAXIMUM_PERTURBED_RECONSTRUCTION_FACTOR_CHANGE = 0.0
MAXIMUM_SIGN_AMPLITUDE_STATE_DEFECT = 1.0e-11
MAXIMUM_SIGN_AMPLITUDE_EXPORT_DEFECT = 1.0e-11

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_prospective_uniform_validation_wp10c9d6c4"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py",
    "scripts/run_causal_inner_monolithic_four_level_wp10c9d6c2.py",
    "scripts/run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/"
    "test_causal_inner_prospective_uniform_validation_wp10c9d6c4.py",
)

OBSERVABLE_NAMES = tuple(wp10c9d6c3.OBSERVABLE_NAMES)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).exists()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c4 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if not (
        summary["work_package"] == "WP10c9d6c3"
        and summary["classification"]
        == "smooth_continuum_four_level_export_direction_certified"
        and summary["passed"]
        and summary["method_passed"]
        and summary["lift_uncertainty_passed"]
        and summary["prospective_uniform_validation_authorized"]
        and not summary["operator_changed"]
        and not summary["direct_operator_redesign_authorized"]
        and not summary["embedded_export_discrimination_authorized"]
        and not summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("WP10c9d6c3 binding classification changed")
    return summary, _load_npz(PARENT_ARRAYS)


def _fixed_profile_hash() -> str:
    payload = json.dumps(
        _plain(
            {
                "calibration": HISTORICAL_PROFILE_DEFINITION,
                "heldouts": ANALYTIC_PROFILE_DEFINITIONS,
                "amplitude_sign_factors": AMPLITUDE_SIGN_FACTORS,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _historical_common_target() -> np.ndarray:
    replay_payload, replay_arrays = (
        wp10c9d6c3.wp10c9d6c._load_replay_inputs()
    )
    native = wp10c9d6c3.wp10c9d6c._configurations(
        replay_payload,
        replay_arrays,
    )
    reference = native[REFERENCE_LABEL]
    return (
        np.asarray(
            reference["initial_directions"]["common_mode"],
            dtype=float,
        ).reshape(-1, 5)
        * np.asarray(
            reference["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
    )


def _historical_profile_fit(
    reference_grid,
    target: np.ndarray,
) -> tuple[dict[int, wp10c9d6c3.SmoothCellAverageProfile], dict]:
    values = np.asarray(target, dtype=float)
    inner_anchor = np.array(values[0], copy=True)
    outer_anchor = np.zeros(5, dtype=float)
    profiles = {}
    reports = {}
    projections = {}
    for degree in (
        HISTORICAL_PROFILE_DEFINITION["primary_degree"],
        HISTORICAL_PROFILE_DEFINITION["independent_degree"],
    ):
        profile, report = wp10c9d6c3._fit_cell_average_profile(
            reference_grid,
            values,
            inner_anchor,
            outer_anchor,
            degree=int(degree),
            quadrature_order=PRIMARY_PROJECTION_ORDER,
        )
        profiles[int(degree)] = profile
        reports[str(degree)] = report
        projections[int(degree)] = wp10c9d6c3._project_callable_to_cells(
            reference_grid,
            profile.evaluate,
            quadrature_order=PRIMARY_PROJECTION_ORDER,
        )
    primary = projections[
        HISTORICAL_PROFILE_DEFINITION["primary_degree"]
    ]
    independent = projections[
        HISTORICAL_PROFILE_DEFINITION["independent_degree"]
    ]
    field_scales = np.maximum(
        np.max(np.abs(values), axis=0),
        np.finfo(float).tiny,
    )
    target_scaled = values / field_scales
    primary_scaled = primary / field_scales
    relative_l2 = float(
        np.linalg.norm(primary_scaled - target_scaled)
        / max(np.linalg.norm(target_scaled), np.finfo(float).tiny)
    )
    cosine = wp10c9d6c3._cosine(primary_scaled, target_scaled)
    representation = float(
        np.max(np.abs(primary - independent) / field_scales)
    )
    return profiles, {
        "definition": HISTORICAL_PROFILE_DEFINITION,
        "fit_reports": reports,
        "relative_l2_fit_defect": relative_l2,
        "fit_cosine": cosine,
        "maximum_primary_independent_representation_difference": (
            representation
        ),
        "inner_anchor": inner_anchor,
        "outer_anchor": outer_anchor,
        "passed": bool(
            relative_l2 <= MAXIMUM_HISTORICAL_FIT_RELATIVE_L2_DEFECT
            and cosine >= MINIMUM_HISTORICAL_FIT_COSINE
            and representation
            <= MAXIMUM_HISTORICAL_REPRESENTATION_DIFFERENCE
        ),
    }


def _component_profile(
    definition: dict,
    radii_over_rg: np.ndarray,
) -> np.ndarray:
    values = np.zeros((radii_over_rg.size, 5), dtype=float)
    for component in definition["components"]:
        envelope = np.exp(
            -0.5
            * (
                np.log(
                    radii_over_rg / float(component["center_over_rg"])
                )
                / float(component["log_width"])
            )
            ** 2
        )
        values += envelope[:, None] * np.asarray(
            component["coefficients"],
            dtype=float,
        )[None, :]
    return values


def _analytic_profile(
    name: str,
    radii: np.ndarray,
    *,
    gravitational_radius: float,
    field_scales: np.ndarray,
    outer_radius: float,
    outgoing_vector: np.ndarray,
) -> np.ndarray:
    physical_radii = np.asarray(radii, dtype=float)
    radii_over_rg = physical_radii / float(gravitational_radius)
    definition = ANALYTIC_PROFILE_DEFINITIONS[name]
    if name == "heldout_first_cell_outgoing":
        envelope = np.exp(
            -0.5
            * (
                np.log(
                    radii_over_rg / float(definition["center_over_rg"])
                )
                / float(definition["log_width"])
            )
            ** 2
        )
        physical = (
            float(definition["amplitude"])
            * envelope[:, None]
            * np.asarray(outgoing_vector, dtype=float)[None, :]
        )
    else:
        physical = (
            _component_profile(definition, radii_over_rg)
            * np.asarray(field_scales, dtype=float)[None, :]
        )
    cutoff = wp10c9d6c3._smooth_cutoff(
        np.log(radii_over_rg),
        float(np.log(outer_radius / gravitational_radius)),
    )
    return cutoff[:, None] * physical


def _background_freeze_defect(
    decisive: dict[str, np.ndarray],
    parent_arrays: dict[str, np.ndarray],
) -> tuple[float, dict[str, float]]:
    keys = (
        "continuum_background_knots",
        "continuum_background_coefficients",
        "continuum_background_inner_anchor",
        "continuum_background_outer_anchor",
        "continuum_perturbation_field_scales",
        "fixed_physical_observable_scales",
    )
    defects = {}
    for key in keys:
        current = np.asarray(decisive[key], dtype=float)
        reference = np.asarray(parent_arrays[key], dtype=float)
        defects[key] = wp10c9d6c3._relative_difference(
            current,
            reference,
        )
    return max(defects.values()), defects


def _build_configurations(
    parent_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray], dict]:
    configurations, decisive, parent_construction = (
        wp10c9d6c3._build_continuum_configurations()
    )
    if not parent_construction["passed"]:
        raise RuntimeError("frozen c3 continuum construction failed")
    decisive["fixed_physical_observable_scales"] = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    freeze_defect, freeze_defects = _background_freeze_defect(
        decisive,
        parent_arrays,
    )

    reference = configurations[REFERENCE_LABEL]
    reference_grid = reference["context"].grid
    background = wp10c9d6c3.SmoothCellAverageProfile(
        knots=np.asarray(
            decisive["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            decisive["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=wp10c9d6c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(reference_grid.gravitational_radius),
    )
    historical_target = _historical_common_target()
    historical_profiles, historical_report = _historical_profile_fit(
        reference_grid,
        historical_target,
    )
    if not historical_report["passed"]:
        raise RuntimeError("historical common calibration fit failed")

    field_scales = np.asarray(
        decisive["continuum_perturbation_field_scales"],
        dtype=float,
    )
    outgoing_radius = (
        float(
            ANALYTIC_PROFILE_DEFINITIONS[
                "heldout_first_cell_outgoing"
            ]["center_over_rg"]
        )
        * float(reference_grid.gravitational_radius)
    )
    outgoing_basis = causal_five_field_characteristic_basis(
        reference["context"],
        outgoing_radius,
        background.evaluate(np.asarray([outgoing_radius]))[0],
        field_scales,
    )
    outgoing_family = ANALYTIC_PROFILE_DEFINITIONS[
        "heldout_first_cell_outgoing"
    ]["characteristic_family"]
    outgoing_index = outgoing_basis.family_labels.index(outgoing_family)
    outgoing_vector = np.asarray(
        outgoing_basis.physical_right_eigenvectors[:, outgoing_index],
        dtype=float,
    )
    outgoing_speed = float(
        outgoing_basis.coordinate_speeds_over_c[outgoing_index]
    )
    if outgoing_speed >= 0.0:
        raise RuntimeError("declared first-cell characteristic is incoming")

    decisive["historical_common_target_cell_averages"] = (
        historical_target
    )
    for degree, profile in historical_profiles.items():
        decisive[f"historical_common_fit_degree_{degree}_knots"] = (
            profile.knots
        )
        decisive[
            f"historical_common_fit_degree_{degree}_coefficients"
        ] = profile.coefficients
    decisive["first_cell_outgoing_physical_vector"] = outgoing_vector
    decisive["first_cell_outgoing_coordinate_speed_over_c"] = np.asarray(
        [outgoing_speed],
        dtype=float,
    )

    maximum_factor_change = 0.0
    profile_peak_cells = {}
    result = {}
    for label in LABELS:
        source = configurations[label]
        grid = source["context"].grid
        directions = dict(source["initial_directions"])
        physical_directions = dict(source["physical_directions"])
        columns = np.asarray(
            source["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
        for name in PERTURBATIONS:
            if name == "historical_common_smooth_fit":
                evaluator = historical_profiles[
                    HISTORICAL_PROFILE_DEFINITION["primary_degree"]
                ].evaluate
            else:
                evaluator = (
                    lambda radii, profile_name=name: _analytic_profile(
                        profile_name,
                        radii,
                        gravitational_radius=grid.gravitational_radius,
                        field_scales=field_scales,
                        outer_radius=float(grid.edges[-1]),
                        outgoing_vector=outgoing_vector,
                    )
                )
            primary = wp10c9d6c3._project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=PRIMARY_PROJECTION_ORDER,
            )
            secondary = wp10c9d6c3._project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=SECONDARY_PROJECTION_ORDER,
            )
            directions[name] = (primary / columns).ravel()
            directions[name + "__projection_order_12"] = (
                secondary / columns
            ).ravel()
            physical_directions[name] = primary
            physical_directions[
                name + "__projection_order_12"
            ] = secondary
            decisive[f"{label}__{name}__physical_direction"] = primary
            decisive[
                f"{label}__{name}__physical_direction_order_12"
            ] = secondary
            normalized = np.linalg.norm(
                primary / field_scales[None, :],
                axis=1,
            )
            profile_peak_cells.setdefault(name, {})[label] = int(
                np.argmax(normalized)
            )
            for factor in (-1.0, -0.5, 0.5, 1.0):
                reconstruction = causal_five_field_reconstruct_face_charts(
                    source["context"],
                    np.asarray(source["base_primitives"], dtype=float)
                    + factor * primary,
                    purpose="flux",
                )
                maximum_factor_change = max(
                    maximum_factor_change,
                    float(
                        np.max(
                            np.abs(
                                reconstruction.admissibility_factors - 1.0
                            )
                        )
                    ),
                )
        result[label] = {
            **source,
            "initial_directions": directions,
            "physical_directions": physical_directions,
        }

    coarse_first_cell_outer_over_rg = float(
        configurations[LABELS[0]]["context"].grid.edges[1]
        / configurations[LABELS[0]][
            "context"
        ].grid.gravitational_radius
    )
    peak_radii_over_rg = {
        label: float(
            configurations[label]["context"].grid.centers[index]
            / configurations[label]["context"].grid.gravitational_radius
        )
        for label, index in profile_peak_cells[
            "heldout_first_cell_outgoing"
        ].items()
    }
    first_cell_peak = bool(
        all(
            radius <= coarse_first_cell_outer_over_rg
            for radius in peak_radii_over_rg.values()
        )
    )
    report = {
        "profile_definition_sha256": _fixed_profile_hash(),
        "parent_continuum_construction": parent_construction,
        "frozen_background_defects": freeze_defects,
        "maximum_frozen_background_defect": freeze_defect,
        "historical_calibration_fit": historical_report,
        "first_cell_outgoing": {
            "family": outgoing_family,
            "reference_radius_over_rg": (
                outgoing_radius / reference_grid.gravitational_radius
            ),
            "coordinate_speed_over_c": outgoing_speed,
            "basis_condition_number": outgoing_basis.condition_number,
            "maximum_eigenpair_defect": (
                outgoing_basis.maximum_eigenpair_defect
            ),
            "peak_cell_indices": profile_peak_cells[
                "heldout_first_cell_outgoing"
            ],
            "peak_radii_over_rg": peak_radii_over_rg,
            "coarse_first_cell_outer_radius_over_rg": (
                coarse_first_cell_outer_over_rg
            ),
            "first_cell_dominated_on_every_grid": first_cell_peak,
        },
        "profile_peak_cell_indices": profile_peak_cells,
        "maximum_perturbed_reconstruction_factor_change": (
            maximum_factor_change
        ),
        "passed": bool(
            freeze_defect <= MAXIMUM_FROZEN_BACKGROUND_DEFECT
            and historical_report["passed"]
            and outgoing_speed < 0.0
            and first_cell_peak
            and maximum_factor_change
            <= MAXIMUM_PERTURBED_RECONSTRUCTION_FACTOR_CHANGE
        ),
    }
    return result, decisive, report


def _propagate_profiles(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    decisive: dict[str, np.ndarray],
) -> tuple[dict, dict, dict]:
    all_histories = {}
    restart_defects = {}
    legacy_restart_defects = {}
    for profile in PERTURBATIONS:
        all_histories[profile] = {}
        restart_defects[profile] = {}
        legacy_restart_defects[profile] = {}
        for variant, suffix in (
            ("primary", ""),
            ("projection_order_12", "__projection_order_12"),
        ):
            histories = {}
            restarts = {}
            legacy_restarts = {}
            for label in LABELS:
                print(
                    "WP10c9d6c4: propagate "
                    f"{profile} {variant} on {label}",
                    flush=True,
                )
                configuration = configurations[label]
                initial = configuration["initial_directions"][
                    profile + suffix
                ]
                state, legacy_restart = wp10c9d6c3.wp10c9d6c._propagate(
                    tangents[label].scaled_generator_per_s,
                    initial,
                    configuration["times"],
                )
                endpoint_envelope_norm = max(
                    float(np.linalg.norm(initial)),
                    float(np.linalg.norm(state[-1])),
                    np.finfo(float).tiny,
                )
                final_norm = float(np.linalg.norm(state[-1]))
                # The inherited restart metric is relative to the final
                # state.  A deliberately exiting packet can make that state
                # nearly zero and inflate a roundoff-sized split/restart
                # discrepancy.  From
                #
                #   d = ||r-f|| / max(||r||, ||f||)
                #
                # one has max(||r||,||f||) <= ||f||/(1-d).  The expression
                # below is therefore a conservative upper bound on the
                # restart error normalized by the initial/final state
                # envelope.  This is stricter than normalization by a
                # potentially amplified intermediate trajectory peak.
                restart = float(
                    legacy_restart
                    * final_norm
                    / max(1.0 - legacy_restart, np.finfo(float).tiny)
                    / endpoint_envelope_norm
                )
                signals = state @ observable_maps[label].T
                histories[label] = {
                    "times": np.asarray(
                        configuration["times"],
                        dtype=float,
                    ),
                    "signals": signals,
                    "state": state,
                    "final_scaled_state": state[-1],
                }
                restarts[label] = restart
                legacy_restarts[label] = legacy_restart
                prefix = f"{profile}__{variant}__{label}__"
                decisive[prefix + "times"] = configuration["times"]
                decisive[prefix + "signals"] = signals
                decisive[prefix + "cumulative"] = (
                    wp10c9d6c3.wp10c9d6c._cumulative(
                        configuration["times"],
                        signals,
                    )
                )
                decisive[prefix + "final_scaled_state"] = state[-1]
            all_histories[profile][variant] = histories
            restart_defects[profile][variant] = restarts
            legacy_restart_defects[profile][variant] = legacy_restarts
    return all_histories, restart_defects, legacy_restart_defects


def _amplitude_sign_report(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    histories: dict,
) -> dict:
    reports = {}
    maximum_state = 0.0
    maximum_export = 0.0
    for profile in PERTURBATIONS:
        reports[profile] = {}
        for label in AMPLITUDE_SIGN_LABELS:
            reference = histories[profile]["primary"][label]
            initial = configurations[label]["initial_directions"][profile]
            factor_reports = {}
            for factor in AMPLITUDE_SIGN_FACTORS:
                state, _restart = wp10c9d6c3.wp10c9d6c._propagate(
                    tangents[label].scaled_generator_per_s,
                    factor * initial,
                    configurations[label]["times"],
                )
                signals = state @ observable_maps[label].T
                expected_state = factor * reference["state"]
                expected_signals = factor * reference["signals"]
                state_defect = wp10c9d6c3._relative_difference(
                    state,
                    expected_state,
                )
                export_defect = wp10c9d6c3._relative_difference(
                    signals,
                    expected_signals,
                )
                maximum_state = max(maximum_state, state_defect)
                maximum_export = max(maximum_export, export_defect)
                factor_reports[str(factor)] = {
                    "state_relative_defect": state_defect,
                    "export_relative_defect": export_defect,
                }
            reports[profile][label] = factor_reports
    return {
        "factors": AMPLITUDE_SIGN_FACTORS,
        "labels": AMPLITUDE_SIGN_LABELS,
        "profile_reports": reports,
        "maximum_state_relative_defect": maximum_state,
        "maximum_export_relative_defect": maximum_export,
        "passed": bool(
            maximum_state <= MAXIMUM_SIGN_AMPLITUDE_STATE_DEFECT
            and maximum_export <= MAXIMUM_SIGN_AMPLITUDE_EXPORT_DEFECT
        ),
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent_summary, parent_arrays = _load_parent()
    configurations, decisive, construction = _build_configurations(
        parent_arrays
    )
    if not construction["passed"]:
        raise RuntimeError("WP10c9d6c4 profile construction failed")

    tangents, observable_maps, method_reports, baselines = (
        wp10c9d6c3._build_tangents(configurations, decisive)
    )
    physical_scales = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    histories, restart_defects, legacy_restart_defects = (
        _propagate_profiles(
            configurations,
            tangents,
            observable_maps,
            decisive,
        )
    )

    lift_reports = {
        name: wp10c9d6c3._lift_uncertainty(
            name,
            configurations,
            tangents,
            observable_maps,
            histories[name],
            physical_scales,
        )
        for name in PERTURBATIONS
    }
    profile_reports = {}
    for name in PERTURBATIONS:
        report, arrays = wp10c9d6c3._profile_report(
            histories[name],
            physical_scales,
        )
        profile_reports[name] = report
        for array_name, values in arrays.items():
            decisive[f"{name}__{array_name}"] = values

    amplitude_sign = _amplitude_sign_report(
        configurations,
        tangents,
        observable_maps,
        histories,
    )
    maximum_restart = max(
        value
        for profile in restart_defects.values()
        for variant in profile.values()
        for value in variant.values()
    )
    maximum_legacy_restart = max(
        value
        for profile in legacy_restart_defects.values()
        for variant in profile.values()
        for value in variant.values()
    )
    method_passed = bool(
        construction["passed"]
        and all(
            report["passed"] for report in method_reports.values()
        )
        and amplitude_sign["passed"]
        and maximum_restart <= MAXIMUM_RESTART_DEFECT
    )
    lift_passed = bool(
        all(report["passed"] for report in lift_reports.values())
    )
    calibration_passed = bool(
        all(
            profile_reports[name]["historical"]["passed"]
            for name in CALIBRATION_PROFILES
        )
    )
    heldout_passed = bool(
        all(
            profile_reports[name]["historical"]["passed"]
            for name in HELDOUT_PROFILES
        )
    )
    strict_passed = bool(
        method_passed
        and lift_passed
        and calibration_passed
        and heldout_passed
    )

    if not method_passed:
        classification = "prospective_uniform_validation_method_failed"
        authorized_next = "none"
    elif not lift_passed:
        classification = "prospective_uniform_lift_uncertainty_unresolved"
        authorized_next = "improve_continuum_projection"
    elif strict_passed:
        classification = (
            "prospective_heldout_uniform_export_validation_certified"
        )
        authorized_next = "embedded_uniform_to_coupled_discrimination"
    elif heldout_passed and not calibration_passed:
        classification = (
            "historical_common_profile_sensitive_heldouts_certified"
        )
        authorized_next = "historical_profile_interaction_audit"
    else:
        classification = "prospective_heldout_uniform_validation_failed"
        authorized_next = "smooth_profile_local_truncation_audit"

    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "meshes": MESHES,
        "labels": LABELS,
        "active_cells": dict(zip(LABELS, ACTIVE_CELLS, strict=True)),
        "reference_label": REFERENCE_LABEL,
        "calibration_profiles": CALIBRATION_PROFILES,
        "heldout_profiles": HELDOUT_PROFILES,
        "historical_profile_definition": HISTORICAL_PROFILE_DEFINITION,
        "analytic_profile_definitions": ANALYTIC_PROFILE_DEFINITIONS,
        "profile_definition_sha256": _fixed_profile_hash(),
        "primary_projection_order": PRIMARY_PROJECTION_ORDER,
        "secondary_projection_order": SECONDARY_PROJECTION_ORDER,
        "amplitude_sign_factors": AMPLITUDE_SIGN_FACTORS,
        "amplitude_sign_labels": AMPLITUDE_SIGN_LABELS,
        "primary_stride": PRIMARY_STRIDE,
        "stride_audits": STRIDE_AUDITS,
        "gates": {
            "minimum_export_order": MINIMUM_EXPORT_ORDER,
            "maximum_fine_physical_difference": (
                MAXIMUM_FINE_PHYSICAL_DIFFERENCE
            ),
            "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
            "minimum_error_cosine": MINIMUM_ERROR_COSINE,
            "maximum_lift_state_relative_difference": (
                MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE
            ),
            "maximum_lift_rate_relative_difference": (
                MAXIMUM_LIFT_RATE_RELATIVE_DIFFERENCE
            ),
            "maximum_lift_to_fine_export_ratio": (
                MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO
            ),
            "maximum_historical_fit_relative_l2_defect": (
                MAXIMUM_HISTORICAL_FIT_RELATIVE_L2_DEFECT
            ),
            "minimum_historical_fit_cosine": (
                MINIMUM_HISTORICAL_FIT_COSINE
            ),
            "maximum_historical_representation_difference": (
                MAXIMUM_HISTORICAL_REPRESENTATION_DIFFERENCE
            ),
            "maximum_sign_amplitude_state_defect": (
                MAXIMUM_SIGN_AMPLITUDE_STATE_DEFECT
            ),
            "maximum_sign_amplitude_export_defect": (
                MAXIMUM_SIGN_AMPLITUDE_EXPORT_DEFECT
            ),
            "restart_defect_normalization": (
                "conservative split/restart upper bound normalized by "
                "the maximum initial/final state norm"
            ),
        },
        "operator_change": False,
        "production_defaults_changed": False,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": strict_passed,
        "audit_executed": True,
        "method_passed": method_passed,
        "lift_uncertainty_passed": lift_passed,
        "calibration_passed": calibration_passed,
        "prospective_heldout_passed": heldout_passed,
        "strict_four_level_export_direction_passed": strict_passed,
        "parent_wp10c9d6c3_classification_preserved": True,
        "parent_classification": parent_summary["classification"],
        "profile_definition_sha256": _fixed_profile_hash(),
        "construction": construction,
        "method_reports": method_reports,
        "baseline_observables": baselines,
        "lift_reports": lift_reports,
        "profile_reports": profile_reports,
        "amplitude_sign_report": amplitude_sign,
        "restart_defects": restart_defects,
        "legacy_final_state_relative_restart_defects": (
            legacy_restart_defects
        ),
        "maximum_restart_defect": maximum_restart,
        "maximum_legacy_final_state_relative_restart_defect": (
            maximum_legacy_restart
        ),
        "historical_profile_interaction_audit_authorized": bool(
            authorized_next == "historical_profile_interaction_audit"
        ),
        "smooth_profile_local_truncation_audit_authorized": bool(
            authorized_next == "smooth_profile_local_truncation_audit"
        ),
        "direct_operator_redesign_authorized": False,
        "embedded_export_discrimination_authorized": bool(
            authorized_next
            == "embedded_uniform_to_coupled_discrimination"
        ),
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "uses_production_generator": False,
        "uses_production_anchor_storage_derivative": False,
        "operator_changed": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "CERTIFIED"
            if strict_passed
            else "DIAGNOSTIC ONLY"
            if method_passed and lift_passed
            else "UNRESOLVED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "generation_command": (
            "PYTHONPATH=src:scripts python3 "
            "scripts/"
            "run_causal_inner_prospective_uniform_validation_wp10c9d6c4.py"
        ),
        "profile_definition_sha256": _fixed_profile_hash(),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            _relative(PARENT_CONFIG): _sha256(PARENT_CONFIG),
            _relative(PARENT_SUMMARY): _sha256(PARENT_SUMMARY),
            _relative(PARENT_ARRAYS): _sha256(PARENT_ARRAYS),
            _relative(PARENT_PROVENANCE): _sha256(PARENT_PROVENANCE),
        },
        "establishes": (
            "whether the unchanged c3 monolithic uniform operator retains "
            "its strict physical-export convergence for a prospectively "
            "declared calibration profile and four diverse smooth held-outs"
        ),
        "does_not_establish": (
            "embedded coupling convergence, nonlinear convergence, "
            "production eligibility, fixed-Q closure, reduced slow-time "
            "evolution, or a causal operator defect"
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    print(
        f"WP10c9d6c4: classification={classification}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
