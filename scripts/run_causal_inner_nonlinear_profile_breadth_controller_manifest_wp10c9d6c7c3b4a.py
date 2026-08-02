#!/usr/bin/env python3
"""Freeze nonlinear profile breadth and a cost-bounded campaign controller."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3  # noqa: E402
import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a  # noqa: E402
import run_causal_inner_embedded_regularity_manifest_wp10c9d6c7c0 as c7c0  # noqa: E402
import run_causal_inner_nonlinear_temporal_refinement_manifest_wp10c9d6c7c3b3a as c3b3a  # noqa: E402
import run_causal_inner_nonlinear_temporal_symmetry_wp10c9d6c7c3b3b4 as c3b3b4  # noqa: E402
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_characteristic_dissipation import (  # noqa: E402
    causal_five_field_coordinate_principal_basis,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_characteristic_phase import (  # noqa: E402
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
    causal_five_field_reconstruct_face_charts,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
    causal_characteristic_purity,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (  # noqa: E402
    causal_packet_spectrum,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b4a"
ANALYZED_BASE_COMMIT = "1e5fcc3900b2d3bd44f792ee768291dab5459f03"
ANALYZED_BASE_PARENT = "fd2ed817c6bc32195b9c476726b9d640c68b8013"
ANALYZED_BASE_TREE = "8c46e12d14920bc37486172df7f57ea9e13611c0"

ARTIFACT = (
    "causal_inner_nonlinear_profile_breadth_controller_manifest_"
    "wp10c9d6c7c3b4a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_profile_breadth_controller_"
    "manifest_wp10c9d6c7c3b4a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_profile_breadth_controller_"
    "manifest_wp10c9d6c7c3b4a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_PROFILE_BREADTH_CONTROLLER_MANIFEST_"
    "WP10C9D6C7C3B4A_2026-08-02.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

COUPLING_PARENT_FACE = c7c0.COUPLING_PARENT_FACE
BUFFER_PARENT_FACE = c7c0.SUPPORT_BUFFER_PARENT_FACE
PRIMARY_PROJECTION_ORDER = c7c0.PRIMARY_PROJECTION_ORDER
SECONDARY_PROJECTION_ORDER = c7c0.SECONDARY_PROJECTION_ORDER
PROFILE_AMPLITUDE = c6a2.PROBE_AMPLITUDE
READINESS_MULTIPLIERS = (1.0, -1.0, 0.5, -0.5)
BINDING_PROPAGATION_MULTIPLIER = 1.0
LAYOUTS = tuple(c7c0.EMBEDDED_LABELS)
REFINEMENT_RATIOS = tuple(c7c0.REFINEMENT_RATIOS)
UNIFORM_LABELS = tuple(c7c0.UNIFORM_LABELS)
TIMESTEP_LEVELS_SECONDS = np.asarray((1.0e-5, 5.0e-6, 2.5e-6))
HORIZON_SECONDS = 4.0e-5
COMMON_OUTPUT_TIMES_SECONDS = np.arange(5, dtype=float) * 1.0e-5

GENERIC_COEFFICIENTS = (0.35, -0.40, 0.50, -0.45, 0.30)
INWARD_MIXED_COEFFICIENTS = (
    1.0 / np.sqrt(2.0),
    1.0 / np.sqrt(2.0),
    0.0,
    0.0,
    0.0,
)
PROFILE_DEFINITIONS = {
    "p4__inward_acoustic": {
        "role": "acoustic_direction_control",
        "family": "inward_acoustic",
        "mixed_coefficients": None,
        "window_power": 4,
        "support_upper_parent_face": COUPLING_PARENT_FACE,
        "coupling_trace_expectation": "active",
    },
    "p4__outward_acoustic": {
        "role": "acoustic_direction_control",
        "family": "outward_acoustic",
        "mixed_coefficients": None,
        "window_power": 4,
        "support_upper_parent_face": COUPLING_PARENT_FACE,
        "coupling_trace_expectation": "active",
    },
    "p3_buffer45__material": {
        "role": "material_contact_control",
        "family": "material",
        "mixed_coefficients": None,
        "window_power": 3,
        "support_upper_parent_face": BUFFER_PARENT_FACE,
        "coupling_trace_expectation": "inactive",
    },
    "p4__inward_shear_acoustic_mix": {
        "role": "mixed_shear_acoustic_control",
        "family": "mixed",
        "mixed_coefficients": INWARD_MIXED_COEFFICIENTS,
        "window_power": 4,
        "support_upper_parent_face": COUPLING_PARENT_FACE,
        "coupling_trace_expectation": "active",
    },
    "p3_buffer45__generic_five_field": {
        "role": "generic_five_family_control",
        "family": "mixed",
        "mixed_coefficients": GENERIC_COEFFICIENTS,
        "window_power": 3,
        "support_upper_parent_face": BUFFER_PARENT_FACE,
        "coupling_trace_expectation": "inactive",
    },
}
PROFILE_NAMES = tuple(PROFILE_DEFINITIONS)

MAXIMUM_H_OVER_R = 0.25
MINIMUM_SCATTERING_OPTICAL_DEPTH = 1.0
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
MAXIMUM_RESTRICTION_DEFECT = 2.0e-12
MAXIMUM_COUPLING_TRACE_JUMP = 1.0e-4
REQUIRED_INCOMING_EXCISION_CHARACTERISTICS = 0
MAXIMUM_EXTERIOR_NORM = 0.0
MINIMUM_ACTIVE_TRACE_FRACTION = c7c0.MINIMUM_ACTIVE_COUPLING_TRACE_FRACTION
MAXIMUM_BUFFERED_TRACE_FRACTION = c7c0.MAXIMUM_BUFFERED_COUPLING_TRACE_FRACTION

PARENT_DIRECTORY = c3b3b4.CANONICAL_DIRECTORY
C7A_DIRECTORY = c7c0.C7A_DIRECTORY
C7C0_DIRECTORY = c7c0.CANONICAL_DIRECTORY
E1_DIRECTORY = c7c0.E1_DIRECTORY
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "profile_breadth_controller_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _source_identity() -> dict[str, str]:
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        "scripts/run_causal_inner_nonlinear_temporal_symmetry_"
        "wp10c9d6c7c3b3b4.py",
        "scripts/run_causal_inner_embedded_regularity_manifest_"
        "wp10c9d6c7c0.py",
        "scripts/run_causal_inner_windowed_contract_wp10c9d6c6a2.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    )
    return {
        path: _sha256(ROOT / path)
        for path in sources
        if (ROOT / path).is_file()
    }


def _validate_parent() -> tuple[dict, dict, dict]:
    parent = _read_json(PARENT_DIRECTORY / "summary.json")
    c7a_summary = _read_json(C7A_DIRECTORY / "summary.json")
    c7c0_summary = _read_json(C7C0_DIRECTORY / "summary.json")
    if (
        not parent["passed"]
        or parent["classification"]
        != "coarse_primary_nonlinear_symmetry_controls_certified_"
        "short_horizon_profile_breadth_controller_manifest_authorized"
        or parent["authorized_next"]
        != "WP10c9d6c7c3b4a_short_horizon_nonlinear_profile_breadth_"
        "and_efficient_controller_manifest"
        or parent["meaningfully_nonlinear_dynamics_certified"]
        or parent["long_nonlinear_physical_ladder_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
        or c7a_summary["classification"]
        != "embedded_layout_and_profile_manifest_frozen_"
        "propagation_authorized"
        or not c7c0_summary["passed"]
    ):
        raise RuntimeError("c3b4a authorization or profile basis changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b4a analyzed identity changed")
    return parent, c7a_summary, c7c0_summary


def _project_profiles():
    configurations, construction_arrays, construction_report = (
        c3._build_continuum_configurations()
    )
    interpolator, characteristic_report, characteristic_arrays = (
        c6a2._build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    if not construction_report["passed"] or not characteristic_report["passed"]:
        raise RuntimeError("continuum background or characteristic field changed")
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    c7a_arrays = _load_npz(c7c0.C7A_ARRAYS)
    inherited_scales = np.asarray(c7a_arrays["field_scales"], dtype=float)
    if _relative_defect(field_scales, inherited_scales) > 2.0e-12:
        raise RuntimeError("profile field scales changed")
    parent_edges = np.asarray(c7a_arrays["parent_grid_edges"], dtype=float)
    lower_radius = float(parent_edges[0])
    coupling_radius = float(parent_edges[COUPLING_PARENT_FACE])
    buffer_radius = float(parent_edges[BUFFER_PARENT_FACE])
    evaluators = {}
    for name, definition in PROFILE_DEFINITIONS.items():
        upper_radius = (
            coupling_radius
            if definition["support_upper_parent_face"]
            == COUPLING_PARENT_FACE
            else buffer_radius
        )
        evaluators[name] = c6a2._probe_evaluator(
            definition,
            interpolator,
            lower_radius=lower_radius,
            upper_radius=upper_radius,
        )
    arrays = {
        "field_scales": field_scales,
        "parent_grid_edges": parent_edges,
        "characteristic_field_radii": np.asarray(
            characteristic_arrays["characteristic_field_radii"],
            dtype=float,
        ),
        "characteristic_field_physical_vectors": np.asarray(
            characteristic_arrays["characteristic_field_physical_vectors"],
            dtype=float,
        ),
    }
    projected = {}
    projection_reports = {name: {} for name in PROFILE_NAMES}
    reference_cell_count = int(
        configurations["uniform_N128"]["context"].grid.centers.size
    )
    for label in UNIFORM_LABELS:
        grid = configurations[label]["context"].grid
        resolution_ratio = int(grid.centers.size // reference_cell_count)
        if grid.centers.size != resolution_ratio * reference_cell_count:
            raise RuntimeError("uniform inner grids are not nested")
        for name, evaluator in evaluators.items():
            primary = c3._project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=PRIMARY_PROJECTION_ORDER,
            )
            secondary = c3._project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=SECONDARY_PROJECTION_ORDER,
            )
            support = (
                PROFILE_DEFINITIONS[name]["support_upper_parent_face"]
                * resolution_ratio
            )
            normalized = primary / field_scales[None, :]
            peak = max(
                float(np.max(np.linalg.norm(normalized, axis=1))),
                np.finfo(float).tiny,
            )
            endpoint_fraction = float(
                max(
                    np.linalg.norm(normalized[0]),
                    np.linalg.norm(normalized[support - 1]),
                )
                / peak
            )
            projected[(name, label)] = primary
            arrays[f"{name}__{label}__primary_physical"] = primary
            projection_reports[name][label] = {
                "projection_defect": _relative_defect(primary, secondary),
                "outside_support_norm": float(np.linalg.norm(primary[support:])),
                "support_endpoint_cell_fraction": endpoint_fraction,
                "array_sha256": causal_array_sha256(primary),
            }
    return (
        configurations,
        interpolator,
        field_scales,
        parent_edges,
        projected,
        projection_reports,
        arrays,
        characteristic_report,
    )


def _spectral_eligibility(
    configurations,
    interpolator,
    field_scales,
    projected,
    projection_reports,
) -> dict:
    eligibility = _read_json(c7c0.E1_CONFIG)["eligibility_contract"]
    grid = configurations["uniform_N128"]["context"].grid
    spacing = float(np.mean(np.diff(np.log(grid.edges))))
    bases = interpolator.evaluate(np.asarray(grid.centers, dtype=float))
    reports = {}
    for name, definition in PROFILE_DEFINITIONS.items():
        values = projected[(name, "uniform_N128")]
        spectrum = causal_packet_spectrum(
            values / field_scales[None, :],
            spacing,
            quantile=float(eligibility["spectral_energy_quantile"]),
        )
        purity = causal_characteristic_purity(
            values,
            bases,
            field_scales,
            np.asarray(grid.cell_measures, dtype=float),
            selected_family=0,
        )
        pure_family = definition["family"] != "mixed"
        selected_index = (
            interpolator.family_labels.index(definition["family"])
            if pure_family
            else None
        )
        global_fraction = (
            float(purity.family_energy_fractions[selected_index])
            if pure_family
            else None
        )
        active_fraction = (
            float(
                causal_characteristic_purity(
                    values,
                    bases,
                    field_scales,
                    np.asarray(grid.cell_measures, dtype=float),
                    selected_family=selected_index,
                ).minimum_active_cell_selected_fraction
            )
            if pure_family
            else None
        )
        theta = float(spectrum.quantile_angular_wavenumber * spacing)
        projection_defect = max(
            float(projection_reports[name][label]["projection_defect"])
            for label in UNIFORM_LABELS
        )
        endpoint = max(
            float(
                projection_reports[name][label][
                    "support_endpoint_cell_fraction"
                ]
            )
            for label in UNIFORM_LABELS
        )
        outside = max(
            float(projection_reports[name][label]["outside_support_norm"])
            for label in UNIFORM_LABELS
        )
        pure_pass = bool(
            not pure_family
            or (
                global_fraction
                >= float(eligibility["minimum_global_family_purity"])
                and active_fraction
                >= float(eligibility["minimum_active_cell_family_purity"])
            )
        )
        passed = bool(
            theta <= float(eligibility["maximum_theta_99"])
            and float(spectrum.nyquist_alias_fraction)
            <= float(eligibility["maximum_nyquist_alias_fraction"])
            and projection_defect
            <= float(eligibility["maximum_projection_replay_defect"])
            and endpoint
            <= float(eligibility["maximum_endpoint_cell_fraction"])
            and outside <= MAXIMUM_EXTERIOR_NORM
            and pure_pass
        )
        reports[name] = {
            "passed": passed,
            "role": definition["role"],
            "family": definition["family"],
            "theta_99": theta,
            "nyquist_alias_fraction": float(spectrum.nyquist_alias_fraction),
            "family_energy_fractions": purity.family_energy_fractions.tolist(),
            "selected_global_family_fraction": global_fraction,
            "minimum_active_cell_selected_fraction": active_fraction,
            "maximum_projection_defect": projection_defect,
            "maximum_support_endpoint_cell_fraction": endpoint,
            "maximum_outside_support_norm": outside,
        }
    return {
        "passed": all(report["passed"] for report in reports.values()),
        "profiles": reports,
        "contract": eligibility,
    }


def _embedded_readiness(field_scales, parent_edges, projected, arrays):
    c7a_arrays = _load_npz(c7c0.C7A_ARRAYS)
    replay_arrays = _load_npz(c7c0.C0E_INPUTS)
    replay_contexts = _read_json(c7c0.C0E_CONTEXTS)
    parent_grid = make_kerr_schild_column_grid_from_edges(
        parent_edges,
        float(
            replay_contexts["contexts"][LAYOUTS[0]][
                "grid_gravitational_radius"
            ]
        ),
    )
    reference_restrictions = {}
    layout_reports = {}
    maximum_restriction = 0.0
    maximum_h = 0.0
    minimum_tau = float("inf")
    minimum_factor = 1.0
    maximum_incoming = 0
    maximum_jump = 0.0
    maximum_exterior = 0.0
    all_trace_expectations = True
    for ratio, label, uniform_label in zip(
        REFINEMENT_RATIOS,
        LAYOUTS,
        UNIFORM_LABELS,
        strict=True,
    ):
        layout = make_causal_embedded_patch_layout(
            parent_grid,
            COUPLING_PARENT_FACE,
            ratio,
        )
        context = c7c0.wp10c9d5a._context_from_payload(
            replay_contexts["contexts"][label],
            replay_arrays,
        )
        base = np.asarray(
            c7a_arrays[f"{label}__spliced_base_primitives"],
            dtype=float,
        )
        if not np.array_equal(context.grid.edges, layout.grid.edges):
            raise RuntimeError("embedded profile-breadth layout changed")
        left_weights, right_weights, reconstruction_defect = (
            c7c0._frozen_quadratic_reconstruction_weights(context, base)
        )
        active = int(layout.coupling_face_index)
        profile_reports = {}
        for name, definition in PROFILE_DEFINITIONS.items():
            inner = projected[(name, uniform_label)]
            packet = np.concatenate(
                (
                    inner,
                    np.zeros((layout.n_cells - active, 5), dtype=float),
                ),
                axis=0,
            )
            arrays[f"{name}__{label}__primary_physical"] = packet
            restricted = restrict_causal_embedded_patch_cell_averages(
                packet,
                layout,
            )
            if name not in reference_restrictions:
                reference_restrictions[name] = np.array(restricted, copy=True)
            restriction = _relative_defect(
                restricted,
                reference_restrictions[name],
            )
            maximum_restriction = max(maximum_restriction, restriction)
            maximum_exterior = max(
                maximum_exterior,
                float(np.linalg.norm(packet[active:])),
            )
            left_trace = left_weights[active] @ packet
            right_trace = right_weights[active] @ packet
            normalized = packet / field_scales[None, :]
            peak = max(
                float(np.max(np.linalg.norm(normalized, axis=1))),
                np.finfo(float).tiny,
            )
            trace_fraction = float(
                max(
                    np.linalg.norm(left_trace / field_scales),
                    np.linalg.norm(right_trace / field_scales),
                )
                / peak
            )
            expected = definition["coupling_trace_expectation"]
            trace_passed = bool(
                trace_fraction >= MINIMUM_ACTIVE_TRACE_FRACTION
                if expected == "active"
                else trace_fraction <= MAXIMUM_BUFFERED_TRACE_FRACTION
            )
            all_trace_expectations = bool(
                all_trace_expectations and trace_passed
            )
            readiness = []
            for multiplier in READINESS_MULTIPLIERS:
                state = base + float(multiplier) * packet
                reconstruction = causal_five_field_reconstruct_face_charts(
                    context,
                    state,
                    purpose="flux",
                )
                local_h = []
                local_tau = []
                for radius, chart in zip(
                    context.grid.centers,
                    state,
                    strict=True,
                ):
                    cell = _cell_state(context, float(radius), chart)
                    local_h.append(
                        cell.thermodynamics.proper_half_thickness
                        / float(radius)
                    )
                    local_tau.append(
                        0.5 * context.kappa * cell.primitive.surface_density
                    )
                basis = causal_five_field_coordinate_principal_basis(
                    context,
                    float(context.grid.edges[0]),
                    reconstruction.right_face_charts[0],
                )
                scale = np.maximum(np.max(np.abs(state), axis=0), 1.0)
                jump = float(
                    np.max(
                        np.abs(
                            reconstruction.right_face_charts[active]
                            - reconstruction.left_face_charts[active]
                        )
                        / scale
                    )
                )
                item = {
                    "multiplier": multiplier,
                    "maximum_h_over_r": max(local_h),
                    "minimum_scattering_optical_depth": min(local_tau),
                    "minimum_reconstruction_factor": float(
                        np.min(reconstruction.admissibility_factors)
                    ),
                    "incoming_excision_characteristics": (
                        basis.incoming_inner_characteristics
                    ),
                    "coupling_trace_jump": jump,
                }
                readiness.append(item)
                maximum_h = max(maximum_h, item["maximum_h_over_r"])
                minimum_tau = min(
                    minimum_tau,
                    item["minimum_scattering_optical_depth"],
                )
                minimum_factor = min(
                    minimum_factor,
                    item["minimum_reconstruction_factor"],
                )
                maximum_incoming = max(
                    maximum_incoming,
                    item["incoming_excision_characteristics"],
                )
                maximum_jump = max(maximum_jump, jump)
            profile_reports[name] = {
                "array_sha256": causal_array_sha256(packet),
                "restriction_to_parent_defect": restriction,
                "maximum_coupling_trace_fraction": trace_fraction,
                "coupling_trace_expectation": expected,
                "coupling_trace_expectation_passed": trace_passed,
                "readiness_variants": readiness,
            }
        layout_reports[label] = {
            "refinement_ratio": ratio,
            "n_cells": int(layout.n_cells),
            "reconstruction_weight_defect": reconstruction_defect,
            "profiles": profile_reports,
        }
    passed = bool(
        maximum_restriction <= MAXIMUM_RESTRICTION_DEFECT
        and maximum_h <= MAXIMUM_H_OVER_R
        and minimum_tau > MINIMUM_SCATTERING_OPTICAL_DEPTH
        and minimum_factor >= MINIMUM_RECONSTRUCTION_FACTOR
        and maximum_incoming == REQUIRED_INCOMING_EXCISION_CHARACTERISTICS
        and maximum_jump <= MAXIMUM_COUPLING_TRACE_JUMP
        and maximum_exterior <= MAXIMUM_EXTERIOR_NORM
        and all_trace_expectations
    )
    return {
        "passed": passed,
        "maximum_restriction_defect": maximum_restriction,
        "maximum_h_over_r": maximum_h,
        "minimum_scattering_optical_depth": minimum_tau,
        "minimum_reconstruction_factor": minimum_factor,
        "maximum_incoming_excision_characteristics": maximum_incoming,
        "maximum_coupling_trace_jump": maximum_jump,
        "maximum_exterior_norm": maximum_exterior,
        "all_coupling_trace_expectations_passed": all_trace_expectations,
        "layouts": layout_reports,
    }


def _campaign_controller() -> dict:
    parent = _read_json(c3b3a.SUMMARY_PATH)
    cost = parent["cost_audit"]
    step_seconds = cost["median_step_seconds_by_layout"]
    coarse, middle, fine = LAYOUTS
    count = len(PROFILE_NAMES)
    stage_steps = {
        "coarse_fixed_step_method_screen": 4 * count,
        "coarse_temporal_refinement": 24 * count,
        "middle_fine_spatial_confirmation": 8 * count,
    }
    stage_hours = {
        "coarse_fixed_step_method_screen": (
            stage_steps["coarse_fixed_step_method_screen"]
            * step_seconds[coarse]
            / 3600.0
        ),
        "coarse_temporal_refinement": (
            stage_steps["coarse_temporal_refinement"]
            * step_seconds[coarse]
            / 3600.0
        ),
        "middle_fine_spatial_confirmation": (
            4.0
            * count
            * (step_seconds[middle] + step_seconds[fine])
            / 3600.0
        ),
    }
    naive_hours = float(
        count
        * int(np.sum(np.rint(HORIZON_SECONDS / TIMESTEP_LEVELS_SECONDS)))
        * sum(step_seconds.values())
        / 3600.0
    )
    staged_hours = float(sum(stage_hours.values()))
    return {
        "passed": staged_hours < naive_hours,
        "kind": "fail_fast_checkpoint_safe_campaign_controller",
        "selected_short_horizon_timestep_seconds": 1.0e-5,
        "selected_step_basis": (
            "all certified b3b1-b3b4 state and Tier-I selected-step "
            "Richardson bounds are below the frozen 0.005 budget"
        ),
        "background_histories_reused_by_hash": True,
        "new_profile_histories_must_not_be_recombined_from_linear_bases": True,
        "stages": [
            {
                "work_package": "WP10c9d6c7c3b4b1",
                "name": "coarse_fixed_step_method_screen",
                "layout": coarse,
                "timestep_seconds": 1.0e-5,
                "profiles": list(PROFILE_NAMES),
                "stop_on_any_method_or_readiness_failure": True,
            },
            {
                "work_package": "WP10c9d6c7c3b4b2",
                "name": "coarse_temporal_refinement",
                "layout": coarse,
                "timestep_seconds": [5.0e-6, 2.5e-6],
                "profiles": list(PROFILE_NAMES),
                "stop_on_any_state_or_Tier_I_temporal_failure": True,
            },
            {
                "work_package": "WP10c9d6c7c3b4b3",
                "name": "middle_fine_spatial_confirmation",
                "layouts": [middle, fine],
                "timestep_seconds": 1.0e-5,
                "profiles": list(PROFILE_NAMES),
                "stop_on_any_state_or_Tier_I_spatial_failure": True,
            },
        ],
        "estimated_stage_cpu_hours": stage_hours,
        "estimated_staged_cpu_hours": staged_hours,
        "estimated_naive_full_matrix_cpu_hours": naive_hours,
        "staged_to_naive_cost_ratio": staged_hours / naive_hours,
        "future_duration_controller_status": (
            "not_yet_authorized; variable-step BDF2 and step-doubling "
            "must be frozen and certified after profile breadth passes"
        ),
    }


def _manifest(spectral: dict, readiness: dict, controller: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "short_horizon_nonlinear_profile_breadth_and_controller_"
            "manifest_frozen_coarse_breadth_screen_authorized"
        ),
        "operator_changed": False,
        "propagation_executed": False,
        "profiles": PROFILE_DEFINITIONS,
        "binding_propagation_multiplier": BINDING_PROPAGATION_MULTIPLIER,
        "initial_readiness_multipliers": list(READINESS_MULTIPLIERS),
        "spectral_eligibility": spectral,
        "initial_physical_readiness": readiness,
        "campaign_controller": controller,
        "binding_contract": {
            "horizon_seconds": HORIZON_SECONDS,
            "common_output_times_seconds": COMMON_OUTPUT_TIMES_SECONDS.tolist(),
            "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS.tolist(),
            "state_and_Tier_I_metrics": (
                "unchanged b3a temporal and b2a spatial response contracts"
            ),
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "checkpoint_roundtrip": "bitwise",
            "split_restart_replay": "bitwise",
            "incoming_excision_characteristics": 0,
            "Tier_II": "diagnostic_nonpromoted",
        },
        "interpretation_limits": {
            "meaningfully_nonlinear_dynamics_certified": False,
            "long_nonlinear_physical_ladder_authorized": False,
            "fixed_q_micro_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "authorized_next": (
            "WP10c9d6c7c3b4b1_coarse_short_horizon_nonlinear_"
            "profile_breadth_screen"
        ),
        "hard_stops": [
            "no profile tuning after this manifest",
            "no duration extension before all breadth stages pass",
            "no meaningful-nonlinearity claim from the fixed 0.01 amplitude",
            "no operator or production-default change",
            "no fixed-Q or reduced evolution",
            "no N1024 rescue",
        ],
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict, manifest: dict) -> str:
    spectral = manifest["spectral_eligibility"]
    readiness = manifest["initial_physical_readiness"]
    controller = manifest["campaign_controller"]
    lines = [
        "# Nonlinear profile-breadth/controller manifest WP10c9d6c7c3b4a",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "This definitions-only package changes no operator and propagates no "
        "state. It freezes five held-out characteristic profiles and a "
        "checkpoint-safe fail-fast campaign.",
        "",
        "## Frozen held-outs",
        "",
    ]
    for name, definition in PROFILE_DEFINITIONS.items():
        item = spectral["profiles"][name]
        lines.append(
            f"- `{name}`: {definition['role']}; theta99 "
            f"`{item['theta_99']:.6f}`; eligible `{item['passed']}`"
        )
    lines.extend(
        [
            "",
            "## Initial physical readiness",
            "",
            f"- maximum H/R: `{readiness['maximum_h_over_r']:.8f}`",
            "- minimum scattering optical depth: "
            f"`{readiness['minimum_scattering_optical_depth']:.8f}`",
            "- minimum reconstruction factor: "
            f"`{readiness['minimum_reconstruction_factor']:.16g}`",
            "- maximum cross-layout restriction defect: "
            f"`{readiness['maximum_restriction_defect']:.3e}`",
            "- maximum coupling trace jump: "
            f"`{readiness['maximum_coupling_trace_jump']:.3e}`",
            "- incoming excision characteristics: "
            f"`{readiness['maximum_incoming_excision_characteristics']}`",
            "",
            "All signs and half amplitudes pass the initial physical gates. "
            "Only the full positive amplitude is binding for propagation; "
            "the package does not claim a measurable nonlinear remainder.",
            "",
            "## Efficient campaign controller",
            "",
            "- staged estimated cost: "
            f"`{controller['estimated_staged_cpu_hours']:.2f} CPU h`",
            "- naive full-matrix estimate: "
            f"`{controller['estimated_naive_full_matrix_cpu_hours']:.2f} CPU h`",
            "- staged/naive ratio: "
            f"`{controller['staged_to_naive_cost_ratio']:.3f}`",
            "- stages: coarse fixed-step method screen; coarse temporal "
            "refinement; middle/fine spatial confirmation",
            "",
            "## Authorized next",
            "",
            f"`{summary['authorized_next']}`",
            "",
            "Duration extension, variable-step control, fixed-Q experiments, "
            "and reduced slow evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parent, _, _ = _validate_parent()
    (
        configurations,
        interpolator,
        field_scales,
        parent_edges,
        projected,
        projection_reports,
        arrays,
        characteristic_report,
    ) = _project_profiles()
    spectral = _spectral_eligibility(
        configurations,
        interpolator,
        field_scales,
        projected,
        projection_reports,
    )
    readiness = _embedded_readiness(
        field_scales,
        parent_edges,
        projected,
        arrays,
    )
    controller = _campaign_controller()
    passed = bool(spectral["passed"] and readiness["passed"] and controller["passed"])
    if not passed:
        raise RuntimeError("profile-breadth manifest readiness failed")
    manifest = _manifest(spectral, readiness, controller)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profile_amplitude": PROFILE_AMPLITUDE,
        "profiles": PROFILE_DEFINITIONS,
        "readiness_multipliers": list(READINESS_MULTIPLIERS),
        "binding_propagation_multiplier": BINDING_PROPAGATION_MULTIPLIER,
        "layouts": list(LAYOUTS),
        "timestep_levels_seconds": TIMESTEP_LEVELS_SECONDS.tolist(),
        "horizon_seconds": HORIZON_SECONDS,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "authorized_next": manifest["authorized_next"],
        "operator_changed": False,
        "propagation_executed": False,
        "parent_classification": parent["classification"],
        "profile_count": len(PROFILE_NAMES),
        "all_profiles_spectrally_eligible": spectral["passed"],
        "all_initial_readiness_variants_passed": readiness["passed"],
        "campaign_controller_passed": controller["passed"],
        "estimated_staged_cpu_hours": controller[
            "estimated_staged_cpu_hours"
        ],
        "meaningfully_nonlinear_dynamics_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": causal_canonical_json_sha256(config),
        "manifest_sha256": causal_canonical_json_sha256(manifest),
        "decisive_arrays_sha256": None,
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in arrays.items()
        },
        "characteristic_field_report": characteristic_report,
    }
    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    _write_json(SUMMARY_PATH, summary)
    input_paths = {
        "parent_summary": PARENT_DIRECTORY / "summary.json",
        "c7a_arrays": c7c0.C7A_ARRAYS,
        "c7c0_summary": C7C0_DIRECTORY / "summary.json",
        "eligibility_config": c7c0.E1_CONFIG,
        "replay_contexts": c7c0.C0E_CONTEXTS,
        "replay_inputs": c7c0.C0E_INPUTS,
        "temporal_manifest_summary": c3b3a.SUMMARY_PATH,
    }
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src /Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
            "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
            "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
            "implementation_parent_tree_sha": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": _source_identity(),
            "input_hashes": {
                name: _sha256(path) for name, path in input_paths.items()
            },
            "scientific_status": "CERTIFIED",
        },
    )
    REPORT_PATH.write_text(_report(summary, manifest), encoding="utf-8")
    names = (
        "config.json",
        "profile_breadth_controller_manifest.json",
        "decisive_arrays.npz",
        "summary.json",
        "provenance.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
