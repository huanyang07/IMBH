#!/usr/bin/env python3
"""Freeze the interface-scattering observability contract.

WP10c9d6c7c1b passed every direct state and physical-export gate for all
sixteen regularized embedded controls, but retained its strict rejection
because five small auxiliary channels missed inherited direction-cosine
gates.  This definitions-only package separates the direct physical
contract from a prospective, invariant interface-scattering contract.

The package propagates no state and changes no operator.  It also performs a
necessary geometry preflight: a compact incident packet must satisfy the
already-certified spectral contract on both sides of the frozen coupling
surface while retaining reconstruction clearance.  No propagation package
is authorized when that bidirectional packet class cannot fit.
"""

from __future__ import annotations

import csv
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (  # noqa: E402
    causal_packet_spectrum,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2a"
ANALYZED_BASE_COMMIT = "c73102812b73f115c1e4f2771be952adc6ea4c00"
ANALYZED_BASE_PARENT = "d9300e938f6636d92518cc242ac16231a65d6716"
ANALYZED_BASE_TREE = "61675425c2a5b3c0c2845e9a271ae95bfe9d2c60"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_scattering_observability_manifest_"
    "wp10c9d6c7c2a.py"
)

COUPLING_PARENT_FACE = 48
PARENT_CELL_COUNT = 64
PREINTERFACE_PARENT_FACE = 45
POSTINTERFACE_PARENT_FACE = 51
RECONSTRUCTION_HALO_CELLS = 3
WINDOW_POWER = 4
MINIMUM_SIGNAL_TO_UNCERTAINTY_RATIO = 5.0
MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE = 0.10
MINIMUM_OBSERVABLE_ERROR_COSINE = 0.90
MINIMUM_OBSERVABLE_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05

C7C0_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_manifest_wp10c9d6c7c0"
)
C7C1A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a"
)
C7C1B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_wp10c9d6c7c1b"
)
C7B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_validation_wp10c9d6c7b"
)
E1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1"
)

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_observability_manifest_wp10c9d6c7c2a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "scattering_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_manifest.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_resolution.py",
    "tests/"
    "test_causal_inner_scattering_observability_manifest_"
    "wp10c9d6c7c2a.py",
)


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


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
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


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if resolved != ANALYZED_BASE_COMMIT:
        raise RuntimeError("analyzed base commit changed")
    if parent != ANALYZED_BASE_PARENT:
        raise RuntimeError("analyzed base parent changed")
    if tree != ANALYZED_BASE_TREE:
        raise RuntimeError("analyzed base tree changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
        "current_head": _git_value("rev-parse", "HEAD"),
        "current_branch": _git_value("branch", "--show-current"),
        "working_tree_status": _git_value("status", "--short"),
    }


def _source_manifest() -> dict[str, str]:
    return {
        relative: _sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _load_parent_contracts() -> tuple[dict, dict]:
    c7b = _read_json(C7B_DIRECTORY / "summary.json")
    c7c0 = _read_json(C7C0_DIRECTORY / "summary.json")
    c7c1a = _read_json(C7C1A_DIRECTORY / "summary.json")
    c7c1b = _read_json(C7C1B_DIRECTORY / "summary.json")
    eligibility = _read_json(E1_DIRECTORY / "config.json")[
        "eligibility_contract"
    ]
    expected = {
        "c7b": (
            c7b,
            "prospective_embedded_profile_validation_failed",
            False,
        ),
        "c7c0": (
            c7c0,
            "endpoint_interface_regularity_manifest_frozen_"
            "uniform_control_preflight_authorized",
            True,
        ),
        "c7c1a": (
            c7c1a,
            "endpoint_interface_regularity_uniform_controls_certified_"
            "embedded_discrimination_authorized",
            True,
        ),
        "c7c1b": (
            c7c1b,
            "no_regularized_embedded_profile_class_selected",
            False,
        ),
    }
    for label, (summary, classification, passed) in expected.items():
        if (
            summary["classification"] != classification
            or bool(summary["passed"]) is not passed
        ):
            raise RuntimeError(f"{label} classification changed")
    preserved = {
        label: {
            "classification": classification,
            "passed": passed,
        }
        for label, (_, classification, passed) in expected.items()
    }
    return preserved, eligibility


def _parent_hashes() -> dict[str, str]:
    paths = (
        C7B_DIRECTORY / "summary.json",
        C7C0_DIRECTORY / "summary.json",
        C7C0_DIRECTORY / "decisive_arrays.npz",
        C7C1A_DIRECTORY / "summary.json",
        C7C1B_DIRECTORY / "config.json",
        C7C1B_DIRECTORY / "summary.json",
        C7C1B_DIRECTORY / "decisive_arrays.npz",
        E1_DIRECTORY / "config.json",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _compact_envelope_scan(
    *,
    maximum_theta: float,
    maximum_alias: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Find the narrowest C3 compact envelope passing inherited spectrum gates."""

    widths = np.arange(4, COUPLING_PARENT_FACE + 1, dtype=np.int64)
    theta = np.empty(widths.size, dtype=float)
    alias = np.empty(widths.size, dtype=float)
    endpoint_fraction = np.empty(widths.size, dtype=float)
    centers = np.arange(PARENT_CELL_COUNT, dtype=float) + 0.5
    for index, width in enumerate(widths):
        values = np.zeros((PARENT_CELL_COUNT, 5), dtype=float)
        active = centers < float(width)
        phase = np.pi * centers[active] / float(width)
        values[active, 0] = np.sin(phase) ** WINDOW_POWER
        spectrum = causal_packet_spectrum(
            values,
            1.0,
            quantile=0.99,
        )
        theta[index] = float(spectrum.quantile_angular_wavenumber)
        alias[index] = float(spectrum.nyquist_alias_fraction)
        peak = max(float(np.max(values[:, 0])), np.finfo(float).tiny)
        endpoint_fraction[index] = float(
            max(values[0, 0], values[int(width) - 1, 0]) / peak
        )
    eligible = (
        (theta <= maximum_theta)
        & (alias <= maximum_alias)
        & (endpoint_fraction <= 5.0e-3)
    )
    minimum_width = (
        int(widths[np.flatnonzero(eligible)[0]])
        if np.any(eligible)
        else None
    )
    report = {
        "template": "cell_centered_zero_extended_sin_power_4",
        "regularity": "C3",
        "spectral_energy_quantile": 0.99,
        "maximum_theta_99": maximum_theta,
        "maximum_nyquist_alias_fraction": maximum_alias,
        "maximum_endpoint_cell_fraction": 5.0e-3,
        "minimum_eligible_support_parent_cells": minimum_width,
        "eligible_width_count": int(np.count_nonzero(eligible)),
        "scan_is_necessary_not_sufficient": True,
        "full_physical_projection_and_family_purity_still_required": True,
    }
    arrays = {
        "support_width_parent_cells": widths,
        "envelope_theta_99": theta,
        "envelope_nyquist_alias_fraction": alias,
        "envelope_endpoint_cell_fraction": endpoint_fraction,
        "envelope_eligible": eligible.astype(np.int8),
    }
    return report, arrays


def _requested_profiles() -> dict:
    common = {
        "endpoint_regularity": "C3_or_better",
        "compact_support": True,
        "actual_finite_volume_projection_spectrum_required": True,
        "spectral_contract_must_hold_on_incident_and_transmitted_sides": True,
        "initial_support_separated_from_interface": True,
        "measurement_windows_precomputed_from_characteristic_travel_times": True,
        "observed_history_may_not_move_binding_windows": True,
        "amplitude_factors": [0.5, 1.0],
        "signs": [-1, 1],
    }
    profiles = {}
    for direction in ("fine_to_coarse", "coarse_to_fine"):
        for family in ("shear", "acoustic", "mixed_shear_acoustic"):
            profiles[f"{direction}__{family}"] = {
                **common,
                "incidence_direction": direction,
                "family": family,
                "binding": True,
                "role": "prospective_interface_scattering",
            }
    profiles["null_selected_channel"] = {
        **common,
        "incidence_direction": "fine_to_coarse",
        "family": "orthogonal_family_null",
        "binding": True,
        "role": "diagnostic_false_positive_floor",
        "selected_incident_channel_exactly_zero": True,
    }
    return profiles


def _tier_contract() -> dict:
    return {
        "tier_I_primary_physics": {
            "observables": [
                "state",
                "inner_mass_flux",
                "inner_angular_momentum_flux",
                "inner_killing_energy_flux",
                "coupling_mass_flux_when_active",
                "coupling_angular_momentum_flux_when_active",
                "coupling_killing_energy_flux_when_active",
                "net_mass_drive",
                "net_angular_momentum_drive",
                "net_killing_energy_drive",
                "cooling",
                "responsive_height_work",
            ],
            "one_shared_conservative_face_flux": True,
            "exact_prefix_and_global_ledgers": True,
            "historical_order_difference_and_cosine_gates_unchanged": True,
        },
        "tier_II_interface_scattering": {
            "primary_observables": [
                "time_integrated_incident_energy_flux",
                "time_integrated_reflected_energy_flux",
                "time_integrated_transmitted_energy_flux",
                "family_leakage",
                "physical_stress_relaxation_dissipation",
                "stored_energy_change",
                "background_gradient_work",
                "responsive_height_work",
                "other_declared_lower_source_work",
                "complete_energy_ledger_residual",
            ],
            "secondary_observables": [
                "pointwise_interface_traction_when_observable",
                "spatial_window_energy",
            ],
            "integrated_flux_is_primary": True,
            "pointwise_traction_requires_observability": True,
        },
        "tier_III_nonlinear": {
            "authorized": False,
            "future_required_cases": [
                "bounded_nonlinear_common_mode",
                "finite_amplitude_interface_crossing_packet",
            ],
            "nonlinear_scaled_residual_maximum": 1.0e-10,
            "dense_colored_and_independent_jvp_checks_required": True,
            "bitwise_bdf2_replay_required": True,
        },
    }


def _energy_contract() -> dict:
    return {
        "state_space": "complete_linearized_five_field_DAE",
        "energy_metric": (
            "descriptor_compatible_positive_physical_symmetrizer"
        ),
        "projectors": (
            "smooth_real_Schur_or_generalized_QZ_invariant_subspaces"
        ),
        "eigenvector_normalization_invariant": True,
        "required_projector_checks": [
            "positive_energy_metric",
            "idempotence",
            "mutual_energy_orthogonality_for_separated_clusters",
            "internal_basis_change_invariance",
            "continuous_subspace_tracking",
        ],
        "scattering_coefficients": {
            "R": "E_reflected / E_incident",
            "T": "E_transmitted / E_incident",
            "interface_delta_R": (
                "R_embedded - R_uniform_continuum_extrapolate"
            ),
            "interface_delta_T": (
                "T_embedded - T_uniform_continuum_extrapolate"
            ),
            "interface_delta_leakage": (
                "leakage_embedded - leakage_uniform_continuum_extrapolate"
            ),
        },
        "complete_balance": (
            "E_incident - E_reflected - E_transmitted "
            "- D_physical - Delta_E_stored - W_background "
            "- W_height - W_other = ledger_residual"
        ),
        "balance_terms_must_be_derived_from_implemented_symmetrized_DAE": True,
        "no_constant_coefficient_R_plus_T_equals_one_assumption": True,
        "each_physical_work_term_recorded_exactly_once": True,
        "uniform_virtual_interface_parent_face": COUPLING_PARENT_FACE,
        "uniform_reference_requires_continuum_extrapolation": True,
    }


def _uncertainty_contract() -> dict:
    return {
        "sources": [
            "continuum_reference",
            "finite_volume_projection",
            "invariant_subspace_choice",
            "measurement_window_placement",
            "time_sampling_and_quadrature",
            "restart_replay",
            "roundoff",
        ],
        "default_combination": (
            "conservative_sum_or_direct_nuisance_sweep_envelope"
        ),
        "root_sum_square_forbidden_without_demonstrated_independence": True,
        "covariance_combination_allowed_only_when_measured_and_stable": True,
        "minimum_signal_to_uncertainty_ratio": (
            MINIMUM_SIGNAL_TO_UNCERTAINTY_RATIO
        ),
        "maximum_reference_uncertainty_to_medium_fine_difference": (
            MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE
        ),
        "ratio_eligibility": (
            "incident_energy_lower_bound_must_be_positive_and "
            "nominal_incident_energy_must_exceed_kappa_times_uncertainty"
        ),
        "ratio_uncertainty": (
            "conservative_interval_bounds_not_near_zero_linearization"
        ),
        "direction_gate": {
            "coarse_medium_error_norm_must_exceed_kappa_uncertainty": True,
            "medium_fine_error_norm_must_exceed_kappa_uncertainty": True,
            "minimum_conservative_cosine_lower_bound": (
                MINIMUM_OBSERVABLE_ERROR_COSINE
            ),
            "below_floor_classification": (
                "direction_not_certifying_because_error_is_below_"
                "observability"
            ),
            "below_floor_is_neither_pass_nor_fail": True,
        },
        "slow_impact_threshold_binding": False,
        "slow_impact_deferred_until_Q_macro_horizon_and_closure_exist": True,
        "c7c1b_not_reclassified": True,
    }


def _propagation_contract() -> dict:
    return {
        "uniform_and_embedded_are_separate_commits": True,
        "uniform_work_package": "WP10c9d6c7c2b",
        "embedded_work_package": "WP10c9d6c7c2c",
        "uniform_must_pass_before_embedded": True,
        "uniform_virtual_interface_matches_embedded_coupling_radius": True,
        "state_and_flux_amplitude_scaling": "linear",
        "characteristic_energy_amplitude_scaling": "quadratic",
        "minimum_rms_order": MINIMUM_OBSERVABLE_ORDER,
        "minimum_maximum_order": MINIMUM_OBSERVABLE_ORDER,
        "minimum_significant_component_order": MINIMUM_OBSERVABLE_ORDER,
        "maximum_fine_normalized_difference": (
            MAXIMUM_FINE_NORMALIZED_DIFFERENCE
        ),
        "minimum_history_cosine": MINIMUM_OBSERVABLE_ERROR_COSINE,
        "minimum_observable_refinement_error_cosine": (
            MINIMUM_OBSERVABLE_ERROR_COSINE
        ),
        "exact_conservative_and_energy_ledgers": True,
        "window_shift_and_time_quadrature_stability": True,
        "branch_B_tier_I_heldouts_frozen_before_any_propagation": True,
    }


def _build() -> tuple[dict, dict[str, np.ndarray]]:
    preserved, eligibility = _load_parent_contracts()
    c7c0_arrays = np.load(
        C7C0_DIRECTORY / "decisive_arrays.npz",
        allow_pickle=False,
    )
    parent_edges = np.asarray(c7c0_arrays["parent_grid_edges"], dtype=float)
    if parent_edges.size != PARENT_CELL_COUNT + 1:
        raise RuntimeError("frozen parent-grid cell count changed")
    coupling_radius = float(parent_edges[COUPLING_PARENT_FACE])
    gravitational_radius = float(parent_edges[0] / 1.8)

    spectral_report, scan_arrays = _compact_envelope_scan(
        maximum_theta=float(eligibility["maximum_theta_99"]),
        maximum_alias=float(
            eligibility["maximum_nyquist_alias_fraction"]
        ),
    )
    minimum_support = spectral_report[
        "minimum_eligible_support_parent_cells"
    ]
    if minimum_support is None:
        raise RuntimeError("no C3 compact envelope passes inherited gates")

    raw_inner_cells = COUPLING_PARENT_FACE
    raw_outer_cells = PARENT_CELL_COUNT - COUPLING_PARENT_FACE
    cleared_inner_cells = (
        raw_inner_cells - 2 * RECONSTRUCTION_HALO_CELLS
    )
    cleared_outer_cells = (
        raw_outer_cells - 2 * RECONSTRUCTION_HALO_CELLS
    )
    geometry = {
        "parent_cell_count": PARENT_CELL_COUNT,
        "coupling_parent_face": COUPLING_PARENT_FACE,
        "coupling_radius": coupling_radius,
        "coupling_radius_over_rg": coupling_radius / gravitational_radius,
        "preinterface_measurement_parent_face": PREINTERFACE_PARENT_FACE,
        "postinterface_measurement_parent_face": POSTINTERFACE_PARENT_FACE,
        "reconstruction_halo_cells_each_end": RECONSTRUCTION_HALO_CELLS,
        "raw_inner_parent_cells": raw_inner_cells,
        "raw_outer_parent_cells": raw_outer_cells,
        "clearance_qualified_inner_parent_cells": cleared_inner_cells,
        "clearance_qualified_outer_parent_cells": cleared_outer_cells,
        "minimum_spectral_support_parent_cells": minimum_support,
        "raw_fine_to_coarse_capacity_passed": (
            raw_inner_cells >= minimum_support
        ),
        "raw_coarse_to_fine_capacity_passed": (
            raw_outer_cells >= minimum_support
        ),
        "clearance_fine_to_coarse_capacity_passed": (
            cleared_inner_cells >= minimum_support
        ),
        "clearance_coarse_to_fine_capacity_passed": (
            cleared_outer_cells >= minimum_support
        ),
    }
    bidirectional_feasible = bool(
        geometry["clearance_fine_to_coarse_capacity_passed"]
        and geometry["clearance_coarse_to_fine_capacity_passed"]
    )
    geometry["bidirectional_compact_packet_class_feasible"] = (
        bidirectional_feasible
    )
    geometry["interpretation"] = (
        "The frozen 64-cell domain has insufficient clearance-qualified "
        "support on both sides of coupling face 48 for the narrowest frozen "
        "C3 sin^4 compact-envelope template that passes the inherited "
        "theta_99 and alias gates. This is a geometry preflight failure for "
        "the declared packet class, not an operator or scattering-diagnostic "
        "failure."
    )

    requested_profiles = _requested_profiles()
    branch_b_heldouts = {
        "tier_I_heldout__inner_broad_mixed": {
            "binding_if_branch_B_reached": True,
            "support_side": "inner",
            "family": "five_field_mixed",
            "endpoint_regularity": "C3_or_better",
            "definition_must_remain_independent_of_c2c_results": True,
        },
        "tier_I_heldout__outer_broad_mixed": {
            "binding_if_branch_B_reached": True,
            "support_side": "outer",
            "family": "five_field_mixed",
            "endpoint_regularity": "C3_or_better",
            "definition_must_remain_independent_of_c2c_results": True,
        },
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_classifications_preserved": preserved,
        "scientific_interpretation": {
            "direct_physical_contract": (
                "passed_for_declared_c7c1b_profiles"
            ),
            "interface_scattering_observability": "unresolved",
            "c7c1b_strict_classification_unchanged": True,
            "interface_redesign_selected": False,
        },
        "certification_tiers": _tier_contract(),
        "energy_and_scattering_contract": _energy_contract(),
        "uncertainty_and_observability_contract": _uncertainty_contract(),
        "measurement_contract": {
            "primary_measurement": (
                "time_integrated_signed_energy_flux_through_fixed_"
                "physical_surfaces"
            ),
            "uniform_virtual_interface_parent_face": (
                COUPLING_PARENT_FACE
            ),
            "preinterface_measurement_parent_face": (
                PREINTERFACE_PARENT_FACE
            ),
            "postinterface_measurement_parent_face": (
                POSTINTERFACE_PARENT_FACE
            ),
            "windows_from_precomputed_continuum_characteristic_travel_times": (
                True
            ),
            "no_observed_peak_repositioning": True,
            "window_nuisance_sweeps_are_uncertainty_only": True,
            "stop_before_boundary_return_contamination": True,
        },
        "inherited_spectral_contract": eligibility,
        "compact_envelope_preflight": spectral_report,
        "geometry_feasibility": geometry,
        "requested_scattering_profiles": requested_profiles,
        "branch_B_tier_I_heldouts": branch_b_heldouts,
        "prospective_propagation_contract": _propagation_contract(),
        "decision_table": {
            "geometry_and_uniform_pass": (
                "authorize_separate_embedded_c2c_discrimination"
            ),
            "tier_I_and_observable_tier_II_pass": (
                "certify_declared_regularized_embedded_scattering_class"
            ),
            "tier_I_pass_tier_II_below_observability": (
                "classify_tier_II_channels_non_certifying_then_run_"
                "pre_frozen_tier_I_heldouts"
            ),
            "observable_integrated_scattering_fails_stably": (
                "authorize_interface_local_truncation_audit_only"
            ),
            "integrated_scattering_passes_pointwise_traction_fails": (
                "revise_prospective_diagnostic_metric_not_operator"
            ),
            "uniform_scattering_fails": (
                "repair_definition_or_reference_before_embedded"
            ),
            "bidirectional_geometry_infeasible": (
                "stop_before_propagation_and_design_one_operator_neutral_"
                "scattering_domain_or_boundary_injection_preflight"
            ),
        },
        "hard_stops": [
            "do_not_amend_or_relabel_c7b_through_c7c1b",
            "do_not_lower_historical_gates",
            "do_not_use_c7c1b_magnitudes_to_set_thresholds",
            "do_not_bind_a_slow_impact_threshold_yet",
            "do_not_use_RSS_without_demonstrated_independence",
            "do_not_tune_endpoint_power_or_buffer_length_again",
            "do_not_redesign_the_interface",
            "do_not_propagate_before_bidirectional_geometry_is_feasible",
            "do_not_run_N1024_as_rescue",
            "do_not_begin_nonlinear_fixed_Q_or_reduced_slow_evolution",
        ],
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(manifest)

    arrays = {
        "parent_grid_edges": parent_edges,
        "field_scales": np.asarray(c7c0_arrays["field_scales"], dtype=float),
        "measurement_parent_faces": np.asarray(
            [
                PREINTERFACE_PARENT_FACE,
                COUPLING_PARENT_FACE,
                POSTINTERFACE_PARENT_FACE,
            ],
            dtype=np.int64,
        ),
        "geometry_capacity_parent_cells": np.asarray(
            [
                raw_inner_cells,
                raw_outer_cells,
                cleared_inner_cells,
                cleared_outer_cells,
                minimum_support,
            ],
            dtype=np.int64,
        ),
        **scan_arrays,
    }
    c7c0_arrays.close()
    return manifest, arrays


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "manifest_sha256": manifest["manifest_sha256"],
        "operator_changed": False,
        "propagation_executed": False,
        "minimum_signal_to_uncertainty_ratio": (
            MINIMUM_SIGNAL_TO_UNCERTAINTY_RATIO
        ),
        "maximum_reference_uncertainty_to_fine_difference": (
            MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE
        ),
        "minimum_observable_error_cosine": (
            MINIMUM_OBSERVABLE_ERROR_COSINE
        ),
        "minimum_observable_order": MINIMUM_OBSERVABLE_ORDER,
        "maximum_fine_normalized_difference": (
            MAXIMUM_FINE_NORMALIZED_DIFFERENCE
        ),
    }


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


def _refresh_canonical_catalog() -> None:
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = _read_json(provenance_path)
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
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
    summary = _read_json(CANONICAL_SUMMARY)
    summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def run() -> dict:
    start = time.perf_counter()
    git_identity = _validate_analyzed_git_identity()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest, arrays = _build()
    config = _config(manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    source_manifest = _source_manifest()
    array_hashes = {
        name: causal_array_sha256(value)
        for name, value in arrays.items()
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "passed": True,
        "classification": (
            "scattering_observability_contract_frozen_"
            "bidirectional_packet_preflight_blocked"
        ),
        "scientific_interpretation": manifest[
            "scientific_interpretation"
        ],
        "historical_classifications_preserved": manifest[
            "historical_classifications_preserved"
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "compact_envelope_preflight": manifest[
            "compact_envelope_preflight"
        ],
        "geometry_feasibility": manifest["geometry_feasibility"],
        "tier_I_direct_physics_status": (
            "passed_for_declared_c7c1b_profiles"
        ),
        "tier_II_scattering_status": "definitions_frozen_not_propagated",
        "uniform_scattering_propagation_authorized": False,
        "embedded_scattering_propagation_authorized": False,
        "bounded_nonlinear_common_mode_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c2a1_operator_neutral_scattering_geometry_"
            "feasibility_design"
        ),
        "operator_changed": False,
        "propagation_executed": False,
        "implementation_source_hashes": source_manifest,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_manifest)
        ),
        "parent_input_hashes": _parent_hashes(),
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": array_hashes,
        "runtime_seconds": time.perf_counter() - start,
    }
    provenance = {
        **git_identity,
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_scattering_observability_manifest_"
            "wp10c9d6c7c2a.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "implementation_source_hashes": source_manifest,
        "parent_input_hashes": summary["parent_input_hashes"],
        "scientific_status": "DIAGNOSTIC ONLY",
        "classification": (
            "scattering_observability_contract_frozen_"
            "bidirectional_packet_preflight_blocked"
        ),
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    _write_json(PROVENANCE_PATH, provenance)
    _write_json(SUMMARY_PATH, summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


if __name__ == "__main__":
    print(json.dumps(_plain(run()), indent=2, sort_keys=True))
