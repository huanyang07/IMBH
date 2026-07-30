#!/usr/bin/env python3
"""Freeze the one-way physical-core interface-scattering scope.

WP10c9d6c7c2a2 certified the manufactured C4 state, invariant projectors,
positive energy, and complete variable-coefficient energy ledger.  It also
proved that every complete coordinate characteristic at the exact interface
core points toward smaller radius.  A bidirectional physical-core experiment
is therefore impossible.

This definitions-only package selects the causal coarse-to-fine route.  It
freezes analytic packet definitions, measurement surfaces, travel-time
windows, uncertainty rules, and future decision gates.  It propagates no
state and changes no physical or numerical operator.
"""

from __future__ import annotations

import csv
import json
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

import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2a3"
ANALYZED_BASE_COMMIT = "1f3570894fc6e41a0770289dc7134356402e17cb"
ANALYZED_BASE_PARENT = "de29e71f05be20c979c52354584b7b694fb26c6e"
ANALYZED_BASE_TREE = "a0ceaf93236b9962c94bd6b7b81f9903f003b05a"
THIS_RUNNER = (
    "scripts/run_causal_inner_scattering_scope_wp10c9d6c7c2a3.py"
)
THIS_TEST = (
    "tests/test_causal_inner_scattering_scope_wp10c9d6c7c2a3.py"
)

C_CGS = 2.99792458e10
PATCH_CELL_COUNT = 98
PATCH_INTERFACE_FACE = 49
PACKET_SUPPORT = (52, 95)
DOWNSTREAM_MEASUREMENT_FACE = 6
UPSTREAM_DIAGNOSTIC_FACE = 92
REFERENCE_LEVELS = (98, 196, 392)
PRIMARY_TIME_SAMPLES = 513
TIME_SAMPLE_COUNTS = (257, 513, 1025)
WINDOW_PADDING_FRACTION = 0.025
WINDOW_PADDING_NUISANCE_FACTORS = (0.5, 1.0, 1.5)
OBSERVABILITY_FACTOR = 5.0

C2A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_observability_manifest_wp10c9d6c7c2a"
)
C2A2_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_energy_wp10c9d6c7c2a2"
)
C7A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_scope_wp10c9d6c7c2a3"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "scope_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (THIS_RUNNER, THIS_TEST)


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
        relative: c2a._sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _parent_hashes() -> dict[str, str]:
    paths = (
        C2A_DIRECTORY / "scattering_manifest.json",
        C2A_DIRECTORY / "summary.json",
        C2A2_DIRECTORY / "decisive_arrays.npz",
        C2A2_DIRECTORY / "method_manifest.json",
        C2A2_DIRECTORY / "summary.json",
        C7A_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _load_inputs() -> tuple[dict, dict, dict[str, np.ndarray], np.ndarray]:
    summary = c2a._read_json(C2A2_DIRECTORY / "summary.json")
    method = c2a._read_json(C2A2_DIRECTORY / "method_manifest.json")
    if (
        summary["classification"]
        != "manufactured_interface_patch_rejected_"
        "unidirectional_characteristic_core"
        or summary["passed"]
        or summary["propagation_executed"]
        or summary["operator_changed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2a3_definitions_only_scattering_scope_revision"
        or summary["manifest_sha256"] != method["manifest_sha256"]
    ):
        raise RuntimeError("c2a2 binding status changed")
    decision = method["binding_decision"]
    required_method_passes = (
        decision["energy_and_projector_method_passed"],
        decision["extension_admissibility_and_hyperbolicity_passed"],
        decision["interface_core_parity_passed"],
        decision["independent_balance_reference_passed"],
        decision["coarse_to_fine_incidence_available"],
    )
    if not all(required_method_passes):
        raise RuntimeError("one-way c2a2 method prerequisites changed")
    if (
        decision["fine_to_coarse_incidence_available"]
        or decision["bidirectional_incidence_passed"]
    ):
        raise RuntimeError("c2a2 causal direction changed")
    with np.load(
        C2A2_DIRECTORY / "decisive_arrays.npz",
        allow_pickle=False,
    ) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
        }
    with np.load(
        C7A_DIRECTORY / "decisive_arrays.npz",
        allow_pickle=False,
    ) as source:
        field_scales = np.asarray(source["field_scales"], dtype=float)
    if field_scales.shape != (5,) or np.any(field_scales <= 0.0):
        raise RuntimeError("invalid physical primitive scales")
    return summary, method, arrays, field_scales


def _compact_envelope(
    log_centers: np.ndarray,
    log_edges: np.ndarray,
) -> np.ndarray:
    left, right = PACKET_SUPPORT
    coordinate = (
        (log_centers - log_edges[left])
        / (log_edges[right] - log_edges[left])
    )
    envelope = np.zeros_like(log_centers)
    active = (coordinate > 0.0) & (coordinate < 1.0)
    envelope[active] = np.sin(np.pi * coordinate[active]) ** 4
    return envelope


def _seed_direction(
    projector: np.ndarray,
    energy: np.ndarray,
    field_scales: np.ndarray,
) -> tuple[np.ndarray, int]:
    scaled = (
        np.diag(1.0 / field_scales)
        @ projector
        @ np.diag(field_scales)
    )
    column = int(np.argmax(np.linalg.norm(scaled, axis=0)))
    direction = projector @ (field_scales * np.eye(5)[column])
    norm = float(np.sqrt(direction @ energy @ direction))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("singular family seed")
    return direction / norm, column


def _packet_from_seed(
    *,
    family: int,
    seed: np.ndarray,
    envelope: np.ndarray,
    projectors: np.ndarray,
    energy: np.ndarray,
    field_scales: np.ndarray,
) -> np.ndarray:
    packet = np.zeros((PATCH_CELL_COUNT, 5), dtype=float)
    previous = None
    for cell in range(PATCH_CELL_COUNT):
        direction = projectors[cell, family] @ seed
        norm = float(np.sqrt(direction @ energy[cell] @ direction))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise RuntimeError("packet seed leaves selected subspace")
        direction /= norm
        if (
            previous is not None
            and np.dot(
                previous / field_scales,
                direction / field_scales,
            )
            < 0.0
        ):
            direction *= -1.0
        packet[cell] = envelope[cell] * direction
        previous = direction
    return packet


def _family_energy_fractions(
    packet: np.ndarray,
    projectors: np.ndarray,
    energy: np.ndarray,
    measures: np.ndarray,
) -> np.ndarray:
    projected = np.einsum(
        "nfij,nj->nfi",
        projectors,
        packet,
        optimize=True,
    )
    family_energy = 0.5 * np.einsum(
        "nfi,nij,nfj,n->f",
        projected,
        energy,
        projected,
        measures,
        optimize=True,
    )
    return family_energy / max(
        float(np.sum(family_energy)),
        np.finfo(float).tiny,
    )


def _travel_time_seconds(
    edges: np.ndarray,
    speeds: np.ndarray,
    family: int,
    source_face: int,
    target_face: int,
) -> float:
    if target_face >= source_face:
        raise ValueError("one-way travel target must be inward")
    cells = np.arange(target_face, source_face)
    selected = np.abs(speeds[cells, family])
    if np.any(selected <= np.finfo(float).tiny):
        raise RuntimeError("zero characteristic speed in travel path")
    return float(np.sum(np.diff(edges)[cells] / (C_CGS * selected)))


def _window(
    edges: np.ndarray,
    speeds: np.ndarray,
    family: int,
    target_face: int,
) -> tuple[float, float]:
    leading = _travel_time_seconds(
        edges,
        speeds,
        family,
        PACKET_SUPPORT[0],
        target_face,
    )
    trailing = _travel_time_seconds(
        edges,
        speeds,
        family,
        PACKET_SUPPORT[1],
        target_face,
    )
    padding = WINDOW_PADDING_FRACTION * (trailing - leading)
    return max(0.0, leading - padding), trailing + padding


def _freeze_packets_and_windows(
    arrays: dict[str, np.ndarray],
    field_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    edges = np.asarray(arrays["patch_edges"], dtype=float)
    centers = np.asarray(arrays["patch_centers"], dtype=float)
    speeds = np.asarray(
        arrays["characteristic_speeds_over_c"],
        dtype=float,
    )
    projectors = np.asarray(
        arrays["normalization_invariant_projectors"],
        dtype=float,
    )
    energy = np.asarray(arrays["primitive_energy_metrics"], dtype=float)
    measures = np.diff(edges)
    envelope = _compact_envelope(np.log(centers), np.log(edges))
    center_cell = int(np.argmax(envelope))

    names = {
        "acoustic": 0,
        "shear": 1,
        "material_null": 2,
    }
    seeds = {}
    seed_columns = {}
    packets = {}
    fractions = {}
    for name, family in names.items():
        seed, column = _seed_direction(
            projectors[center_cell, family],
            energy[center_cell],
            field_scales,
        )
        packet = _packet_from_seed(
            family=family,
            seed=seed,
            envelope=envelope,
            projectors=projectors,
            energy=energy,
            field_scales=field_scales,
        )
        seeds[name] = seed
        seed_columns[name] = column
        packets[name] = packet
        fractions[name] = _family_energy_fractions(
            packet,
            projectors,
            energy,
            measures,
        )
    packets["mixed_shear_acoustic"] = (
        packets["acoustic"] + packets["shear"]
    ) / np.sqrt(2.0)
    fractions["mixed_shear_acoustic"] = _family_energy_fractions(
        packets["mixed_shear_acoustic"],
        projectors,
        energy,
        measures,
    )
    packets["zero_null"] = np.zeros_like(packets["acoustic"])

    interface_windows = {
        family: _window(
            edges,
            speeds,
            index,
            PATCH_INTERFACE_FACE,
        )
        for family, index in (("acoustic", 0), ("shear", 1))
    }
    downstream_windows = {
        family: _window(
            edges,
            speeds,
            index,
            DOWNSTREAM_MEASUREMENT_FACE,
        )
        for family, index in (("acoustic", 0), ("shear", 1))
    }
    interface_windows["mixed_shear_acoustic"] = (
        min(value[0] for value in interface_windows.values()),
        max(value[1] for value in interface_windows.values()),
    )
    downstream_windows["mixed_shear_acoustic"] = (
        min(value[0] for value in downstream_windows.values()),
        max(value[1] for value in downstream_windows.values()),
    )
    experiment_end = max(
        value[1] for value in downstream_windows.values()
    )
    primary_time = np.linspace(0.0, experiment_end, PRIMARY_TIME_SAMPLES)
    windows_array = np.asarray(
        [
            (
                *interface_windows[name],
                *downstream_windows[name],
            )
            for name in (
                "acoustic",
                "shear",
                "mixed_shear_acoustic",
            )
        ],
        dtype=float,
    )
    available = np.all(speeds < 0.0, axis=1)
    maximum_positive_speed = float(np.max(speeds))

    report = {
        "route": "one_way_coarse_to_fine_physical_core",
        "exact_physical_core_retained": True,
        "manufactured_extension_is_physical_background": False,
        "packet_support_faces": list(PACKET_SUPPORT),
        "interface_face": PATCH_INTERFACE_FACE,
        "downstream_measurement_face": DOWNSTREAM_MEASUREMENT_FACE,
        "upstream_diagnostic_face": UPSTREAM_DIAGNOSTIC_FACE,
        "all_characteristics_inward_over_patch": bool(np.all(available)),
        "maximum_characteristic_speed_over_c": maximum_positive_speed,
        "positive_speed_family_count_everywhere": 0,
        "reflection_coefficient_defined": False,
        "reflection_reason": (
            "the positive-speed characteristic subspace is empty; report "
            "upstream contamination rather than a physical reflection ratio"
        ),
        "seed_center_cell": center_cell,
        "seed_primitive_columns": seed_columns,
        "initial_family_energy_fractions": {
            key: value.tolist()
            for key, value in fractions.items()
        },
        "amplitude_factors": [0.5, 1.0],
        "signs": [-1, 1],
        "interface_windows_seconds": {
            key: list(value)
            for key, value in interface_windows.items()
        },
        "downstream_windows_seconds": {
            key: list(value)
            for key, value in downstream_windows.items()
        },
        "experiment_end_seconds": experiment_end,
        "primary_time_sample_count": PRIMARY_TIME_SAMPLES,
        "time_sample_counts_for_stability": list(TIME_SAMPLE_COUNTS),
        "window_padding_fraction": WINDOW_PADDING_FRACTION,
        "window_padding_nuisance_factors": list(
            WINDOW_PADDING_NUISANCE_FACTORS
        ),
        "windows_derived_before_propagation": True,
        "observed_histories_may_not_move_windows": True,
    }
    decisive = {
        "patch_edges": edges,
        "patch_centers": centers,
        "interface_characteristic_speeds_over_c": np.asarray(
            arrays["interface_characteristic_speeds_over_c"],
            dtype=float,
        ),
        "packet_envelope": envelope,
        "packet_seed__acoustic": seeds["acoustic"],
        "packet_seed__shear": seeds["shear"],
        "packet_seed__material_null": seeds["material_null"],
        "packet__acoustic": packets["acoustic"],
        "packet__shear": packets["shear"],
        "packet__mixed_shear_acoustic": packets[
            "mixed_shear_acoustic"
        ],
        "packet__material_null": packets["material_null"],
        "packet__zero_null": packets["zero_null"],
        "initial_family_energy_fractions__acoustic": fractions[
            "acoustic"
        ],
        "initial_family_energy_fractions__shear": fractions["shear"],
        "initial_family_energy_fractions__mixed": fractions[
            "mixed_shear_acoustic"
        ],
        "initial_family_energy_fractions__material_null": fractions[
            "material_null"
        ],
        "measurement_faces": np.asarray(
            (
                DOWNSTREAM_MEASUREMENT_FACE,
                PATCH_INTERFACE_FACE,
                UPSTREAM_DIAGNOSTIC_FACE,
            ),
            dtype=np.int64,
        ),
        "travel_windows_seconds": windows_array,
        "primary_time_samples_seconds": primary_time,
        "reference_levels": np.asarray(REFERENCE_LEVELS, dtype=np.int64),
    }
    return report, decisive


def _scope_manifest(parent: dict, packet_report: dict) -> dict:
    inherited = c2a._read_json(
        C2A_DIRECTORY / "scattering_manifest.json"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "parent_classification_preserved": parent["classification"],
        "historical_classifications_preserved": True,
        "selected_route": "one_way_coarse_to_fine_physical_core",
        "route_claim": (
            "operator-neutral method-level one-way transmission through "
            "the exact physical interface core"
        ),
        "physical_scope": {
            "exact_interface_core": True,
            "full_manufactured_extension": False,
            "bidirectional_physical_scattering": False,
            "generic_bidirectional_method_stress_test": (
                "deferred_to_a_separate_nonphysical_work_package"
            ),
            "routes_must_not_be_combined": True,
        },
        "packet_and_window_contract": packet_report,
        "certification_tiers": {
            "tier_I_primary_physics": inherited[
                "certification_tiers"
            ]["tier_I_primary_physics"],
            "tier_II_one_way_transport": {
                "primary_observables": [
                    "time_integrated_incident_energy_at_interface",
                    "time_integrated_transmitted_energy_at_downstream_face",
                    "transmission_coefficient_T",
                    "target_family_transmission",
                    "family_leakage",
                    "physical_stress_relaxation_dissipation",
                    "stored_energy_change_between_faces_6_and_49",
                    "background_gradient_work",
                    "responsive_height_work",
                    "other_declared_lower_source_work",
                    "complete_energy_ledger_residual",
                ],
                "secondary_observables": [
                    "embedded_minus_uniform_upstream_contamination",
                    "pointwise_interface_traction_when_observable",
                    "spatial_window_energy",
                ],
                "reflection_coefficient_is_not_defined": True,
                "no_R_plus_T_equals_one_assumption": True,
                "transmission_definition": (
                    "T = E_transmitted_face_6 / E_incident_face_49"
                ),
                "interface_induced_transmission": (
                    "Delta_T = T_embedded - "
                    "T_uniform_continuum_extrapolate"
                ),
                "energy_balance": (
                    "E_incident - E_transmitted - D_physical - "
                    "Delta_E_stored - W_background - W_height - "
                    "W_other = ledger_residual"
                ),
            },
            "tier_III_nonlinear": inherited[
                "certification_tiers"
            ]["tier_III_nonlinear"],
        },
        "uncertainty_and_observability": {
            "component_bounds": [
                "continuum_reference",
                "finite_volume_projection",
                "invariant_subspace_choice",
                "window_placement",
                "time_sampling_and_quadrature",
                "restart_replay",
                "roundoff",
            ],
            "default_combination": (
                "conservative_sum_of_deterministic_component_bounds"
            ),
            "RSS_forbidden_without_demonstrated_independence": True,
            "measured_covariance_allowed_only_if_stable_and_documented": True,
            "observability_factor": OBSERVABILITY_FACTOR,
            "transmission_ratio_requires_observable_incident_energy": True,
            "error_direction_cosine_binding_condition": (
                "both refinement-error norms must exceed "
                "observability_factor times their frozen uncertainty bounds"
            ),
            "below_floor_classification": (
                "direction_not_certifying_because_error_is_below_"
                "observability"
            ),
            "maximum_reference_uncertainty_to_medium_fine_difference": 0.1,
            "no_slow_impact_threshold": True,
        },
        "uniform_c2b1_contract": {
            "work_package": (
                "WP10c9d6c7c2b1_one_way_uniform_scattering_validation"
            ),
            "reference_levels": list(REFERENCE_LEVELS),
            "same_C4_coefficient_field_at_every_level": True,
            "virtual_interface_matches_patch_face_49": True,
            "analytic_packet_reprojection_required_at_every_level": True,
            "packet_cases": (
                "acoustic_shear_mixed_times_two_signs_times_two_amplitudes"
            ),
            "material_family_null_control": True,
            "zero_state_null_control": True,
            "state_and_flux_amplitude_scaling": "linear",
            "energy_amplitude_scaling": "quadratic",
            "minimum_rms_order": 0.75,
            "minimum_maximum_order": 0.75,
            "minimum_significant_component_order": 0.75,
            "maximum_fine_normalized_difference": 0.05,
            "minimum_history_cosine": 0.9,
            "minimum_observable_refinement_error_cosine": 0.9,
            "maximum_energy_ledger_relative_defect": 1.0e-10,
            "exact_conservative_ledgers": True,
            "window_and_time_quadrature_stability": True,
        },
        "conditional_embedded_c2c1_contract": {
            "authorized_now": False,
            "authorized_only_after_uniform_c2b1_passes": True,
            "work_package": (
                "WP10c9d6c7c2c1_one_way_embedded_scattering_"
                "discrimination"
            ),
            "outer_incident_grid_fixed": True,
            "inner_transmitted_refinement_factors": [1, 2, 4],
            "same_packet_definitions_surfaces_windows_and_gates": True,
            "one_shared_MJE_interface_flux": True,
            "uniform_continuum_extrapolate_is_reference": True,
        },
        "decision_table": {
            "uniform_tier_I_and_observable_tier_II_pass": (
                "authorize_one_way_embedded_c2c1"
            ),
            "uniform_observable_definition_or_ledger_fails": (
                "repair_diagnostic_or_reference_before_embedded"
            ),
            "uniform_integrated_transport_passes_pointwise_fails": (
                "revise_prospective_pointwise_metric_not_operator"
            ),
            "embedded_T_and_energy_balance_pass": (
                "certify_declared_one_way_embedded_scattering_class"
            ),
            "embedded_T_fails_with_stable_interface_local_mechanism": (
                "authorize_interface_local_truncation_audit_only"
            ),
            "tier_I_passes_tier_II_below_observability": (
                "classify_tier_II_non_certifying_then_run_pre_frozen_"
                "tier_I_heldouts"
            ),
        },
        "hard_stops": [
            "do_not_relabel_c2a2_or_c7c1b",
            "do_not_claim_bidirectional_physical_scattering",
            "do_not_define_R_for_an_empty_positive_speed_subspace",
            "do_not_change_characteristic_signs_or_fit_the_background",
            "do_not_change_the_interface_operator",
            "do_not_start_embedded_before_uniform_c2b1_passes",
            "do_not_start_nonlinear_fixed_Q_or_reduced_evolution",
            "do_not_run_N1024_as_rescue",
        ],
        "binding_decision": {
            "definitions_only_scope_is_internally_consistent": True,
            "exact_physical_core_retained": True,
            "one_way_uniform_c2b1_authorized": True,
            "bidirectional_physical_propagation_authorized": False,
            "generic_bidirectional_method_test_authorized": False,
            "embedded_c2c1_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": (
            "one_way_physical_core_scattering_scope_frozen_"
            "uniform_validation_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c2b1_one_way_uniform_scattering_validation"
        ),
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(manifest)
    return manifest


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{c2a._sha256(path)}  {path.name}")
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
        provenance = c2a._read_json(provenance_path)
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "case": case.name,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": c2a._sha256(path),
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
    canonical_summary = c2a._read_json(CANONICAL_SUMMARY)
    canonical_summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    c2a._write_json(CANONICAL_SUMMARY, canonical_summary)


def run() -> dict:
    start = time.perf_counter()
    git_identity = _validate_analyzed_git_identity()
    parent_summary, parent_method, arrays, field_scales = _load_inputs()
    packet_report, decisive = _freeze_packets_and_windows(
        arrays,
        field_scales,
    )
    manifest = _scope_manifest(parent_summary, packet_report)
    source_hashes = _source_manifest()
    parent_hashes = _parent_hashes()

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "route": "one_way_coarse_to_fine_physical_core",
        "reference_levels": list(REFERENCE_LEVELS),
        "packet_support_faces": list(PACKET_SUPPORT),
        "measurement_faces": [
            DOWNSTREAM_MEASUREMENT_FACE,
            PATCH_INTERFACE_FACE,
            UPSTREAM_DIAGNOSTIC_FACE,
        ],
        "time_sample_counts": list(TIME_SAMPLE_COUNTS),
        "window_padding_fraction": WINDOW_PADDING_FRACTION,
        "window_padding_nuisance_factors": list(
            WINDOW_PADDING_NUISANCE_FACTORS
        ),
        "observability_factor": OBSERVABILITY_FACTOR,
        "propagation_executed": False,
    }
    c2a._write_json(CONFIG_PATH, config)
    c2a._write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    decisive_hashes = {
        name: causal_array_sha256(value)
        for name, value in decisive.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "passed": True,
        "classification": manifest["classification"],
        "authorized_next": manifest["authorized_next"],
        "operator_changed": False,
        "propagation_executed": False,
        "selected_route": manifest["selected_route"],
        "parent_classification_preserved": parent_summary[
            "classification"
        ],
        "parent_manifest_sha256": parent_method["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "packet_and_window_contract": packet_report,
        "binding_decision": manifest["binding_decision"],
        "decisive_array_hashes": decisive_hashes,
        "decisive_arrays_sha256": c2a._sha256(DECISIVE_ARRAYS),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_hashes)
        ),
        "parent_input_hashes": parent_hashes,
        "runtime_seconds": time.perf_counter() - start,
    }
    c2a._write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED",
        "classification": manifest["classification"],
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **git_identity,
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_scattering_scope_"
            "wp10c9d6c7c2a3.py"
        ),
        "implementation_source_hashes": source_hashes,
        "parent_input_hashes": parent_hashes,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    c2a._write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run()
