#!/usr/bin/env python3
"""Select an operator-neutral geometry for interface-scattering audits.

WP10c9d6c7c2a froze a physical scattering/observability contract but found
that its declared C3 compact packet needs 43 parent cells.  With a three-cell
clearance at both ends, each incident side needs 49 parent-cell equivalents.
The current physical layout provides only 48 inner and 16 outer cells.

This definitions-only package compares:

* extension of the physical radial domain;
* characteristic injection on the existing domain;
* a two-sided manufactured variable-coefficient interface patch.

It propagates no state and changes no physical or numerical operator.
"""

from __future__ import annotations

import csv
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

import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2a1"
ANALYZED_BASE_COMMIT = "73f902622834d13981d36e22aa21e13fefb9df8b"
ANALYZED_BASE_PARENT = "c73102812b73f115c1e4f2771be952adc6ea4c00"
ANALYZED_BASE_TREE = "ae82502e89fdaf0a28235b9010df31f9ade67267"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_scattering_geometry_wp10c9d6c7c2a1.py"
)

PACKET_SUPPORT_CELLS = 43
CLEARANCE_CELLS_EACH_END = 3
REQUIRED_CELLS_PER_SIDE = (
    PACKET_SUPPORT_CELLS + 2 * CLEARANCE_CELLS_EACH_END
)
PATCH_CELLS_PER_SIDE = REQUIRED_CELLS_PER_SIDE
PATCH_CELL_COUNT = 2 * PATCH_CELLS_PER_SIDE
PATCH_INTERFACE_FACE = PATCH_CELLS_PER_SIDE
PHYSICAL_INTERFACE_FACE = 48
PHYSICAL_PARENT_CELL_COUNT = 64
PHYSICAL_CORE_LOWER_FACE = 42
PHYSICAL_CORE_UPPER_FACE = 54
PATCH_CORE_LOWER_FACE = (
    PATCH_INTERFACE_FACE
    - (PHYSICAL_INTERFACE_FACE - PHYSICAL_CORE_LOWER_FACE)
)
PATCH_CORE_UPPER_FACE = (
    PATCH_INTERFACE_FACE
    + (PHYSICAL_CORE_UPPER_FACE - PHYSICAL_INTERFACE_FACE)
)
PATCH_LEFT_SUPPORT = (
    CLEARANCE_CELLS_EACH_END,
    CLEARANCE_CELLS_EACH_END + PACKET_SUPPORT_CELLS,
)
PATCH_RIGHT_SUPPORT = (
    PATCH_CELL_COUNT
    - CLEARANCE_CELLS_EACH_END
    - PACKET_SUPPORT_CELLS,
    PATCH_CELL_COUNT - CLEARANCE_CELLS_EACH_END,
)
PATCH_MEASUREMENT_FACES = (
    2 * CLEARANCE_CELLS_EACH_END,
    PATCH_INTERFACE_FACE,
    PATCH_CELL_COUNT - 2 * CLEARANCE_CELLS_EACH_END,
)
COEFFICIENT_TRANSITION_CELLS = 12

C2A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_observability_manifest_wp10c9d6c7c2a"
)
C3_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_geometry_wp10c9d6c7c2a1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "geometry_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/"
    "run_causal_inner_scattering_observability_manifest_"
    "wp10c9d6c7c2a.py",
    "tests/"
    "test_causal_inner_scattering_geometry_wp10c9d6c7c2a1.py",
)


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
        C2A_DIRECTORY / "config.json",
        C2A_DIRECTORY / "scattering_manifest.json",
        C2A_DIRECTORY / "decisive_arrays.npz",
        C2A_DIRECTORY / "summary.json",
        C3_DIRECTORY / "config.json",
        C3_DIRECTORY / "decisive_arrays.npz",
        C3_DIRECTORY / "summary.json",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _load_inputs() -> tuple[dict, dict, dict[str, np.ndarray]]:
    summary = c2a._read_json(C2A_DIRECTORY / "summary.json")
    manifest = c2a._read_json(C2A_DIRECTORY / "scattering_manifest.json")
    if (
        summary["classification"]
        != "scattering_observability_contract_frozen_"
        "bidirectional_packet_preflight_blocked"
        or not summary["passed"]
        or summary["uniform_scattering_propagation_authorized"]
        or summary["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise RuntimeError("c2a binding status changed")
    arrays = {}
    with np.load(
        C2A_DIRECTORY / "decisive_arrays.npz",
        allow_pickle=False,
    ) as source:
        arrays.update(
            {name: np.asarray(source[name]) for name in source.files}
        )
    with np.load(
        C3_DIRECTORY / "decisive_arrays.npz",
        allow_pickle=False,
    ) as source:
        arrays["continuum_background_knots"] = np.asarray(
            source["continuum_background_knots"],
            dtype=float,
        )
    return summary, manifest, arrays


def _physical_extension_report(
    parent_edges: np.ndarray,
    background_knots: np.ndarray,
) -> tuple[dict, np.ndarray]:
    log_spacing = float(np.mean(np.diff(np.log(parent_edges))))
    coupling_log_radius = float(
        np.log(parent_edges[PHYSICAL_INTERFACE_FACE])
    )
    extended_log_edges = coupling_log_radius + log_spacing * np.arange(
        -REQUIRED_CELLS_PER_SIDE,
        REQUIRED_CELLS_PER_SIDE + 1,
        dtype=float,
    )
    extended_edges = np.exp(extended_log_edges)
    gravitational_radius = float(parent_edges[0] / 1.8)
    certified_lower = float(np.exp(background_knots[0]))
    certified_upper = float(np.exp(background_knots[-1]))
    required_lower = float(extended_edges[0])
    required_upper = float(extended_edges[-1])
    geometry_passed = bool(
        extended_edges.size == 2 * REQUIRED_CELLS_PER_SIDE + 1
        and np.isclose(
            extended_edges[REQUIRED_CELLS_PER_SIDE],
            parent_edges[PHYSICAL_INTERFACE_FACE],
            rtol=2.0e-14,
            atol=0.0,
        )
    )
    background_coverage = bool(
        required_lower >= certified_lower
        and required_upper <= certified_upper
    )
    report = {
        "option": "extended_physical_domain",
        "operator_changed": False,
        "geometry_passed": geometry_passed,
        "cells_per_side": REQUIRED_CELLS_PER_SIDE,
        "required_inner_radius_over_rg": (
            required_lower / gravitational_radius
        ),
        "required_outer_radius_over_rg": (
            required_upper / gravitational_radius
        ),
        "certified_background_inner_radius_over_rg": (
            certified_lower / gravitational_radius
        ),
        "certified_background_outer_radius_over_rg": (
            certified_upper / gravitational_radius
        ),
        "certified_background_coverage_passed": background_coverage,
        "new_inner_excision_causality_certified": False,
        "independent_stationary_background_extension_available": False,
        "passed": bool(
            geometry_passed
            and background_coverage
            and False
        ),
        "selected": False,
        "reason_not_selected": (
            "geometry fits, but the required 1.728-94.479 rg domain lies "
            "outside the independently certified 1.8-12.777 rg background; "
            "new inner-boundary causality is also uncertified"
        ),
    }
    return report, extended_edges


def _existing_injection_report() -> dict:
    raw_inner = PHYSICAL_INTERFACE_FACE
    raw_outer = PHYSICAL_PARENT_CELL_COUNT - PHYSICAL_INTERFACE_FACE
    outer_boundary_roundtrip = 2 * raw_outer
    interface_surface_roundtrip = 2 * (
        int(c2a.POSTINTERFACE_PARENT_FACE)
        - PHYSICAL_INTERFACE_FACE
    )
    return {
        "option": "existing_domain_characteristic_injection",
        "operator_changed": False,
        "minimum_resolved_pulse_extent_parent_cell_times": (
            PACKET_SUPPORT_CELLS
        ),
        "outer_cells_available": raw_outer,
        "outer_boundary_to_interface_roundtrip_parent_cell_times": (
            outer_boundary_roundtrip
        ),
        "postinterface_surface_roundtrip_parent_cell_times": (
            interface_surface_roundtrip
        ),
        "coarse_to_fine_nonoverlap_at_outer_boundary_passed": bool(
            PACKET_SUPPORT_CELLS < outer_boundary_roundtrip
        ),
        "coarse_to_fine_nonoverlap_at_postinterface_surface_passed": bool(
            PACKET_SUPPORT_CELLS < interface_surface_roundtrip
        ),
        "fine_to_coarse_injection_at_excision_allowed": False,
        "injected_energy_ledger_implemented": False,
        "passed": False,
        "selected": False,
        "reason_not_selected": (
            "the 43-cell resolved pulse is longer than both available "
            "coarse-side return separations, while injection through the "
            "outgoing excision surface would violate the causal contract"
        ),
    }


def _manufactured_patch_report(
    patch_edges: np.ndarray,
    parent_edges: np.ndarray,
) -> dict:
    spacing_defect = float(
        np.max(
            np.abs(
                np.diff(np.log(patch_edges))
                - np.mean(np.diff(np.log(parent_edges)))
            )
        )
    )
    interface_defect = float(
        abs(
            patch_edges[PATCH_INTERFACE_FACE]
            - parent_edges[PHYSICAL_INTERFACE_FACE]
        )
        / parent_edges[PHYSICAL_INTERFACE_FACE]
    )
    left_clearance = int(PATCH_LEFT_SUPPORT[0])
    right_clearance = int(
        PATCH_INTERFACE_FACE - PATCH_LEFT_SUPPORT[1]
    )
    opposite_left_clearance = int(
        PATCH_RIGHT_SUPPORT[0] - PATCH_INTERFACE_FACE
    )
    opposite_right_clearance = int(
        PATCH_CELL_COUNT - PATCH_RIGHT_SUPPORT[1]
    )
    geometry_passed = bool(
        patch_edges.size == PATCH_CELL_COUNT + 1
        and spacing_defect <= 1.0e-14
        and interface_defect <= 2.0e-14
        and min(
            left_clearance,
            right_clearance,
            opposite_left_clearance,
            opposite_right_clearance,
        )
        >= CLEARANCE_CELLS_EACH_END
    )
    incident_reflected_separation = int(
        2
        * (
            PATCH_INTERFACE_FACE
            - PATCH_MEASUREMENT_FACES[0]
        )
    )
    return {
        "option": "manufactured_variable_coefficient_interface_patch",
        "operator_changed": False,
        "physical_background_claimed": False,
        "method_level_interface_audit_only": True,
        "parent_equivalent_cell_count": PATCH_CELL_COUNT,
        "parent_equivalent_cells_per_side": PATCH_CELLS_PER_SIDE,
        "interface_face": PATCH_INTERFACE_FACE,
        "physical_core_parent_faces": [
            PHYSICAL_CORE_LOWER_FACE,
            PHYSICAL_CORE_UPPER_FACE,
        ],
        "patch_core_faces": [
            PATCH_CORE_LOWER_FACE,
            PATCH_CORE_UPPER_FACE,
        ],
        "left_packet_support_faces": list(PATCH_LEFT_SUPPORT),
        "right_packet_support_faces": list(PATCH_RIGHT_SUPPORT),
        "measurement_faces": list(PATCH_MEASUREMENT_FACES),
        "minimum_packet_clearance_cells": min(
            left_clearance,
            right_clearance,
            opposite_left_clearance,
            opposite_right_clearance,
        ),
        "incident_reflected_separation_parent_cell_times": (
            incident_reflected_separation
        ),
        "resolved_pulse_extent_parent_cell_times": PACKET_SUPPORT_CELLS,
        "travel_window_nonoverlap_capacity_passed": bool(
            incident_reflected_separation > PACKET_SUPPORT_CELLS
        ),
        "maximum_log_spacing_defect": spacing_defect,
        "relative_interface_radius_defect": interface_defect,
        "geometry_passed": geometry_passed,
        "coefficient_extension_implemented": False,
        "energy_ledger_implemented": False,
        "passed": geometry_passed,
        "selected": geometry_passed,
        "selection_scope": (
            "definitions_only_route_selection; c2a2 method gates remain "
            "binding before propagation"
        ),
    }


def _coefficient_extension_contract() -> dict:
    return {
        "exact_physical_core": {
            "parent_faces": [
                PHYSICAL_CORE_LOWER_FACE,
                PHYSICAL_CORE_UPPER_FACE,
            ],
            "patch_faces": [
                PATCH_CORE_LOWER_FACE,
                PATCH_CORE_UPPER_FACE,
            ],
            "interface_state_measure_and_principal_matrix_parity_required": (
                True
            ),
            "maximum_interface_parity_defect": 1.0e-12,
        },
        "extension_chart": "implemented_five_field_primitive_chart",
        "endpoint_jet_order": 4,
        "transition_regularity": "C4",
        "transition_parent_cells": COEFFICIENT_TRANSITION_CELLS,
        "construction": (
            "fourth_order_endpoint_Taylor_jet_blended_to_constant_far_"
            "state_with_C4_degree9_smootherstep"
        ),
        "degree9_smootherstep_coefficients_low_to_high": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            126.0,
            -420.0,
            540.0,
            -315.0,
            70.0,
        ],
        "must_recompute_all_physical_matrices_from_extended_state": True,
        "direct_matrix_interpolation_forbidden": True,
        "admissibility_required_everywhere": True,
        "hyperbolicity_and_separated_subspaces_required_everywhere": True,
        "background_gradient_and_height_work_ledgered": True,
        "uniform_and_embedded_use_identical_coefficient_field": True,
        "uniform_subtraction_removes_manufactured_background_scattering": True,
        "no_residual_subtraction": True,
        "no_fitted_coefficient": True,
    }


def _c2a2_contract() -> dict:
    return {
        "work_package": (
            "WP10c9d6c7c2a2_manufactured_scattering_energy_preflight"
        ),
        "propagation_authorized": False,
        "required_before_uniform_propagation": [
            "construct_C4_extended_state_and_recompute_physical_matrices",
            "certify_primitive_admissibility",
            "certify_real_separated_characteristic_clusters",
            "derive_complete_symmetrized_DAE_energy_identity",
            "implement_normalization_invariant_Schur_or_QZ_projectors",
            "implement_incident_reflected_transmitted_flux_ledgers",
            "verify_interface_core_parity",
            "verify_constant_and_variable_coefficient_manufactured_balances",
            "verify_null_channel_and_amplitude_scaling",
            "freeze_exact_packets_surfaces_windows_and_hashes",
        ],
        "maximum_interface_core_parity_defect": 1.0e-12,
        "maximum_projector_idempotence_defect": 1.0e-12,
        "maximum_energy_ledger_relative_defect": 1.0e-10,
        "maximum_constant_state_residual": 1.0e-12,
        "minimum_signal_to_uncertainty_ratio": 5.0,
        "maximum_reference_uncertainty_to_fine_difference": 0.10,
        "uniform_c2b_authorized_only_if_every_gate_passes": True,
    }


def _build() -> tuple[dict, dict[str, np.ndarray]]:
    parent_summary, parent_manifest, inputs = _load_inputs()
    parent_edges = np.asarray(inputs["parent_grid_edges"], dtype=float)
    background_knots = np.asarray(
        inputs["continuum_background_knots"],
        dtype=float,
    )
    if (
        int(
            parent_summary["geometry_feasibility"][
                "minimum_spectral_support_parent_cells"
            ]
        )
        != PACKET_SUPPORT_CELLS
    ):
        raise RuntimeError("frozen packet support changed")
    if CLEARANCE_CELLS_EACH_END != int(
        parent_summary["geometry_feasibility"][
            "reconstruction_halo_cells_each_end"
        ]
    ):
        raise RuntimeError("frozen clearance changed")

    physical_report, physical_edges = _physical_extension_report(
        parent_edges,
        background_knots,
    )
    injection_report = _existing_injection_report()
    patch_log_spacing = float(np.mean(np.diff(np.log(parent_edges))))
    patch_coupling_log = float(
        np.log(parent_edges[PHYSICAL_INTERFACE_FACE])
    )
    patch_edges = np.exp(
        patch_coupling_log
        + patch_log_spacing
        * np.arange(
            -PATCH_CELLS_PER_SIDE,
            PATCH_CELLS_PER_SIDE + 1,
            dtype=float,
        )
    )
    patch_report = _manufactured_patch_report(
        patch_edges,
        parent_edges,
    )
    option_reports = {
        "extended_physical_domain": physical_report,
        "existing_domain_characteristic_injection": injection_report,
        "manufactured_variable_coefficient_interface_patch": patch_report,
    }
    selected = [
        name
        for name, report in option_reports.items()
        if report["selected"]
    ]
    if selected != [
        "manufactured_variable_coefficient_interface_patch"
    ]:
        raise RuntimeError("geometry selection is not unique")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "parent_manifest_sha256": parent_manifest["manifest_sha256"],
        "parent_classification_preserved": (
            parent_summary["classification"]
        ),
        "required_geometry": {
            "packet_support_parent_cells": PACKET_SUPPORT_CELLS,
            "clearance_cells_each_end": CLEARANCE_CELLS_EACH_END,
            "required_parent_cells_per_side": REQUIRED_CELLS_PER_SIDE,
            "both_incidence_directions_required": True,
        },
        "option_reports": option_reports,
        "selected_route": selected[0],
        "selected_route_scope": (
            "operator-neutral method-level interface scattering only"
        ),
        "coefficient_extension_contract": (
            _coefficient_extension_contract()
        ),
        "c2a2_method_preflight_contract": _c2a2_contract(),
        "scientific_limits": {
            "physical_radial_background_certified": False,
            "actual_physical_embedded_scattering_certified": False,
            "direct_c7c1b_Tier_I_result_unchanged": True,
            "c7c1b_strict_rejection_unchanged": True,
            "manufactured_patch_cannot_by_itself_authorize_nonlinear_work": (
                True
            ),
        },
        "decision_after_c2a2": {
            "all_method_gates_pass": (
                "authorize_uniform_c2b_on_exact_frozen_manufactured_patch"
            ),
            "extension_admissibility_or_hyperbolicity_fails": (
                "reject_manufactured_route_and_stop"
            ),
            "energy_identity_or_projector_fails": (
                "repair_diagnostic_definition_before_propagation"
            ),
            "interface_core_parity_fails": (
                "repair_patch_construction_not_interface_operator"
            ),
        },
        "hard_stops": [
            "do_not_propagate_in_c2a1",
            "do_not_change_the_interface_operator",
            "do_not_claim_a_physical_background_extension",
            "do_not_extrapolate_the_c3_background_as_physical_truth",
            "do_not_use_existing_domain_injection_with_overlapping_windows",
            "do_not_relax_packet_spectral_or_clearance_gates",
            "do_not_reclassify_c7c1b",
            "do_not_begin_embedded_nonlinear_fixed_Q_or_reduced_evolution",
        ],
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(
        manifest
    )
    arrays = {
        "original_parent_grid_edges": parent_edges,
        "certified_background_knot_log_radii": background_knots,
        "extended_physical_candidate_edges": physical_edges,
        "manufactured_patch_edges": patch_edges,
        "manufactured_patch_core_face_map": np.asarray(
            [
                [
                    PHYSICAL_CORE_LOWER_FACE,
                    PHYSICAL_INTERFACE_FACE,
                    PHYSICAL_CORE_UPPER_FACE,
                ],
                [
                    PATCH_CORE_LOWER_FACE,
                    PATCH_INTERFACE_FACE,
                    PATCH_CORE_UPPER_FACE,
                ],
            ],
            dtype=np.int64,
        ),
        "manufactured_patch_support_faces": np.asarray(
            [PATCH_LEFT_SUPPORT, PATCH_RIGHT_SUPPORT],
            dtype=np.int64,
        ),
        "manufactured_patch_measurement_faces": np.asarray(
            PATCH_MEASUREMENT_FACES,
            dtype=np.int64,
        ),
        "option_passed_selected_flags": np.asarray(
            [
                [
                    int(option_reports[name]["passed"]),
                    int(option_reports[name]["selected"]),
                ]
                for name in option_reports
            ],
            dtype=np.int8,
        ),
    }
    return manifest, arrays


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "manifest_sha256": manifest["manifest_sha256"],
        "operator_changed": False,
        "propagation_executed": False,
        "packet_support_parent_cells": PACKET_SUPPORT_CELLS,
        "clearance_cells_each_end": CLEARANCE_CELLS_EACH_END,
        "required_parent_cells_per_side": REQUIRED_CELLS_PER_SIDE,
        "selected_route": manifest["selected_route"],
    }


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
            if not path.is_file():
                continue
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
    summary = c2a._read_json(CANONICAL_SUMMARY)
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
    c2a._write_json(CANONICAL_SUMMARY, summary)


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
    option_reports = manifest["option_reports"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "passed": True,
        "classification": (
            "manufactured_interface_patch_geometry_selected_"
            "energy_preflight_authorized"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_classification_preserved": manifest[
            "parent_classification_preserved"
        ],
        "selected_route": manifest["selected_route"],
        "selected_route_scope": manifest["selected_route_scope"],
        "option_reports": option_reports,
        "required_geometry": manifest["required_geometry"],
        "operator_changed": False,
        "propagation_executed": False,
        "uniform_scattering_propagation_authorized": False,
        "embedded_scattering_propagation_authorized": False,
        "bounded_nonlinear_common_mode_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest[
            "c2a2_method_preflight_contract"
        ]["work_package"],
        "implementation_source_hashes": source_manifest,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_manifest)
        ),
        "parent_input_hashes": _parent_hashes(),
        "decisive_arrays_path": str(DECISIVE_ARRAYS.relative_to(ROOT)),
        "decisive_arrays_sha256": c2a._sha256(DECISIVE_ARRAYS),
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
            "scripts/run_causal_inner_scattering_geometry_"
            "wp10c9d6c7c2a1.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "implementation_source_hashes": source_manifest,
        "parent_input_hashes": summary["parent_input_hashes"],
        "scientific_status": "DIAGNOSTIC ONLY",
        "classification": summary["classification"],
    }
    c2a._write_json(CONFIG_PATH, config)
    c2a._write_json(MANIFEST_PATH, manifest)
    c2a._write_json(PROVENANCE_PATH, provenance)
    c2a._write_json(SUMMARY_PATH, summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    return summary


if __name__ == "__main__":
    print(json.dumps(c2a._plain(run()), indent=2, sort_keys=True))
