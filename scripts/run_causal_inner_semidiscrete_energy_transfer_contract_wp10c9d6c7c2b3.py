#!/usr/bin/env python3
"""Freeze the positive semidiscrete energy-transfer contract.

WP10c9d6c7c2b2 certified the exact semidiscrete energy identity but showed
that a ratio of two local face powers is not a certifiable transmission
observable for the descriptor-reduced operator.  This definitions-only
package replaces that rejected ratio with positive stored energy in fixed
physical source and receiving bands.

No state is propagated and no physical or numerical operator is changed.
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
import run_causal_inner_scattering_scope_wp10c9d6c7c2a3 as c2a3  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b3"
ANALYZED_BASE_COMMIT = "6b144ecf325efbd428b25b14f0f388d8e0369515"
ANALYZED_BASE_PARENT = "51a32ff686cea3b91d7f5056c464004399318172"
ANALYZED_BASE_TREE = "a73b21a30c7e9666084b38044c33372b5cd2243f"

REFERENCE_LEVELS = c2a3.REFERENCE_LEVELS
SOURCE_BAND_FACES = c2a3.PACKET_SUPPORT
RECEIVING_BAND_FACES = (
    c2a3.DOWNSTREAM_MEASUREMENT_FACE,
    c2a3.PATCH_INTERFACE_FACE,
)
UPSTREAM_DIAGNOSTIC_BAND_FACES = (
    c2a3.PACKET_SUPPORT[1],
    c2a3.PATCH_CELL_COUNT,
)
RECEIVING_BAND_NUISANCE_FACES = (
    (5, 48),
    (5, 49),
    (6, 48),
    (6, 49),
    (7, 48),
    (7, 49),
)
PRIMARY_FAMILIES = ("acoustic", "shear", "mixed_shear_acoustic")
FAMILY_INDICES = {
    "acoustic": (0,),
    "shear": (1,),
    "mixed_shear_acoustic": (0, 1),
    "material_null": (2,),
}
MAXIMUM_ALGEBRAIC_DEFECT = 1.0e-10
MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE = 0.10
MINIMUM_OBSERVABLE_ERROR_COSINE = 0.90
MINIMUM_OBSERVABLE_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05
OBSERVABILITY_FACTOR = c2a3.OBSERVABILITY_FACTOR

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_semidiscrete_energy_transfer_contract_"
    "wp10c9d6c7c2b3.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_semidiscrete_energy_transfer_contract_"
    "wp10c9d6c7c2b3.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_SEMIDISCRETE_ENERGY_TRANSFER_CONTRACT_"
    "WP10C9D6C7C2B3_RESULTS_2026-07-30.md"
)

C2B2_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2"
)
SCOPE_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_scope_wp10c9d6c7c2a3"
)
ENERGY_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_energy_wp10c9d6c7c2a2"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_semidiscrete_energy_transfer_contract_wp10c9d6c7c2b3"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "transfer_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
REPORT_PATH = ROOT / REPORT_RELATIVE
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _source_manifest() -> dict[str, str]:
    return {
        relative: c2a._sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _input_hashes() -> dict[str, str]:
    paths = (
        C2B2_DIRECTORY / "config.json",
        C2B2_DIRECTORY / "summary.json",
        C2B2_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        ENERGY_DIRECTORY / "method_manifest.json",
        ENERGY_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): c2a._sha256(path)
        for path in paths
    }


def _validate_parent() -> tuple[dict, dict, dict, dict[str, np.ndarray]]:
    parent = c2a._read_json(C2B2_DIRECTORY / "summary.json")
    decision = parent["binding_decision"]
    if (
        parent["classification"]
        != "exact_semidiscrete_energy_identity_certified_"
        "local_face_transmission_not_certifiable"
        or not parent["passed"]
        or not decision["semidiscrete_energy_identity_passed"]
        or decision["local_face_transmission_contract_certified"]
        or decision["genuine_uniform_transport_error_selected"]
        or decision["embedded_c2c1_authorized"]
        or decision["operator_or_interface_redesign_authorized"]
        or parent["authorized_next"]
        != "WP10c9d6c7c2b3_definitions_only_semidiscrete_energy_"
        "transfer_contract"
    ):
        raise RuntimeError("WP10c9d6c7c2b2 binding status changed")
    scope = c2a._read_json(SCOPE_DIRECTORY / "scope_manifest.json")
    scope_arrays = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    energy_arrays = _load_npz(ENERGY_DIRECTORY / "decisive_arrays.npz")
    return parent, scope, scope_arrays, energy_arrays


def _band_energy(
    state: np.ndarray,
    metric: np.ndarray,
    log_widths: np.ndarray,
    lower_face: int,
    upper_face: int,
) -> float:
    lower = int(lower_face)
    upper = int(upper_face)
    return float(
        0.5
        * np.einsum(
            "ni,nij,nj,n->",
            state[lower:upper],
            metric[lower:upper],
            state[lower:upper],
            log_widths[lower:upper],
            optimize=True,
        )
    )


def _family_band_energies(
    state: np.ndarray,
    projectors: np.ndarray,
    metric: np.ndarray,
    log_widths: np.ndarray,
    lower_face: int,
    upper_face: int,
) -> np.ndarray:
    projected = np.einsum(
        "nfij,nj->nfi",
        projectors,
        state,
        optimize=True,
    )
    lower = int(lower_face)
    upper = int(upper_face)
    return 0.5 * np.einsum(
        "nfi,nij,nfj,n->f",
        projected[lower:upper],
        metric[lower:upper],
        projected[lower:upper],
        log_widths[lower:upper],
        optimize=True,
    )


def _arrival_windows(travel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    primary = np.column_stack(
        (
            np.zeros(3, dtype=float),
            travel[:, 3],
        )
    )
    padding = float(c2a3.WINDOW_PADDING_FRACTION)
    padded_span = travel[:, 3] - travel[:, 2]
    raw_span = padded_span / (1.0 + 2.0 * padding)
    raw_trailing = travel[:, 3] - padding * raw_span
    nuisance = np.stack(
        [
            np.column_stack(
                (
                    np.zeros(3, dtype=float),
                    raw_trailing + padding * factor * raw_span,
                )
            )
            for factor in c2a3.WINDOW_PADDING_NUISANCE_FACTORS
        ],
        axis=0,
    )
    return primary, nuisance


def _freeze_positive_reference(
    scope_arrays: dict[str, np.ndarray],
    energy_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    edges = np.asarray(scope_arrays["patch_edges"], dtype=float)
    log_widths = np.diff(np.log(edges))
    metric = np.asarray(
        energy_arrays["primitive_energy_metrics"],
        dtype=float,
    )
    projectors = np.asarray(
        energy_arrays["normalization_invariant_projectors"],
        dtype=float,
    )
    if (
        edges.shape != (c2a3.PATCH_CELL_COUNT + 1,)
        or metric.shape != (c2a3.PATCH_CELL_COUNT, 5, 5)
        or projectors.shape != (c2a3.PATCH_CELL_COUNT, 5, 5, 5)
    ):
        raise RuntimeError("frozen c2a2/c2a3 geometry changed")

    packet_names = (
        "acoustic",
        "shear",
        "mixed_shear_acoustic",
        "material_null",
        "zero_null",
    )
    total_initial = []
    receiving_initial = []
    upstream_initial = []
    family_initial = []
    partition_defects = []
    reports = {}
    for name in packet_names:
        packet = np.asarray(scope_arrays[f"packet__{name}"], dtype=float)
        total = _band_energy(
            packet,
            metric,
            log_widths,
            *SOURCE_BAND_FACES,
        )
        receiving = _band_energy(
            packet,
            metric,
            log_widths,
            *RECEIVING_BAND_FACES,
        )
        upstream = _band_energy(
            packet,
            metric,
            log_widths,
            *UPSTREAM_DIAGNOSTIC_BAND_FACES,
        )
        family = _family_band_energies(
            packet,
            projectors,
            metric,
            log_widths,
            *SOURCE_BAND_FACES,
        )
        partition = abs(float(np.sum(family)) - total) / max(
            abs(total),
            np.finfo(float).tiny,
        )
        total_initial.append(total)
        receiving_initial.append(receiving)
        upstream_initial.append(upstream)
        family_initial.append(family)
        partition_defects.append(partition)
        reports[name] = {
            "initial_source_band_energy": total,
            "initial_receiving_band_energy": receiving,
            "initial_upstream_diagnostic_band_energy": upstream,
            "initial_family_energies": family.tolist(),
            "family_partition_relative_defect": partition,
            "target_family_indices": list(FAMILY_INDICES.get(name, ())),
        }

    positive = np.asarray(total_initial[:4], dtype=float)
    zero_energy = float(total_initial[4])
    maximum_partition = float(np.max(partition_defects[:4]))
    maximum_receiving = float(np.max(np.abs(receiving_initial)))
    minimum_metric_eigenvalue = float(
        min(np.min(np.linalg.eigvalsh(block)) for block in metric)
    )
    reference_passed = bool(
        np.all(positive > np.finfo(float).tiny)
        and zero_energy == 0.0
        and maximum_receiving == 0.0
        and minimum_metric_eigenvalue > 0.0
        and maximum_partition <= MAXIMUM_ALGEBRAIC_DEFECT
    )

    primary_windows, nuisance_windows = _arrival_windows(
        np.asarray(scope_arrays["travel_windows_seconds"], dtype=float)
    )
    level_band_faces = np.asarray(
        [
            (
                cells,
                RECEIVING_BAND_FACES[0] * (cells // REFERENCE_LEVELS[0]),
                RECEIVING_BAND_FACES[1] * (cells // REFERENCE_LEVELS[0]),
                SOURCE_BAND_FACES[0] * (cells // REFERENCE_LEVELS[0]),
                SOURCE_BAND_FACES[1] * (cells // REFERENCE_LEVELS[0]),
                UPSTREAM_DIAGNOSTIC_BAND_FACES[0]
                * (cells // REFERENCE_LEVELS[0]),
                UPSTREAM_DIAGNOSTIC_BAND_FACES[1]
                * (cells // REFERENCE_LEVELS[0]),
            )
            for cells in REFERENCE_LEVELS
        ],
        dtype=np.int64,
    )
    nuisance_faces = np.asarray(RECEIVING_BAND_NUISANCE_FACES, dtype=np.int64)
    decisive = {
        "patch_edges": edges,
        "reference_levels": np.asarray(REFERENCE_LEVELS, dtype=np.int64),
        "level_band_faces": level_band_faces,
        "receiving_band_nuisance_faces_N98": nuisance_faces,
        "primary_arrival_windows_seconds": primary_windows,
        "arrival_window_nuisance_seconds": nuisance_windows,
        "primary_time_samples_seconds": np.asarray(
            scope_arrays["primary_time_samples_seconds"],
            dtype=float,
        ),
        "initial_total_energy": np.asarray(total_initial, dtype=float),
        "initial_receiving_band_energy": np.asarray(
            receiving_initial,
            dtype=float,
        ),
        "initial_upstream_band_energy": np.asarray(
            upstream_initial,
            dtype=float,
        ),
        "initial_family_energy": np.asarray(family_initial, dtype=float),
        "family_partition_relative_defect": np.asarray(
            partition_defects,
            dtype=float,
        ),
    }
    report = {
        "passed": reference_passed,
        "packet_order": list(packet_names),
        "per_packet": reports,
        "minimum_positive_initial_energy": float(np.min(positive)),
        "zero_null_initial_energy": zero_energy,
        "maximum_initial_receiving_band_energy": maximum_receiving,
        "minimum_energy_metric_eigenvalue": minimum_metric_eigenvalue,
        "maximum_family_partition_relative_defect": maximum_partition,
        "source_band_faces_N98": list(SOURCE_BAND_FACES),
        "receiving_band_faces_N98": list(RECEIVING_BAND_FACES),
        "upstream_diagnostic_band_faces_N98": list(
            UPSTREAM_DIAGNOSTIC_BAND_FACES
        ),
        "receiving_band_nuisance_faces_N98": [
            list(item) for item in RECEIVING_BAND_NUISANCE_FACES
        ],
        "primary_arrival_windows_seconds": {
            family: primary_windows[index].tolist()
            for index, family in enumerate(PRIMARY_FAMILIES)
        },
        "arrival_windows_derived_before_propagation": True,
        "bands_fixed_in_physical_space_and_nested_across_levels": True,
    }
    return report, decisive


def _transfer_manifest(parent: dict, reference: dict) -> dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_classifications_preserved": True,
        "parent_classification_preserved": parent["classification"],
        "rejected_contract": {
            "local_incident_face_power": True,
            "local_transmitted_face_power": True,
            "ratio_of_two_local_face_powers": True,
            "reason": (
                "descriptor inversion makes the exact control-energy dual "
                "nonlocal, so two selected faces do not define a closed "
                "positive transport balance"
            ),
        },
        "positive_reference": reference,
        "prospective_observable": {
            "name": "normalized_time_averaged_receiving_band_energy",
            "definition": (
                "A_total = integral_window E_receiving_total(t) dt / "
                "(window_duration * E_initial_source)"
            ),
            "receiving_energy": (
                "E_receiving_total(t) = 1/2 sum_cells "
                "delta_p.T H delta_p Delta_lnR over fixed faces [6,49]"
            ),
            "strictly_nonnegative": True,
            "need_not_be_bounded_by_one": True,
            "reason_not_bounded": (
                "variable-background, descriptor, relaxation and lower-"
                "source work can amplify or attenuate stored packet energy"
            ),
            "normalization": (
                "the positive initial source-band energy computed on the "
                "same grid from the analytic reprojection"
            ),
            "primary_history": (
                "positive receiving-band total-energy history divided by "
                "initial source-band energy"
            ),
            "peak_companion": (
                "maximum receiving-band energy divided by initial source-"
                "band energy"
            ),
        },
        "family_transfer": {
            "definition": (
                "project the state with normalization-invariant, energy-"
                "orthogonal projectors before evaluating the same positive "
                "band energy"
            ),
            "target_indices": {
                family: list(FAMILY_INDICES[family])
                for family in PRIMARY_FAMILIES
            },
            "target_arrival": (
                "sum of arrival energies in the declared target family "
                "indices"
            ),
            "opposite_family_leakage": (
                "sum of arrival energies outside the declared target "
                "family indices"
            ),
            "partition_identity": (
                "target arrival + opposite-family leakage = total arrival"
            ),
        },
        "energy_ledger": {
            "primary_identity": (
                "retain the exact c2b2 descriptor-dual semidiscrete stored-"
                "energy, ten-block and shared-face identity"
            ),
            "local_face_ratio_is_not_reintroduced": True,
            "report_complete_block_work": True,
            "report_stored_energy_change": True,
            "report_time_integration_residual": True,
        },
        "geometry_and_windows": {
            "source_band_faces_N98": list(SOURCE_BAND_FACES),
            "receiving_band_faces_N98": list(RECEIVING_BAND_FACES),
            "upstream_diagnostic_band_faces_N98": list(
                UPSTREAM_DIAGNOSTIC_BAND_FACES
            ),
            "receiving_band_nuisance_faces_N98": [
                list(item) for item in RECEIVING_BAND_NUISANCE_FACES
            ],
            "reference_levels": list(REFERENCE_LEVELS),
            "bands_scale_by_exact_nested_integer_factor": True,
            "arrival_windows": reference[
                "primary_arrival_windows_seconds"
            ],
            "windows_derived_from_c2a3_characteristic_travel_times": True,
            "observed_histories_may_not_move_windows": True,
            "time_sample_counts": list(c2a3.TIME_SAMPLE_COUNTS),
            "time_stride_diagnostics": [1, 2, 4],
            "window_padding_nuisance_factors": list(
                c2a3.WINDOW_PADDING_NUISANCE_FACTORS
            ),
        },
        "future_profile_contract": {
            "binding_families": list(PRIMARY_FAMILIES),
            "amplitude_factors": [0.5, 1.0],
            "signs": [-1, 1],
            "binding_case_count": 12,
            "material_null_control": True,
            "zero_null_control": True,
            "analytic_reprojection_at_every_level": True,
            "state_and_flux_scaling": "linear",
            "energy_scaling": "quadratic",
            "sign_symmetry_required_for_energy": True,
        },
        "uncertainty_contract": {
            "combination": (
                "conservative deterministic sum or directly measured "
                "nuisance envelope; no root-sum-square without demonstrated "
                "independence"
            ),
            "components": [
                "continuum_reference",
                "analytic_projection",
                "invariant_subspace_choice",
                "receiving_band_placement",
                "arrival_window_padding",
                "time_sampling",
                "restart",
                "roundoff",
            ],
            "maximum_reference_uncertainty_to_medium_fine_difference": (
                MAXIMUM_REFERENCE_UNCERTAINTY_TO_FINE_DIFFERENCE
            ),
            "observability_factor": OBSERVABILITY_FACTOR,
            "direction_cosine_binding_rule": (
                "both refinement-error norms must exceed the frozen "
                "uncertainty envelope by the observability factor"
            ),
            "no_slow_impact_threshold": True,
        },
        "future_uniform_gates": {
            "minimum_RMS_order": MINIMUM_OBSERVABLE_ORDER,
            "minimum_maximum_order": MINIMUM_OBSERVABLE_ORDER,
            "minimum_component_order": MINIMUM_OBSERVABLE_ORDER,
            "maximum_fine_normalized_difference": (
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            "minimum_history_cosine": MINIMUM_OBSERVABLE_ERROR_COSINE,
            "minimum_observable_error_cosine": (
                MINIMUM_OBSERVABLE_ERROR_COSINE
            ),
            "maximum_energy_ledger_defect": MAXIMUM_ALGEBRAIC_DEFECT,
            "exact_null_and_amplitude_scaling": True,
            "exact_restart_replay": True,
        },
        "decision_table": {
            "all_uniform_arrival_energy_contracts_pass": (
                "authorize c2c2 one-way embedded arrival-energy "
                "discrimination"
            ),
            "one_uniform_arrival_energy_contract_fails": (
                "freeze the failing profile and authorize only a local "
                "truncation/observable audit; do not redesign"
            ),
            "uncertainty_or_partition_contract_fails": (
                "repair the observable definition before interpretation"
            ),
            "local_face_ratio_only_changes": (
                "nonbinding; the c2b1/c2b2 rejection remains unchanged"
            ),
        },
        "hard_stops": [
            "do_not_relabel_c2b1_or_c2b2",
            "do_not_reintroduce_a_local_face_transmission_ratio",
            "do_not_change_the_operator_or_interface",
            "do_not_propagate_in_c2b3",
            "do_not_run_embedded_or_nonlinear_work",
            "do_not_run_N1024",
            "do_not_begin_fixed_Q_or_reduced_slow_time_evolution",
        ],
        "binding_decision": {
            "c2b1_rejection_preserved": True,
            "c2b2_interpretation_preserved": True,
            "positive_initial_reference_certified": reference["passed"],
            "positive_fixed_band_arrival_contract_frozen": reference["passed"],
            "local_face_transmission_contract_certified": False,
            "uniform_c2b4_authorized": reference["passed"],
            "embedded_c2c1_authorized": False,
            "embedded_c2c2_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": (
            "positive_fixed_band_arrival_energy_contract_frozen_"
            "uniform_validation_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c2b4_one_way_uniform_arrival_energy_validation"
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
    parent, scope, scope_arrays, energy_arrays = _validate_parent()
    reference, decisive = _freeze_positive_reference(
        scope_arrays,
        energy_arrays,
    )
    if not reference["passed"]:
        raise RuntimeError("positive stored-energy reference failed")
    manifest = _transfer_manifest(parent, reference)
    source_hashes = _source_manifest()
    parent_hashes = _input_hashes()

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "source_band_faces_N98": list(SOURCE_BAND_FACES),
        "receiving_band_faces_N98": list(RECEIVING_BAND_FACES),
        "upstream_diagnostic_band_faces_N98": list(
            UPSTREAM_DIAGNOSTIC_BAND_FACES
        ),
        "receiving_band_nuisance_faces_N98": [
            list(item) for item in RECEIVING_BAND_NUISANCE_FACES
        ],
        "reference_levels": list(REFERENCE_LEVELS),
        "primary_families": list(PRIMARY_FAMILIES),
        "family_indices": {
            name: list(indices)
            for name, indices in FAMILY_INDICES.items()
        },
        "gates": manifest["future_uniform_gates"],
        "uncertainty_contract": manifest["uncertainty_contract"],
        "operator_changed": False,
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
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "passed": True,
        "classification": manifest["classification"],
        "authorized_next": manifest["authorized_next"],
        "operator_changed": False,
        "propagation_executed": False,
        "parent_classification": parent["classification"],
        "scope_classification": scope["classification"],
        "positive_reference": reference,
        "binding_decision": manifest["binding_decision"],
        "manifest_sha256": manifest["manifest_sha256"],
        "decisive_array_hashes": decisive_hashes,
        "decisive_arrays_sha256": c2a._sha256(DECISIVE_ARRAYS),
        "config_sha256": c2a._sha256(CONFIG_PATH),
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
            "scripts/run_causal_inner_semidiscrete_energy_transfer_contract_"
            "wp10c9d6c7c2b3.py"
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
