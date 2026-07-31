#!/usr/bin/env python3
"""Freeze the revised uniform arrival/transfer recertification contract.

WP10c9d6c7c2b6a is definitions-only.  It preserves the c2b4 rejection and
the c2b5a/c2b5b diagnostic classifications, changes no operator, and
propagates no state.  The package replaces the ill-conditioned absolute
arrival-history accuracy test and the raw local opposite-family leakage gate
with response-relative histories, projector-qualified target energy, and the
exact covariant family-transfer ledger certified in c2b5b.
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
WORK_PACKAGE = "WP10c9d6c7c2b6a"
ANALYZED_BASE_COMMIT = "7857051b4292c3101456019210f7437c73fd8621"
ANALYZED_BASE_PARENT = "dc27efe1b414143cec67a050f8dc5c9ccff69ee4"
ANALYZED_BASE_TREE = "7926e567e4ba325bb9aa47541c0ca808ab23f3ec"

LEVELS = (98, 196, 392)
SOURCE_BAND_FACES_N98 = (52, 95)
RECEIVING_BAND_FACES_N98 = (6, 49)
PRIMARY_FAMILIES = ("acoustic", "shear", "mixed_shear_acoustic")
CALIBRATION_BASES = PRIMARY_FAMILIES
HELDOUT_BASES = (
    "difference_shear_acoustic",
    "shear_weighted_shear_acoustic",
)
BINDING_BASES = CALIBRATION_BASES + HELDOUT_BASES
AMPLITUDE_FACTORS = (0.5, 1.0)
SIGNS = (-1, 1)
TARGET_FAMILY_INDICES = {
    "acoustic": (0,),
    "shear": (1,),
    "mixed_shear_acoustic": (0, 1),
    "difference_shear_acoustic": (0, 1),
    "shear_weighted_shear_acoustic": (0, 1),
}

MINIMUM_ORDER = 0.75
MAXIMUM_FINE_RESPONSE_RELATIVE_DIFFERENCE = 0.05
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_OBSERVABLE_ERROR_COSINE = 0.90
MAXIMUM_CONTINUUM_HISTORY_TO_FINE_DIFFERENCE = 0.10
MAXIMUM_PROJECTOR_ALGEBRA_DEFECT = 2.0e-9
MAXIMUM_EQUIVALENT_LOCAL_PROJECTOR_DEFECT = 2.0e-8
MAXIMUM_TRANSFER_CLOSURE_DEFECT = 2.0e-9
MAXIMUM_CONTINUUM_ACTION_DIFFERENCE = 2.0e-5
MAXIMUM_SCALING_DEFECT = 1.0e-10
OBSERVABILITY_FACTOR = 5.0

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_revised_arrival_contract_manifest_"
    "wp10c9d6c7c2b6a.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_revised_arrival_contract_manifest_"
    "wp10c9d6c7c2b6a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_REVISED_ARRIVAL_CONTRACT_MANIFEST_"
    "WP10C9D6C7C2B6A_RESULTS_2026-07-30.md"
)

B4_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_arrival_energy_wp10c9d6c7c2b4"
)
B5A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a"
)
B5B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b"
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
C2B3_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_semidiscrete_energy_transfer_contract_wp10c9d6c7c2b3"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_revised_arrival_contract_manifest_wp10c9d6c7c2b6a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "contract_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (THIS_RUNNER, THIS_TEST)


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


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


def _sha256(path: Path) -> str:
    return c2a._sha256(path)


def _source_manifest() -> dict[str, str]:
    return {
        relative: _sha256(ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
    }


def _input_hashes() -> dict[str, str]:
    paths = (
        B4_DIRECTORY / "summary.json",
        B5A_DIRECTORY / "summary.json",
        B5B_DIRECTORY / "summary.json",
        B5B_DIRECTORY / "config.json",
        B5B_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
        SCOPE_DIRECTORY / "decisive_arrays.npz",
        ENERGY_DIRECTORY / "method_manifest.json",
        ENERGY_DIRECTORY / "decisive_arrays.npz",
        C2B3_DIRECTORY / "transfer_manifest.json",
        C2B3_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _validate_predecessors() -> tuple[dict, dict, dict]:
    b4 = _read_json(B4_DIRECTORY / "summary.json")
    b5a = _read_json(B5A_DIRECTORY / "summary.json")
    b5b = _read_json(B5B_DIRECTORY / "summary.json")
    if (
        b4["classification"]
        != "one_way_uniform_arrival_energy_validation_failed_"
        "embedded_discrimination_blocked"
        or b5a["classification"]
        != "arrival_history_conditioning_and_horizon_audit_complete_"
        "shear_family_transfer_audit_required"
        or b5b["classification"]
        != "raw_local_family_leakage_projector_rotation_sensitive_"
        "revised_transfer_observable_manifest_authorized"
        or b5b["authorized_next"]
        != "WP10c9d6c7c2b6a_revised_uniform_arrival_contract_manifest"
        or not b5b["binding_decision"]["revised_uniform_manifest_authorized"]
        or b5b["binding_decision"][
            "uniform_recertification_propagation_authorized"
        ]
        or b5b["binding_decision"]["embedded_authorized"]
        or b5b["binding_decision"][
            "operator_or_interface_redesign_authorized"
        ]
    ):
        raise RuntimeError("c2b4/c2b5a/c2b5b binding status changed")
    return b4, b5a, b5b


def _band_energy(
    state: np.ndarray,
    metric: np.ndarray,
    log_widths: np.ndarray,
) -> float:
    lower, upper = SOURCE_BAND_FACES_N98
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


def _family_energy(
    state: np.ndarray,
    projectors: np.ndarray,
    metric: np.ndarray,
    log_widths: np.ndarray,
) -> np.ndarray:
    projected = np.einsum("nfij,nj->nfi", projectors, state, optimize=True)
    lower, upper = SOURCE_BAND_FACES_N98
    return 0.5 * np.einsum(
        "nfi,nij,nfj,n->f",
        projected[lower:upper],
        metric[lower:upper],
        projected[lower:upper],
        log_widths[lower:upper],
        optimize=True,
    )


def _freeze_profiles() -> tuple[dict, dict[str, np.ndarray]]:
    scope = _load_npz(SCOPE_DIRECTORY / "decisive_arrays.npz")
    energy = _load_npz(ENERGY_DIRECTORY / "decisive_arrays.npz")
    acoustic = np.asarray(scope["packet__acoustic"], dtype=float)
    shear = np.asarray(scope["packet__shear"], dtype=float)
    mixed = np.asarray(scope["packet__mixed_shear_acoustic"], dtype=float)
    difference = (acoustic - shear) / np.sqrt(2.0)
    shear_weighted = 0.5 * acoustic + np.sqrt(3.0) * 0.5 * shear
    packets = {
        "acoustic": acoustic,
        "shear": shear,
        "mixed_shear_acoustic": mixed,
        "difference_shear_acoustic": difference,
        "shear_weighted_shear_acoustic": shear_weighted,
    }
    edges = np.asarray(scope["patch_edges"], dtype=float)
    log_widths = np.diff(np.log(edges))
    metric = np.asarray(energy["primitive_energy_metrics"], dtype=float)
    projectors = np.asarray(
        energy["normalization_invariant_projectors"],
        dtype=float,
    )

    initial_total = []
    initial_family = []
    reports = {}
    decisive: dict[str, np.ndarray] = {
        "reference_levels": np.asarray(LEVELS, dtype=np.int64),
        "patch_edges_N98": edges,
        "source_band_faces_N98": np.asarray(
            SOURCE_BAND_FACES_N98,
            dtype=np.int64,
        ),
        "receiving_band_faces_N98": np.asarray(
            RECEIVING_BAND_FACES_N98,
            dtype=np.int64,
        ),
    }
    for name, packet in packets.items():
        total = _band_energy(packet, metric, log_widths)
        family = _family_energy(
            packet,
            projectors,
            metric,
            log_widths,
        )
        target = TARGET_FAMILY_INDICES[name]
        target_fraction = float(np.sum(family[list(target)]) / total)
        partition = abs(float(np.sum(family)) - total) / total
        decisive[f"packet__{name}"] = packet
        decisive[f"initial_family_energy__{name}"] = family
        initial_total.append(total)
        initial_family.append(family)
        reports[name] = {
            "role": (
                "calibration"
                if name in CALIBRATION_BASES
                else "prospective_heldout"
            ),
            "target_family_indices": list(target),
            "initial_source_energy": total,
            "initial_target_family_fraction": target_fraction,
            "initial_family_partition_relative_defect": partition,
            "packet_sha256": causal_array_sha256(packet),
        }

    variant_rows = []
    for name in BINDING_BASES:
        for amplitude in AMPLITUDE_FACTORS:
            for sign in SIGNS:
                variant_rows.append(
                    (
                        BINDING_BASES.index(name),
                        float(amplitude),
                        int(sign),
                    )
                )
    decisive["binding_variant_table"] = np.asarray(
        variant_rows,
        dtype=float,
    )
    decisive["initial_total_energy"] = np.asarray(initial_total, dtype=float)
    decisive["initial_family_energy"] = np.asarray(
        initial_family,
        dtype=float,
    )
    maximum_partition = max(
        item["initial_family_partition_relative_defect"]
        for item in reports.values()
    )
    minimum_target_fraction = min(
        item["initial_target_family_fraction"] for item in reports.values()
    )
    passed = bool(
        min(initial_total) > np.finfo(float).tiny
        and maximum_partition <= MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
        and minimum_target_fraction >= 1.0 - 1.0e-10
        and len(variant_rows) == 20
    )
    report = {
        "passed": passed,
        "binding_base_count": len(BINDING_BASES),
        "binding_variant_count": len(variant_rows),
        "calibration_bases": list(CALIBRATION_BASES),
        "prospective_heldout_bases": list(HELDOUT_BASES),
        "amplitude_factors": list(AMPLITUDE_FACTORS),
        "signs": list(SIGNS),
        "maximum_initial_family_partition_relative_defect": (
            maximum_partition
        ),
        "minimum_initial_target_family_fraction": minimum_target_fraction,
        "per_profile": reports,
        "heldouts_frozen_before_recertification_propagation": True,
        "amplitude_and_sign_variants_are_controls_not_independent_profiles": (
            True
        ),
    }
    return report, decisive


def _contract_manifest(
    profile_manifest: dict,
    b4: dict,
    b5a: dict,
    b5b: dict,
) -> dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_classifications_preserved": {
            "WP10c9d6c7c2b4": b4["classification"],
            "WP10c9d6c7c2b5a": b5a["classification"],
            "WP10c9d6c7c2b5b": b5b["classification"],
        },
        "scientific_interpretation": {
            "uniform_operator_rejected": False,
            "positive_energy_rejected": False,
            "tier_I_direct_physics_rejected": False,
            "old_absolute_arrival_history_contract_rejected": True,
            "raw_local_opposite_family_leakage_certifying": False,
            "reason": (
                "c2b5a identified an initial-energy absolute normalization "
                "that becomes ill-conditioned at large physical gain; c2b5b "
                "showed that raw local family leakage mixes spatial "
                "projector rotation while all local DAE blocks converge"
            ),
        },
        "profile_manifest": profile_manifest,
        "certification_tiers": {
            "tier_I_primary_physics": {
                "binding": True,
                "observables": [
                    "five_field_state_history",
                    "inner_M_J_E_export",
                    "coupling_M_J_E_export",
                    "net_M_J_E_drive",
                    "cooling",
                    "responsive_height_work",
                    "exact_conservative_ledgers",
                ],
                "contract": "unchanged_from_c2b4",
            },
            "tier_II_arrival_and_transfer": {
                "binding": True,
                "observables": [
                    "positive_total_receiving_band_energy",
                    "projector_qualified_target_family_energy",
                    "total_covariant_transfer_work",
                    "target_receiver_covariant_transfer_work",
                    "exact_block_source_receiver_transfer_closure",
                    "stored_energy_and_physical_work_balance",
                ],
            },
            "tier_II_diagnostics_noncertifying_alone": [
                "raw_local_opposite_family_stored_energy",
                "individual_block_dominance",
                "frozen_receiving_band_midpoint_family_energy",
                "pointwise_interface_stress",
            ],
            "tier_III_nonlinear": {
                "binding": False,
                "authorized": False,
            },
        },
        "arrival_history_definition": {
            "reported_physical_gain": (
                "G_h(t)=E_receiving,h(t)/E_initial_source,h"
            ),
            "physical_gain_may_exceed_one": True,
            "binding_accuracy_normalization": (
                "response-relative, using max(max_t(abs(G_reference)),1)"
            ),
            "reference": {
                "primary": (
                    "independent 769-node continuum/collocation trajectory"
                ),
                "secondary": (
                    "independent 513-node continuum/collocation trajectory"
                ),
                "maximum_primary_secondary_difference_to_medium_fine": (
                    MAXIMUM_CONTINUUM_HISTORY_TO_FINE_DIFFERENCE
                ),
                "no_grid_trajectory_may_be_called_exact": True,
                "stop_if_independent_history_reference_unavailable": True,
            },
            "amplitude_shape_separation": {
                "amplitude": (
                    "maximum and time-averaged response over the frozen "
                    "travel-time window"
                ),
                "unit_shape": (
                    "G_h(t)/max_t(abs(G_h(t))) for an observable response"
                ),
                "peak_time_reported_separately": True,
            },
            "old_initial_energy_absolute_0p05_gate_reused": False,
            "c2b4_result_reclassified": False,
        },
        "projector_contract": {
            "physical_definition": (
                "local descriptor-compatible energy-orthogonal invariant "
                "subspaces on the frozen background"
            ),
            "required_algorithms": [
                "overlap_tracked_local_generalized_eigenvector_projector",
                "local_polynomial_spectral_projector",
            ],
            "maximum_projector_algebra_defect": (
                MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            ),
            "maximum_equivalent_local_projector_difference": (
                MAXIMUM_EQUIVALENT_LOCAL_PROJECTOR_DEFECT
            ),
            "hard_stop_on_unresolved_spectral_cluster": True,
            "common_high_resolution_local_field": (
                "cross_grid_definition_sensitivity_diagnostic"
            ),
            "frozen_midpoint_projector": (
                "rotation_diagnostic_only_not_equivalent_uncertainty"
            ),
            "raw_opposite_family_leakage_is_noncertifying": True,
        },
        "covariant_transfer_contract": {
            "family_rate": (
                "dE_f/dt = 0.5 u.T (G.T Q_f + Q_f G) u"
            ),
            "block_source_receiver_tensor": (
                "T_brs=<P_r u,H G_b P_s u> with the certified exact "
                "descriptor-reduced ten-block generator"
            ),
            "binding_integrals": [
                "total_receiver_work",
                "target_family_receiver_work",
            ],
            "reported_nonbinding_integrals": [
                "opposite_family_receiver_work",
                "individual_block_source_receiver_work",
            ],
            "maximum_family_partition_power_and_block_closure_defect": (
                MAXIMUM_TRANSFER_CLOSURE_DEFECT
            ),
            "one_large_block_is_not_causal_evidence": True,
            "no_block_specific_intervention_without_two_pair_truncation": True,
        },
        "uncertainty_contract": {
            "combination": (
                "conservative deterministic sum or directly measured "
                "nuisance envelope; root-sum-square is forbidden unless "
                "independence or measured covariance is demonstrated"
            ),
            "components": [
                "independent_continuum_history_reference",
                "analytic_finite_volume_projection",
                "equivalent_local_projector_algorithm",
                "receiving_band_placement",
                "arrival_window_placement",
                "time_sampling_and_quadrature",
                "restart",
                "roundoff",
            ],
            "observability_factor": OBSERVABILITY_FACTOR,
            "error_direction_rule": (
                "both refinement-error norms must exceed their complete "
                "conservative envelopes by the observability factor; "
                "otherwise direction is non-certifying, not pass or fail"
            ),
            "maximum_reference_uncertainty_to_medium_fine_difference": (
                MAXIMUM_CONTINUUM_HISTORY_TO_FINE_DIFFERENCE
            ),
            "no_slow_impact_threshold": True,
        },
        "uniform_gates": {
            "minimum_RMS_order": MINIMUM_ORDER,
            "minimum_maximum_order": MINIMUM_ORDER,
            "minimum_component_order": MINIMUM_ORDER,
            "maximum_fine_response_relative_difference": (
                MAXIMUM_FINE_RESPONSE_RELATIVE_DIFFERENCE
            ),
            "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
            "minimum_observable_refinement_error_cosine": (
                MINIMUM_OBSERVABLE_ERROR_COSINE
            ),
            "maximum_continuum_history_to_medium_fine_difference": (
                MAXIMUM_CONTINUUM_HISTORY_TO_FINE_DIFFERENCE
            ),
            "maximum_continuum_action_difference": (
                MAXIMUM_CONTINUUM_ACTION_DIFFERENCE
            ),
            "minimum_continuum_action_truncation_order": MINIMUM_ORDER,
            "maximum_scaling_and_sign_symmetry_defect": (
                MAXIMUM_SCALING_DEFECT
            ),
            "maximum_transfer_closure_defect": (
                MAXIMUM_TRANSFER_CLOSURE_DEFECT
            ),
            "exact_restart_replay": True,
            "exact_conservative_ledgers": True,
        },
        "geometry_and_windows": {
            "reference_levels": list(LEVELS),
            "source_band_faces_N98": list(SOURCE_BAND_FACES_N98),
            "receiving_band_faces_N98": list(RECEIVING_BAND_FACES_N98),
            "reuse_exact_c2b3_travel_time_windows": True,
            "windows_may_not_move_from_observed_histories": True,
            "time_stride_diagnostics": [1, 2, 4],
            "c2b5a_terminal_tail_contract_retained": True,
        },
        "decision_table": {
            "all_tier_I_and_binding_tier_II_and_heldouts_pass": (
                "certify revised uniform arrival/transfer class and "
                "authorize a definitions-only embedded manifest"
            ),
            "raw_local_leakage_fails_but_binding_transfer_passes": (
                "report raw leakage as projector-rotation diagnostic; it "
                "does not reject the revised uniform class"
            ),
            "projector_or_transfer_identity_fails": (
                "repair the observable definition; no embedded work"
            ),
            "independent_continuum_history_unavailable_or_fails": (
                "stop uniform recertification; no embedded interpretation"
            ),
            "binding_calibration_or_heldout_profile_fails": (
                "freeze the exact failure and authorize local DAE/"
                "observable audit only; do not tune the contract"
            ),
            "stable_two_pair_block_truncation_failure": (
                "authorize only the selected local block audit"
            ),
        },
        "hard_stops": [
            "do_not_relabel_c2b4_c2b5a_or_c2b5b",
            "do_not_use_raw_local_leakage_as_a_standalone_gate",
            "do_not_call_the_frozen_midpoint_projector_physical",
            "do_not_change_the_operator_or_interface",
            "do_not_propagate_in_c2b6a",
            "do_not_run_embedded_or_nonlinear_work",
            "do_not_run_N1024",
            "do_not_begin_fixed_Q_or_reduced_slow_time_evolution",
        ],
        "binding_decision": {
            "historical_classifications_preserved": True,
            "profile_manifest_certified": profile_manifest["passed"],
            "revised_uniform_contract_frozen": profile_manifest["passed"],
            "uniform_b6b_recertification_authorized": (
                profile_manifest["passed"]
            ),
            "raw_local_family_leakage_certifying": False,
            "embedded_authorized": False,
            "operator_or_interface_redesign_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": (
            "revised_uniform_arrival_transfer_contract_frozen_"
            "recertification_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c2b6b_revised_uniform_arrival_transfer_"
            "recertification"
        ),
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(manifest)
    return manifest


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
            if path.is_file():
                rows.append(
                    {
                        "case": case.name,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "scientific_status": status,
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
    canonical_summary = _read_json(CANONICAL_SUMMARY)
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
    _write_json(CANONICAL_SUMMARY, canonical_summary)


def run() -> dict:
    start = time.perf_counter()
    git_identity = _validate_analyzed_git_identity()
    b4, b5a, b5b = _validate_predecessors()
    profile_manifest, decisive = _freeze_profiles()
    if not profile_manifest["passed"]:
        raise RuntimeError("prospective profile manifest failed")
    manifest = _contract_manifest(profile_manifest, b4, b5a, b5b)
    source_hashes = _source_manifest()
    input_hashes = _input_hashes()

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "reference_levels": list(LEVELS),
        "binding_bases": list(BINDING_BASES),
        "calibration_bases": list(CALIBRATION_BASES),
        "prospective_heldout_bases": list(HELDOUT_BASES),
        "amplitude_factors": list(AMPLITUDE_FACTORS),
        "signs": list(SIGNS),
        "target_family_indices": {
            name: list(indices)
            for name, indices in TARGET_FAMILY_INDICES.items()
        },
        "uniform_gates": manifest["uniform_gates"],
        "uncertainty_contract": manifest["uncertainty_contract"],
        "projector_contract": manifest["projector_contract"],
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
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
        "historical_classifications_preserved": (
            manifest["historical_classifications_preserved"]
        ),
        "profile_manifest": profile_manifest,
        "binding_decision": manifest["binding_decision"],
        "manifest_sha256": manifest["manifest_sha256"],
        "decisive_array_hashes": decisive_hashes,
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "config_sha256": _sha256(CONFIG_PATH),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_hashes)
        ),
        "input_hashes": input_hashes,
        "runtime_seconds": time.perf_counter() - start,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED",
        "classification": manifest["classification"],
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **git_identity,
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_revised_arrival_contract_manifest_"
            "wp10c9d6c7c2b6a.py"
        ),
        "implementation_source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run()
