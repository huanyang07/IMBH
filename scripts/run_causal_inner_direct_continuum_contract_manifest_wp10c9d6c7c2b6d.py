#!/usr/bin/env python3
"""Freeze the direct-continuum uniform arrival contract.

WP10c9d6c7c2b6d is definitions-only.  It preserves the b6b rejection and
the b6c no-redesign diagnosis, changes no operator, and propagates no state.
The next validation compares every finite-grid arrival history directly with
independent N769/N513 continuum histories.  Four previously unseen angular
mixtures are frozen before propagation.
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

import run_causal_inner_revised_arrival_contract_manifest_wp10c9d6c7c2b6a as b6a  # noqa: E402
import run_causal_inner_scattering_observability_manifest_wp10c9d6c7c2a as c2a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2b6d"
ANALYZED_BASE_COMMIT = "523100839171bf672319ec2185a5a69e18da1f02"
ANALYZED_BASE_PARENT = "51c584a859b238eb528569179c570e6fd128707c"
ANALYZED_BASE_TREE = "cbd9366e8ebba9ee0a3cdbc64f89e5eac70459f9"

LEVELS = (98, 196, 392)
CONTINUUM_NODES = (513, 769)
SOURCE_BAND_FACES_N98 = b6a.SOURCE_BAND_FACES_N98
RECEIVING_BAND_FACES_N98 = b6a.RECEIVING_BAND_FACES_N98
CALIBRATION_BASES = b6a.BINDING_BASES
HELDOUT_ANGLES_DEGREES = (22.5, 67.5, 112.5, 157.5)
HELDOUT_BASES = tuple(
    f"angle_{str(angle).replace('.', 'p')}_acoustic_shear"
    for angle in HELDOUT_ANGLES_DEGREES
)
BINDING_BASES = CALIBRATION_BASES + HELDOUT_BASES
AMPLITUDE_FACTORS = (0.5, 1.0)
SIGNS = (-1, 1)

MINIMUM_DIRECT_ERROR_ORDER = 0.75
MAXIMUM_FINE_DIRECT_RESPONSE_RELATIVE_RMS = 0.05
MAXIMUM_FINE_DIRECT_RESPONSE_RELATIVE_MAXIMUM = 0.05
MAXIMUM_DIRECT_SCALAR_RESPONSE_RELATIVE_ERROR = 0.05
MAXIMUM_PEAK_TIME_WINDOW_FRACTION = 0.05
MINIMUM_FINE_CONTINUUM_HISTORY_COSINE = 0.90
MAXIMUM_REFERENCE_TO_FINE_DIRECT_ERROR_RATIO = 0.10
MAXIMUM_CONTINUUM_ACTION_DIFFERENCE = 2.0e-5
MAXIMUM_PROJECTOR_ALGEBRA_DEFECT = 2.0e-9
MAXIMUM_EQUIVALENT_LOCAL_PROJECTOR_DEFECT = 2.0e-8
MAXIMUM_TRANSFER_CLOSURE_DEFECT = 2.0e-9
MAXIMUM_SCALING_DEFECT = 1.0e-10
MINIMUM_TIER_I_ORDER = 0.75
MAXIMUM_TIER_I_FINE_DIFFERENCE = 0.05
MINIMUM_TIER_I_HISTORY_COSINE = 0.90
MINIMUM_TIER_I_ERROR_COSINE = 0.90

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_direct_continuum_contract_manifest_"
    "wp10c9d6c7c2b6d.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_direct_continuum_contract_manifest_"
    "wp10c9d6c7c2b6d.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_DIRECT_CONTINUUM_CONTRACT_MANIFEST_"
    "WP10C9D6C7C2B6D_RESULTS_2026-07-30.md"
)

B6C_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_failure_localization_wp10c9d6c7c2b6c"
)
B6B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b"
)
B6A_DIRECTORY = b6a.CANONICAL_DIRECTORY
ENERGY_DIRECTORY = b6a.ENERGY_DIRECTORY
SCOPE_DIRECTORY = b6a.SCOPE_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d"
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


def _sha256(path: Path) -> str:
    return c2a._sha256(path)


def _validate_inputs() -> tuple[dict, dict, dict]:
    b6a_summary = _read_json(B6A_DIRECTORY / "summary.json")
    b6b_summary = _read_json(B6B_DIRECTORY / "summary.json")
    b6c_summary = _read_json(B6C_DIRECTORY / "summary.json")
    if (
        b6a_summary["classification"]
        != "revised_uniform_arrival_transfer_contract_frozen_"
        "recertification_authorized"
        or b6b_summary["classification"]
        != "revised_uniform_arrival_transfer_recertification_failed_"
        "embedded_blocked"
        or b6c_summary["classification"]
        != "direct_continuum_arrival_errors_contract_pairwise_rotation_"
        "preasymptotic_no_redesign"
        or not b6c_summary["passed"]
        or b6c_summary["authorized_next"]
        != "WP10c9d6c7c2b6d_direct_continuum_arrival_contract_manifest"
        or not b6c_summary["binding_decision"][
            "all_failed_channels_contract_directly_to_N769"
        ]
        or b6c_summary["binding_decision"][
            "stable_noncontracting_DAE_mechanism_selected"
        ]
        or b6c_summary["binding_decision"][
            "operator_or_interface_redesign_authorized"
        ]
        or not b6c_summary["binding_decision"][
            "definitions_only_direct_continuum_manifest_authorized"
        ]
    ):
        raise RuntimeError("b6a/b6b/b6c binding status changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2b6d analyzed identity changed")
    return b6a_summary, b6b_summary, b6c_summary


def _freeze_profiles() -> tuple[dict, dict[str, np.ndarray]]:
    old = _load_npz(B6A_DIRECTORY / "decisive_arrays.npz")
    energy = _load_npz(ENERGY_DIRECTORY / "decisive_arrays.npz")
    acoustic = np.asarray(old["packet__acoustic"], dtype=float)
    shear = np.asarray(old["packet__shear"], dtype=float)
    packets = {
        name: np.asarray(old[f"packet__{name}"], dtype=float)
        for name in CALIBRATION_BASES
    }
    coefficients = {}
    for angle, name in zip(
        HELDOUT_ANGLES_DEGREES, HELDOUT_BASES, strict=True
    ):
        radians = np.deg2rad(angle)
        pair = np.asarray([np.cos(radians), np.sin(radians)])
        coefficients[name] = pair
        packets[name] = pair[0] * acoustic + pair[1] * shear

    edges = np.asarray(old["patch_edges_N98"], dtype=float)
    widths = np.diff(np.log(edges))
    metric = np.asarray(energy["primitive_energy_metrics"], dtype=float)
    projectors = np.asarray(
        energy["normalization_invariant_projectors"],
        dtype=float,
    )
    reports = {}
    total_energies = []
    family_energies = []
    decisive = {
        "reference_levels": np.asarray(LEVELS, dtype=np.int64),
        "continuum_nodes": np.asarray(CONTINUUM_NODES, dtype=np.int64),
        "patch_edges_N98": edges,
        "source_band_faces_N98": np.asarray(
            SOURCE_BAND_FACES_N98, dtype=np.int64
        ),
        "receiving_band_faces_N98": np.asarray(
            RECEIVING_BAND_FACES_N98, dtype=np.int64
        ),
        "heldout_angles_degrees": np.asarray(
            HELDOUT_ANGLES_DEGREES, dtype=float
        ),
        "heldout_acoustic_shear_coefficients": np.asarray(
            [coefficients[name] for name in HELDOUT_BASES], dtype=float
        ),
    }
    for name, packet in packets.items():
        total = b6a._band_energy(packet, metric, widths)
        family = b6a._family_energy(
            packet, projectors, metric, widths
        )
        target = (0, 1) if name in HELDOUT_BASES else tuple(
            b6a.TARGET_FAMILY_INDICES[name]
        )
        fraction = float(np.sum(family[list(target)]) / total)
        partition = abs(float(np.sum(family)) - total) / total
        decisive[f"packet__{name}"] = packet
        decisive[f"initial_family_energy__{name}"] = family
        total_energies.append(total)
        family_energies.append(family)
        reports[name] = {
            "role": (
                "prospective_heldout"
                if name in HELDOUT_BASES
                else "historical_calibration"
            ),
            "acoustic_shear_coefficients": (
                coefficients[name].tolist()
                if name in HELDOUT_BASES
                else None
            ),
            "target_family_indices": list(target),
            "initial_source_energy": total,
            "initial_target_family_fraction": fraction,
            "initial_family_partition_relative_defect": partition,
            "packet_sha256": causal_array_sha256(packet),
        }
    variants = np.asarray(
        [
            (BINDING_BASES.index(name), amplitude, sign)
            for name in BINDING_BASES
            for amplitude in AMPLITUDE_FACTORS
            for sign in SIGNS
        ],
        dtype=float,
    )
    decisive["binding_variant_table"] = variants
    decisive["initial_total_energy"] = np.asarray(total_energies)
    decisive["initial_family_energy"] = np.asarray(family_energies)
    maximum_partition = max(
        item["initial_family_partition_relative_defect"]
        for item in reports.values()
    )
    minimum_fraction = min(
        item["initial_target_family_fraction"]
        for item in reports.values()
    )
    passed = bool(
        len(packets) == 9
        and len(HELDOUT_BASES) == 4
        and variants.shape == (36, 3)
        and min(total_energies) > np.finfo(float).tiny
        and maximum_partition <= MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
        and minimum_fraction >= 1.0 - 1.0e-10
    )
    return (
        {
            "passed": passed,
            "binding_base_count": len(BINDING_BASES),
            "binding_variant_count": int(variants.shape[0]),
            "historical_calibration_bases": list(CALIBRATION_BASES),
            "prospective_heldout_bases": list(HELDOUT_BASES),
            "heldout_angles_degrees": list(HELDOUT_ANGLES_DEGREES),
            "amplitude_factors": list(AMPLITUDE_FACTORS),
            "signs": list(SIGNS),
            "maximum_initial_family_partition_relative_defect": (
                maximum_partition
            ),
            "minimum_initial_target_family_fraction": minimum_fraction,
            "heldouts_frozen_before_propagation": True,
            "amplitude_and_sign_variants_are_controls": True,
            "per_profile": reports,
        },
        decisive,
    )


def _contract_manifest(
    profiles: dict,
    b6a_summary: dict,
    b6b_summary: dict,
    b6c_summary: dict,
) -> dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "historical_classifications_preserved": {
            "WP10c9d6c7c2b6a": b6a_summary["classification"],
            "WP10c9d6c7c2b6b": b6b_summary["classification"],
            "WP10c9d6c7c2b6c": b6c_summary["classification"],
        },
        "scientific_interpretation": {
            "b6b_reclassified": False,
            "uniform_operator_rejected": False,
            "numerical_redesign_selected": False,
            "pairwise_error_direction_is_a_continuum_accuracy_gate": False,
            "reason": (
                "b6c showed that every frozen b6b failure contracts "
                "directly toward N769 and that no stable noncontracting "
                "DAE block is selected; therefore a prospective direct-"
                "continuum contract is tested on new unseen profiles"
            ),
        },
        "profile_manifest": profiles,
        "certification_tiers": {
            "tier_I_primary_physics": {
                "binding": True,
                "contract": "unchanged_pairwise_state_and_13_export_contract",
                "minimum_order": MINIMUM_TIER_I_ORDER,
                "maximum_fine_normalized_difference": (
                    MAXIMUM_TIER_I_FINE_DIFFERENCE
                ),
                "minimum_history_cosine": MINIMUM_TIER_I_HISTORY_COSINE,
                "minimum_refinement_error_cosine": (
                    MINIMUM_TIER_I_ERROR_COSINE
                ),
            },
            "tier_II_direct_continuum_arrival": {
                "binding": True,
                "observables": [
                    "positive_total_receiving_band_energy_gain",
                    "projector_qualified_target_family_energy_gain",
                    "unit_shape_history",
                    "time_average",
                    "peak_amplitude",
                    "peak_time",
                ],
            },
            "tier_II_covariant_transfer": {
                "binding": True,
                "contract": "unchanged_exact_block_source_receiver_transfer",
                "maximum_closure_defect": MAXIMUM_TRANSFER_CLOSURE_DEFECT,
            },
            "noncertifying_diagnostics": [
                "pairwise_refinement_error_direction_for_arrival_histories",
                "raw_local_opposite_family_stored_energy",
                "individual_block_dominance",
                "pointwise_interface_stress",
            ],
            "tier_III_nonlinear": {
                "binding": False,
                "authorized": False,
            },
        },
        "direct_continuum_contract": {
            "primary_reference": "independent_N769_continuum_history",
            "secondary_reference": "independent_N513_continuum_history",
            "finite_levels": list(LEVELS),
            "history_error": (
                "e_h(t)=G_h(t)-G_N769(t), evaluated on the same frozen "
                "travel-time samples and physical window"
            ),
            "response_scale": "max(max_t(abs(G_N769)),1)",
            "direct_error_orders": [
                "log2(norm(e_N98)/norm(e_N196))",
                "log2(norm(e_N196)/norm(e_N392))",
            ],
            "minimum_weighted_RMS_error_order": (
                MINIMUM_DIRECT_ERROR_ORDER
            ),
            "minimum_maximum_error_order": MINIMUM_DIRECT_ERROR_ORDER,
            "maximum_N392_response_relative_RMS_error": (
                MAXIMUM_FINE_DIRECT_RESPONSE_RELATIVE_RMS
            ),
            "maximum_N392_response_relative_maximum_error": (
                MAXIMUM_FINE_DIRECT_RESPONSE_RELATIVE_MAXIMUM
            ),
            "minimum_N392_continuum_history_cosine": (
                MINIMUM_FINE_CONTINUUM_HISTORY_COSINE
            ),
            "maximum_N392_peak_response_relative_error": (
                MAXIMUM_DIRECT_SCALAR_RESPONSE_RELATIVE_ERROR
            ),
            "maximum_N392_time_average_response_relative_error": (
                MAXIMUM_DIRECT_SCALAR_RESPONSE_RELATIVE_ERROR
            ),
            "maximum_N392_peak_time_error_fraction_of_window": (
                MAXIMUM_PEAK_TIME_WINDOW_FRACTION
            ),
            "pairwise_error_direction_reported_not_binding": True,
            "no_grid_trajectory_called_exact": True,
        },
        "continuum_reference_contract": {
            "maximum_N769_N513_history_difference_to_N392_direct_error": (
                MAXIMUM_REFERENCE_TO_FINE_DIRECT_ERROR_RATIO
            ),
            "maximum_continuum_action_difference": (
                MAXIMUM_CONTINUUM_ACTION_DIFFERENCE
            ),
            "stop_if_either_reference_is_unavailable": True,
            "reference_must_use_complete_DAE_generator_blocks": True,
            "reference_reuse_requires_source_and_configuration_hash_match": (
                True
            ),
        },
        "projector_and_transfer_contract": {
            "physical_projectors": (
                "local descriptor-compatible energy-orthogonal invariant "
                "subspaces"
            ),
            "equivalent_projector_algorithms": [
                "overlap_tracked_generalized_eigenvector",
                "local_polynomial_spectral_projector",
            ],
            "maximum_projector_algebra_defect": (
                MAXIMUM_PROJECTOR_ALGEBRA_DEFECT
            ),
            "maximum_equivalent_local_projector_difference": (
                MAXIMUM_EQUIVALENT_LOCAL_PROJECTOR_DEFECT
            ),
            "maximum_covariant_transfer_closure_defect": (
                MAXIMUM_TRANSFER_CLOSURE_DEFECT
            ),
            "hard_stop_on_unresolved_spectral_cluster": True,
        },
        "uncertainty_contract": {
            "combination": (
                "conservative deterministic sum or measured nuisance "
                "envelope; no RSS without demonstrated independence"
            ),
            "components": [
                "N769_N513_continuum_reference",
                "finite_volume_projection",
                "equivalent_projector_algorithm",
                "receiving_band_placement",
                "arrival_window_placement",
                "time_sampling_and_quadrature",
                "restart",
                "roundoff",
            ],
            "maximum_reference_to_N392_direct_error_ratio": (
                MAXIMUM_REFERENCE_TO_FINE_DIRECT_ERROR_RATIO
            ),
            "no_slow_impact_threshold": True,
        },
        "geometry_and_windows": {
            "source_band_faces_N98": list(SOURCE_BAND_FACES_N98),
            "receiving_band_faces_N98": list(RECEIVING_BAND_FACES_N98),
            "reuse_exact_c2b3_travel_time_windows": True,
            "windows_may_not_move_from_observed_histories": True,
            "time_stride_diagnostics": [1, 2, 4],
            "terminal_tail_contract_retained": True,
        },
        "decision_table": {
            "all_calibration_and_unseen_profiles_pass": (
                "certify the declared uniform direct-continuum class and "
                "authorize a definitions-only embedded manifest"
            ),
            "calibration_passes_but_unseen_profile_fails": (
                "freeze the exact unseen failure and run one local DAE/"
                "observable audit; do not tune gates or profiles"
            ),
            "calibration_profile_fails": (
                "preserve the failure and re-audit the direct reference; "
                "no embedded work"
            ),
            "continuum_reference_fails": (
                "repair or strengthen the independent reference before "
                "interpreting finite-grid results"
            ),
            "stable_noncontracting_DAE_block_is_selected": (
                "authorize only that local block audit"
            ),
            "only_pairwise_error_direction_rotates": (
                "report it as pre-asymptotic diagnostics; do not reject a "
                "passing direct-continuum contract"
            ),
        },
        "hard_stops": [
            "do_not_relabel_b6b_or_b6c",
            "do_not_tune_the_four_unseen_profiles_after_propagation",
            "do_not_change_the_operator_or_interface",
            "do_not_propagate_in_b6d",
            "do_not_run_embedded_before_uniform_b6e_passes",
            "do_not_run_nonlinear_or_N1024",
            "do_not_begin_fixed_Q_or_reduced_slow_time_evolution",
        ],
        "binding_decision": {
            "historical_classifications_preserved": True,
            "profile_manifest_certified": profiles["passed"],
            "direct_continuum_contract_frozen": profiles["passed"],
            "uniform_b6e_recertification_authorized": profiles["passed"],
            "operator_or_interface_redesign_authorized": False,
            "embedded_authorized": False,
            "nonlinear_authorized": False,
            "fixed_Q_or_reduction_authorized": False,
        },
        "classification": (
            "direct_continuum_arrival_contract_frozen_uniform_"
            "recertification_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c2b6e_direct_continuum_uniform_recertification"
        ),
    }
    manifest["manifest_sha256"] = causal_canonical_json_sha256(manifest)
    return manifest


def _input_hashes() -> dict[str, str]:
    paths = (
        B6A_DIRECTORY / "summary.json",
        B6A_DIRECTORY / "contract_manifest.json",
        B6A_DIRECTORY / "decisive_arrays.npz",
        B6B_DIRECTORY / "summary.json",
        B6B_DIRECTORY / "config.json",
        B6B_DIRECTORY / "decisive_arrays.npz",
        B6C_DIRECTORY / "summary.json",
        B6C_DIRECTORY / "config.json",
        B6C_DIRECTORY / "decisive_arrays.npz",
        ENERGY_DIRECTORY / "method_manifest.json",
        ENERGY_DIRECTORY / "decisive_arrays.npz",
        SCOPE_DIRECTORY / "scope_manifest.json",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def _refresh_catalog() -> None:
    rows = []
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
    summary = _read_json(CANONICAL_SUMMARY)
    summary.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _write_report(summary: dict) -> None:
    profiles = summary["profile_manifest"]
    lines = [
        "# Causal inner direct-continuum arrival contract manifest "
        "WP10c9d6c7c2b6d",
        "",
        "## Result",
        "",
        "This definitions-only package passes. It changes no operator and "
        "propagates no state.",
        "",
        "The b6b rejection and b6c no-redesign diagnosis are preserved. "
        "The next uniform experiment will compare N98/N196/N392 directly "
        "with independent N769/N513 continuum histories rather than making "
        "the rotating pairwise error direction a convergence gate.",
        "",
        "## Frozen prospective profiles",
        "",
        (
            f"The manifest contains `{profiles['binding_base_count']}` base "
            f"profiles and `{profiles['binding_variant_count']}` exact "
            "sign/amplitude variants."
        ),
        "",
        "| New held-out | Acoustic coefficient | Shear coefficient |",
        "|---|---:|---:|",
    ]
    for name in HELDOUT_BASES:
        item = profiles["per_profile"][name]
        acoustic, shear = item["acoustic_shear_coefficients"]
        lines.append(
            f"| `{name}` | `{acoustic:.8f}` | `{shear:.8f}` |"
        )
    lines.extend(
        [
            "",
            "## Binding direct-continuum gates",
            "",
            "- Direct weighted-RMS and maximum error orders must be at "
            "least `0.75` on both pairs.",
            "- N392 direct RMS and maximum errors must each be at most "
            "`0.05` of the N769 response scale.",
            "- Peak and time-average errors must be at most `0.05`; peak "
            "time must agree within `0.05` of the frozen window duration.",
            "- N769/N513 reference uncertainty must be at most `0.10` of "
            "the N392 direct error.",
            "- Tier I, projector algebra, covariant transfer, ledgers, "
            "scaling, restart, and tail gates remain binding.",
            "",
            "Pairwise arrival-error direction remains reported but is not "
            "binding. No b6b result is retroactively passed.",
            "",
            "## Decision",
            "",
            "Only a complete pass of all five calibration profiles and all "
            "four unseen held-outs may certify the uniform class and "
            "authorize a definitions-only embedded manifest.",
            "",
            "Embedded, nonlinear, fixed-Q, and reduced slow-time evolution "
            "remain blocked.",
            "",
            "Classification:",
            "",
            f"`{summary['classification']}`",
            "",
            "Authorized next:",
            "",
            f"`{summary['authorized_next']}`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    b6a_summary, b6b_summary, b6c_summary = _validate_inputs()
    profiles, decisive = _freeze_profiles()
    if not profiles["passed"]:
        raise RuntimeError("prospective profile manifest failed")
    manifest = _contract_manifest(
        profiles, b6a_summary, b6b_summary, b6c_summary
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "propagation_executed": False,
        "reference_levels": list(LEVELS),
        "continuum_nodes": list(CONTINUUM_NODES),
        "calibration_bases": list(CALIBRATION_BASES),
        "prospective_heldout_bases": list(HELDOUT_BASES),
        "heldout_angles_degrees": list(HELDOUT_ANGLES_DEGREES),
        "amplitude_factors": list(AMPLITUDE_FACTORS),
        "signs": list(SIGNS),
        "direct_continuum_gates": manifest[
            "direct_continuum_contract"
        ],
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).is_file()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "embedded_or_nonlinear_propagation_executed": False,
        "historical_classifications_preserved": manifest[
            "historical_classifications_preserved"
        ],
        "profile_manifest": profiles,
        "binding_decision": manifest["binding_decision"],
        "manifest_sha256": manifest["manifest_sha256"],
        "classification": manifest["classification"],
        "authorized_next": manifest["authorized_next"],
        "passed": True,
        "config_sha256": _sha256(CONFIG_PATH),
        "contract_manifest_file_sha256": _sha256(MANIFEST_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_hashes)
        ),
        "input_hashes": _input_hashes(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "DEFINITIONS CERTIFIED",
        "classification": summary["classification"],
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "source_parent_tree": ANALYZED_BASE_TREE,
        "implementation_worktree_head": _git_value("rev-parse", "HEAD"),
        "implementation_source_hashes": source_hashes,
        "input_hashes": _input_hashes(),
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_direct_continuum_contract_manifest_"
            "wp10c9d6c7c2b6d.py"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    _write_json(PROVENANCE_PATH, provenance)
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    return summary


def main() -> None:
    summary = run()
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": summary["classification"],
                "passed": summary["passed"],
                "authorized_next": summary["authorized_next"],
                "binding_decision": summary["binding_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
