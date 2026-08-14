#!/usr/bin/env python3
"""Freeze the leading-two plus HMM/guard-complement architecture.

Definitions and analysis only.  No tangent/nonlinear trajectory or fixed-Q
reaction is advanced.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15 as c4f15  # noqa: E402
import run_causal_inner_face36_six_mode_fine_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f20 as c4f20  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f21"
ARTIFACT = (
    "causal_inner_face36_leading_two_plus_hmm_manifest_"
    "wp10c9d6c7c3b5c4f21"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_leading_two_plus_hmm_manifest_"
    "wp10c9d6c7c3b5c4f21.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_leading_two_plus_hmm_manifest_"
    "wp10c9d6c7c3b5c4f21.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_LEADING_TWO_PLUS_HMM_MANIFEST_"
    "WP10C9D6C7C3B5C4F21_2026-08-13.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "hybrid_architecture_manifest.json"
METRICS_PATH = CANONICAL_DIRECTORY / "architecture_metrics.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authorization() -> dict:
    summary = _read(c4f20.SUMMARY_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f21_definitions_only_leading_two_plus_"
        "HMM_closure_manifest"
    )
    if (
        summary["passed"]
        or not summary["fine"]["passed"]
        or summary["cross_resolution"][
            "minimum_leading_block_projector_cosine"
        ]
        < 0.95
        or summary["cross_resolution"][
            "minimum_full_subspace_projector_cosine"
        ]
        >= 0.90
        or summary["classification"]
        != (
            "face36_six_mode_dynamic_coordinate_rejected_"
            "leading_two_plus_HMM_manifest_authorized"
        )
        or summary["authorized_next"] != expected
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["nonlinear_retained_mode_pilot_authorized"]
    ):
        raise RuntimeError("c4f21 authorization changed")
    return summary


def _cosine_and_difference(left, right) -> tuple[float, float]:
    a = np.asarray(left).ravel()
    b = np.asarray(right).ravel()
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    cosine = float(
        np.dot(a, b)
        / max(float(np.linalg.norm(a) * np.linalg.norm(b)), np.finfo(float).tiny)
    )
    return cosine, float(np.linalg.norm(a - b) / scale)


def _architecture_metrics(parent: dict):
    middle = c4f20._middle_with_recovered_amplitudes()
    with np.load(c4f20.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        fine_amplitudes = np.asarray(arrays["fine_amplitude_transitions_aligned"])
        fine_outputs = np.asarray(arrays["fine_face36_outputs_aligned"])
        fine_directions = np.asarray(arrays["fine_state_directions"])
        complement_state_fraction = np.asarray(
            arrays["fine_complement_state_fractions"]
        )
        complement_output_fraction = np.asarray(
            arrays["fine_complement_face36_fractions"]
        )
        full_outputs = np.asarray(arrays["fine_face36_outputs"])

    leading_amplitude = _cosine_and_difference(
        middle["amplitude_transitions"][:, :2, :2],
        fine_amplitudes[:, :2, :2],
    )
    leading_output = _cosine_and_difference(
        middle["face36_outputs"][:, :2], fine_outputs[:, :2]
    )
    weak_amplitude = _cosine_and_difference(
        middle["amplitude_transitions"][:, 2:, 2:],
        fine_amplitudes[:, 2:, 2:],
    )
    weak_output = _cosine_and_difference(
        middle["face36_outputs"][:, 2:], fine_outputs[:, 2:]
    )

    _layout, configuration, trajectory = c4f20.c4f13._layout_data("fine")
    columns = np.asarray(configuration["columns"]).reshape(
        trajectory["states"].shape[1:]
    )
    full_state_norms = np.asarray(
        [
            [
                np.linalg.norm(direction.ravel() / columns.ravel())
                for direction in directions
            ]
            for directions in fine_directions
        ]
    )
    complement_state_norms = complement_state_fraction * full_state_norms
    full_output_norms = np.linalg.norm(full_outputs, axis=2)
    complement_output_norms = complement_output_fraction * full_output_norms
    two_mode = parent["static_six_mode_output_reconstruction"]
    earlier = _read(c4f15.SUMMARY_PATH)["two_mode_output_reconstruction"]
    metrics = {
        "minimum_leading_block_projector_cosine": parent["cross_resolution"][
            "minimum_leading_block_projector_cosine"
        ],
        "minimum_full_six_mode_projector_cosine": parent["cross_resolution"][
            "minimum_full_subspace_projector_cosine"
        ],
        "leading_amplitude_history_cosine": leading_amplitude[0],
        "leading_amplitude_history_relative_difference": leading_amplitude[1],
        "leading_face36_history_cosine": leading_output[0],
        "leading_face36_history_relative_difference": leading_output[1],
        "weak_amplitude_history_cosine": weak_amplitude[0],
        "weak_amplitude_history_relative_difference": weak_amplitude[1],
        "weak_face36_history_cosine": weak_output[0],
        "weak_face36_history_relative_difference": weak_output[1],
        "fine_complement_global_state_relative_norm": float(
            np.linalg.norm(complement_state_norms)
            / max(np.linalg.norm(full_state_norms), np.finfo(float).tiny)
        ),
        "fine_complement_global_face36_relative_norm": float(
            np.linalg.norm(complement_output_norms)
            / max(np.linalg.norm(full_output_norms), np.finfo(float).tiny)
        ),
        "fine_complement_state_fraction_p95": float(
            np.quantile(complement_state_fraction, 0.95)
        ),
        "fine_complement_face36_fraction_p95": float(
            np.quantile(complement_output_fraction, 0.95)
        ),
        "six_mode_static_output_weighted_RMS_error": two_mode[
            "maximum_output_weighted_RMS_error"
        ],
        "six_mode_static_maximum_significant_direction_error": two_mode[
            "maximum_significant_direction_error"
        ],
        "earlier_two_mode_static_output_weighted_RMS_error": earlier[
            "maximum_output_weighted_RMS_error"
        ],
        "earlier_two_mode_static_maximum_significant_direction_error": earlier[
            "maximum_significant_direction_error"
        ],
    }
    stored = {
        "middle_leading_amplitude_history": middle[
            "amplitude_transitions"
        ][:, :2, :2],
        "fine_leading_amplitude_history_aligned": fine_amplitudes[:, :2, :2],
        "middle_leading_face36_history": middle["face36_outputs"][:, :2],
        "fine_leading_face36_history_aligned": fine_outputs[:, :2],
        "fine_complement_state_fractions": complement_state_fraction,
        "fine_complement_face36_fractions": complement_output_fraction,
    }
    return metrics, stored


def _manifest(metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "leading_two_plus_HMM_architecture_manifest_frozen_"
            "analysis_only_fixed_Q_preflight_authorized"
        ),
        "definitions_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "hybrid_state": {
            "macro_coordinates": "Q3_exterior_domain_M_J_E",
            "explicit_memory_coordinates": "a2_cross_grid_stable_leading_block",
            "guard_microstate": (
                "all_remaining_DAE_state_including_rotatable_weak_block_"
                "fine_only_complement_and_storage_histories"
            ),
            "formal_state": "Q3_plus_a2_plus_Z_guard",
            "raw_face48_exchange_forbidden": True,
            "slow_exchange": "certified_face36_exterior_partition",
            "architecture_metrics": metrics,
        },
        "interpretation": {
            "leading_two_state_coordinate_supported": True,
            "direct_two_mode_output_closure_supported": False,
            "six_mode_explicit_coordinate_supported": False,
            "weak_vectors_are_individually_named_physical_modes": False,
            "guard_complement_may_be_discarded": False,
            "HMM_microstate_supplies_binding_face36_output": True,
            "reason": (
                "leading_state_amplitudes_are_cross_grid_stable_but_their_"
                "direct_face36_magnitude_is_not;_the_weak_block_is_resolution_"
                "dependent_and_remains_part_of_the_guard_microstate"
            ),
        },
        "fixed_Q_constraint": {
            "Q": "Q3_exterior_domain_M_J_E",
            "C": "DQ3",
            "reaction": "ledger_derived_B_Q_from_c4f15",
            "continuous_KKT": "[M,-B_Q;DQ3,0]*[p_dot,lambda]=[-R,0]",
            "a2_is_not_constrained": True,
            "constraint_reaction_must_annihilate_a2_dual": True,
            "record_multiplier_work_in_M_J_E_ledgers": True,
            "manual_primitive_freezing_forbidden": True,
            "Euclidean_projection_forbidden": True,
        },
        "authorized_analysis_only_preflight": {
            "work_package": (
                "WP10c9d6c7c3b5c4f22_analysis_only_leading_two_plus_HMM_"
                "fixed_Q_constraint_preflight"
            ),
            "representative_Q_state": "committed_middle_20ms_base_state",
            "also_check_fine_endpoint_without_propagation": True,
            "new_nonlinear_trajectory": False,
            "new_tangent_trajectory": False,
            "construct_descriptor_consistent_KKT": True,
            "construct_fixed_Q_projected_local_tangent": True,
            "screen_equal_Q_lifts_as_block_RHS": 24,
            "lift_classes": [
                "two_explicit_leading_directions",
                "four_rotatable_weak_enrichment_directions",
                "fine_only_complement_directions",
                "smooth_random_Q3_null_guard_directions",
            ],
            "evaluate_face36_output_observability": True,
            "compute_finite_time_singular_and_transient_growth_diagnostics": True,
            "do_not_assume_guard_mixing_or_decay": True,
        },
        "prospective_preflight_gates": {
            "maximum_DQ_M_inverse_BQ_identity_defect": 1.0e-10,
            "maximum_KKT_linear_solve_relative_defect": 1.0e-10,
            "maximum_reaction_ledger_relative_defect": 1.0e-12,
            "maximum_reaction_support_relative_defect": 1.0e-12,
            "maximum_a2_dual_reaction_annihilation_defect": 1.0e-10,
            "maximum_a2_dual_biorthogonality_defect": 1.0e-10,
            "maximum_projected_block_solve_relative_defect": 1.0e-10,
            "maximum_face36_directional_JVP_relative_defect": 1.0e-8,
            "incoming_excision_characteristics": 0,
        },
        "conditional_nonlinear_pilot_contract": {
            "layout": "middle_primary",
            "one_constrained_nonlinear_base_per_Q": True,
            "all_lifts_screened_by_block_tangent_first": True,
            "maximum_full_nonlinear_anchor_lifts": 2,
            "fine_full_trajectory_automatic": False,
            "fine_residual_correction_then_short_shadow": True,
            "adaptive_microburst_windows_ms": [2.0, 5.0, 10.0, 20.0],
            "extend_window_only_if_guard_statistics_not_converged": True,
            "binding_outputs": [
                "instantaneous_face36_exterior_partition",
                "cumulative_face36_exterior_partition",
                "window_mean_face36_exterior_partition",
                "a2_transition_history",
                "constraint_reaction_ledgers",
            ],
        },
        "decision": {
            "fixed_Q_preflight_fails": "stop_before_any_constrained_microburst",
            "guard_contracts_or_mixes_rapidly": (
                "authorize_Q3_plus_a2_closure_with_short_HMM_burst"
            ),
            "guard_has_low_rank_persistent_observable_memory": (
                "augment_a2_only_after_new_cross_grid_coordinate_gate"
            ),
            "guard_is_broad_but_window_mean_converges": (
                "retain_HMM_microburst_closure"
            ),
            "guard_mean_depends_on_lift_or_has_long_memory": (
                "retain_inner_solver_or_explicit_memory_kernel"
            ),
            "multiple_guard_attractors": "add_discrete_branch_or_hysteresis_state",
        },
        "cost_contract": {
            "analysis_only_preflight_wall_hours": [1.0, 3.0],
            "first_middle_one_Q_pilot_wall_hours": [6.0, 15.0],
            "one_factorization_24_RHS": True,
            "no_full_nonlinear_run_for_every_lift": True,
            "no_automatic_fine_or_50ms_trajectory": True,
            "adaptive_window_extension": True,
        },
        "hard_stops": [
            "do_not_fit_a_direct_two_mode_face36_output_law",
            "do_not_promote_the_four_weak_vectors_as_coordinates",
            "do_not_discard_the_guard_or_fine_only_complement",
            "do_not_freeze_primitives_manually",
            "do_not_run_every_lift_nonlinearly",
            "do_not_use_raw_face48_as_slow_exchange",
            "do_not_start_50ms_or_reduced_slow_evolution",
        ],
        "fixed_Q_constraint_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f22_analysis_only_leading_two_plus_HMM_"
            "fixed_Q_constraint_preflight"
        ),
    }


def _catalog(summary: dict) -> None:
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
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def main() -> None:
    parent = _authorization()
    metrics, stored = _architecture_metrics(parent)
    if (
        metrics["minimum_leading_block_projector_cosine"] < 0.95
        or metrics["leading_amplitude_history_cosine"] < 0.95
        or metrics["leading_amplitude_history_relative_difference"] > 0.10
        or metrics["minimum_full_six_mode_projector_cosine"] >= 0.90
    ):
        raise RuntimeError("c4f21 leading/weak architecture decision changed")
    manifest = _manifest(metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "architecture_metrics": metrics,
        "leading_two_state_coordinate_supported": True,
        "direct_two_mode_output_closure_supported": False,
        "six_mode_explicit_coordinate_supported": False,
        "guard_HMM_required": True,
        "fixed_Q_constraint_preflight_authorized": True,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "guard_complement_retained": True,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "macro_dimension": 3,
            "explicit_memory_dimension": 2,
            "representative_Q_time_seconds": 0.020,
            "screened_equal_Q_lifts": 24,
            "shared_exchange_parent_face": 36,
            "raw_face48_exchange_forbidden": True,
        },
    )
    _write(MANIFEST_PATH, manifest)
    np.savez_compressed(METRICS_PATH, **stored)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 leading-two plus HMM architecture manifest\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        "The leading two-dimensional state block is retained explicitly: its "
        "minimum cross-grid projector cosine is "
        f"`{metrics['minimum_leading_block_projector_cosine']:.6f}`, and its "
        "stable-dual amplitude history has cosine/difference "
        f"`{metrics['leading_amplitude_history_cosine']:.6f}` / "
        f"`{metrics['leading_amplitude_history_relative_difference']:.6f}`.\n\n"
        "It is not a direct output closure. The leading face-36 history "
        f"difference is `{metrics['leading_face36_history_relative_difference']:.6f}`, "
        "while the weak block is not cross-grid stable. The unresolved state, "
        "fine-only complement, and storage histories therefore remain an HMM "
        "guard microstate that supplies the binding face-36 export.\n\n"
        "The next package may perform only an analysis-only descriptor-consistent "
        "fixed-Q constraint/projected-tangent preflight at the committed 20 ms "
        "state. No constrained nonlinear microburst, 50 ms run, or reduced slow "
        "evolution is authorized.\n",
        encoding="utf-8",
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "parent_summary_sha256": _sha(c4f20.SUMMARY_PATH),
            "parent_arrays_sha256": _sha(c4f20.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: (
                    _sha(ROOT / THIS_TEST) if (ROOT / THIS_TEST).exists() else None
                ),
            },
        },
    )
    files = (CONFIG_PATH, MANIFEST_PATH, METRICS_PATH, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
