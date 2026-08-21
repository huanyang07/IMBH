#!/usr/bin/env python3
"""Freeze the branch-first hybrid impulse-map architecture after rank 16."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_hidden_tangent_wp10c9d6c7c3b5c4f25dk as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dl"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dm"
PARENT_COMMIT = "b650fdc365e2b4e7c2bee7781e71b26af2db66f2"
PARENT_PARENT = "625be2bbdec0c59881eb30ebda1bf0bd397bc882"
PARENT_TREE = "5af04159d26b96687bf1f5b10f297f31819d4490"
CLASSIFICATION = (
    "rank16_transition_internal_candidate_reconciled_"
    "branch_first_hybrid_impulse_sampling_architecture_frozen"
)

FIDUCIAL_CYCLE_SECONDS = 578880.0
MAXIMUM_ONLINE_MACROSTEPS = 100000
MINIMUM_AVERAGE_MACROSTEP_SECONDS = (
    FIDUCIAL_CYCLE_SECONDS / MAXIMUM_ONLINE_MACROSTEPS
)

ARTIFACT = (
    "causal_inner_branch_first_hybrid_impulse_architecture_"
    "wp10c9d6c7c3b5c4f25dl"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_branch_first_hybrid_impulse_architecture_"
    "wp10c9d6c7c3b5c4f25dl.py"
)
THIS_TEST = (
    "tests/test_causal_inner_branch_first_hybrid_impulse_architecture_"
    "wp10c9d6c7c3b5c4f25dl.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_BRANCH_FIRST_HYBRID_IMPULSE_"
    "ARCHITECTURE_WP10C9D6C7C3B5C4F25DL_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_SUMMARY = parent.CANONICAL_DIRECTORY / "summary.json"
PARENT_METRICS = parent.CANONICAL_DIRECTORY / "transition_hidden_tangent_metrics.json"
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "transition_hidden_tangent_arrays.npz"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("branch-first architecture parent changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("branch-first architecture lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("branch-first architecture tree changed")
    hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(PARENT_SUMMARY)
    payload = _read(PARENT_METRICS)
    metrics = payload["metrics"]
    checks = payload["checks"]
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    candidates = metrics["candidate_rank_metrics"]
    if (
        not summary["passed"]
        or summary["classification"] != parent.ENRICHED_CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["selected_hidden_rank"] != 16
        or summary["new_exact_fixed_Q_rate_calls"] != 0
        or summary["new_complete_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or summary["new_chart_retractions"] != 0
        or summary["propagated_states"] != 0
        or summary["sealed_16ms_opened"]
        or not summary["full470_offline_transition_reference_preserved"]
        or not all(checks.values())
        or [candidate["rank"] for candidate in candidates] != [8, 12, 16]
        or candidates[0]["hidden_tangent_invariance_relative_defect"] <= 0.1
        or candidates[1]["hidden_tangent_invariance_relative_defect"] <= 0.1
        or candidates[2]["hidden_tangent_invariance_relative_defect"] > 0.1
        or candidates[2]["hidden_physical_tangent_energy_capture"] < 0.9
    ):
        raise RuntimeError("rank-adaptive transition tangent result changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"transition tangent source changed: {relative}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("branch-first architecture requires a clean tracked tree")
    return {
        "parent_hashes": hashes,
        "parent_classification": summary["classification"],
        "rank8_invariance_defect": candidates[0][
            "hidden_tangent_invariance_relative_defect"
        ],
        "rank12_invariance_defect": candidates[1][
            "hidden_tangent_invariance_relative_defect"
        ],
        "rank16_invariance_defect": candidates[2][
            "hidden_tangent_invariance_relative_defect"
        ],
    }


def _architecture() -> dict:
    payload = _read(PARENT_METRICS)
    candidates = payload["metrics"]["candidate_rank_metrics"]
    rank16 = candidates[-1]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "evidence_reconciliation": {
            "rate_action_basis_rank": 8,
            "rate_action_basis_role": "kinematic_snapshot_compression_only",
            "rank8_tangent_invariance_relative_defect": candidates[0][
                "hidden_tangent_invariance_relative_defect"
            ],
            "rank12_tangent_invariance_relative_defect": candidates[1][
                "hidden_tangent_invariance_relative_defect"
            ],
            "selected_transition_internal_rank": 16,
            "rank16_tangent_invariance_relative_defect": rank16[
                "hidden_tangent_invariance_relative_defect"
            ],
            "rank16_hidden_physical_tangent_energy_capture": rank16[
                "hidden_physical_tangent_energy_capture"
            ],
            "rank16_current_primary_physical_action_energy_capture": rank16[
                "current_primary_gauge_fixed_physical_action_energy_capture"
            ],
            "rank16_nonstable_eigenvalue_count_diagnostic": rank16[
                "reduced_nonstable_eigenvalue_count_diagnostic"
            ],
            "transition_spectrum_is_branch_stability_evidence": False,
            "rank16_is_a_certified_nonlinear_transition_model": False,
        },
        "mathematical_architecture": {
            "online_state": "s_b=(U80,a2,m_b,branch_label)",
            "online_branch_flow": (
                "branch_specific_conservative_second_order_macro_integrator"
            ),
            "online_event": (
                "certified_fold_or_fast_stability_surface_with_hysteresis"
            ),
            "online_transition": (
                "one_conservative_entry_to_exit_reset_map_evaluation"
            ),
            "online_transition_ODE": False,
            "online_exact_truth_calls": 0,
            "online_fast_microsteps": 0,
            "offline_transition_reference": "full_exact_fixed_Q_y470_dynamics",
            "offline_transition_feature_state": (
                "rank16_dual_consistent_hidden_coordinates_plus_event_parameters"
            ),
            "offline_full470_fallback_required": True,
            "transition_map": (
                "T_b:(s_minus,theta_event)->(s_plus,Delta_face_flux,"
                "Delta_sources,Delta_constraint_work,duration,b_plus)"
            ),
            "discrete_conservation": (
                "macro_jump_equals_integrated_single_valued_face_flux_plus_"
                "physical_sources_plus_constraint_work"
            ),
        },
        "branch_first_dependency": {
            "reason": (
                "entry_and_exit_sections_cannot_be_defined_from_an_"
                "unclassified_transition_checkpoint"
            ),
            "required_branch_labels": ["cold", "hot"],
            "branch_candidate_must_not_be_the_exact_20ms_transition_anchor": True,
            "candidate_selection_first_uses_saved_revealed_arrays_only": True,
            "branch_truth_execution_requires_a_separate_prospective_manifest": True,
            "branch_root_fail_fast_preflight_hidden_fraction_max": 0.25,
            "branch_certification_gates": {
                "critical_manifold_invariance_relative_defect_max": 0.1,
                "fast_to_effective_slow_spectral_gap_ratio_min": 10.0,
                "fast_block_requires_strictly_negative_real_spectrum": True,
                "all_fixed_Q_physical_storage_reaction_and_ledger_gates": True,
                "state_robustness_requires_cold_and_hot_pass": True,
            },
        },
        "prospective_branch_candidate_screen": {
            "work_package": AUTHORIZED_NEXT,
            "truth_policy": "saved_revealed_arrays_only",
            "new_exact_fixed_Q_rate_calls_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "new_chart_retractions_equal": 0,
            "propagated_states_equal": 0,
            "sealed_16ms_truth_calls_equal": 0,
            "candidate_inputs": (
                "existing_nonsealed_full_model_or_accepted_fixed_Q_checkpoint_"
                "states_with_hash_locked_times_and_ledgers"
            ),
            "selection_features": [
                "distance_before_or_after_the_identified_transition_sector",
                "rank16_hidden_coordinate_amplitude",
                "saved_rate_or_secant_hidden_fraction_proxy",
                "physical_guard_margin",
                "reconstruction_margin",
                "availability_of_complete_authentic_history_and_provenance",
            ],
            "must_select_distinct_cold_and_hot_candidates": True,
            "must_stop_if_no_nontransition_candidates_are_supported": True,
        },
        "later_offline_transition_sampling": {
            "authorized_in_this_package": False,
            "prerequisite": "both_branch_certificates_and_entry_exit_sections",
            "internal_coordinate_policy": (
                "rank16_is_initial_candidate_with_rank_adaptation_and_full470_"
                "fallback"
            ),
            "training_outputs": [
                "exit_macrostate",
                "integrated_single_valued_face_flux",
                "integrated_physical_sources",
                "integrated_constraint_work",
                "transition_duration",
                "destination_branch",
            ],
            "heldout_requirements": [
                "entry_state_holdout",
                "event_parameter_holdout",
                "transition_direction_holdout",
                "spatial_layout_holdout_before_predictive_use",
            ],
            "surrogate_must_enforce_conservation_by_construction": True,
        },
        "online_runtime_contract": {
            "fiducial_cycle_seconds": FIDUCIAL_CYCLE_SECONDS,
            "maximum_macrosteps_per_cycle": MAXIMUM_ONLINE_MACROSTEPS,
            "minimum_average_macrostep_seconds": (
                MINIMUM_AVERAGE_MACROSTEP_SECONDS
            ),
            "maximum_transition_map_evaluations_per_event": 1,
            "full_transition_microintegration_online": False,
            "target_wall_time_for_one_cycle": "several_days_not_months",
        },
        "authorization_boundaries": {
            "branch_candidate_saved_array_screen_authorized": True,
            "branch_truth_execution_authorized": False,
            "transition_truth_campaign_authorized": False,
            "transition_impulse_surrogate_fit_authorized": False,
            "online_hybrid_solver_authorized": False,
            "exploratory_cycle_authorized": False,
            "predictive_cycle_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "parent_summary": _sha(PARENT_SUMMARY),
            "parent_metrics": _sha(PARENT_METRICS),
            "parent_arrays": _sha(PARENT_ARRAYS),
        },
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
                    "sha256": _sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_parent(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("branch-first architecture already exists")
    architecture = _architecture()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "branch_first_hybrid_impulse_architecture.json",
        architecture,
    )
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_commit": PARENT_COMMIT,
            "parent_parent": PARENT_PARENT,
            "parent_tree": PARENT_TREE,
            **frozen,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "selected_transition_internal_rank": 16,
        "rank8_rate_basis_preserved_as_kinematic_diagnostic": True,
        "full470_offline_transition_reference_preserved": True,
        "branch_first_execution_order_frozen": True,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "branch_truth_execution_authorized": False,
        "transition_truth_campaign_authorized": False,
        "online_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (THIS_RUNNER, THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Branch-first hybrid impulse architecture WP10c9d6c7c3b5c4f25dl",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The saved action family is rank-8 compressible, but its local dynamics are not rank-8 invariant. Rank 12 remains above the frozen 0.1 invariance gate; residual-driven rank 16 passes with a 4.256319e-03 invariance defect and 99.890276% physical tangent capture.",
                "",
                "Rank 16 is therefore the initial offline transition-internal coordinate candidate. It is not a branch model and not yet a nonlinear impulse-map model. The full y470 dynamics remain the offline reference and fallback.",
                "",
                "The execution order is now branch-first: identify distinct cold and hot candidates from nonsealed saved arrays, certify both branch critical states and their normally hyperbolic fast blocks, then define entry/exit sections, and only afterward authorize transition sampling.",
                "",
                f"The online target remains at most {MAXIMUM_ONLINE_MACROSTEPS} macrosteps for a {FIDUCIAL_CYCLE_SECONDS:.0f} s cycle, requiring an average macrostep of at least {MINIMUM_AVERAGE_MACROSTEP_SECONDS:.6f} s. Online transition microintegration remains forbidden; one conservative reset-map evaluation is allowed per event.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`, a saved-revealed-array branch-candidate screen. No new truth, branch root, transition campaign, online solver, microburst, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
