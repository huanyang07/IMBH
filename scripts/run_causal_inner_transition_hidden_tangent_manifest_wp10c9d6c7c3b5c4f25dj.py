#!/usr/bin/env python3
"""Freeze a saved-generator tangent diagnostic for the transition basis."""

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

import run_causal_inner_transition_hidden_basis_screen_wp10c9d6c7c3b5c4f25di as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25dj"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25dk"
PARENT_COMMIT = "634b79181ed3e5e90e1ddd15f0002f966e597c89"
PARENT_PARENT = "2647308f0ab2eb360fb680bba930033235e4584d"
PARENT_TREE = "3ab5b81c9192c133b32714139d8439b582b9ffd5"
CLASSIFICATION = (
    "common_transition_hidden_tangent_diagnostic_manifest_frozen_"
    "saved_complete_generator_execution_authorized"
)

COORDINATE_DIMENSION = 470
PHYSICAL_DIMENSION = 560
GAUGE_DIMENSION = 90
HIDDEN_DIMENSION = 388
INITIAL_HIDDEN_RANK = 8
MAXIMUM_HIDDEN_RANK = 128
ENRICHMENT_RANKS = (8, 12, 16, 24, 32, 48, 64, 96, 128)
HESSIAN_STEPS = (2.0e-4, 1.0e-4, 5.0e-5)
SIGNED_PHYSICAL_COMPONENT_RADIUS = 2.5e-3
MAXIMUM_COORDINATE_JACOBIAN_EVALUATIONS = 400

COMPLETE_JVP_DEFECT_GATE = 1.0e-10
GENERATOR_CLOSURE_GATE = 1.0e-12
CHART_CONDITION_GATE = 1.0e7
TANGENT_STEP_CONVERGENCE_GATE = 1.0e-3
TANGENT_INVARIANCE_GATE = 0.1
PHYSICAL_TANGENT_CAPTURE_GATE = 0.9
MACRO_ANNIHILATION_GATE = 5.0e-12
ORTHONORMALITY_GATE = 5.0e-12

ARTIFACT = (
    "causal_inner_transition_hidden_tangent_manifest_wp10c9d6c7c3b5c4f25dj"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_transition_hidden_tangent_manifest_"
    "wp10c9d6c7c3b5c4f25dj.py"
)
THIS_TEST = (
    "tests/test_causal_inner_transition_hidden_tangent_manifest_"
    "wp10c9d6c7c3b5c4f25dj.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_HIDDEN_TANGENT_"
    "MANIFEST_WP10C9D6C7C3B5C4F25DJ_2026-08-20.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

PARENT_SUMMARY = parent.CANONICAL_DIRECTORY / "summary.json"
PARENT_CONTRACT = parent.CANONICAL_DIRECTORY / "selected_hidden_basis_contract.json"
PARENT_ARRAYS = parent.CANONICAL_DIRECTORY / "hidden_basis_screen_arrays.npz"
SAVED_DESCRIPTOR_DIRECTORY = (
    ROOT
    / "results/canonical/causal_inner_pathwise_closure_descriptor_pilot_"
    "wp10c9d6c7c3b5c4f25c"
)
SAVED_GENERATOR = SAVED_DESCRIPTOR_DIRECTORY / "descriptor_A.npz"
SAVED_GENERATOR_METRICS = SAVED_DESCRIPTOR_DIRECTORY / "assembly_metrics.json"
SAVED_GENERATOR_PROVENANCE = SAVED_DESCRIPTOR_DIRECTORY / "provenance.json"
SAVED_GENERATOR_SEED_LOCK = SAVED_DESCRIPTOR_DIRECTORY / "seed_lock.json"
SAVED_CHECKPOINT = (
    ROOT
    / "results/canonical/causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l/checkpoint_warm_3.npz"
)
EXACT_CHART_RUNNER = (
    "scripts/run_causal_inner_exact_geometric_470_chart_preflight_"
    "wp10c9d6c7c3b5c4f25de.py"
)
FIXED_Q_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
TANGENT_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py"
)


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_parent(*, require_clean: bool) -> dict:
    if _git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("transition-tangent parent commit changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^") != PARENT_PARENT:
        raise RuntimeError("transition-tangent parent lineage changed")
    if _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("transition-tangent parent tree changed")

    parent_hashes = _checksums(parent.CANONICAL_DIRECTORY)
    summary = _read(PARENT_SUMMARY)
    contract = _read(PARENT_CONTRACT)
    provenance = _read(parent.CANONICAL_DIRECTORY / "provenance.json")
    arrays = _load_npz(PARENT_ARRAYS)
    if (
        not summary["passed"]
        or not summary["saved_arrays_only"]
        or summary["classification"] != parent.COMMON_CLASSIFICATION
        or summary["selected_hidden_rank"] != INITIAL_HIDDEN_RANK
        or summary["selected_basis_source"] != "prior_seed_only"
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["new_exact_fixed_Q_rate_calls"] != 0
        or summary["new_complete_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or summary["propagated_states"] != 0
        or summary["sealed_16ms_opened"]
        or not contract["interpretation"][
            "common_transition_hidden_basis_candidate_supported"
        ]
        or contract["interpretation"]["basis_is_a_certified_transition_dynamics_model"]
        or contract["prospective_next_manifest"]["work_package"] != WORK_PACKAGE
        or arrays["selected_hidden_basis388"].shape
        != (HIDDEN_DIMENSION, INITIAL_HIDDEN_RANK)
    ):
        raise RuntimeError("common hidden-basis authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"hidden-basis source changed: {relative}")

    descriptor_hashes = _checksums(SAVED_DESCRIPTOR_DIRECTORY)
    metrics = _read(SAVED_GENERATOR_METRICS)
    generator_provenance = _read(SAVED_GENERATOR_PROVENANCE)
    seed_lock = _read(SAVED_GENERATOR_SEED_LOCK)
    generator = _load_npz(SAVED_GENERATOR)
    checkpoint = _load_npz(SAVED_CHECKPOINT)
    if (
        descriptor_hashes["descriptor_A.npz"] != _sha(SAVED_GENERATOR)
        or metrics["exact_continuous_descriptor_assemblies"] != 1
        or metrics["complete_JVP_relative_defect"] > COMPLETE_JVP_DEFECT_GATE
        or metrics["descriptor_generator_closure_relative_defect"]
        > GENERATOR_CLOSURE_GATE
        or metrics["new_nonlinear_roots"] != 0
        or metrics["propagated_states"] != 0
        or metrics["seed_checkpoint_sha256"] != _sha(SAVED_CHECKPOINT)
        or seed_lock["seed_checkpoint_sha256"] != _sha(SAVED_CHECKPOINT)
        or generator["complete_fixed_Q_generator"].shape
        != (PHYSICAL_DIMENSION, PHYSICAL_DIMENSION)
        or generator["fixed_Q_rate"].shape != (PHYSICAL_DIMENSION,)
        or checkpoint["current_primitive_charts"].shape != (112, 5)
        or float(checkpoint["elapsed_time_seconds"]) != 0.020000599999999997
        or int(checkpoint["completed_steps"]) != 6
    ):
        raise RuntimeError("saved complete generator evidence changed")
    for relative in (FIXED_Q_SOURCE, TANGENT_SOURCE):
        if _sha(ROOT / relative) != generator_provenance["source_hashes"][relative]:
            raise RuntimeError(f"saved generator source changed: {relative}")
    for name, expected in provenance["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("transition-tangent manifest requires a clean tracked tree")
    return {
        "parent_hashes": parent_hashes,
        "descriptor_hashes": descriptor_hashes,
        "parent_classification": summary["classification"],
        "saved_generator_classification": _read(
            SAVED_DESCRIPTOR_DIRECTORY / "summary.json"
        )["classification"],
        "saved_generator_complete_JVP_relative_defect": metrics[
            "complete_JVP_relative_defect"
        ],
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "objective": (
            "test_local_tangent_closure_of_the_rank8_common_transition_hidden_"
            "basis_at_the_saved_20p0006ms_accepted_transition_checkpoint"
        ),
        "input_separation": {
            "selected_basis": (
                "rank8_dual_consistent_hidden_basis_from_WP10c9d6c7c3b5c4f25di"
            ),
            "tangent_reference": (
                "hash_locked_complete_fixed_Q_generator_at_20p0006ms"
            ),
            "old_rejected_resolved_lifting_reused": False,
            "old_rejected_transfer_model_reused": False,
            "saved_generator_source_matches_current_fixed_Q_and_tangent_sources": True,
            "full_y470_offline_reference_and_fallback_preserved": True,
        },
        "exact_local_operator": {
            "coordinate_chart": "exact_geometric_y470_chart_evaluated_at_checkpoint",
            "checkpoint_time_seconds": 0.020000599999999997,
            "checkpoint_role": "accepted_unclassified_transition_checkpoint",
            "gauge": "canonical_null_basis_of_checkpoint_coordinate_Jacobian",
            "basis_action": "V470=Z388_B8",
            "physical_lift": "W560=[J470;N90^T]^{-1}[V470;0]",
            "coordinate_field_tangent": (
                "K_y_V=J470_A560_W560_plus_DJ470[W560]_fixed_Q_rate560"
            ),
            "coordinate_Hessian_term_required": True,
            "hidden_tangent_action": "K_h_B=Q388_K_y_V",
            "reduced_tangent": "K8=B8^T_K_h_B",
            "rank8_invariance_defect": (
                "norm_F((I-B8_B8^T)_K_h_B)/max(norm_F(K_h_B),tiny)"
            ),
            "transition_spectrum_role": "diagnostic_only_not_branch_stability",
        },
        "coordinate_Hessian_audit": {
            "method": (
                "central_coordinate_Jacobian_difference_along_each_gauge_fixed_"
                "physical_basis_direction"
            ),
            "signed_physical_component_radius": SIGNED_PHYSICAL_COMPONENT_RADIUS,
            "dimensionless_step_ladder": list(HESSIAN_STEPS),
            "response_convergence_measure": (
                "adjacent_relative_Frobenius_defect_of_hidden_tangent_actions"
            ),
            "maximum_adjacent_relative_defect": TANGENT_STEP_CONVERGENCE_GATE,
            "new_fixed_Q_rate_calls": 0,
            "new_complete_generator_assemblies": 0,
            "chart_retractions": 0,
        },
        "rank_adaptive_fallback": {
            "candidate_hidden_ranks": list(ENRICHMENT_RANKS),
            "construction": (
                "block_Arnoldi_enrichment_by_orthonormalized_hidden_tangent_"
                "residual_directions"
            ),
            "maximum_rank": MAXIMUM_HIDDEN_RANK,
            "new_basis_vectors_must_remain_in_kernel_of_R82": True,
            "rate_action_and_physical_localization_gates_remain_binding": True,
            "maximum_coordinate_Jacobian_evaluations": (
                MAXIMUM_COORDINATE_JACOBIAN_EVALUATIONS
            ),
        },
        "binding_gates": {
            "saved_complete_JVP_relative_defect_max": COMPLETE_JVP_DEFECT_GATE,
            "saved_generator_closure_relative_defect_max": GENERATOR_CLOSURE_GATE,
            "checkpoint_coordinate_chart_condition_number_max": CHART_CONDITION_GATE,
            "tangent_step_convergence_relative_defect_max": (
                TANGENT_STEP_CONVERGENCE_GATE
            ),
            "selected_hidden_tangent_invariance_relative_defect_max": (
                TANGENT_INVARIANCE_GATE
            ),
            "selected_hidden_physical_tangent_energy_capture_min": (
                PHYSICAL_TANGENT_CAPTURE_GATE
            ),
            "selected_action_macro_annihilation_infinity_max": (
                MACRO_ANNIHILATION_GATE
            ),
            "selected_basis_orthonormality_infinity_max": ORTHONORMALITY_GATE,
            "maximum_selected_hidden_rank": MAXIMUM_HIDDEN_RANK,
        },
        "execution_budget": {
            "new_exact_fixed_Q_rate_evaluations_equal": 0,
            "new_complete_generator_assemblies_equal": 0,
            "new_nonlinear_roots_equal": 0,
            "new_chart_retractions_equal": 0,
            "propagated_states_equal": 0,
            "sealed_16ms_truth_calls_equal": 0,
            "maximum_coordinate_Jacobian_evaluations": (
                MAXIMUM_COORDINATE_JACOBIAN_EVALUATIONS
            ),
        },
        "fail_fast_sequence": [
            "validate_all_parent_and_saved_generator_hashes",
            "rebuild_the_exact_checkpoint_coordinate_Jacobian_and_gauge",
            "audit_the_rank8_Hessian_step_ladder",
            "test_rank8_hidden_and_physical_tangent_closure",
            "only_if_rank8_fails_apply_saved_operator_block_Arnoldi_enrichment",
            "stop_at_the_first_rank_that_passes_or_at_rank128",
            "construct_no_transition_trajectory_or_online_model",
        ],
        "decision": {
            "rank8_passes": {
                "classification": (
                    "common_rank8_transition_hidden_tangent_candidate_supported"
                ),
                "authorizes_only": (
                    "definitions_only_nonlinear_transition_impulse_sampling_design"
                ),
            },
            "enriched_rank_at_most_128_passes": {
                "classification": (
                    "rank_adaptive_common_transition_hidden_tangent_candidate_"
                    "supported"
                ),
                "authorizes_only": (
                    "definitions_only_rank_adaptive_transition_sampling_design"
                ),
            },
            "no_rank_at_most_128_passes": {
                "classification": (
                    "transition_hidden_tangent_reduction_rejected_full470_"
                    "offline_impulse_reference_required"
                ),
                "authorizes_only": (
                    "definitions_only_full470_offline_impulse_sampling_design"
                ),
            },
            "chart_or_saved_operator_audit_fails": {
                "classification": "transition_tangent_diagnostic_infrastructure_failed",
                "authorizes_only": "diagnosis_only",
            },
        },
        "authorization_boundaries": {
            "this_package_executes_the_tangent_diagnostic": False,
            "transition_trajectory_authorized": False,
            "branch_root_authorized": False,
            "transition_truth_campaign_authorized": False,
            "online_transition_ODE_authorized": False,
            "online_solver_authorized": False,
            "physical_microburst_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decisive_input_hashes": {
            "parent_summary": _sha(PARENT_SUMMARY),
            "parent_contract": _sha(PARENT_CONTRACT),
            "parent_arrays": _sha(PARENT_ARRAYS),
            "saved_generator": _sha(SAVED_GENERATOR),
            "saved_generator_metrics": _sha(SAVED_GENERATOR_METRICS),
            "saved_generator_provenance": _sha(SAVED_GENERATOR_PROVENANCE),
            "saved_generator_seed_lock": _sha(SAVED_GENERATOR_SEED_LOCK),
            "saved_checkpoint": _sha(SAVED_CHECKPOINT),
            "exact_chart_runner": _sha(ROOT / EXACT_CHART_RUNNER),
            "fixed_Q_source": _sha(ROOT / FIXED_Q_SOURCE),
            "tangent_source": _sha(ROOT / TANGENT_SOURCE),
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
        raise RuntimeError("transition hidden-tangent manifest already exists")
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "transition_hidden_tangent_contract.json", contract)
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
        "selected_parent_hidden_rank": INITIAL_HIDDEN_RANK,
        "saved_complete_generator_reuse_frozen": True,
        "old_rejected_resolved_lifting_reused": False,
        "coordinate_Hessian_term_required": True,
        "new_exact_fixed_Q_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "new_chart_retractions": 0,
        "propagated_states": 0,
        "sealed_16ms_opened": False,
        "transition_tangent_executed": False,
        "transition_truth_campaign_authorized": False,
        "online_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        EXACT_CHART_RUNNER,
        FIXED_Q_SOURCE,
        TANGENT_SOURCE,
    )
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
                "# Transition hidden-tangent manifest WP10c9d6c7c3b5c4f25dj",
                "",
                "## Classification",
                "",
                f"`{CLASSIFICATION}`",
                "",
                "The rank-8 common transition basis passed the saved action screen. This definitions-only package freezes its first dynamics test; it performs no tangent execution or new truth call.",
                "",
                "The execution reuses only the hash-locked complete fixed-Q generator and accepted 20.0006 ms checkpoint from WP10c9d6c7c3b5c4f25c. It explicitly does not reuse that package's rejected 82-dimensional lifting or transfer model. The generator's fixed-Q and tangent source hashes are identical to the current sources.",
                "",
                "The exact coordinate tangent includes both J A W and the coordinate-Hessian action DJ[W] f. Three central coordinate-Jacobian steps audit the Hessian response without a new rate, generator, root, chart retraction, propagation, or sealed 16 ms call.",
                "",
                "Rank 8 is tested first. Only if it fails may a saved-operator block-Arnoldi enrichment proceed through ranks 12, 16, 24, 32, 48, 64, 96, and 128. The full y470 reference remains mandatory even if a reduced tangent candidate passes.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`, the frozen saved-generator tangent execution. No transition trajectory, truth campaign, branch root, online transition ODE, microburst, or reduced slow evolution is authorized.",
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
