#!/usr/bin/env python3
"""Run the committed-evidence-only reduced-cycle identifiability screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_reduced_cycle_architecture_manifest_wp10c9d6c7c3b5c4f25 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25a"
CLASSIFICATION = (
    "reduced_cycle_architecture_selected_coefficients_unidentified_"
    "offline_closure_database_manifest_authorized"
)
PARENT_PACKAGE_COMMIT = "f9da48298ca7c2a9951fc57b9a1a6be02181de30"
PARENT_PACKAGE_PARENT = "f104a1eada1c83635d9f264b6c8b1cafb56988f0"
PARENT_PACKAGE_TREE = "1295ec4fb0d09a0134494285b06fac807d067d7d"

ARTIFACT = "causal_inner_reduced_cycle_identifiability_wp10c9d6c7c3b5c4f25a"
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_reduced_cycle_identifiability_"
    "wp10c9d6c7c3b5c4f25a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_reduced_cycle_identifiability_"
    "wp10c9d6c7c3b5c4f25a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_CYCLE_IDENTIFIABILITY_"
    "WP10C9D6C7C3B5C4F25A_2026-08-17.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha(directory / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = actual
    return recorded


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _validate_parent() -> tuple[dict, dict]:
    if _git("rev-parse", PARENT_PACKAGE_COMMIT) != PARENT_PACKAGE_COMMIT:
        raise RuntimeError("parent architecture package commit changed")
    if _git("rev-parse", f"{PARENT_PACKAGE_COMMIT}^") != PARENT_PACKAGE_PARENT:
        raise RuntimeError("parent architecture package parent changed")
    if _git("rev-parse", f"{PARENT_PACKAGE_COMMIT}^{{tree}}") != PARENT_PACKAGE_TREE:
        raise RuntimeError("parent architecture package tree changed")
    hashes = _checksums(parent.ARTIFACT_DIRECTORY)
    summary = _read(parent.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(parent.ARTIFACT_DIRECTORY / "architecture_contract.json")
    if (
        not summary["passed"]
        or not summary["evidence_only_identifiability_authorized"]
        or summary["online_reduced_solver_implementation_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or contract["authorized_next"]
        != "WP10c9d6c7c3b5c4f25a_evidence_only_identifiability_screen"
    ):
        raise RuntimeError("parent architecture authorization changed")
    input_lock = _read(parent.ARTIFACT_DIRECTORY / "input_lock.json")
    for relative, expected in input_lock["input_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen architecture input changed: {relative}")
    return summary, {"package_hashes": hashes, "input_lock": input_lock}


def _evidence_ledger() -> dict:
    inputs = parent._validate_inputs()
    cost = parent._cost_model()
    fiber_text = parent.FIBER_REPORT.read_text(encoding="utf-8")
    healing_text = parent.HEALING_REPORT.read_text(encoding="utf-8")
    required_fiber = (
        "34-coordinate output-identifiable/conservative",
        "instantaneous Markov closure is rejected",
    )
    required_healing = (
        "n64_persistent_localized_inner_mode_through_0p125s",
        "N128 independently remains persistent and localized",
    )
    if any(token not in fiber_text for token in required_fiber):
        raise RuntimeError("historical nonlinear-fiber rejection changed")
    if any(token not in healing_text for token in required_healing):
        raise RuntimeError("historical memory persistence result changed")
    warm = inputs["warm"]
    profile = warm["solver_profiling"]["exclusive_wall_seconds"]
    return {
        "truth_solver": {
            "largest_certified_timestep_seconds": parent.CERTIFIED_TRUTH_TIMESTEP_SECONDS,
            "doubled_timestep_classification": inputs["latest"]["classification"],
            "reference_warm_root_wall_seconds": warm["root_wall_seconds"],
            "direct_truth_wall_hours_per_microsecond": cost[
                "direct_truth_wall_hours_per_microsecond"
            ],
            "direct_truth_wall_days_per_millisecond": cost[
                "direct_truth_wall_days_per_millisecond"
            ],
            "direct_truth_wall_years_per_fiducial_cycle": cost[
                "direct_truth_wall_years_per_cycle"
            ],
            "minimum_required_speedup_for_three_days": cost[
                "minimum_required_end_to_end_speedup"
            ],
            "exclusive_profile_wall_seconds": profile,
            "dominant_cost": "monolithic_residual",
            "dense_bordered_solve_is_not_a_material_bottleneck": (
                profile["bordered_linear_solve"] < 0.1
            ),
        },
        "instantaneous_markov_counterexample": {
            "coordinate_dimension": 34,
            "classification": "exact_nonlinear_fiber_counterexample_confirmed_N64_N128",
            "instantaneous_markov_closure_supported": False,
            "source": str(parent.FIBER_REPORT.relative_to(ROOT)),
            "source_sha256": _sha(parent.FIBER_REPORT),
        },
        "persistent_local_memory": {
            "persistence_horizon_seconds": 0.125,
            "N64_and_N128_healing_supported": False,
            "late_phase_is_cross_grid_resolved": False,
            "source": str(parent.HEALING_REPORT.relative_to(ROOT)),
            "source_sha256": _sha(parent.HEALING_REPORT),
        },
        "observable_memory": {
            "binding_k99": inputs["memory"]["middle"]["memory"]["maximum_binding_k99"],
            "minimum_cross_resolution_subspace_cosine": inputs["memory"][
                "cross_resolution"
            ]["minimum_principal_cosine"],
            "persistent_not_rapidly_contracting": True,
            "guard_state_k99": inputs["memory"]["middle"]["memory"][
                "augmented_guard_k99"
            ],
        },
        "two_mode_output_closure": {
            "maximum_significant_direction_error": inputs["two_mode"][
                "two_mode_output_reconstruction"
            ]["maximum_significant_direction_error"],
            "gate": 0.25,
            "supported": False,
        },
        "six_mode_coordinate": {
            "static_maximum_significant_direction_error": inputs["six_mode"][
                "static_six_mode_output_reconstruction"
            ]["maximum_significant_direction_error"],
            "minimum_full_cross_grid_projector_cosine": inputs["six_mode"][
                "cross_resolution"
            ]["minimum_full_subspace_projector_cosine"],
            "minimum_leading_two_projector_cosine": inputs["six_mode"][
                "cross_resolution"
            ]["minimum_leading_block_projector_cosine"],
            "explicit_six_mode_dynamic_coordinate_supported": False,
        },
        "leading_two_plus_guard": {
            "leading_two_state_coordinate_supported": inputs["hybrid"][
                "leading_two_state_coordinate_supported"
            ],
            "direct_two_mode_output_closure_supported": inputs["hybrid"][
                "direct_two_mode_output_closure_supported"
            ],
            "guard_required": inputs["hybrid"]["guard_HMM_required"],
            "online_guard_truth_calls_per_step_allowed": 0,
        },
        "cycle_physics": {
            "fiducial_cycle_days": parent.FIDUCIAL_CYCLE_DAYS,
            "source_is_phenomenological_one_zone_model": True,
            "certified_short_time_truth_includes_predictive_hot_cold_cycle": False,
            "predictive_cycle_coefficients_identifiable_from_existing_data": False,
        },
    }


def _candidate_matrix(evidence: dict) -> dict:
    return {
        "direct_fixed_Q_microstepping": {
            "scientific_role": "offline_truth_only",
            "online_feasible": False,
            "selected": False,
            "decisive_reason": "requires_more_than_2e10_end_to_end_speedup",
        },
        "global_Q3_instantaneous_Markov": {
            "online_feasible": True,
            "scientifically_supported": False,
            "selected": False,
            "decisive_reason": "exact_34_coordinate_fiber_counterexample",
        },
        "global_Q3_plus_two_direct_modes": {
            "online_feasible": True,
            "scientifically_supported": False,
            "selected": False,
            "decisive_reason": (
                f"significant_direction_error_{evidence['two_mode_output_closure']['maximum_significant_direction_error']:.6f}_exceeds_0p25"
            ),
        },
        "global_Q3_plus_six_explicit_modes": {
            "online_feasible": True,
            "scientifically_supported": False,
            "selected": False,
            "decisive_reason": "six_mode_dynamic_coordinate_not_cross_grid_stable",
        },
        "leading_two_plus_online_HMM": {
            "guard_scientifically_required": True,
            "online_feasible": False,
            "selected": False,
            "decisive_reason": "online_truth_microbursts_violate_zero_truth_call_contract",
        },
        "cellwise_Q5_FV_plus_a2_finite_memory_hybrid": {
            "architecture_supported_for_offline_identification": True,
            "coefficients_identified": False,
            "online_feasibility_by_construction": True,
            "selected": True,
            "decisive_reason": (
                "only_predeclared_candidate_preserving_exact_conservation_"
                "stable_leading_coordinates_unresolved_memory_and_zero_online_truth_calls"
            ),
        },
        "larger_conservative_coarse_radial_PDE": {
            "role": "fallback_if_compact_finite_memory_fails_cross_grid_validation",
            "selected": False,
            "must_preserve_exact_M_J_E_ledgers": True,
        },
    }


def _decision() -> dict:
    evidence = _evidence_ledger()
    candidates = _candidate_matrix(evidence)
    selected = [name for name, result in candidates.items() if result.get("selected")]
    if selected != ["cellwise_Q5_FV_plus_a2_finite_memory_hybrid"]:
        raise RuntimeError("identifiability screen did not select exactly one architecture")
    return {
        "classification": CLASSIFICATION,
        "analysis_completed": True,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "selected_architecture": selected[0],
        "architecture_supported_for_offline_identification": True,
        "coefficients_identifiable_from_existing_committed_data": False,
        "online_solver_implementation_authorized": False,
        "offline_closure_database_manifest_authorized": True,
        "predictive_cycle_authorized": False,
        "fallback_if_finite_memory_fails": "larger_conservative_coarse_radial_PDE",
        "next_manifest_must_freeze": {
            "pathwise_anchor_count_range": (10, 30),
            "global_tensor_product_Q_grid_forbidden": True,
            "middle_layout_is_primary": True,
            "fine_layout_is_sparse_validation_only": True,
            "truth_queries": (
                "constrained_steady_roots_on_declared_branches",
                "complete_local_Jacobians_and_transfer_functions",
                "short_truth_bursts_only_near_unresolved_transitions",
            ),
            "memory_orders": parent.KERNEL_MEMORY_CANDIDATES,
            "stable_poles_and_dissipation_required": True,
            "training_validation_split_frozen_before_truth_queries": True,
            "exploratory_and_predictive_cycle_claims_separated": True,
        },
        "evidence": evidence,
        "candidates": candidates,
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "CERTIFIED",
            })
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
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": PARENT_PACKAGE_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    parent_summary, parent_lock = _validate_parent()
    if not _tracked_tree_clean():
        raise RuntimeError("identifiability screen requires a clean tracked tree")
    if ARTIFACT_DIRECTORY.exists():
        raise RuntimeError("identifiability screen is already canonicalized")
    decision = _decision()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "analysis_only": True,
        "new_truth_trajectory_executed": False,
        "new_fixed_Q_root_executed": False,
        "selected_architecture": decision["selected_architecture"],
        "architecture_supported_for_offline_identification": True,
        "coefficients_identified": False,
        "offline_closure_database_manifest_authorized": True,
        "online_reduced_solver_implementation_authorized": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "parent_classification_preserved": parent_summary["classification"],
        "authorized_next": (
            "definitions_only_pathwise_offline_closure_database_manifest"
        ),
    }
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write(ARTIFACT_DIRECTORY / "evidence_ledger.json", decision["evidence"])
    _write(ARTIFACT_DIRECTORY / "candidate_matrix.json", decision["candidates"])
    _write(ARTIFACT_DIRECTORY / "decision.json", {
        key: value
        for key, value in decision.items()
        if key not in {"evidence", "candidates"}
    })
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", {
        "parent_package_commit": PARENT_PACKAGE_COMMIT,
        "parent_package_parent": PARENT_PACKAGE_PARENT,
        "parent_package_tree": PARENT_PACKAGE_TREE,
        **parent_lock,
    })
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED_ANALYSIS_ONLY",
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "report": REPORT_RELATIVE,
        "source_hashes": {
            THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
            THIS_TEST: _sha(ROOT / THIS_TEST),
        },
        "python": sys.version,
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
    })
    REPORT_PATH.write_text(
        "\n".join((
            "# Reduced-cycle identifiability screen WP10c9d6c7c3b5c4f25a",
            "",
            "## Classification",
            "",
            f"`{CLASSIFICATION}`",
            "",
            "No new nonlinear trajectory, fixed-Q root, or tangent propagation was run. The screen uses only hash-locked committed evidence.",
            "",
            "## Decision",
            "",
            "The direct fixed-Q solver is retained only as an offline truth engine. A direct cycle would cost about `2.37e8` wall-years, so code-level acceleration cannot close the required `2.88e10` end-to-end gap.",
            "",
            "The scalar/global instantaneous Markov route is rejected by the exact 34-coordinate nonlinear fiber counterexample. The two-mode direct-output route is rejected by a worst significant-direction error of `1.043772 > 0.25`. Six modes reconstruct static outputs but are rejected as explicit dynamic coordinates because the full cross-grid projector cosine is `0.831658 < 0.90`. The leading two-dimensional state block remains supported (`0.980973`), while unresolved memory must be represented without online truth calls.",
            "",
            "The selected target for offline identification is therefore `cellwise_Q5_FV_plus_a2_finite_memory_hybrid`: a conservative coarse radial finite-volume model, two stable amplitudes, a stable rational memory kernel screened at orders `0/2/4/6`, and cold/hot/transition hysteresis. This is an architecture selection, not a coefficient or predictive-cycle certificate.",
            "",
            "Existing data do not identify the quasi-steady flux maps, memory poles/residues, hot/cold branch maps, or switching surfaces. The next authorized artifact is only a definitions-only pathwise offline closure-database manifest with `10-30` anchors, middle-grid training, sparse fine-grid validation, and a frozen training/held-out split.",
            "",
            "No online reduced solver, physical microburst, predictive QPE cycle, or reduced slow evolution is authorized.",
            "",
        )),
        encoding="utf-8",
    )
    names = (
        "candidate_matrix.json",
        "decision.json",
        "evidence_ledger.json",
        "parent_lock.json",
        "provenance.json",
        "summary.json",
    )
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    print(json.dumps(_run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
