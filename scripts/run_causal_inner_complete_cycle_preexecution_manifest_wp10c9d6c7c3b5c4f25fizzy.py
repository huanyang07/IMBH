#!/usr/bin/env python3
"""Freeze the final reduced-hybrid complete-cycle pre-execution contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_reduced_hybrid_cycle_kernel_certificate_wp10c9d6c7c3b5c4f25fizzx1 as parent  # noqa: E402


WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = (
    "reduced_hybrid_complete_cycle_preexecution_architecture_frozen_"
    "external_physics_blocked"
)
AUTHORIZED_NEXT = (
    "physical_WP10c9d6c7c3b5c4f25fizzy1_"
    "complete_cycle_preexecution_readiness_certificate"
)
ARTIFACT = "causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzy"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_COMPLETE_CYCLE_PREEXECUTION_MANIFEST_WP10C9D6C7C3B5C4F25FIZZY_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzy.py"
THIS_TEST = "tests/test_causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzy.py"
PARENT_SHA256 = "6c111eb534be10e9919e8978711e21667e4e92b8eaaedb0cde47481131afe8fb"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return parent._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    checksum = utility._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
    if checksum != PARENT_SHA256:
        raise RuntimeError("reduced hybrid cycle-kernel certificate changed")
    hashes = utility._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utility._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utility._read_json(parent.CANONICAL_DIRECTORY / "kernel_metrics.json")
    if (
        not summary["passed"]
        or not summary["reduced_hybrid_cycle_kernel_certified"]
        or not summary["production_adapter_structure_certified"]
        or not summary["online_cost_model_certified_on_synthetic_fixture"]
        or not summary["synthetic_fixture_only"]
        or summary["physical_model_complete"]
        or summary["physical_payloads_acquired"]
        or summary["heldout_physical_validation_complete"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["complete_cycle_steps"] != 0
    ):
        raise RuntimeError("reduced hybrid cycle-kernel classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("complete-cycle preexecution manifest needs a clean tracked tree")
    return hashes, metrics


def _contract(parent_metrics):
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "supersedes": {
            "artifact": "causal_inner_complete_cycle_preexecution_manifest_wp10c9d6c7c3b5c4f25fizzn",
            "reason": (
                "subsequent structure certificates selected a five-coordinate "
                "normally-hyperbolic hybrid slow atlas instead of evolving all "
                "1232 fast coordinates online"
            ),
        },
        "selected_mathematical_architecture": {
            "name": "structure-preserving normally-hyperbolic reduced hybrid cycle",
            "online_state": "y=(Q1,Q2,Q3,Q4,unwrapped_phase) plus one discrete mode",
            "smooth_dynamics": (
                "Qdot=C*b(Q,phase,mode), phidot=omega(phase); all values are "
                "convex interpolants of a hash-locked physical driver atlas"
            ),
            "algebraic_fast_reconstruction": (
                "z_star(Q,phase,mode) is reconstructed from a mode-pure branch "
                "simplex and corrected only in the certified minimum-norm "
                "conservation normal"
            ),
            "normal_hyperbolicity": (
                "every accepted endpoint re-audits positive stable spectral gap, "
                "four-coordinate source kernel, symmetric radial ports, and 0/11 "
                "inner/outer characteristic counts"
            ),
            "hybrid_events": (
                "oriented codimension-one guard sheets are localized with dense "
                "output; duration, phase advance, ledger impulse, constitutive "
                "reset, and destination mode are one atomic transition"
            ),
            "online_integrator": "adaptive Dormand-Prince 5(4) with cubic-Hermite event localization",
            "online_prohibitions": {
                "truth_residual_calls": 0,
                "fixed_Q_microsteps": 0,
                "large_nonlinear_roots": 0,
                "unlocked_extrapolation": 0,
                "nearest_neighbor_fallback": 0,
            },
        },
        "already_certified": {
            "four_invariant_conservation_geometry": True,
            "native_1232_state_global_entropy_boundary_action": True,
            "convex_driver_interpolation_and_ledger_closure": True,
            "branch_invariant_correction_and_structure_preservation": True,
            "five_vertex_event_guard_sheet_geometry": True,
            "finite_duration_phase_advancing_conservative_resets": True,
            "fifth_order_smooth_reduced_integrator": True,
            "two_event_hybrid_sequence_and_bitwise_restart": True,
            "production_adapter_code_path_and_endpoint_reaudit": True,
            "fail_closed_synthetic_production_rejection": True,
            "synthetic_100000_step_cost_projection_wall_days": parent_metrics[
                "projected_100000_step_wall_days"
            ],
        },
        "required_external_physical_evidence": {
            "physical_cycle_bundle_v2": [
                "complete nonsynthetic model metadata and source citations",
                "canonical SHA-256 for metadata, driver, branch, events, heldout truth, and topology",
                "full [0,2*pi] driver coverage for every physical mode",
                "overlapping mode-pure branch simplices over the entire prospective Q tube",
                "positive trust radii and stable spectral gaps at every anchor",
                "physical inner excision and outer incoming characteristic data",
                "outgoing transition guard sheets with direction and class",
                "event duration, integrated phase advance, ledger impulse, constitutive jump, and destination margin",
            ],
            "independent_spatial_holdout": (
                "at least one independently resolved >=112-cell truth package not "
                "used for fitting"
            ),
            "independent_sequence_or_cycle_holdout": (
                "a prospectively withheld event sequence or full cycle with exact "
                "mode order, event times, endpoint, port action, and ledger truth"
            ),
            "physical_runtime_benchmark": (
                "at least 1000 driver, branch, guard, and reset queries on the exact "
                "production payload and machine/thread configuration"
            ),
            "initial_condition": (
                "one hash-locked physical Q, phase, mode, transition inventory, and "
                "checkpoint provenance"
            ),
        },
        "heldout_readiness_gates": {
            "maximum_branch_state_relative_defect": 2.0e-2,
            "maximum_branch_rate_relative_defect": 5.0e-2,
            "maximum_port_action_relative_defect": 5.0e-2,
            "maximum_event_time_relative_defect": 2.0e-2,
            "maximum_event_post_state_relative_defect": 5.0e-2,
            "maximum_event_ledger_relative_defect": 2.0e-2,
            "maximum_sequence_endpoint_relative_defect": 5.0e-2,
            "maximum_sequence_ledger_relative_defect": 2.0e-2,
            "discrete_modes_and_event_order_exact": True,
            "restart_suffix_replay_bitwise": True,
        },
        "single_cycle_execution_contract": {
            "terminal_condition": "unwrapped phase equals initial phase plus 2*pi",
            "physical_cycle_seconds": 578880.0,
            "maximum_online_macrosteps": 100000,
            "required_average_physical_seconds_per_macrostep": 5.7888,
            "maximum_wall_days": 3.0,
            "relative_tolerance": 1.0e-8,
            "absolute_tolerance": {
                "Q_coordinates": "1e-10 times the four physical Q scales",
                "unwrapped_phase": 1.0e-10,
            },
            "maximum_step_ratio": 2.0,
            "checkpoint_interval_accepted_steps": 100,
            "required_cycle_outputs": [
                "every accepted reduced checkpoint and endpoint structure audit",
                "every localized event entry/exit and atomic reset record",
                "cumulative smooth, boundary, distributed, and event ledgers",
                "minimum atlas barycentric weight, trust margin, spectral gap, and guard margin",
                "mode/event sequence and final phase defect",
                "wall profile and physical seconds per wall second",
                "bitwise suffix replay from at least three cycle checkpoints",
            ],
            "binding_acceptance": [
                "phase advances exactly one cycle within tolerance",
                "discrete mode returns to the declared cycle mode",
                "event order matches the physical heldout inventory",
                "every query remains inside a certified hull and trust region",
                "every endpoint passes invariant, source, port, boundary, and physical guards",
                "Q change equals cumulative smooth plus event ledger within 2e-10 relative",
                "no truth call, microstep, large nonlinear root, extrapolation, or fallback occurs",
                "wall time and step count remain inside the frozen budget",
                "checkpoint hashes close and suffix replays are bitwise",
            ],
            "important_non_gate": (
                "Q is a slow evolving state and is not required to return to its "
                "initial value after one physical cycle"
            ),
        },
        "current_readiness": {
            "mathematical_architecture_verified": True,
            "production_adapter_structure_verified": True,
            "physical_cycle_bundle_v2_acquired": False,
            "physical_payload_hashes_complete": False,
            "independent_spatial_holdout_complete": False,
            "independent_sequence_or_cycle_holdout_complete": False,
            "physical_runtime_benchmark_complete": False,
            "physical_initial_condition_locked": False,
            "complete_cycle_execution_ready": False,
        },
        "decision": {
            "status": "blocked_before_complete_cycle_execution",
            "blocking_condition": (
                "the external nonsynthetic physical driver/branch/event bundle, "
                "independent heldouts, and production runtime benchmark do not exist"
            ),
            "required_next_artifact": AUTHORIZED_NEXT,
            "complete_cycle_runner_command_available": False,
            "complete_cycle_execution_authorized": False,
            "complete_cycle_steps": 0,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utility._sha256(path),
                    "scientific_status": "SUPPORTED",
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
    catalog = utility._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": CLASSIFICATION,
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": utility._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _freeze():
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("final complete-cycle preexecution manifest already exists")
    hashes, metrics = _validate_parent(require_clean=True)
    utility = _u()
    contract = _contract(metrics)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "preexecution_contract.json", contract)
    utility._write_json(
        CANONICAL_DIRECTORY / "readiness_status.json", contract["current_readiness"]
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "supersedes_legacy_complete_cycle_preexecution_manifest": True,
        "mathematical_architecture_verified": True,
        "production_adapter_structure_verified": True,
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "heldout_physical_validation_complete": False,
        "complete_cycle_execution_ready": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "blocking_condition": contract["decision"]["blocking_condition"],
        "required_next_artifact": AUTHORIZED_NEXT,
        "authorized_next": AUTHORIZED_NEXT,
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_SHA256,
            "parent_hashes": hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Final reduced-hybrid complete-cycle pre-execution contract\n\n"
        f"Classification: `{CLASSIFICATION}`.\n\n"
        "The selected online architecture is now the five-coordinate, normally-"
        "hyperbolic reduced hybrid system: four conserved slow variables and "
        "unwrapped phase are integrated; the 1232-state fast branch is reconstructed "
        "algebraically and structurally re-audited; oriented event sheets apply "
        "finite-duration conservative resets atomically. Its code path, restart, "
        "event sequence, and synthetic online cost have been certified.\n\n"
        "Complete-cycle execution remains blocked for one concrete reason: no "
        "nonsynthetic cycle-wide physical driver/branch/event bundle, independent "
        "spatial and sequence holdouts, or production-payload runtime benchmark has "
        "been supplied. The contract records the exact evidence and thresholds that "
        "must pass next. No complete-cycle command exists and no cycle step was run.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {
                name: utility._sha256(ROOT / name) for name in sources
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
