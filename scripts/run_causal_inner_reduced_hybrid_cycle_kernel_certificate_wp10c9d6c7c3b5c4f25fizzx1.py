#!/usr/bin/env python3
"""Certify the reduced-hybrid production adapter and online cost model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cycle_physical_driver_branch_and_event_interpolator_structure_certificate_wp10c9d6c7c3b5c4f25fizzv1 as atlas_certificate  # noqa: E402
import run_causal_inner_reduced_hybrid_cycle_kernel_manifest_wp10c9d6c7c3b5c4f25fizzx as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_cycle_kernel import (  # noqa: E402
    CycleAtlasKernel,
    CycleKernelCheckpoint,
    CycleKernelTransitionSpec,
    integrate_cycle_kernel,
    load_cycle_kernel_checkpoint,
    require_production_cycle_metadata,
    save_cycle_kernel_checkpoint,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_reduced_hybrid_cycle import (  # noqa: E402
    ReducedHybridCheckpoint,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "reduced_hybrid_cycle_kernel_structure_and_cost_certified_synthetic_fixture_only"
)
FAIL_CLASSIFICATION = "reduced_hybrid_cycle_kernel_or_cost_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = "causal_inner_reduced_hybrid_cycle_kernel_certificate_wp10c9d6c7c3b5c4f25fizzx1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_REDUCED_HYBRID_CYCLE_KERNEL_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZX1_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_reduced_hybrid_cycle_kernel_certificate_wp10c9d6c7c3b5c4f25fizzx1.py"
THIS_TEST = "tests/test_causal_inner_reduced_hybrid_cycle_kernel_certificate_wp10c9d6c7c3b5c4f25fizzx1.py"
PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_cycle_kernel.py"
PHYSICAL_TEST = "tests/test_causal_inner_cycle_kernel.py"
PARENT_SHA256 = "55c57e9a7d4d4329977494b0726e59f75e46f2dc18999dc740bd6597d7090ae6"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    checksum = utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
    if checksum != PARENT_SHA256:
        raise RuntimeError("cycle kernel manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "cycle_kernel_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["reduced_hybrid_cycle_kernel_certified"]
        or summary["physical_model_complete"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["prefix_and_cost_certificate"]["complete_cycle_steps"] != 0
    ):
        raise RuntimeError("cycle kernel contract changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle kernel certificate needs a clean tracked tree")
    return hashes, contract


def _fixture():
    _, source_branch, _, _, conservation, normal = atlas_certificate._fixture()
    q_scales = np.asarray((1.0e-3, 1.2e-3, 0.8e-3, 1.5e-3))
    q0 = 0.02 * q_scales
    phase_scale = 1.0

    q_nodes = np.asarray([q0] + [q0 + 0.2 * q_scales * np.eye(4)[i] for i in range(4)])
    q_simplices = np.asarray(((0, 1, 2, 3, 4),), dtype=int)
    phase_nodes = np.linspace(0.0, 2.0 * np.pi, 9)
    driver_shape = (len(phase_nodes), len(q_nodes), 2)
    forcing = np.zeros(driver_shape + (1232,))
    ledgers = np.zeros(driver_shape + (4,))
    incoming = np.zeros(driver_shape + (11,))
    driver = {
        "phase_nodes": phase_nodes,
        "phase_rate_per_second": np.ones(len(phase_nodes)),
        "retained_invariant_nodes4": q_nodes,
        "mode_labels": np.asarray(("cold_fixture", "hot_fixture")),
        "slow_forcing1232_per_second": forcing,
        "distributed_source_ledger_rate4": ledgers,
        "boundary_ledger_rate4": ledgers.copy(),
        "outer_incoming_characteristics11": incoming,
    }

    segments = ((0, 0.0, 1.0), (1, 1.1, 2.0), (0, 2.1, 2.7))
    anchor_q = []
    anchor_phase = []
    anchor_mode = []
    branch_simplices = []
    branch_simplex_modes = []
    for mode, phase_start, phase_end in segments:
        start = len(anchor_q)
        anchor_q.append(q0.copy())
        anchor_phase.append(phase_start)
        anchor_mode.append(mode)
        for coordinate in range(4):
            anchor_q.append(q0 + 0.1 * q_scales * np.eye(4)[coordinate])
            anchor_phase.append(phase_start)
            anchor_mode.append(mode)
        anchor_q.append(q0.copy())
        anchor_phase.append(phase_end)
        anchor_mode.append(mode)
        branch_simplices.append(np.arange(start, start + 6))
        branch_simplex_modes.append(mode)
    anchor_q = np.asarray(anchor_q)
    anchor_count = len(anchor_q)
    states = np.asarray([normal @ value for value in anchor_q])
    radial_template = source_branch["radial_matrices112x11x11"][0]
    source_template = source_branch["source_matrices112x11x11"][0]
    branch = {
        "anchor_states1232": states,
        "anchor_phase": np.asarray(anchor_phase),
        "anchor_invariants4": anchor_q,
        "anchor_mode_index": np.asarray(anchor_mode),
        "radial_matrices112x11x11": np.repeat(radial_template[None], anchor_count, axis=0),
        "source_matrices112x11x11": np.repeat(source_template[None], anchor_count, axis=0),
        "forcing1232_per_second": np.zeros((anchor_count, 1232)),
        "trust_radii": np.full(anchor_count, 2.0),
        "stable_spectral_gaps_per_second": np.full(anchor_count, 0.4),
        "guard_margins": np.full((anchor_count, 2), 0.5),
    }

    event_q = []
    event_phase = []
    event_class = []
    event_source = []
    event_destination = []
    event_simplices = []
    for transition_class, (phase, source, destination) in enumerate(
        ((1.0, 0, 1), (2.0, 1, 0))
    ):
        start = len(event_q)
        event_q.append(q0.copy())
        event_phase.append(phase)
        event_class.append(transition_class)
        event_source.append(source)
        event_destination.append(destination)
        for coordinate in range(4):
            event_q.append(q0 + 0.1 * q_scales * np.eye(4)[coordinate])
            event_phase.append(phase)
            event_class.append(transition_class)
            event_source.append(source)
            event_destination.append(destination)
        event_simplices.append(np.arange(start, start + 5))
    event_q = np.asarray(event_q)
    event_count = len(event_q)
    guard_normals = np.zeros((event_count, 5))
    guard_normals[:, 4] = 1.0
    events = {
        "pre_states1232": np.asarray([normal @ value for value in event_q]),
        "pre_invariants4": event_q,
        "phase": np.asarray(event_phase),
        "source_mode_index": np.asarray(event_source),
        "destination_mode_index": np.asarray(event_destination),
        "transition_class_index": np.asarray(event_class),
        "duration_seconds": np.full(event_count, 0.1),
        "integrated_phase_advance": np.full(event_count, 0.1),
        "destination_guard_margin": np.full(event_count, 0.2),
        "integrated_ledger_impulse4": np.zeros((event_count, 4)),
        "ledger_null_constitutive_jump1232": np.zeros((event_count, 1232)),
        "reduced_guard_normals5": guard_normals,
        "reduced_guard_offsets": -np.asarray(event_phase) / phase_scale,
    }
    additions = {
        "q_simplices": q_simplices,
        "q_scales": q_scales,
        "branch_simplices": np.asarray(branch_simplices),
        "branch_simplex_modes": np.asarray(branch_simplex_modes),
        "phase_scale": phase_scale,
        "event_simplices": np.asarray(event_simplices),
        "event_simplex_classes": np.asarray((0, 1)),
    }
    metadata = {
        "synthetic_fixture": True,
        "physical_model_complete": False,
        "physical_payload_hashes_complete": False,
        "heldout_physical_validation_complete": False,
        "independent_spatial_holdout_complete": False,
        "independent_sequence_or_cycle_holdout_complete": False,
        "physical_bundle_sha256": "synthetic-fixture",
    }
    specs = (
        CycleKernelTransitionSpec("cold_to_hot", 0, 0, 1, 1),
        CycleKernelTransitionSpec("hot_to_cold", 1, 1, 0, 1),
    )
    kernel = CycleAtlasKernel(
        metadata,
        driver,
        branch,
        events,
        additions,
        conservation,
        normal,
        specs,
        require_physical=False,
    )
    initial = ReducedHybridCheckpoint(
        np.concatenate((q0, [0.0])),
        0.0,
        0,
        1.0e-6,
        np.zeros(4),
        np.zeros(4),
        0,
        0,
        0,
    )
    return kernel, initial


def _checkpoint_equal(left: ReducedHybridCheckpoint, right: ReducedHybridCheckpoint) -> bool:
    return bool(
        np.array_equal(left.state5, right.state5)
        and left.time_seconds == right.time_seconds
        and left.mode_index == right.mode_index
        and left.next_timestep_seconds == right.next_timestep_seconds
        and np.array_equal(left.cumulative_smooth_ledger4, right.cumulative_smooth_ledger4)
        and np.array_equal(left.cumulative_event_ledger4, right.cumulative_event_ledger4)
        and left.accepted_steps == right.accepted_steps
        and left.rejected_steps == right.rejected_steps
        and left.completed_events == right.completed_events
    )


def _mean_time(callable_, repetitions: int) -> float:
    began = time.perf_counter()
    for _ in range(int(repetitions)):
        callable_()
    return (time.perf_counter() - began) / int(repetitions)


def _certificate():
    began = time.perf_counter()
    _, contract = _validate_parent()
    kernel, initial = _fixture()
    tolerances = np.full(5, 1.0e-10)
    prefix = integrate_cycle_kernel(
        kernel,
        initial,
        end_time_seconds=2.5,
        absolute_tolerance=tolerances,
        relative_tolerance=1.0e-9,
        maximum_accepted_steps=128,
    )
    after_first = next(
        value
        for value in prefix.reduced.accepted_checkpoints
        if value.completed_events == 1
    )
    contract_digest = "f" * 64
    wrapped = CycleKernelCheckpoint(
        after_first, kernel.physical_bundle_sha256, contract_digest
    )
    with tempfile.TemporaryDirectory(prefix="cycle_kernel_checkpoint_") as directory:
        path = Path(directory) / "checkpoint.npz"
        io_start = time.perf_counter()
        save_cycle_kernel_checkpoint(wrapped, path)
        write_seconds = time.perf_counter() - io_start
        io_start = time.perf_counter()
        loaded = load_cycle_kernel_checkpoint(
            path,
            expected_physical_bundle_sha256=kernel.physical_bundle_sha256,
            expected_kernel_contract_sha256=contract_digest,
        )
        read_seconds = time.perf_counter() - io_start
        checkpoint_bitwise = _checkpoint_equal(loaded.reduced, after_first)
        try:
            load_cycle_kernel_checkpoint(
                path,
                expected_physical_bundle_sha256="wrong",
                expected_kernel_contract_sha256=contract_digest,
            )
        except ValueError:
            hash_mismatch_rejected = True
        else:
            hash_mismatch_rejected = False
    replay_kernel, _ = _fixture()
    replay = integrate_cycle_kernel(
        replay_kernel,
        loaded.reduced,
        end_time_seconds=2.5,
        absolute_tolerance=tolerances,
        relative_tolerance=1.0e-9,
        maximum_accepted_steps=128,
    )
    suffix_bitwise = _checkpoint_equal(replay.reduced.checkpoint, prefix.reduced.checkpoint)

    query_state = initial.state5.copy()
    query_state[4] = 0.5
    event_state = initial.state5.copy()
    event_state[4] = 1.0
    repetitions = int(contract["prefix_and_cost_certificate"]["minimum_benchmark_queries"])
    driver_seconds = _mean_time(lambda: kernel.rhs(0.0, query_state, 0), repetitions)
    branch_seconds = _mean_time(lambda: kernel.branch_value(query_state, 0), repetitions)
    guard_seconds = _mean_time(lambda: kernel.guard_value(query_state, 0), repetitions)
    reset_seconds = _mean_time(
        lambda: kernel.event_reset(1.0, event_state, kernel.transition_specs[0]),
        repetitions,
    )
    checkpoint_seconds_per_step = (write_seconds + read_seconds) / 100.0
    step_seconds = (
        contract["prefix_and_cost_certificate"]["assumed_rhs_queries_per_step"]
        * driver_seconds
        + contract["prefix_and_cost_certificate"]
        ["assumed_endpoint_branch_queries_per_step"]
        * branch_seconds
        + guard_seconds
        + (2.0 / 100000.0) * reset_seconds
        + checkpoint_seconds_per_step
    )
    projected_steps = contract["cost_projection"]["maximum_online_macrosteps"]
    projected_wall_seconds = projected_steps * step_seconds
    projected_wall_days = projected_wall_seconds / 86400.0

    fake_production = dict(kernel.metadata)
    fake_production.update(
        {
            "physical_model_complete": True,
            "physical_payload_hashes_complete": True,
            "heldout_physical_validation_complete": True,
            "independent_spatial_holdout_complete": True,
            "independent_sequence_or_cycle_holdout_complete": True,
        }
    )
    try:
        require_production_cycle_metadata(fake_production)
    except ValueError:
        synthetic_production_rejected = True
    else:
        synthetic_production_rejected = False

    gates = contract["prefix_and_cost_certificate"]
    event_names = [value.name for value in prefix.reduced.events]
    minimum_weight = min(value.minimum_barycentric_weight for value in prefix.endpoint_audits)
    max_invariant = max(value.invariant_relative_defect for value in prefix.endpoint_audits)
    max_symmetry = max(value.maximum_radial_symmetry_defect for value in prefix.endpoint_audits)
    max_source_positive = max(
        value.maximum_source_entropy_positive_eigenvalue
        for value in prefix.endpoint_audits
    )
    min_nullity = min(value.minimum_source_nullity for value in prefix.endpoint_audits)
    min_gap = min(value.fast_spectral_gap_per_second for value in prefix.endpoint_audits)
    boundary_counts = sorted(
        {(value.inner_incoming_count, value.outer_incoming_count) for value in prefix.endpoint_audits}
    )
    passed = bool(
        len(prefix.reduced.accepted_checkpoints) >= gates["minimum_prefix_accepted_steps"]
        and len(prefix.reduced.events) >= gates["minimum_prefix_events"]
        and len(prefix.endpoint_audits) >= gates["minimum_endpoint_structure_audits"]
        and event_names == ["cold_to_hot", "hot_to_cold"]
        and prefix.reduced_ledger_relative_defect
        <= gates["maximum_prefix_ledger_relative_defect"]
        and minimum_weight >= -2.0e-12
        and max_invariant <= 2.0e-12
        and max_symmetry <= 2.0e-12
        and max_source_positive <= 2.0e-12
        and min_nullity >= 4
        and min_gap > 0.0
        and boundary_counts == [(0, 11)]
        and checkpoint_bitwise
        and suffix_bitwise
        and hash_mismatch_rejected
        and synthetic_production_rejected
        and projected_wall_days <= gates["maximum_projected_100000_step_wall_days"]
    )
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "synthetic_fixture_only": True,
        "prefix_accepted_steps": len(prefix.reduced.accepted_checkpoints),
        "prefix_rejected_steps": prefix.reduced.checkpoint.rejected_steps,
        "prefix_event_count": len(prefix.reduced.events),
        "prefix_event_names": event_names,
        "endpoint_structure_audit_count": len(prefix.endpoint_audits),
        "minimum_endpoint_barycentric_weight": minimum_weight,
        "maximum_endpoint_invariant_relative_defect": max_invariant,
        "maximum_endpoint_radial_symmetry_defect": max_symmetry,
        "maximum_endpoint_source_positive_eigenvalue": max_source_positive,
        "minimum_endpoint_source_nullity": min_nullity,
        "minimum_endpoint_fast_spectral_gap_per_second": min_gap,
        "endpoint_boundary_incoming_counts": [list(value) for value in boundary_counts],
        "prefix_reduced_ledger_relative_defect": prefix.reduced_ledger_relative_defect,
        "checkpoint_roundtrip_bitwise": checkpoint_bitwise,
        "restart_suffix_replay_bitwise": suffix_bitwise,
        "checkpoint_hash_mismatch_rejected": hash_mismatch_rejected,
        "synthetic_production_metadata_rejected": synthetic_production_rejected,
        "benchmark_queries_per_operation": repetitions,
        "mean_driver_rhs_query_wall_seconds": driver_seconds,
        "mean_branch_endpoint_query_wall_seconds": branch_seconds,
        "mean_guard_query_wall_seconds": guard_seconds,
        "mean_event_reset_query_wall_seconds": reset_seconds,
        "checkpoint_write_wall_seconds": write_seconds,
        "checkpoint_read_wall_seconds": read_seconds,
        "projected_100000_step_wall_seconds": projected_wall_seconds,
        "projected_100000_step_wall_days": projected_wall_days,
        "projected_physical_seconds_per_wall_second": (
            contract["cost_projection"]["fiducial_period_seconds"]
            / projected_wall_seconds
        ),
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "heldout_physical_validation_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "certificate_wall_seconds": time.perf_counter() - began,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "prefix_final_state5": prefix.reduced.checkpoint.state5,
        "replay_final_state5": replay.reduced.checkpoint.state5,
        "event_entry_states5": np.asarray(
            [value.entry_state5 for value in prefix.reduced.events]
        ),
        "event_exit_states5": np.asarray(
            [value.exit_state5 for value in prefix.reduced.events]
        ),
        "endpoint_audit_times_seconds": np.asarray(
            [value.time_seconds for value in prefix.endpoint_audits]
        ),
    }
    return metrics, arrays


def _update(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utility._sha256(path),
                    "scientific_status": status,
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
        "classification": summary["classification"],
        "passed": summary["passed"],
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


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cycle kernel certificate already exists")
    hashes, _ = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "kernel_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "kernel_arrays.npz", **arrays)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "reduced_hybrid_cycle_kernel_certified": metrics["passed"],
        "production_adapter_structure_certified": metrics["passed"],
        "online_cost_model_certified_on_synthetic_fixture": metrics["passed"],
        "synthetic_fixture_only": True,
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "heldout_physical_validation_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": metrics["authorized_next"],
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_artifact": manifest.ARTIFACT,
            "manifest_checksum_manifest_sha256": PARENT_SHA256,
            "manifest_hashes": hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Reduced hybrid cycle-kernel certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"The adapter advanced `{metrics['prefix_accepted_steps']}` accepted "
        f"synthetic prefix steps through `{metrics['prefix_event_count']}` finite "
        "events, re-audited every endpoint, closed the reduced ledger, and replayed "
        "a restarted suffix bitwise. Production metadata rejects this fixture. "
        f"The measured online projection is `{metrics['projected_100000_step_wall_days']:.6e}` "
        "wall days for 100,000 macrosteps.\n\n"
        "This is a code-path and cost certificate, not physical calibration. The "
        "physical bundle and independent heldouts remain absent; no complete-cycle "
        "step occurred.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {
                name: utility._sha256(ROOT / name) for name in sources
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
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
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
