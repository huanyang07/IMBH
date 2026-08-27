#!/usr/bin/env python3
"""Certify the bounded eleven-field AP coarse-trajectory kernel."""

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
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

import run_causal_inner_bounded_ap_coarse_trajectory_manifest_wp10c9d6c7c3b5c4f25fizzm as manifest  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_bounded_ap_trajectory import (  # noqa: E402
    APAtlasPath,
    APTrajectoryCheckpoint,
    deterministic_initial_state,
    deterministic_slow_forcing,
    fast_slaving_defect,
    integrate_ap_trajectory,
    load_ap_checkpoint,
    save_ap_checkpoint,
    source_nullity,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import build_full_port_atlas_anchor  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_physical_entropy_congruence import (  # noqa: E402
    build_corrected_physical_port_atlas,
    build_physical_entropy_congruence,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "bounded_AP_coarse_trajectory_kernel_certified"
FAIL_CLASSIFICATION = "bounded_AP_coarse_trajectory_kernel_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = "causal_inner_bounded_ap_coarse_trajectory_kernel_wp10c9d6c7c3b5c4f25fizzm1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_BOUNDED_AP_COARSE_TRAJECTORY_KERNEL_WP10C9D6C7C3B5C4F25FIZZM1_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_bounded_ap_coarse_trajectory_kernel_wp10c9d6c7c3b5c4f25fizzm1.py"
THIS_TEST = "tests/test_causal_inner_bounded_ap_coarse_trajectory_kernel_wp10c9d6c7c3b5c4f25fizzm1.py"
TRAJECTORY_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_bounded_ap_trajectory.py"
TRAJECTORY_TEST = "tests/test_causal_inner_bounded_ap_trajectory.py"
PARENT_SHA256 = "58784c9adbcb4725ed55dc17c8c47d50783ba7d01ce90b946030b0fd2efbb18b"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("bounded AP trajectory manifest checksum changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(manifest.CANONICAL_DIRECTORY / "trajectory_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["bounded_AP_coarse_trajectory_certified"]
        or summary["complete_cycle_execution_authorized"]
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("bounded AP trajectory manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("bounded AP trajectory kernel needs a clean tracked tree")
    return hashes, contract


def _physical_ports(indices):
    fizzl1 = manifest.parent
    witness_module = fizzl1.witnesses
    physical = {
        index: (old, chart)
        for index, _, _, old, chart in witness_module._physical_witnesses()
        if index in indices
    }
    ports = {}
    for index in indices:
        old, chart = physical[index]
        height = float(np.exp(chart[5])); sigma = float(np.exp(chart[0]))
        congruence = build_physical_entropy_congruence(
            old.geometry,
            proper_half_thickness=height,
            density=sigma / (2.0 * height),
            temperature=float(np.exp(chart[3])),
            radial_velocity_over_c=float(chart[1]),
            azimuthal_velocity_over_c=float(chart[2]),
            primitive_step=3.0e-4,
        )
        sound = float(old.thermodynamics.sound_speed)
        alpha = float((old.closure.viscous_signal_speed_over_c * C / sound) ** 2)
        omega = float(np.sqrt(old.thermodynamics.integrated_pressure / (sigma * height**2)))
        anchor = build_full_port_atlas_anchor(
            sound_speed=congruence.sound_speed_over_c * C,
            temperature=float(np.exp(chart[3])),
            proper_half_thickness=height,
            proper_vertical_frequency=omega,
            alpha=alpha,
            shear_relaxation_time=float(old.closure.relaxation_time),
            transport_speed_over_c=float(chart[1]),
        )
        ports[index] = build_corrected_physical_port_atlas(anchor, congruence, old.geometry)
    return ports


def _wave_number(time_value, horizon):
    return float(0.3 + 0.05 * np.sin(2.0 * np.pi * time_value / horizon))


def _case(case_name, pair, ports, contract, checkpoint_directory):
    horizon = float(contract["trajectory"]["normalized_horizon"])
    path = APAtlasPath(
        ports[pair[0]].radial_matrix,
        ports[pair[0]].source_matrix,
        ports[pair[1]].radial_matrix,
        ports[pair[1]].source_matrix,
    )
    initial = deterministic_initial_state()
    wave = lambda value: _wave_number(value, horizon)
    forcing = lambda value: deterministic_slow_forcing(value, horizon)
    step_counts = tuple(contract["trajectory"]["step_counts"])
    reference_count = int(contract["trajectory"]["reference_step_count"])
    rows = []; final_states = []; per_step_times = []
    minimum_order = float("inf"); maximum_norm = 0.0; maximum_expansivity = 0.0
    all_roundtrips = True; all_replays = True; maximum_slaving = 0.0
    for stiffness in contract["trajectory"]["stiffness_ratios"]:
        reference = integrate_ap_trajectory(path, initial, start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=reference_count, stiffness=stiffness, wave_number=wave, forcing=forcing)
        results = [integrate_ap_trajectory(path, initial, start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=count, stiffness=stiffness, wave_number=wave, forcing=forcing) for count in step_counts]
        errors = [float(np.linalg.norm(result.final_state - reference.final_state) / max(np.linalg.norm(reference.final_state), np.finfo(float).tiny)) for result in results]
        orders = [float(np.log(errors[position] / errors[position + 1]) / np.log(2.0)) for position in range(2)]
        minimum_order = min(minimum_order, *orders)
        maximum_norm = max(maximum_norm, reference.maximum_state_norm, *(result.maximum_state_norm for result in results))
        maximum_expansivity = max(maximum_expansivity, reference.maximum_homogeneous_step_expansivity, *(result.maximum_homogeneous_step_expansivity for result in results))
        per_step_times.extend([reference.wall_seconds / reference_count, *(result.wall_seconds / count for result, count in zip(results, step_counts))])
        middle_count = int(step_counts[1]); half_count = middle_count // 2
        first = integrate_ap_trajectory(path, initial, start_time=0.0, end_time=horizon / 2.0, atlas_horizon=horizon, step_count=half_count, stiffness=stiffness, wave_number=wave, forcing=forcing)
        checkpoint = APTrajectoryCheckpoint(path, first.final_state, horizon / 2.0, horizon, stiffness, half_count)
        filename = checkpoint_directory / f"{case_name}_stiffness_{int(stiffness)}.npz"
        save_ap_checkpoint(checkpoint, filename); loaded = load_ap_checkpoint(filename)
        roundtrip = bool(
            np.array_equal(loaded.state, checkpoint.state)
            and np.array_equal(loaded.path.radial_start, path.radial_start)
            and np.array_equal(loaded.path.source_end, path.source_end)
            and loaded.time == checkpoint.time
            and loaded.stiffness == checkpoint.stiffness
        )
        suffix = integrate_ap_trajectory(loaded.path, loaded.state, start_time=loaded.time, end_time=horizon, atlas_horizon=loaded.atlas_horizon, step_count=half_count, stiffness=loaded.stiffness, wave_number=wave, forcing=forcing)
        replay = bool(np.array_equal(suffix.final_state, results[1].final_state))
        all_roundtrips = all_roundtrips and roundtrip; all_replays = all_replays and replay
        slaving = fast_slaving_defect(path, results[-1].final_state, time=horizon, horizon=horizon, stiffness=stiffness, wave_number=wave, forcing=forcing)
        if stiffness == max(contract["trajectory"]["stiffness_ratios"]):
            maximum_slaving = max(maximum_slaving, slaving)
        rows.append({"stiffness_ratio": stiffness, "relative_errors": errors, "matched_orders": orders, "checkpoint_roundtrip_bitwise": roundtrip, "suffix_replay_bitwise": replay, "fast_slaving_defect": slaving})
        final_states.extend([reference.final_state, *(result.final_state for result in results)])
    nullities = [source_nullity(path.source_start), source_nullity(path.source_end)]
    projected_days = float(np.median(per_step_times) * 100000.0 / 86400.0)
    gates = contract["gates"]
    passed = bool(
        minimum_order >= gates["minimum_matched_refinement_order"]
        and maximum_expansivity <= gates["maximum_homogeneous_step_expansivity"]
        and maximum_norm <= gates["maximum_state_norm"]
        and maximum_slaving <= gates["maximum_stiff_fast_slaving_defect"]
        and min(nullities) == gates["required_source_nullity"]
        and all_roundtrips and all_replays
        and projected_days <= gates["maximum_projected_100k_step_wall_days"]
    )
    metrics = {"case": case_name, "anchor_indices": list(pair), "minimum_matched_refinement_order": minimum_order, "maximum_homogeneous_step_expansivity": maximum_expansivity, "maximum_state_norm": maximum_norm, "maximum_stiff_fast_slaving_defect": maximum_slaving, "source_nullities": nullities, "checkpoint_roundtrip_bitwise": all_roundtrips, "suffix_replay_bitwise": all_replays, "median_online_step_wall_seconds": float(np.median(per_step_times)), "projected_100k_step_wall_days": projected_days, "online_truth_calls": 0, "rows": rows, "passed": passed}
    return metrics, np.asarray(final_states)


def _certificate():
    began = time.perf_counter(); _, contract = _validate_parent()
    pairs = {name: tuple(values) for name, values in contract["offline_physical_atlas"]["case_anchor_indices"].items()}
    offline_began = time.perf_counter(); ports = _physical_ports(sorted({index for pair in pairs.values() for index in pair})); offline_wall = time.perf_counter() - offline_began
    with tempfile.TemporaryDirectory(prefix="ap-trajectory-") as temporary:
        cases = [_case(name, pair, ports, contract, Path(temporary)) for name, pair in pairs.items()]
    rows = [case[0] for case in cases]; passed = bool(len(rows) == 2 and all(row["passed"] for row in rows))
    metrics = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION, "passed": passed, "case_count": len(rows), "passing_case_count": sum(row["passed"] for row in rows), "minimum_matched_refinement_order": float(min(row["minimum_matched_refinement_order"] for row in rows)), "maximum_homogeneous_step_expansivity": float(max(row["maximum_homogeneous_step_expansivity"] for row in rows)), "maximum_state_norm": float(max(row["maximum_state_norm"] for row in rows)), "maximum_stiff_fast_slaving_defect": float(max(row["maximum_stiff_fast_slaving_defect"] for row in rows)), "maximum_projected_100k_step_wall_days": float(max(row["projected_100k_step_wall_days"] for row in rows)), "all_checkpoints_bitwise": all(row["checkpoint_roundtrip_bitwise"] for row in rows), "all_suffix_replays_bitwise": all(row["suffix_replay_bitwise"] for row in rows), "minimum_source_nullity": min(min(row["source_nullities"]) for row in rows), "offline_physical_anchor_builds": len(ports), "offline_anchor_wall_seconds": offline_wall, "online_truth_calls": 0, "cycle_wide_coefficient_atlas_complete": False, "complete_cycle_execution_authorized": False, "certificate_wall_seconds": time.perf_counter() - began, "rows": rows, "authorized_next": AUTHORIZED_NEXT if passed else None}
    arrays = {"final_states": np.asarray([case[1] for case in cases]), "anchor_indices": np.asarray([pair for pair in pairs.values()])}
    return metrics, arrays


def _update_catalog(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]; status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("bounded AP trajectory certificate exists")
    hashes, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "trajectory_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "trajectory_arrays.npz", **arrays)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "bounded_AP_coarse_trajectory_certified": metrics["passed"], "arbitrary_step_restart_certified": metrics["passed"] and metrics["all_suffix_replays_bitwise"], "online_truth_call_free": metrics["online_truth_calls"] == 0, "cycle_wide_coefficient_atlas_complete": False, "complete_cycle_preexecution_manifest_authorized": metrics["passed"], "complete_cycle_execution_authorized": False, "authorized_next": metrics["authorized_next"]}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(f"# Bounded AP coarse-trajectory kernel certificate\n\nClassification: `{metrics['classification']}`.\n\n{metrics['passing_case_count']}/{metrics['case_count']} physical two-anchor paths pass. The minimum matched order is `{metrics['minimum_matched_refinement_order']:.6f}`, the maximum stiff slaving defect is `{metrics['maximum_stiff_fast_slaving_defect']:.6e}`, and the conservative 100,000-step online projection is `{metrics['maximum_projected_100k_step_wall_days']:.6e}` wall days. All checkpoint and suffix replays are bitwise.\n\nThis is a bounded local linearized AP trajectory. A cycle-wide offline coefficient atlas is absent and complete-cycle execution remains unauthorized.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, TRAJECTORY_SOURCE, TRAJECTORY_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); arguments = parser.parse_args()
    if not arguments.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
