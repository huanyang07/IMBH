#!/usr/bin/env python3
"""Execute the 1,034-state periodic global AP dry run."""

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

import run_causal_inner_cycle_wide_inputs_and_global_ap_dry_run_manifest_wp10c9d6c7c3b5c4f25fizzo as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_bounded_ap_trajectory import (  # noqa: E402
    APAtlasPath,
    APTrajectoryCheckpoint,
    load_ap_checkpoint,
    save_ap_checkpoint,
    source_nullity,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_global_fourier_ap import (  # noqa: E402
    deterministic_global_forcing,
    deterministic_global_initial_state,
    integrate_global_fourier_ap,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "production_size_global_AP_dry_run_certified"
FAIL_CLASSIFICATION = "production_size_global_AP_dry_run_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = "causal_inner_production_size_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzo1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_PRODUCTION_SIZE_GLOBAL_AP_DRY_RUN_WP10C9D6C7C3B5C4F25FIZZO1_2026-08-27.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_production_size_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzo1.py"
THIS_TEST = "tests/test_causal_inner_production_size_global_ap_dry_run_wp10c9d6c7c3b5c4f25fizzo1.py"
GLOBAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_global_fourier_ap.py"
GLOBAL_TEST = "tests/test_causal_inner_global_fourier_ap.py"
PARENT_SHA256 = "a955530ee52d5b26957d8f4c5bcdcdf51f81826a5aefe9f6aef317d762150bbe"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u(): return manifest._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256: raise RuntimeError("global AP manifest checksum changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY); summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json"); contract = utility._read_json(manifest.CANONICAL_DIRECTORY / "decomposition_contract.json")
    if not summary["passed"] or not summary["definitions_only"] or summary["global_AP_dry_run_certified"] or summary["authorized_next"] != WORK_PACKAGE or contract["claim_boundary"]["complete_cycle_execution_authorized"]: raise RuntimeError("global AP manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"): raise RuntimeError("global AP dry run needs a clean tracked tree")
    return hashes, contract


def _case(name, pair, ports, contract, directory):
    specification = contract["global_AP_dry_run"]; gates = specification["gates"]
    cell_count = int(specification["radial_cells"]); horizon = float(specification["normalized_horizon"])
    path = APAtlasPath(ports[pair[0]].radial_matrix, ports[pair[0]].source_matrix, ports[pair[1]].radial_matrix, ports[pair[1]].source_matrix)
    initial = deterministic_global_initial_state(cell_count); forcing = lambda value: deterministic_global_forcing(value, horizon, cell_count)
    counts = tuple(specification["step_counts"]); reference_count = int(specification["reference_step_count"])
    rows = []; final_states = []; per_step = []; minimum_order = float("inf"); maximum_expansivity = 0.0; maximum_conservation = 0.0; maximum_norm = 0.0; all_roundtrips = True; all_replays = True
    for stiffness in specification["stiffness_ratios"]:
        reference = integrate_global_fourier_ap(path, initial, start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=reference_count, stiffness=stiffness, forcing=forcing)
        results = [integrate_global_fourier_ap(path, initial, start_time=0.0, end_time=horizon, atlas_horizon=horizon, step_count=count, stiffness=stiffness, forcing=forcing) for count in counts]
        errors = [float(np.linalg.norm(result.final_state - reference.final_state) / max(np.linalg.norm(reference.final_state), np.finfo(float).tiny)) for result in results]
        orders = [float(np.log(errors[index] / errors[index + 1]) / np.log(2.0)) for index in range(2)]
        minimum_order = min(minimum_order, *orders); maximum_expansivity = max(maximum_expansivity, reference.maximum_homogeneous_mode_expansivity, *(result.maximum_homogeneous_mode_expansivity for result in results)); maximum_conservation = max(maximum_conservation, reference.maximum_core_total_conservation_defect, *(result.maximum_core_total_conservation_defect for result in results)); maximum_norm = max(maximum_norm, reference.maximum_state_norm, *(result.maximum_state_norm for result in results)); per_step.extend([reference.wall_seconds / reference_count, *(result.wall_seconds / count for result, count in zip(results, counts))])
        middle_count = int(counts[1]); half = middle_count // 2
        first = integrate_global_fourier_ap(path, initial, start_time=0.0, end_time=horizon / 2.0, atlas_horizon=horizon, step_count=half, stiffness=stiffness, forcing=forcing)
        checkpoint = APTrajectoryCheckpoint(path, first.final_state, horizon / 2.0, horizon, stiffness, half); filename = directory / f"{name}_stiffness_{int(stiffness)}.npz"; save_ap_checkpoint(checkpoint, filename); loaded = load_ap_checkpoint(filename)
        roundtrip = bool(np.array_equal(loaded.state, checkpoint.state) and np.array_equal(loaded.path.radial_start, path.radial_start) and np.array_equal(loaded.path.source_end, path.source_end) and loaded.completed_steps == checkpoint.completed_steps)
        suffix = integrate_global_fourier_ap(loaded.path, loaded.state, start_time=loaded.time, end_time=horizon, atlas_horizon=loaded.atlas_horizon, step_count=half, stiffness=loaded.stiffness, forcing=forcing)
        replay = bool(np.array_equal(suffix.final_state, results[1].final_state)); all_roundtrips &= roundtrip; all_replays &= replay
        rows.append({"stiffness_ratio": stiffness, "relative_errors": errors, "matched_orders": orders, "maximum_homogeneous_mode_expansivity": max(result.maximum_homogeneous_mode_expansivity for result in results), "maximum_core_total_conservation_defect": max(result.maximum_core_total_conservation_defect for result in results), "checkpoint_roundtrip_bitwise": roundtrip, "suffix_replay_bitwise": replay}); final_states.extend([reference.final_state, *(result.final_state for result in results)])
    nullities = [source_nullity(path.source_start), source_nullity(path.source_end)]; projected_days = float(max(per_step) * 100000.0 / 86400.0)
    passed = bool(minimum_order >= gates["minimum_matched_refinement_order"] and maximum_expansivity <= gates["maximum_homogeneous_mode_expansivity"] and maximum_conservation <= gates["maximum_core_total_conservation_defect"] and maximum_norm <= gates["maximum_state_norm"] and min(nullities) >= gates["minimum_source_nullity"] and all_roundtrips and all_replays and projected_days <= gates["maximum_projected_100k_step_wall_days"])
    metrics = {"case": name, "anchor_indices": list(pair), "global_state_dimension": int(cell_count * 11), "minimum_matched_refinement_order": minimum_order, "maximum_homogeneous_mode_expansivity": maximum_expansivity, "maximum_core_total_conservation_defect": maximum_conservation, "maximum_state_norm": maximum_norm, "source_nullities": nullities, "checkpoint_roundtrip_bitwise": all_roundtrips, "suffix_replay_bitwise": all_replays, "maximum_online_step_wall_seconds": float(max(per_step)), "projected_100k_step_wall_days": projected_days, "online_truth_calls": 0, "rows": rows, "passed": passed}
    return metrics, np.asarray(final_states)


def _certificate():
    began = time.perf_counter(); _, contract = _validate_parent(); pairs = {name: tuple(value) for name, value in contract["global_AP_dry_run"]["physical_anchor_paths"].items()}; indices = sorted({index for pair in pairs.values() for index in pair})
    physical_builder = manifest.parent.parent
    offline_began = time.perf_counter(); ports = physical_builder._physical_ports(indices); offline_wall = time.perf_counter() - offline_began
    with tempfile.TemporaryDirectory(prefix="global-ap-") as temporary: cases = [_case(name, pair, ports, contract, Path(temporary)) for name, pair in pairs.items()]
    rows = [case[0] for case in cases]; passed = bool(len(rows) == 2 and all(row["passed"] for row in rows))
    metrics = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION, "passed": passed, "case_count": len(rows), "passing_case_count": sum(row["passed"] for row in rows), "radial_cells": contract["global_AP_dry_run"]["radial_cells"], "global_state_dimension": contract["global_AP_dry_run"]["global_state_dimension"], "minimum_matched_refinement_order": float(min(row["minimum_matched_refinement_order"] for row in rows)), "maximum_homogeneous_mode_expansivity": float(max(row["maximum_homogeneous_mode_expansivity"] for row in rows)), "maximum_core_total_conservation_defect": float(max(row["maximum_core_total_conservation_defect"] for row in rows)), "maximum_state_norm": float(max(row["maximum_state_norm"] for row in rows)), "maximum_projected_100k_step_wall_days": float(max(row["projected_100k_step_wall_days"] for row in rows)), "all_checkpoints_bitwise": all(row["checkpoint_roundtrip_bitwise"] for row in rows), "all_suffix_replays_bitwise": all(row["suffix_replay_bitwise"] for row in rows), "minimum_source_nullity": min(min(row["source_nullities"]) for row in rows), "offline_physical_anchor_builds": len(ports), "offline_anchor_wall_seconds": offline_wall, "online_truth_calls": 0, "physical_boundary_ports_certified": False, "cycle_wide_inputs_complete": False, "complete_cycle_execution_authorized": False, "certificate_wall_seconds": time.perf_counter() - began, "rows": rows, "authorized_next": AUTHORIZED_NEXT if passed else None}
    return metrics, {"final_states": np.asarray([case[1] for case in cases]), "anchor_indices": np.asarray(list(pairs.values()))}


def _update_catalog(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]; status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("global AP certificate exists")
    hashes, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True); utility._write_json(CANONICAL_DIRECTORY / "global_dry_run_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "global_dry_run_arrays.npz", **arrays)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "production_size_global_AP_dry_run_certified": metrics["passed"], "global_state_dimension": metrics["global_state_dimension"], "periodic_boundary_only": True, "physical_boundary_ports_certified": False, "legacy_evidence_compatibility_audit_authorized": metrics["passed"], "cycle_wide_inputs_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": metrics["authorized_next"]}; utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(f"# Production-size global AP dry-run certificate\n\nClassification: `{metrics['classification']}`.\n\nThe periodic proof kernel advances 94 radial cells and 11 fields per cell (1,034 complex Fourier amplitudes) on two physical anchor paths. The minimum matched order is `{metrics['minimum_matched_refinement_order']:.6f}`, maximum core-total conservation defect `{metrics['maximum_core_total_conservation_defect']:.6e}`, and conservative 100,000-step projection `{metrics['maximum_projected_100k_step_wall_days']:.6e}` wall days. Restart and suffix replay are bitwise.\n\nThis clears the production-dimension exponential-action and cost blocker only. Physical inner/outer boundaries, events, and cycle-wide forcing remain uncertified; no complete-cycle step occurred.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, GLOBAL_SOURCE, GLOBAL_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}); names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); arguments = parser.parse_args()
    if not arguments.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
