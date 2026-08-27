#!/usr/bin/env python3
"""Certify the native nonperiodic SBP/SAT global AP boundary action."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonperiodic_native_global_ap_boundary_action_structure_manifest_wp10c9d6c7c3b5c4f25fizzu as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_nonperiodic_global_ap import (  # noqa: E402
    NonperiodicGlobalAPCheckpoint,
    audit_nonperiodic_global_ap_operator,
    build_nonperiodic_global_ap_operator,
    load_nonperiodic_global_ap_checkpoint,
    midpoint_affine_step,
    save_nonperiodic_global_ap_checkpoint,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "nonperiodic_native_global_AP_boundary_action_structure_certified"
FAIL_CLASSIFICATION = "nonperiodic_native_global_AP_boundary_action_structure_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_nonperiodic_native_global_ap_boundary_action_structure_"
    "certificate_wp10c9d6c7c3b5c4f25fizzu1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONPERIODIC_NATIVE_GLOBAL_AP_"
    "BOUNDARY_ACTION_STRUCTURE_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZU1_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_nonperiodic_native_global_ap_boundary_action_"
    "structure_certificate_wp10c9d6c7c3b5c4f25fizzu1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonperiodic_native_global_ap_boundary_action_"
    "structure_certificate_wp10c9d6c7c3b5c4f25fizzu1.py"
)
PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_nonperiodic_global_ap.py"
PHYSICAL_TEST = "tests/test_causal_inner_nonperiodic_global_ap.py"
PARENT_SHA256 = "e64b2be1427874b674c9c47bfa6d3adc74b5a9144f0c814db95469f290944c0e"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("nonperiodic boundary-action manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "boundary_action_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["nonperiodic_global_AP_boundary_action_certified"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or contract["certificate"]["complete_cycle_steps"] != 0
    ):
        raise RuntimeError("nonperiodic boundary-action contract changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("nonperiodic boundary-action certificate needs a clean tracked tree")
    return hashes, contract


def _native_profile_zero():
    ports = manifest.ports
    with np.load(
        ports.CANONICAL_DIRECTORY / "prefix_port_payloads.npz", allow_pickle=False
    ) as payload:
        profiles = np.asarray(payload["selected_profile_indices"], dtype=int)
        selected = profiles == 0
        cells = np.asarray(payload["selected_cell_indices"][selected], dtype=int)
        radial = np.asarray(
            payload["corrected_radial_matrices11x11"][selected], dtype=float
        )
        source = np.asarray(payload["source_matrices11x11"][selected], dtype=float)
    if not np.array_equal(cells, np.arange(112)):
        raise RuntimeError("profile zero is not a complete ordered native radial state")
    context = ports._physical_context()
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    return cells, radial, source, measures


def _certificate():
    began = time.perf_counter(); _, contract = _validate_parent()
    load_began = time.perf_counter(); cells, radial, source, measures = _native_profile_zero(); load_seconds = time.perf_counter() - load_began
    build_began = time.perf_counter(); operator = build_nonperiodic_global_ap_operator(radial, source, measures); build_seconds = time.perf_counter() - build_began
    audit_began = time.perf_counter(); audit = audit_nonperiodic_global_ap_operator(operator); audit_seconds = time.perf_counter() - audit_began

    rng = np.random.default_rng(2026082704)
    initial = rng.normal(scale=1.0e-3, size=operator.state_dimension)
    outer_first = rng.normal(scale=2.0e-4, size=operator.outer_control_dimension)
    outer_second = rng.normal(scale=2.0e-4, size=operator.outer_control_dimension)
    rate_bound = max(
        float(np.max(np.abs(operator.generator).sum(axis=1))),
        np.finfo(float).tiny,
    )
    timestep = 0.05 / rate_bound
    first = midpoint_affine_step(operator, initial, outer_first, timestep)
    homogeneous = midpoint_affine_step(
        operator, initial, np.zeros(operator.outer_control_dimension), timestep
    )
    uninterrupted = midpoint_affine_step(
        operator, first.state, outer_second, timestep
    )
    with tempfile.TemporaryDirectory(prefix="native_nonperiodic_global_ap_") as directory:
        path = Path(directory) / "checkpoint.npz"
        checkpoint = NonperiodicGlobalAPCheckpoint(
            first.state, outer_second, timestep, 1
        )
        save_nonperiodic_global_ap_checkpoint(checkpoint, path)
        loaded = load_nonperiodic_global_ap_checkpoint(path)
        checkpoint_bitwise = bool(
            np.array_equal(loaded.state, checkpoint.state)
            and np.array_equal(
                loaded.outer_incoming_amplitudes,
                checkpoint.outer_incoming_amplitudes,
            )
            and loaded.elapsed_time_seconds == checkpoint.elapsed_time_seconds
            and loaded.completed_steps == checkpoint.completed_steps
        )
        replay = midpoint_affine_step(
            operator,
            loaded.state,
            loaded.outer_incoming_amplitudes,
            timestep,
        )
    replay_bitwise = bool(np.array_equal(replay.state, uninterrupted.state))
    gates = contract["certificate"]
    passed = bool(
        audit.passed
        and len(cells) == gates["minimum_native_cells"]
        and operator.state_dimension == 1232
        and audit.sbp_adjoint_defect <= gates["maximum_sbp_identity_defect"]
        and audit.sbp_constant_defect <= gates["maximum_sbp_identity_defect"]
        and audit.energy_identity_relative_defect
        <= gates["maximum_energy_identity_relative_defect"]
        and audit.maximum_homogeneous_entropy_growth_eigenvalue
        <= gates["maximum_homogeneous_entropy_growth_eigenvalue"]
        and audit.affine_action_relative_defect
        <= gates["maximum_affine_action_defect"]
        and audit.minimum_source_nullity >= gates["minimum_source_nullity"]
        and audit.inner_incoming_count == gates["inner_incoming_count"]
        and audit.outer_incoming_count == gates["outer_incoming_count"]
        and first.entropy_ledger_relative_defect <= 5.0e-12
        and homogeneous.entropy_after <= homogeneous.entropy_before + 2.0e-14
        and checkpoint_bitwise
        and replay_bitwise
    )
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "native_radial_cells": operator.cell_count,
        "fields_per_cell": operator.field_count,
        "global_state_dimension": operator.state_dimension,
        "sparse_generator_nonzeros": int(operator.generator.nnz),
        "outer_control_nonzeros": int(operator.outer_control.nnz),
        "outer_control_dimension": operator.outer_control_dimension,
        "operator_audit": asdict(audit),
        "test_timestep_seconds": timestep,
        "affine_midpoint_entropy_ledger_relative_defect": first.entropy_ledger_relative_defect,
        "homogeneous_entropy_before": homogeneous.entropy_before,
        "homogeneous_entropy_after": homogeneous.entropy_after,
        "checkpoint_roundtrip_bitwise": checkpoint_bitwise,
        "suffix_replay_bitwise": replay_bitwise,
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "radial_accuracy_certified": False,
        "nonlinear_atlas_interpolation_certified": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "native_profile_load_wall_seconds": load_seconds,
        "operator_build_wall_seconds": build_seconds,
        "operator_audit_wall_seconds": audit_seconds,
        "certificate_wall_seconds": time.perf_counter() - began,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "native_cell_indices": cells,
        "normalized_entropy_weights": operator.entropy_weights,
        "interface_viscosities": operator.viscosity,
        "outer_control_dense": operator.outer_control.toarray(),
        "initial_state": initial,
        "first_affine_state": first.state,
        "uninterrupted_suffix_state": uninterrupted.state,
        "replayed_suffix_state": replay.state,
        "outer_first": outer_first,
        "outer_second": outer_second,
    }
    return metrics, arrays


def _update(summary: dict) -> None:
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("nonperiodic boundary-action certificate already exists")
    hashes, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "boundary_action_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "boundary_action_arrays.npz", **arrays)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "nonperiodic_global_AP_boundary_action_certified": metrics["passed"],
        "pure_inner_excision_certified": metrics["passed"],
        "eleven_characteristic_outer_affine_loading_certified": metrics["passed"],
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": metrics["authorized_next"],
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    audit = metrics["operator_audit"]; REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(
        "# Nonperiodic native global AP boundary-action certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"The 112-cell x 11-field sparse generator has {metrics['sparse_generator_nonzeros']} nonzeros. The exact weighted energy identity closes at `{audit['energy_identity_relative_defect']:.6e}`; the largest homogeneous entropy-growth eigenvalue is `{audit['maximum_homogeneous_entropy_growth_eigenvalue']:.6e}`. Source nullity is `{audit['minimum_source_nullity']}`, the incoming boundary counts are `{audit['inner_incoming_count']}` inner and `{audit['outer_incoming_count']}` outer, and affine checkpoint suffix replay is bitwise.\n\n"
        "This certifies the nonperiodic action structure only. Physical cycle inputs, radial accuracy, nonlinear atlas interpolation, physical events, heldout cycle prediction, and complete-cycle execution remain blocked. No cycle step occurred.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {name: utility._sha256(ROOT / name) for name in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
