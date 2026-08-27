#!/usr/bin/env python3
"""Certify entropy stability and order of the frozen split discretization."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

import run_causal_inner_entropy_stable_split_discretization_manifest_wp10c9d6c7c3b5c4f25fizzh as manifest  # noqa: E402
import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_entropy_split_discretization import (  # noqa: E402
    audit_frozen_split_operators,
    build_frozen_split_operators,
    midpoint_cayley_matrix,
    strang_split_step,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import (  # noqa: E402
    build_full_port_atlas_anchor,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "entropy_stable_split_discretization_kernel_certified"
FAIL_CLASSIFICATION = "entropy_stable_split_discretization_kernel_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = "causal_inner_entropy_stable_split_discretization_kernel_wp10c9d6c7c3b5c4f25fizzh1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_STABLE_SPLIT_DISCRETIZATION_KERNEL_WP10C9D6C7C3B5C4F25FIZZH1_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_entropy_stable_split_discretization_kernel_wp10c9d6c7c3b5c4f25fizzh1.py"
THIS_TEST = "tests/test_causal_inner_entropy_stable_split_discretization_kernel_wp10c9d6c7c3b5c4f25fizzh1.py"
PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_entropy_split_discretization.py"
PHYSICAL_TEST = "tests/test_causal_inner_entropy_split_discretization.py"
PARENT_SHA256 = "936011e6d129ea4a961ad7986ae3a710e14ef3dca389103a3dd87e4547157863"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(*, require_clean: bool = False) -> tuple[dict, dict]:
    utils = _u()
    if utils._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("split discretization manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(manifest.CANONICAL_DIRECTORY / "discretization_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["kernel"]["trajectory_steps"] != 0
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("split discretization contract changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("split discretization kernel needs a clean tracked tree")
    return hashes, contract


def _physical_anchor(old_state, chart7):
    height = float(np.exp(chart7[5]))
    temperature = float(np.exp(chart7[3]))
    surface_mass = float(np.exp(chart7[0]))
    sound_speed = float(old_state.thermodynamics.sound_speed)
    omega = float(np.sqrt(old_state.thermodynamics.integrated_pressure / (surface_mass * height**2)))
    alpha = float((old_state.closure.viscous_signal_speed_over_c * C / sound_speed) ** 2)
    anchor = build_full_port_atlas_anchor(
        sound_speed=sound_speed,
        temperature=temperature,
        proper_half_thickness=height,
        proper_vertical_frequency=omega,
        alpha=alpha,
        shear_relaxation_time=float(old_state.closure.relaxation_time),
        transport_speed_over_c=float(chart7[1]),
    )
    return anchor, omega


def _split_matrix(operators, timestep: float) -> np.ndarray:
    source_half = midpoint_cayley_matrix(operators.source_generator, 0.5 * timestep)
    transport = midpoint_cayley_matrix(operators.transport_generator, timestep)
    return source_half @ transport @ source_half


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    _, contract = _validate_parent()
    witness_began = time.perf_counter()
    physical_witnesses = list(witnesses._physical_witnesses())
    witness_seconds = time.perf_counter() - witness_began
    rows = []
    charts = []
    radii = []
    operator_defects = []
    ledger_defects = []
    order_minima = []
    all_errors = []
    all_orders = []
    for index, label, radius, old_state, chart7 in physical_witnesses:
        anchor, omega = _physical_anchor(old_state, chart7)
        crossing = min(1.0 / omega, float(old_state.closure.relaxation_time))
        operators = build_frozen_split_operators(anchor, cell_count=7, cell_light_crossing_seconds=crossing)
        audit = audit_frozen_split_operators(operators)
        audit_values = asdict(audit)
        operator_defect = max(audit_values.values())
        generator = operators.transport_generator + operators.source_generator
        rate_scale = max(float(np.linalg.norm(generator, ord=2)), np.finfo(float).tiny)
        horizon = 0.2 / rate_scale
        rng = np.random.default_rng(9173 + index)
        probes = [rng.normal(size=77) for _ in range(3)]
        probe_ledgers = []
        for probe in probes:
            probe /= np.linalg.norm(probe)
            step = strang_split_step(operators, probe, 0.05 / rate_scale)
            probe_ledgers.append(step.total_ledger_relative_defect)
        reference_probe = probes[0] / np.linalg.norm(probes[0])
        exact = expm(horizon * generator) @ reference_probe
        errors = []
        for step_count in (1, 2, 4, 8):
            dt = horizon / step_count
            split_matrix = _split_matrix(operators, dt)
            numerical = np.linalg.matrix_power(split_matrix, step_count) @ reference_probe
            errors.append(float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), np.finfo(float).tiny)))
        orders = [float(np.log(errors[i] / errors[i + 1]) / np.log(2.0)) for i in range(3)]
        minimum_order = min(orders)
        maximum_ledger = max(probe_ledgers)
        passed = bool(audit.passed and maximum_ledger <= 2e-12 and minimum_order >= 1.8)
        rows.append({"index": index, "label": label, "radius_cm": radius, "cell_light_crossing_seconds": crossing, "rate_scale_per_second": rate_scale, "operator_audit": audit_values, "maximum_split_ledger_relative_defect": maximum_ledger, "matched_horizon_relative_errors": errors, "adjacent_observed_orders": orders, "passed": passed})
        charts.append(chart7)
        radii.append(radius)
        operator_defects.append(operator_defect)
        ledger_defects.append(maximum_ledger)
        order_minima.append(minimum_order)
        all_errors.append(errors)
        all_orders.append(orders)
    passed = bool(len(rows) == 47 and all(row["passed"] for row in rows))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "physical_witness_count": len(rows),
        "passing_witness_count": sum(row["passed"] for row in rows),
        "maximum_operator_structure_defect": float(max(operator_defects)),
        "maximum_split_energy_ledger_relative_defect": float(max(ledger_defects)),
        "minimum_matched_horizon_observed_order": float(min(order_minima)),
        "spatial_entropy_stability_certified": passed,
        "midpoint_heat_ledger_certified": passed,
        "second_order_split_composition_certified": passed,
        "physical_port_atlas_preserved": True,
        "nonlinear_trust_region_certified": False,
        "trajectory_steps": 0,
        "complete_cycle_execution_authorized": False,
        "witness_construction_wall_seconds": witness_seconds,
        "certificate_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "witness_charts7": np.asarray(charts),
        "witness_radii_cm": np.asarray(radii),
        "operator_structure_defects": np.asarray(operator_defects),
        "split_energy_ledger_relative_defects": np.asarray(ledger_defects),
        "matched_horizon_relative_errors": np.asarray(all_errors),
        "adjacent_observed_orders": np.asarray(all_orders),
    }
    del contract
    return metrics, arrays


def _update(summary: dict) -> None:
    utils = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("split discretization certificate already exists")
    hashes, _ = _validate_parent(require_clean=True)
    utils = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "kernel_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "kernel_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "physical_port_atlas_preserved": True, "entropy_stable_split_discretization_certified": metrics["passed"], "nonlinear_trust_region_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": metrics["authorized_next"]}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Entropy-stable split-discretization kernel certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"All {metrics['passing_witness_count']}/{metrics['physical_witness_count']} physical anchors pass. "
        f"The maximum discrete energy-ledger defect is `{metrics['maximum_split_energy_ledger_relative_defect']:.6e}`, "
        f"and the minimum matched-horizon order is `{metrics['minimum_matched_horizon_observed_order']:.6f}`.\n\n"
        "This is a frozen-coefficient periodic proof kernel. It does not certify nonlinear atlas motion, boundaries, a trajectory, or complete-cycle execution.\n\n"
        f"Authorized next: `{metrics['authorized_next']}`.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utils._git("rev-parse", "HEAD"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
