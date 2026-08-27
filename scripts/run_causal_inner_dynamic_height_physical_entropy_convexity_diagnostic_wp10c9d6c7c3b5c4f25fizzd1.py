#!/usr/bin/env python3
"""Execute the frozen physical-entropy convexity diagnostic for height."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_dynamic_height_convex_legendre_manifest_wp10c9d6c7c3b5c4f25fizzd as manifest  # noqa: E402
import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_dynamic_height_legendre import (  # noqa: E402
    centered_dynamic_height_entropy_hessian,
    dynamic_height_entropy_state,
    equilibrium_dynamic_height_conserved,
    height_force_identity_defect,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "dynamic_height_common_potential_convex"
FAIL_CLASSIFICATION = "dynamic_height_common_potential_convexity_obstructed"
ARTIFACT = (
    "causal_inner_dynamic_height_physical_entropy_convexity_diagnostic_"
    "wp10c9d6c7c3b5c4f25fizzd1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_DYNAMIC_HEIGHT_PHYSICAL_"
    "ENTROPY_CONVEXITY_DIAGNOSTIC_WP10C9D6C7C3B5C4F25FIZZD1_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_dynamic_height_physical_entropy_convexity_"
    "diagnostic_wp10c9d6c7c3b5c4f25fizzd1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_dynamic_height_physical_entropy_convexity_"
    "diagnostic_wp10c9d6c7c3b5c4f25fizzd1.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_dynamic_height_legendre.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_dynamic_height_legendre.py"
PARENT_SHA256 = "7018994326e5a624cc50533d8dbb5a004054d46ebcc55abc3e8a3f7b09e53071"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(*, require_clean: bool = False) -> tuple[dict, dict]:
    utils = _u()
    checksum = manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_SHA256:
        raise RuntimeError("dynamic-height diagnostic manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        manifest.CANONICAL_DIRECTORY / "convexity_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["diagnostic"]["centered_Hessian_step_factors"]
        != [0.002, 0.001, 0.0005]
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("dynamic-height diagnostic contract changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("height convexity diagnostic needs a clean tracked tree")
    return hashes, contract


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    _, contract = _validate_parent()
    factors = tuple(contract["diagnostic"]["centered_Hessian_step_factors"])
    minimum_gate = float(
        contract["diagnostic"]["positive_minimum_eigenvalue_gate"]
    )
    symmetry_gate = float(contract["diagnostic"]["Hessian_symmetry_gate"])
    force_gate = float(contract["diagnostic"]["force_identity_relative_gate"])
    rows = []
    charts = []
    radii = []
    minimum_eigenvalues = []
    maximum_eigenvalues = []
    force_defects = []
    symmetry_defects = []
    worst_matrix = None
    worst_vector = None
    witness_began = time.perf_counter()
    physical_witnesses = list(witnesses._physical_witnesses())
    witness_seconds = time.perf_counter() - witness_began
    for index, label, radius, old_state, chart7 in physical_witnesses:
        surface_mass = float(np.exp(chart7[0]))
        temperature = float(np.exp(chart7[3]))
        height = float(np.exp(chart7[5]))
        omega = float(
            np.sqrt(
                old_state.thermodynamics.integrated_pressure
                / (surface_mass * height**2)
            )
        )
        conserved = equilibrium_dynamic_height_conserved(
            surface_mass=surface_mass,
            temperature=temperature,
            proper_half_thickness=height,
            proper_vertical_frequency=omega,
        )
        state = dynamic_height_entropy_state(
            conserved,
            proper_vertical_frequency=omega,
            temperature_seed=temperature,
        )
        force_defect = height_force_identity_defect(
            state, proper_vertical_frequency=omega
        )
        audits = [
            centered_dynamic_height_entropy_hessian(
                conserved,
                proper_vertical_frequency=omega,
                temperature_seed=temperature,
                step_factor=factor,
            )
            for factor in factors
        ]
        minima = np.asarray(
            [audit.equilibrated_eigenvalues[0] for audit in audits]
        )
        maxima = np.asarray(
            [audit.equilibrated_eigenvalues[-1] for audit in audits]
        )
        symmetries = np.asarray([audit.symmetry_defect for audit in audits])
        stable_sign = bool(np.all(minima > 0.0) or np.all(minima < 0.0))
        row = {
            "index": index,
            "label": label,
            "radius_cm": radius,
            "proper_vertical_frequency_per_second": omega,
            "minimum_equilibrated_eigenvalues": minima.tolist(),
            "maximum_equilibrated_eigenvalues": maxima.tolist(),
            "stable_minimum_eigenvalue_sign": stable_sign,
            "maximum_Hessian_symmetry_defect": float(np.max(symmetries)),
            "height_force_identity_relative_defect": force_defect,
            "common_potential_convex_at_witness": bool(
                np.all(minima >= minimum_gate)
            ),
        }
        rows.append(row)
        charts.append(chart7)
        radii.append(radius)
        minimum_eigenvalues.append(minima)
        maximum_eigenvalues.append(maxima)
        force_defects.append(force_defect)
        symmetry_defects.append(float(np.max(symmetries)))
        if worst_matrix is None or minima[1] < worst_matrix[0]:
            worst_matrix = (minima[1], audits[1].equilibrated_hessian)
            worst_vector = audits[1].equilibrated_eigenvectors[:, 0]
    minimum_eigenvalues_array = np.asarray(minimum_eigenvalues)
    all_stable = all(row["stable_minimum_eigenvalue_sign"] for row in rows)
    convex_witnesses = sum(row["common_potential_convex_at_witness"] for row in rows)
    passed = bool(
        len(rows) == 47
        and convex_witnesses == 47
        and all_stable
        and max(symmetry_defects) <= symmetry_gate
        and max(force_defects) <= force_gate
    )
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = manifest.PASS_NEXT if passed else manifest.FAILURE_NEXT
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "audit_completed": True,
        "physical_witness_count": len(rows),
        "convex_witness_count": convex_witnesses,
        "stable_sign_witness_count": sum(
            row["stable_minimum_eigenvalue_sign"] for row in rows
        ),
        "global_minimum_equilibrated_entropy_Hessian_eigenvalue": float(
            np.min(minimum_eigenvalues_array)
        ),
        "global_maximum_equilibrated_entropy_Hessian_eigenvalue": float(
            np.max(np.asarray(maximum_eigenvalues))
        ),
        "maximum_Hessian_symmetry_defect": float(max(symmetry_defects)),
        "maximum_height_force_identity_relative_defect": float(
            max(force_defects)
        ),
        "step_factors": list(factors),
        "one_piece_common_potential_rejected": not passed,
        "fixed_height_physical_potential_preserved": True,
        "failure_scope": (
            "one-piece convex Legendre completion with Z_H=surface_mass*H; "
            "not the fixed-height equilibrium potential"
        ),
        "trajectory_steps": 0,
        "complete_cycle_execution_authorized": False,
        "witness_construction_wall_seconds": witness_seconds,
        "diagnostic_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": authorized_next,
    }
    assert worst_matrix is not None and worst_vector is not None
    arrays = {
        "witness_charts7": np.asarray(charts),
        "witness_radii_cm": np.asarray(radii),
        "minimum_equilibrated_eigenvalues": minimum_eigenvalues_array,
        "maximum_equilibrated_eigenvalues": np.asarray(maximum_eigenvalues),
        "height_force_identity_relative_defects": np.asarray(force_defects),
        "worst_middle_factor_equilibrated_Hessian": np.asarray(worst_matrix[1]),
        "worst_middle_factor_negative_eigenvector": np.asarray(worst_vector),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _u()
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
                    "sha256": utils._sha256(path),
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
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
    catalog = utils._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("dynamic-height convexity diagnostic already exists")
    hashes, _ = _validate_parent(require_clean=True)
    utils = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "diagnostic_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "diagnostic_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "audit_completed": True,
        "fixed_height_physical_potential_preserved": True,
        "dynamic_height_common_potential_certified": metrics["passed"],
        "split_architecture_manifest_authorized": not metrics["passed"],
        "trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_artifact": manifest.ARTIFACT,
            "manifest_checksum_manifest_sha256": PARENT_SHA256,
            "manifest_hashes": hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Dynamic-height physical-entropy convexity diagnostic\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"The exact candidate was evaluated on {metrics['physical_witness_count']} "
        "frozen physical witnesses at all three prospective Hessian steps. "
        f"Only {metrics['convex_witness_count']} witnesses are convex; the global "
        "minimum diagonally equilibrated eigenvalue is "
        f"`{metrics['global_minimum_equilibrated_entropy_Hessian_eigenvalue']:.6e}`. "
        f"The height-force identity remains accurate to "
        f"`{metrics['maximum_height_force_identity_relative_defect']:.6e}`.\n\n"
        "This rejects the one-piece common-potential height completion. The "
        "certified fixed-height gas+radiation potential is preserved. No failed "
        "state is propagated and no trajectory was executed.\n\n"
        f"Authorized next: `{metrics['authorized_next']}`.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
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
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
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
