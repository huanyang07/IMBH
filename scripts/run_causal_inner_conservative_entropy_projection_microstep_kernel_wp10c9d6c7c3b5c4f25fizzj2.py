#!/usr/bin/env python3
"""Certify the bounded conservative entropy-projection microstep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_conservative_entropy_projection_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj0 as manifest  # noqa: E402
import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_conservative_entropy_projection_microstep import (  # noqa: E402
    EquilibriumPrimitiveSeed,
    conservative_entropy_projected_midpoint_microstep,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_nonlinear_port_atlas import (  # noqa: E402
    equilibrium_entropy_point_from_primitive,
    equilibrium_temporal_conserved,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "conservative_entropy_projection_microstep_kernel_certified"
FAIL_CLASSIFICATION = "conservative_entropy_projection_microstep_kernel_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_conservative_entropy_projection_microstep_kernel_"
    "wp10c9d6c7c3b5c4f25fizzj2"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_CONSERVATIVE_ENTROPY_PROJECTION_MICROSTEP_KERNEL_"
    "WP10C9D6C7C3B5C4F25FIZZJ2_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_conservative_entropy_projection_microstep_kernel_"
    "wp10c9d6c7c3b5c4f25fizzj2.py"
)
THIS_TEST = (
    "tests/test_causal_inner_conservative_entropy_projection_microstep_kernel_"
    "wp10c9d6c7c3b5c4f25fizzj2.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_conservative_entropy_projection_microstep.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_conservative_entropy_projection_microstep.py"
ATLAS_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_nonlinear_port_atlas.py"
)
PARENT_SHA256 = "edbffee210790f9ed6a8b073f86914d8808cafcf8c4e44c8eba2e942ac3a0663"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SELECTED_WITNESSES = (0, 10, 20, 30, 40, 46)
BASE_PATTERN = np.asarray(
    (
        (-0.002, 0.001, -0.0003, 0.0002),
        (0.001, -0.002, 0.0002, -0.0003),
        (0.001, 0.001, 0.0001, 0.0001),
    ),
    dtype=float,
)
MIXED_PATTERN = np.asarray(
    (
        (0.0015, -0.0005, 0.00025, 0.00015),
        (-0.0005, 0.0015, -0.0001, -0.0003),
        (-0.001, -0.001, -0.00015, 0.00015),
    ),
    dtype=float,
)
PATCH_PATTERNS = (BASE_PATTERN, MIXED_PATTERN)


def _u():
    return manifest._u()


def _validate_parent(require_clean: bool = False):
    utility = _u()
    checksum = utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
    if checksum != PARENT_SHA256:
        raise RuntimeError("entropy-projection manifest checksum changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "projection_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["prior_microstep_manifest_superseded_before_execution"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["kernel"]["trajectory_authorized"]
        or summary["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("entropy-projection manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("entropy-projection kernel needs a clean tracked tree")
    return hashes, contract


def _make_patch(old, chart, pattern):
    height = float(np.exp(chart[5]))
    base = np.asarray(
        (
            float(np.exp(chart[0])) / (2.0 * height),
            float(np.exp(chart[3])),
            float(chart[1]),
            float(chart[2]),
        )
    )
    points = []
    seeds = []
    for delta in pattern:
        seed = EquilibriumPrimitiveSeed(
            base[0] * np.exp(delta[0]),
            base[1] * np.exp(delta[1]),
            base[2] + delta[2],
            base[3] + delta[3],
        )
        seeds.append(seed)
        points.append(
            equilibrium_entropy_point_from_primitive(
                old.geometry,
                density=seed.density,
                temperature=seed.temperature,
                proper_half_thickness=height,
                radial_velocity_over_c=seed.radial_velocity_over_c,
                azimuthal_velocity_over_c=seed.azimuthal_velocity_over_c,
            )
        )
    return height, base, tuple(points), tuple(seeds)


def _trust_radius(base, seeds):
    return max(
        max(
            abs(np.log(seed.density / base[0])) / 0.01,
            abs(np.log(seed.temperature / base[1])) / 0.01,
            abs(seed.radial_velocity_over_c - base[2]) / 0.002,
            abs(seed.azimuthal_velocity_over_c - base[3]) / 0.002,
        )
        for seed in seeds
    )


def _advance(old, height, points, seeds, courant, count):
    results = []
    maximum_trust = 0.0
    for _ in range(count):
        result = conservative_entropy_projected_midpoint_microstep(
            geometry=old.geometry,
            proper_half_thickness=height,
            points=points,
            seeds=seeds,
            courant_factor=courant,
        )
        results.append(result)
        if not result.passed:
            return None, tuple(results), maximum_trust
        points, seeds = result.points, result.seeds
    endpoint = np.asarray(
        [equilibrium_temporal_conserved(point) for point in points], dtype=float
    )
    return endpoint, tuple(results), maximum_trust


def _result_metrics(result):
    return {
        "projection_theta": result.projection_theta,
        "correction_relative_norm": result.correction_relative_norm,
        "proposal_entropy_relative_defect": result.proposal_entropy_relative_defect,
        "projection_entropy_slope": result.projection_entropy_slope,
        "maximum_recovery_residual": result.maximum_recovery_residual,
        "conservation_relative_defect": result.conservation_relative_defect,
        "entropy_relative_defect": result.entropy_relative_defect,
        "passed": result.passed,
    }


def _certificate():
    began = time.perf_counter()
    _, contract = _validate_parent()
    physical = {
        index: (label, radius, old, chart)
        for index, label, radius, old, chart in witnesses._physical_witnesses()
        if index in SELECTED_WITNESSES
    }
    rows = []
    endpoint_arrays = []
    all_results = []
    all_orders = []
    all_trust = []
    for witness_index in SELECTED_WITNESSES:
        label, radius, old, chart = physical[witness_index]
        for patch_index, pattern in enumerate(PATCH_PATTERNS):
            height, base, initial_points, initial_seeds = _make_patch(
                old, chart, pattern
            )
            initial_trust = _trust_radius(base, initial_seeds)
            full, full_results, _ = _advance(
                old, height, initial_points, initial_seeds, 0.02, 1
            )
            half, half_results, _ = _advance(
                old, height, initial_points, initial_seeds, 0.01, 2
            )
            quarter, quarter_results, _ = _advance(
                old, height, initial_points, initial_seeds, 0.005, 4
            )
            results = full_results + half_results + quarter_results
            all_results.extend(results)
            endpoint_ready = full is not None and half is not None and quarter is not None
            if endpoint_ready:
                scales = np.maximum(np.max(np.abs(quarter), axis=0), 1.0)
                coarse_defect = float(np.linalg.norm((full - half) / scales))
                refined_defect = float(np.linalg.norm((half - quarter) / scales))
                order = float(np.log2(coarse_defect / refined_defect))
                endpoint_arrays.append(np.stack((full, half, quarter)))
            else:
                coarse_defect = float("inf")
                refined_defect = float("inf")
                order = float("-inf")
                endpoint_arrays.append(np.full((3, 3, 4), np.nan))
            final_seeds = []
            for result in results:
                final_seeds.extend(result.seeds)
            trust = max(initial_trust, _trust_radius(base, final_seeds))
            passed = bool(
                endpoint_ready
                and all(result.passed for result in results)
                and order >= contract["kernel"]["step_halving_order_gate"]
                and trust <= contract["kernel"]["trust_radius_gate"]
            )
            all_orders.append(order)
            all_trust.append(trust)
            rows.append(
                {
                    "witness_index": witness_index,
                    "witness_label": label,
                    "radius_cm": radius,
                    "patch_index": patch_index,
                    "initial_trust_radius_fraction": initial_trust,
                    "maximum_trust_radius_fraction": trust,
                    "matched_coarse_defect": coarse_defect,
                    "matched_refined_defect": refined_defect,
                    "matched_step_halving_order": order,
                    "full_step": [_result_metrics(result) for result in full_results],
                    "two_half_steps": [
                        _result_metrics(result) for result in half_results
                    ],
                    "four_quarter_steps": [
                        _result_metrics(result) for result in quarter_results
                    ],
                    "passed": passed,
                }
            )
    passed = bool(len(rows) == 12 and all(row["passed"] for row in rows))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "physical_anchor_count": len(physical),
        "patch_count": len(rows),
        "passing_patch_count": sum(row["passed"] for row in rows),
        "microstep_evaluation_count": len(all_results),
        "minimum_matched_step_halving_order": float(min(all_orders)),
        "maximum_trust_radius_fraction": float(max(all_trust)),
        "maximum_recovery_residual": float(
            max(result.maximum_recovery_residual for result in all_results)
        ),
        "maximum_conservation_relative_defect": float(
            max(result.conservation_relative_defect for result in all_results)
        ),
        "maximum_entropy_relative_defect": float(
            max(result.entropy_relative_defect for result in all_results)
        ),
        "maximum_projection_correction_relative_norm": float(
            max(result.correction_relative_norm for result in all_results)
        ),
        "maximum_absolute_projection_theta": float(
            max(abs(result.projection_theta) for result in all_results)
        ),
        "minimum_projection_entropy_slope": float(
            min(result.projection_entropy_slope for result in all_results)
        ),
        "all_projection_entropy_slopes_nonpositive": bool(
            all(result.projection_entropy_slope <= 0.0 for result in all_results)
        ),
        "accepted_state_only": True,
        "trajectory_steps": 0,
        "complete_cycle_execution_authorized": False,
        "certificate_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "selected_witness_indices": np.asarray(SELECTED_WITNESSES),
        "patch_patterns": np.asarray(PATCH_PATTERNS),
        "matched_endpoint_conserved": np.asarray(endpoint_arrays),
        "matched_step_halving_orders": np.asarray(all_orders),
        "trust_radius_fractions": np.asarray(all_trust),
    }
    return metrics, arrays


def _update_catalog(summary):
    utility = _u()
    rows = list(
        csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))
    )
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
        raise RuntimeError("entropy-projection microstep certificate exists")
    hashes, _ = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "kernel_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "kernel_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "conservative_entropy_projection_microstep_certified": metrics["passed"],
        "prior_rejections_preserved": True,
        "short_restartable_nonlinear_atlas_trajectory_manifest_authorized": metrics[
            "passed"
        ],
        "trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
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
        "# Conservative entropy-projection microstep certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"{metrics['passing_patch_count']}/{metrics['patch_count']} physical "
        "patches pass. The minimum matched-step order is "
        f"`{metrics['minimum_matched_step_halving_order']:.6f}`, maximum "
        "conservation defect is "
        f"`{metrics['maximum_conservation_relative_defect']:.6e}`, maximum "
        "entropy defect is "
        f"`{metrics['maximum_entropy_relative_defect']:.6e}`, and maximum "
        "normalized projection correction is "
        f"`{metrics['maximum_projection_correction_relative_norm']:.6e}`.\n\n"
        "All repeated steps are matched-endpoint kernel audits, not an "
        "authorized physical trajectory. Complete-cycle execution remains "
        "blocked.\n\n"
        f"Authorized next: `{metrics['authorized_next']}`.\n",
        encoding="utf-8",
    )
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        PHYSICAL_SOURCE,
        PHYSICAL_TEST,
        ATLAS_SOURCE,
        REPORT_RELATIVE,
    )
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {
                source: utility._sha256(ROOT / source) for source in sources
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
    _update_catalog(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("choose --run")
    metrics, arrays = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
