#!/usr/bin/env python3
"""Certify native conservative reset and dense guard-localization structure."""

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

import run_causal_inner_cycle_wide_missing_input_acquisition_and_event_reset_manifest_wp10c9d6c7c3b5c4f25fizzs as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_conservative_hybrid_event import (  # noqa: E402
    audit_entropy_ledger_reset,
    audit_entropy_ledger_reset_geometry,
    build_entropy_ledger_reset_geometry,
    cubic_hermite_dense_state,
    localize_bracketed_guard,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "native_1232_state_conservative_entropy_reset_and_guard_localization_"
    "structure_certified_physical_event_calibration_missing"
)
FAIL_CLASSIFICATION = "conservative_entropy_reset_or_guard_localization_structure_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_conservative_entropy_reset_and_guard_localization_structure_"
    "certificate_wp10c9d6c7c3b5c4f25fizzs1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_ENTROPY_RESET_AND_"
    "GUARD_LOCALIZATION_STRUCTURE_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZS1_"
    "2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_conservative_entropy_reset_and_guard_localization_"
    "structure_certificate_wp10c9d6c7c3b5c4f25fizzs1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_conservative_entropy_reset_and_guard_localization_"
    "structure_certificate_wp10c9d6c7c3b5c4f25fizzs1.py"
)
RESET_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_conservative_hybrid_event.py"
)
RESET_TEST = "tests/test_causal_inner_conservative_hybrid_event.py"
PARENT_SHA256 = "d04caa4f700bcc4416e824c5188dce508fb7384692f078386f43b464f1102310"
PREFIX_SHA256 = "4b491cbba2440f6106da7ae69c54c494ecaa5c15137f8fd4aa808cf305b3d9c6"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(require_clean: bool = False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("cycle input/reset manifest changed")
    parent_hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "acquisition_and_reset_contract.json"
    )
    prefix = manifest.parent
    if utility._sha256(prefix.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PREFIX_SHA256:
        raise RuntimeError("prefix port payload certificate changed")
    prefix_hashes = utility._validate_checksums(prefix.CANONICAL_DIRECTORY)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["reset_and_guard_structure_certified"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or contract["claim_boundary"]["complete_cycle_steps"] != 0
        or contract["required_physical_inputs"]["impact_guard_and_reset_truth"]
        or contract["required_physical_inputs"]["hot_exit_guard_and_reset_truth"]
    ):
        raise RuntimeError("cycle input/reset manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("reset/localization certificate needs a clean tracked tree")
    return parent_hashes, prefix_hashes, contract


def _profile_zero_payload():
    prefix = manifest.parent
    with np.load(prefix.CANONICAL_DIRECTORY / "prefix_port_payloads.npz", allow_pickle=False) as payload:
        profile_indices = np.asarray(payload["selected_profile_indices"], dtype=int)
        positions = np.flatnonzero(profile_indices == 0)
        cells = np.asarray(payload["selected_cell_indices"], dtype=int)[positions]
        scales = np.asarray(payload["conserved_scales4"], dtype=float)[positions]
        inverse_roots = np.asarray(
            payload["scaled_entropy_inverse_square_roots"], dtype=float
        )[positions]
    order = np.argsort(cells)
    cells = cells[order]
    if not np.array_equal(cells, np.arange(112)):
        raise RuntimeError("profile-zero physical anchor is not unique in every native cell")
    return positions[order], cells, scales[order], inverse_roots[order]


def _assemble_native_ledger_map(
    cell_measures: np.ndarray,
    conserved_scales: np.ndarray,
    inverse_entropy_roots: np.ndarray,
):
    measures = np.asarray(cell_measures, dtype=float)
    scales = np.asarray(conserved_scales, dtype=float)
    roots = np.asarray(inverse_entropy_roots, dtype=float)
    if measures.shape != (112,) or scales.shape != (112, 4) or roots.shape != (112, 4, 4):
        raise ValueError("native reset inputs have the wrong shape")
    if np.any(measures <= 0.0) or np.any(~np.isfinite(measures)):
        raise ValueError("native cell measures must be positive and finite")
    normalized_measures = measures / np.sum(measures)
    physical = np.zeros((4, 112 * 11), dtype=float)
    for cell in range(112):
        local_increment_map = np.diag(scales[cell]) @ roots[cell]
        physical[:, 11 * cell : 11 * cell + 4] = (
            normalized_measures[cell] * local_increment_map
        )
    row_scales = np.linalg.norm(physical, axis=1)
    if np.any(row_scales <= 0.0) or np.any(~np.isfinite(row_scales)):
        raise RuntimeError("native physical ledger rows cannot be normalized")
    scaled = physical / row_scales[:, None]
    entropy_weights = np.repeat(normalized_measures, 11)
    return physical, scaled, row_scales, entropy_weights, normalized_measures


def _guard_structure():
    state_dimension = 112 * 11
    start_time = 7.25
    timestep = 5.7888
    crossing_fraction = 0.371
    coefficients = np.zeros((4, state_dimension), dtype=float)
    coefficients[0, 0] = -crossing_fraction
    coefficients[1, 0] = 1.0 / timestep
    coefficients[:, 1] = np.asarray((0.07, -0.03, 0.004, -0.0002))
    coefficients[:, 17] = np.asarray((-0.02, 0.05, -0.003, 0.0001))

    def state(elapsed):
        x = float(elapsed)
        return (
            coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * x**2
            + coefficients[3] * x**3
        )

    def rate(elapsed):
        x = float(elapsed)
        return coefficients[1] + 2.0 * coefficients[2] * x + 3.0 * coefficients[3] * x**2

    left_state = state(0.0)
    right_state = state(timestep)
    left_rate = rate(0.0)
    right_rate = rate(timestep)
    guard = lambda value, _time: float(value[0])
    full = localize_bracketed_guard(
        guard,
        left_state,
        right_state,
        left_rate,
        right_rate,
        start_time=start_time,
        timestep=timestep,
        orientation="negative_to_positive",
    )
    half_step = 0.5 * timestep
    midpoint_state = state(half_step)
    midpoint_rate = rate(half_step)
    half = localize_bracketed_guard(
        guard,
        left_state,
        midpoint_state,
        left_rate,
        midpoint_rate,
        start_time=start_time,
        timestep=half_step,
        orientation="negative_to_positive",
    )
    exact_time = start_time + crossing_fraction * timestep
    exact_state = state(crossing_fraction * timestep)
    dense_midpoint = cubic_hermite_dense_state(
        left_state,
        right_state,
        left_rate,
        right_rate,
        timestep=timestep,
        fraction=0.5,
    )
    with tempfile.TemporaryDirectory(prefix="imbh_guard_replay_") as directory:
        checkpoint = Path(directory) / "guard_checkpoint.npz"
        np.savez(
            checkpoint,
            left_state=left_state,
            right_state=right_state,
            left_rate=left_rate,
            right_rate=right_rate,
            start_time=np.asarray(start_time),
            timestep=np.asarray(timestep),
        )
        with np.load(checkpoint, allow_pickle=False) as payload:
            replay = localize_bracketed_guard(
                guard,
                payload["left_state"],
                payload["right_state"],
                payload["left_rate"],
                payload["right_rate"],
                start_time=float(payload["start_time"]),
                timestep=float(payload["timestep"]),
                orientation="negative_to_positive",
            )
    metrics = {
        "known_crossing_fraction": crossing_fraction,
        "localized_fraction": full.fraction,
        "event_time_absolute_defect": abs(full.event_time - exact_time),
        "event_state_infinity_defect": float(np.linalg.norm(full.event_state - exact_state, ord=np.inf)),
        "guard_absolute_defect": abs(full.guard_value),
        "full_to_half_event_time_absolute_defect": abs(full.event_time - half.event_time),
        "full_to_half_event_state_infinity_defect": float(
            np.linalg.norm(full.event_state - half.event_state, ord=np.inf)
        ),
        "dense_midpoint_infinity_defect": float(
            np.linalg.norm(dense_midpoint - state(half_step), ord=np.inf)
        ),
        "checkpoint_replay_bitwise": bool(
            full.fraction == replay.fraction
            and full.event_time == replay.event_time
            and full.guard_value == replay.guard_value
            and np.array_equal(full.event_state, replay.event_state)
        ),
        "orientation": full.orientation,
        "full_iterations": full.iterations,
        "half_iterations": half.iterations,
    }
    arrays = {
        "guard_coefficients4x1232": coefficients,
        "guard_left_state": left_state,
        "guard_right_state": right_state,
        "guard_left_rate": left_rate,
        "guard_right_rate": right_rate,
        "guard_event_state": full.event_state,
    }
    return metrics, arrays


def _certificate():
    began = time.perf_counter()
    _, _, contract = _validate_parent()
    positions, cells, scales, inverse_roots = _profile_zero_payload()
    context_began = time.perf_counter()
    context = manifest.parent._physical_context()
    context_wall = time.perf_counter() - context_began
    cell_measures = np.asarray(context.grid.cell_measures, dtype=float)
    physical_map, scaled_map, row_scales, weights, normalized_measures = (
        _assemble_native_ledger_map(cell_measures, scales, inverse_roots)
    )
    geometry = build_entropy_ledger_reset_geometry(scaled_map, weights)
    geometry_audit = audit_entropy_ledger_reset_geometry(geometry)

    rng = np.random.default_rng(2026082701)
    impulses = rng.normal(scale=2.0e-3, size=(32, 4))
    constitutive = rng.normal(scale=3.0e-2, size=(32, 112 * 11))
    reset_audits = [
        audit_entropy_ledger_reset(geometry, impulse, jump)
        for impulse, jump in zip(impulses, constitutive, strict=True)
    ]
    reset_jumps = np.asarray(
        [
            geometry.reset_jump(impulse, jump)
            for impulse, jump in zip(impulses, constitutive, strict=True)
        ]
    )
    scaled_realized = reset_jumps @ scaled_map.T
    physical_impulses = impulses * row_scales[None, :]
    physical_realized = reset_jumps @ physical_map.T
    scaled_defects = np.linalg.norm(scaled_realized - impulses, axis=1) / np.maximum(
        np.linalg.norm(impulses, axis=1), 1.0
    )
    physical_component_relative = np.max(
        np.abs(physical_realized - physical_impulses)
        / np.maximum(np.abs(physical_impulses), row_scales[None, :] * 1.0e-12),
        axis=1,
    )
    minimum_jumps = np.asarray(
        [geometry.minimum_norm_jump(impulse) for impulse in impulses]
    )
    projected_constitutive = reset_jumps - minimum_jumps

    with tempfile.TemporaryDirectory(prefix="imbh_reset_replay_") as directory:
        checkpoint = Path(directory) / "reset_checkpoint.npz"
        np.savez(checkpoint, impulses=impulses, constitutive=constitutive)
        with np.load(checkpoint, allow_pickle=False) as payload:
            replay_jumps = np.asarray(
                [
                    geometry.reset_jump(impulse, jump)
                    for impulse, jump in zip(
                        payload["impulses"], payload["constitutive"], strict=True
                    )
                ]
            )
    reset_replay_bitwise = bool(np.array_equal(reset_jumps, replay_jumps))
    guard_metrics, guard_arrays = _guard_structure()
    maximum_reset_ledger_defect = max(
        audit.ledger_relative_defect for audit in reset_audits
    )
    maximum_projected_ledger_defect = max(
        audit.projected_constitutive_ledger_defect for audit in reset_audits
    )
    maximum_orthogonality_defect = max(
        audit.minimum_to_null_weighted_orthogonality_defect for audit in reset_audits
    )
    all_reset_audits_passed = all(audit.passed for audit in reset_audits)
    physical_calibration_missing = contract["conservative_reset_structure"][
        "physical_calibration_missing"
    ]
    passed = bool(
        geometry.ledger_dimension == 4
        and geometry.state_dimension == 1232
        and geometry_audit.ledger_rank == 4
        and geometry_audit.passed
        and all_reset_audits_passed
        and maximum_reset_ledger_defect <= 2.0e-12
        and maximum_projected_ledger_defect <= 2.0e-12
        and maximum_orthogonality_defect <= 2.0e-12
        and float(np.max(scaled_defects)) <= 2.0e-12
        and float(np.max(physical_component_relative)) <= 2.0e-10
        and reset_replay_bitwise
        and guard_metrics["event_time_absolute_defect"] <= 2.0e-12
        and guard_metrics["event_state_infinity_defect"] <= 2.0e-12
        and guard_metrics["guard_absolute_defect"] <= 2.0e-12
        and guard_metrics["full_to_half_event_time_absolute_defect"] <= 2.0e-12
        and guard_metrics["full_to_half_event_state_infinity_defect"] <= 2.0e-12
        and guard_metrics["dense_midpoint_infinity_defect"] <= 2.0e-12
        and guard_metrics["checkpoint_replay_bitwise"]
        and guard_metrics["orientation"] == "negative_to_positive"
        and len(physical_calibration_missing) == 4
    )
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "native_radial_cells": 112,
        "global_state_dimension": 1232,
        "ledger_dimension": 4,
        "profile_zero_anchor_count": len(positions),
        "geometry_audit": asdict(geometry_audit),
        "all_reset_audits_passed": all_reset_audits_passed,
        "sampled_reset_count": len(reset_audits),
        "maximum_reset_ledger_relative_defect": maximum_reset_ledger_defect,
        "maximum_projected_constitutive_ledger_defect": maximum_projected_ledger_defect,
        "maximum_minimum_to_null_weighted_orthogonality_defect": maximum_orthogonality_defect,
        "maximum_scaled_direct_ledger_defect": float(np.max(scaled_defects)),
        "maximum_physical_component_relative_ledger_defect": float(
            np.max(physical_component_relative)
        ),
        "minimum_realized_to_minimum_weighted_norm_ratio": float(
            min(
                audit.realized_jump_norm / max(audit.minimum_norm_jump, np.finfo(float).tiny)
                for audit in reset_audits
            )
        ),
        "reset_checkpoint_replay_bitwise": reset_replay_bitwise,
        "guard_localization": guard_metrics,
        "context_initialization_wall_seconds": context_wall,
        "total_certificate_wall_seconds": time.perf_counter() - began,
        "physical_guard_surface_calibrated": False,
        "physical_ledger_impulse_calibrated": False,
        "physical_constitutive_jump_calibrated": False,
        "destination_mode_and_event_duration_calibrated": False,
        "reset_and_guard_structure_certified": passed,
        "events_and_resets_physically_calibrated": False,
        "cycle_wide_physical_inputs_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "profile_zero_anchor_positions": positions,
        "native_cell_indices": cells,
        "native_cell_measures": cell_measures,
        "normalized_cell_measures": normalized_measures,
        "conserved_scales4": scales,
        "scaled_entropy_inverse_square_roots4x4": inverse_roots,
        "physical_conservation_map4x1232": physical_map,
        "ledger_row_scales4": row_scales,
        "scaled_conservation_map4x1232": scaled_map,
        "entropy_weights1232": weights,
        "minimum_norm_normal1232x4": geometry.minimum_norm_normal,
        "reset_impulses32x4": impulses,
        "constitutive_candidates32x1232": constitutive,
        "minimum_norm_jumps32x1232": minimum_jumps,
        "projected_constitutive_jumps32x1232": projected_constitutive,
        "realized_reset_jumps32x1232": reset_jumps,
        **guard_arrays,
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
        raise RuntimeError("reset/localization structure certificate exists")
    parent_hashes, prefix_hashes, _ = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "reset_and_guard_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "reset_and_guard_arrays.npz", **arrays)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "native_radial_cells": 112,
        "global_state_dimension": 1232,
        "reset_and_guard_structure_certified": metrics[
            "reset_and_guard_structure_certified"
        ],
        "events_and_resets_physically_calibrated": False,
        "cycle_wide_physical_inputs_complete": False,
        "heldout_cycle_validation_complete": False,
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
            "manifest_hashes": parent_hashes,
            "prefix_port_artifact": manifest.parent.ARTIFACT,
            "prefix_port_checksum_manifest_sha256": PREFIX_SHA256,
            "prefix_port_hashes": prefix_hashes,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Conservative entropy-reset and guard-localization structure certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        "The native 112-cell, 1,232-coordinate entropy state now has a four-ledger "
        "weighted minimum-entropy reset normal and matrix-free ledger-null projection. "
        f"All `{metrics['sampled_reset_count']}` deterministic reset probes close the "
        "scaled and unscaled physical ledger maps and reproduce bitwise from checkpoint.\n\n"
        "Cubic-Hermite dense output localizes one synthetic transverse guard crossing, "
        "agrees between full- and half-interval brackets, and replays bitwise. Unbracketed, "
        "multiply crossed, tangential, and wrong-orientation events fail closed in the unit "
        "contract.\n\n"
        "This is a structure certificate only. No physical guard surface, ledger impulse, "
        "constitutive jump, destination mode, or event duration has been calibrated. Cycle-"
        "wide forcing and outer loading remain missing; no complete-cycle step is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, RESET_SOURCE, RESET_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {source: utility._sha256(ROOT / source) for source in sources},
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update(summary)
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
