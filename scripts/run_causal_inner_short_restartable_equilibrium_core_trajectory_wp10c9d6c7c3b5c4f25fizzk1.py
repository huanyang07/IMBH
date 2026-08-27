#!/usr/bin/env python3
"""Execute the short restartable equilibrium-core trajectory certificate."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import run_causal_inner_conservative_entropy_projection_microstep_kernel_wp10c9d6c7c3b5c4f25fizzj2 as microsteps  # noqa: E402
import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
import run_causal_inner_short_restartable_nonlinear_atlas_trajectory_manifest_wp10c9d6c7c3b5c4f25fizzk as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_equilibrium_core_trajectory import (  # noqa: E402
    advance_equilibrium_core_trajectory,
    audit_equilibrium_core_trajectory,
    initialize_equilibrium_core_trajectory,
    load_equilibrium_core_trajectory_checkpoint,
    save_equilibrium_core_trajectory_checkpoint,
    trajectory_primitive_array,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_nonlinear_port_atlas import (  # noqa: E402
    equilibrium_temporal_conserved,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "short_restartable_equilibrium_core_trajectory_certified"
FAIL_CLASSIFICATION = "short_restartable_equilibrium_core_trajectory_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_short_restartable_equilibrium_core_trajectory_"
    "wp10c9d6c7c3b5c4f25fizzk1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_SHORT_RESTARTABLE_EQUILIBRIUM_CORE_TRAJECTORY_"
    "WP10C9D6C7C3B5C4F25FIZZK1_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_short_restartable_equilibrium_core_trajectory_"
    "wp10c9d6c7c3b5c4f25fizzk1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_short_restartable_equilibrium_core_trajectory_"
    "wp10c9d6c7c3b5c4f25fizzk1.py"
)
TRAJECTORY_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_equilibrium_core_trajectory.py"
)
TRAJECTORY_TEST = "tests/test_causal_inner_equilibrium_core_trajectory.py"
MICROSTEP_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_conservative_entropy_projection_microstep.py"
)
PARENT_SHA256 = "67f86f3054c0ebba0783e0c7d876133e5a6842dc750b6de172fd5859362ac6a9"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CASES = (
    ("primary", 0, 0),
    ("held_out", 30, 1),
)


def _u():
    return manifest._u()


def _validate_parent(require_clean=False):
    utility = _u()
    checksum = utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
    if checksum != PARENT_SHA256:
        raise RuntimeError("short trajectory manifest checksum changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "trajectory_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["equilibrium_core_trajectory_certified"]
        or contract["claim_boundary"]["physical_time_horizon_claimed"]
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("short trajectory manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("short trajectory execution needs a clean tracked tree")
    return hashes, contract


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


def _endpoint_conserved(state):
    return np.asarray(
        [equilibrium_temporal_conserved(point) for point in state.points],
        dtype=float,
    )


def _advance(state, *, courant, count, base):
    maximum_trust = _trust_radius(base, state.seeds)
    maximum_step_entropy = 0.0
    maximum_step_conservation = 0.0
    maximum_step_recovery = 0.0
    maximum_correction = 0.0
    for _ in range(count):
        advanced = advance_equilibrium_core_trajectory(
            state, courant_factor=courant
        )
        if not advanced.accepted:
            return None, {
                "accepted": False,
                "failure_step": state.accepted_steps + 1,
            }
        result = advanced.microstep
        state = advanced.state
        maximum_trust = max(maximum_trust, _trust_radius(base, state.seeds))
        maximum_step_entropy = max(
            maximum_step_entropy, result.entropy_relative_defect
        )
        maximum_step_conservation = max(
            maximum_step_conservation, result.conservation_relative_defect
        )
        maximum_step_recovery = max(
            maximum_step_recovery, result.maximum_recovery_residual
        )
        maximum_correction = max(
            maximum_correction, result.correction_relative_norm
        )
    cumulative = audit_equilibrium_core_trajectory(state)
    return state, {
        "accepted": True,
        "accepted_steps": state.accepted_steps,
        "accumulated_courant_time": state.accumulated_courant_time,
        "maximum_trust_radius_fraction": maximum_trust,
        "maximum_step_entropy_relative_defect": maximum_step_entropy,
        "maximum_step_conservation_relative_defect": maximum_step_conservation,
        "maximum_step_recovery_residual": maximum_step_recovery,
        "maximum_projection_correction_relative_norm": maximum_correction,
        "cumulative_conservation_relative_defect": (
            cumulative.cumulative_conservation_relative_defect
        ),
        "cumulative_entropy_relative_defect": (
            cumulative.cumulative_entropy_relative_defect
        ),
    }


def _initial_case(witness_index, patch_index):
    physical = {
        index: (label, radius, old, chart)
        for index, label, radius, old, chart in witnesses._physical_witnesses()
    }
    label, radius, old, chart = physical[witness_index]
    pattern = microsteps.PATCH_PATTERNS[patch_index]
    height, base, points, seeds = microsteps._make_patch(old, chart, pattern)
    state = initialize_equilibrium_core_trajectory(
        geometry=old.geometry,
        proper_half_thickness=height,
        points=points,
        seeds=seeds,
    )
    return label, radius, base, state


def _case_certificate(case_name, witness_index, patch_index, contract):
    label, radius, base, initial = _initial_case(witness_index, patch_index)
    coarse, coarse_metrics = _advance(
        initial, courant=0.02, count=16, base=base
    )
    middle_first, middle_first_metrics = _advance(
        initial, courant=0.01, count=16, base=base
    )
    if coarse is None or middle_first is None:
        return {
            "case": case_name,
            "witness_index": witness_index,
            "passed": False,
            "coarse": coarse_metrics,
            "middle_first": middle_first_metrics,
        }, np.full((3, 3, 4), np.nan), np.empty((0, 4))

    with tempfile.TemporaryDirectory(prefix="equilibrium_core_restart_") as directory:
        checkpoint = Path(directory) / "halfway.npz"
        save_equilibrium_core_trajectory_checkpoint(checkpoint, middle_first)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restarted = load_equilibrium_core_trajectory_checkpoint(checkpoint)
        roundtrip = bool(
            restarted.accepted_steps == middle_first.accepted_steps
            and restarted.accumulated_courant_time
            == middle_first.accumulated_courant_time
            and restarted.initial_entropy_decimal
            == middle_first.initial_entropy_decimal
            and np.array_equal(
                trajectory_primitive_array(restarted),
                trajectory_primitive_array(middle_first),
            )
            and np.array_equal(
                restarted.initial_conserved_total,
                middle_first.initial_conserved_total,
            )
        )
        middle, middle_tail_metrics = _advance(
            middle_first, courant=0.01, count=16, base=base
        )
        replay, replay_metrics = _advance(
            restarted, courant=0.01, count=16, base=base
        )
    if middle is None or replay is None:
        return {
            "case": case_name,
            "witness_index": witness_index,
            "passed": False,
            "checkpoint_roundtrip_bitwise": roundtrip,
            "middle_tail": middle_tail_metrics,
            "replay": replay_metrics,
        }, np.full((3, 3, 4), np.nan), np.empty((0, 4))
    suffix_replay = bool(
        np.array_equal(
            trajectory_primitive_array(middle),
            trajectory_primitive_array(replay),
        )
    )
    fine, fine_metrics = _advance(
        initial, courant=0.005, count=64, base=base
    )
    if fine is None:
        return {
            "case": case_name,
            "witness_index": witness_index,
            "passed": False,
            "fine": fine_metrics,
        }, np.full((3, 3, 4), np.nan), trajectory_primitive_array(middle_first)
    endpoints = np.stack(
        (
            _endpoint_conserved(coarse),
            _endpoint_conserved(middle),
            _endpoint_conserved(fine),
        )
    )
    scales = np.maximum(np.max(np.abs(endpoints[2]), axis=0), 1.0)
    coarse_defect = float(np.linalg.norm((endpoints[0] - endpoints[1]) / scales))
    refined_defect = float(np.linalg.norm((endpoints[1] - endpoints[2]) / scales))
    order = float(np.log2(coarse_defect / refined_defect))
    middle_metrics = dict(middle_tail_metrics)
    middle_metrics.update(
        {
            "maximum_trust_radius_fraction": max(
                middle_first_metrics["maximum_trust_radius_fraction"],
                middle_tail_metrics["maximum_trust_radius_fraction"],
            ),
            "maximum_step_entropy_relative_defect": max(
                middle_first_metrics["maximum_step_entropy_relative_defect"],
                middle_tail_metrics["maximum_step_entropy_relative_defect"],
            ),
            "maximum_step_conservation_relative_defect": max(
                middle_first_metrics[
                    "maximum_step_conservation_relative_defect"
                ],
                middle_tail_metrics["maximum_step_conservation_relative_defect"],
            ),
            "maximum_step_recovery_residual": max(
                middle_first_metrics["maximum_step_recovery_residual"],
                middle_tail_metrics["maximum_step_recovery_residual"],
            ),
            "maximum_projection_correction_relative_norm": max(
                middle_first_metrics[
                    "maximum_projection_correction_relative_norm"
                ],
                middle_tail_metrics["maximum_projection_correction_relative_norm"],
            ),
        }
    )
    gate = contract["gates"]
    ladder_metrics = (coarse_metrics, middle_metrics, fine_metrics)
    passed = bool(
        roundtrip
        and suffix_replay
        and order >= gate["minimum_matched_endpoint_order"]
        and max(
            item["cumulative_conservation_relative_defect"]
            for item in ladder_metrics
        )
        <= gate["maximum_cumulative_conservation_defect"]
        and max(item["cumulative_entropy_relative_defect"] for item in ladder_metrics)
        <= gate["maximum_cumulative_entropy_defect"]
        and max(item["maximum_trust_radius_fraction"] for item in ladder_metrics)
        <= gate["maximum_trust_radius_fraction"]
    )
    row = {
        "case": case_name,
        "witness_index": witness_index,
        "witness_label": label,
        "radius_cm": radius,
        "patch_index": patch_index,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_roundtrip_bitwise": roundtrip,
        "suffix_replay_bitwise": suffix_replay,
        "matched_coarse_defect": coarse_defect,
        "matched_refined_defect": refined_defect,
        "matched_endpoint_order": order,
        "coarse": coarse_metrics,
        "middle": middle_metrics,
        "fine": fine_metrics,
        "passed": passed,
    }
    return row, endpoints, trajectory_primitive_array(middle_first)


def _certificate():
    began = time.perf_counter()
    _, contract = _validate_parent()
    rows = []
    endpoints = []
    checkpoints = []
    for case in CASES:
        row, endpoint, checkpoint = _case_certificate(*case, contract)
        rows.append(row)
        endpoints.append(endpoint)
        checkpoints.append(checkpoint)
        if not row["passed"]:
            break
    passed = bool(len(rows) == len(CASES) and all(row["passed"] for row in rows))
    valid = [row for row in rows if row.get("matched_endpoint_order") is not None]
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "case_count": len(rows),
        "passing_case_count": sum(row["passed"] for row in rows),
        "minimum_matched_endpoint_order": float(
            min(row["matched_endpoint_order"] for row in valid)
        ) if valid else None,
        "maximum_cumulative_conservation_relative_defect": float(
            max(
                item[level]["cumulative_conservation_relative_defect"]
                for item in valid
                for level in ("coarse", "middle", "fine")
            )
        ) if valid else None,
        "maximum_cumulative_entropy_relative_defect": float(
            max(
                item[level]["cumulative_entropy_relative_defect"]
                for item in valid
                for level in ("coarse", "middle", "fine")
            )
        ) if valid else None,
        "maximum_trust_radius_fraction": float(
            max(
                item[level]["maximum_trust_radius_fraction"]
                for item in valid
                for level in ("coarse", "middle", "fine")
            )
        ) if valid else None,
        "all_checkpoint_roundtrips_bitwise": bool(
            valid and all(row["checkpoint_roundtrip_bitwise"] for row in valid)
        ),
        "all_suffix_replays_bitwise": bool(
            valid and all(row["suffix_replay_bitwise"] for row in valid)
        ),
        "physical_time_horizon_claimed": False,
        "full_eleven_field_trajectory_certified": False,
        "complete_cycle_execution_authorized": False,
        "certificate_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "matched_endpoint_conserved": np.asarray(endpoints),
        "halfway_checkpoint_primitives": np.asarray(checkpoints),
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
        raise RuntimeError("short trajectory certificate exists")
    hashes, _ = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "trajectory_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "trajectory_arrays.npz", **arrays)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "short_restartable_equilibrium_core_trajectory_certified": metrics[
            "passed"
        ],
        "full_eleven_field_trajectory_certified": False,
        "physical_entropy_congruence_manifest_authorized": metrics["passed"],
        "physical_time_horizon_claimed": False,
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
    verdict = "passes" if metrics["passed"] else "fails"
    REPORT_PATH.write_text(
        "# Short restartable equilibrium-core trajectory certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"The two-case 0.32-Courant-time trajectory {verdict}. Minimum matched "
        f"endpoint order: `{metrics['minimum_matched_endpoint_order']}`. Maximum "
        "cumulative conservation and entropy defects are "
        f"`{metrics['maximum_cumulative_conservation_relative_defect']}` and "
        f"`{metrics['maximum_cumulative_entropy_relative_defect']}`.\n\n"
        "This is a restartable nonlinear four-current core trajectory, not a "
        "full eleven-field or physical-time trajectory. The physical entropy "
        "congruence remains the next required bridge. Complete-cycle execution "
        "is not authorized.\n",
        encoding="utf-8",
    )
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        TRAJECTORY_SOURCE,
        TRAJECTORY_TEST,
        MICROSTEP_SOURCE,
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
