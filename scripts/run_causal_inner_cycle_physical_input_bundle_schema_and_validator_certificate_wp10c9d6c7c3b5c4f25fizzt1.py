#!/usr/bin/env python3
"""Certify the cycle physical-input bundle schema with a synthetic fixture."""

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

import run_causal_inner_cycle_wide_physical_driver_boundary_loading_and_event_truth_acquisition_manifest_wp10c9d6c7c3b5c4f25fizzt as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_cycle_physical_input import (  # noqa: E402
    load_cycle_physical_input_bundle,
    save_cycle_physical_input_bundle,
    validate_cycle_physical_input_bundle,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "cycle_physical_input_bundle_schema_and_fail_closed_validator_certified_"
    "synthetic_fixture_only_external_physics_missing"
)
FAIL_CLASSIFICATION = "cycle_physical_input_bundle_schema_or_validator_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_cycle_physical_input_bundle_schema_and_validator_certificate_"
    "wp10c9d6c7c3b5c4f25fizzt1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_PHYSICAL_INPUT_BUNDLE_SCHEMA_"
    "AND_VALIDATOR_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZT1_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_cycle_physical_input_bundle_schema_and_validator_"
    "certificate_wp10c9d6c7c3b5c4f25fizzt1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_cycle_physical_input_bundle_schema_and_validator_"
    "certificate_wp10c9d6c7c3b5c4f25fizzt1.py"
)
INPUT_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_cycle_physical_input.py"
)
INPUT_TEST = "tests/test_causal_inner_cycle_physical_input.py"
PARENT_SHA256 = "af99169da4ceb552d48779245f8505a6a78ada7b4b54541f6940c9b1a8f1b1d4"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(require_clean: bool = False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("physical-input acquisition manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "physical_input_acquisition_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["physical_model_complete"]
        or summary["physical_payloads_acquired"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or not all(contract["current_missing_payloads"].values())
    ):
        raise RuntimeError("physical-input acquisition classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-input validator certificate needs a clean tracked tree")
    return hashes, contract


def _reset_geometry_arrays():
    reset = manifest.parent
    with np.load(
        reset.CANONICAL_DIRECTORY / "reset_and_guard_arrays.npz", allow_pickle=False
    ) as payload:
        conservation = np.asarray(payload["scaled_conservation_map4x1232"], dtype=float)
        normal = np.asarray(payload["minimum_norm_normal1232x4"], dtype=float)
    return conservation, normal


def _synthetic_bundle():
    """Return a deterministic schema fixture that makes no physical claim."""

    conservation, normal = _reset_geometry_arrays()
    rng = np.random.default_rng(2026082702)
    period = 578880.0
    phases = np.linspace(0.0, 2.0 * np.pi, 9)
    q_nodes = rng.normal(scale=2.0e-3, size=(3, 4))
    modes = np.asarray(("cold_fixture", "hot_fixture", "recovery_fixture"))
    common = (len(phases), len(q_nodes), len(modes))
    base = rng.normal(scale=2.0e-4, size=common[1:] + (1232,))
    cosine = rng.normal(scale=5.0e-5, size=common[1:] + (1232,))
    sine = rng.normal(scale=4.0e-5, size=common[1:] + (1232,))
    forcing = np.asarray(
        [base + np.cos(phase) * cosine + np.sin(phase) * sine for phase in phases]
    )
    total_ledger = np.einsum("as,pqms->pqma", conservation, forcing)
    distributed = 0.65 * total_ledger
    boundary_ledger = total_ledger - distributed
    incoming_base = rng.normal(scale=1.0e-3, size=common[1:] + (11,))
    incoming_cosine = rng.normal(scale=2.0e-4, size=common[1:] + (11,))
    incoming_sine = rng.normal(scale=2.0e-4, size=common[1:] + (11,))
    incoming = np.asarray(
        [
            incoming_base
            + np.cos(phase) * incoming_cosine
            + np.sin(phase) * incoming_sine
            for phase in phases
        ]
    )
    driver = {
        "phase_nodes": phases,
        "phase_rate_per_second": np.full(len(phases), 2.0 * np.pi / period),
        "retained_invariant_nodes4": q_nodes,
        "mode_labels": modes,
        "slow_forcing1232_per_second": forcing,
        "distributed_source_ledger_rate4": distributed,
        "boundary_ledger_rate4": boundary_ledger,
        "outer_incoming_characteristics11": incoming,
    }

    n_anchor = 10
    states = rng.normal(scale=3.0e-3, size=(n_anchor, 1232))
    radial_diagonal = np.linspace(-0.6, 0.7, 11)
    radial = np.broadcast_to(
        np.diag(radial_diagonal), (n_anchor, 112, 11, 11)
    ).copy()
    source_diagonal = np.concatenate((np.zeros(4), -np.linspace(0.5, 2.0, 7)))
    source = np.broadcast_to(
        np.diag(source_diagonal), (n_anchor, 112, 11, 11)
    ).copy()
    tangents = rng.normal(size=(n_anchor, 1237))
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    branch = {
        "anchor_states1232": states,
        "anchor_phase": np.linspace(0.1, 2.0 * np.pi - 0.1, n_anchor),
        "anchor_invariants4": states @ conservation.T,
        "anchor_mode_index": np.arange(n_anchor) % len(modes),
        "radial_matrices112x11x11": radial,
        "source_matrices112x11x11": source,
        "forcing1232_per_second": rng.normal(scale=1.0e-4, size=(n_anchor, 1232)),
        "trust_radii": np.linspace(0.02, 0.05, n_anchor),
        "stable_spectral_gaps_per_second": np.linspace(0.4, 0.8, n_anchor),
        "guard_margins": rng.normal(size=(n_anchor, 4)),
        "pseudo_arclength_tangents": tangents,
    }

    n_event = 8
    pre = rng.normal(scale=2.0e-3, size=(n_event, 1232))
    impulses = rng.normal(scale=8.0e-4, size=(n_event, 4))
    constitutive_candidate = rng.normal(scale=4.0e-4, size=(n_event, 1232))
    constitutive = constitutive_candidate - (
        constitutive_candidate @ conservation.T
    ) @ normal.T
    normal_jumps = impulses @ normal.T
    post = pre + normal_jumps + constitutive
    source_modes = np.arange(n_event) % len(modes)
    destination_modes = (source_modes + 1) % len(modes)
    events = {
        "pre_states1232": pre,
        "post_states1232": post,
        "pre_invariants4": pre @ conservation.T,
        "phase": np.linspace(0.25, 2.0 * np.pi - 0.25, n_event),
        "source_mode_index": source_modes,
        "destination_mode_index": destination_modes,
        "duration_seconds": np.linspace(0.2, 1.1, n_event),
        "integrated_ledger_impulse4": impulses,
        "ledger_null_constitutive_jump1232": constitutive,
        "guard_value_and_direction": np.column_stack(
            (np.zeros(n_event), np.where(np.arange(n_event) % 2 == 0, 1.0, -1.0))
        ),
    }
    heldout = {
        "withheld_branch_anchor_indices": np.asarray((8, 9), dtype=int),
        "withheld_event_indices": np.asarray((6, 7), dtype=int),
        "withheld_phase_windows": np.asarray(((0.9, 1.3), (4.2, 4.7))),
        "sequence_mode_indices": np.asarray((0, 1, 2, 0), dtype=int),
        "sequence_ledger_increments4": rng.normal(scale=1.0e-3, size=(3, 4)),
        "spatial_truth_grid_cells": np.asarray(224, dtype=int),
    }
    metadata = {
        "schema_version": 1,
        "physical_model_id": "synthetic_schema_fixture_not_physics",
        "physical_model_complete": False,
        "synthetic_fixture": True,
        "period_seconds": period,
        "unit_system": "cgs",
        "source_citations": ["internal deterministic schema fixture"],
        "source_code_commit": "synthetic-fixture-20260827",
        "split_frozen_before_fit": True,
        "training_phase_indices": [0, 1, 3, 4, 5, 7, 8],
        "heldout_phase_indices": [2, 6],
        "training_invariant_indices": [0, 1],
        "heldout_invariant_indices": [2],
        "training_branch_anchor_indices": list(range(8)),
        "heldout_branch_anchor_indices": [8, 9],
        "training_event_indices": list(range(6)),
        "heldout_event_indices": [6, 7],
    }
    return metadata, driver, branch, events, heldout, conservation


def _all_bitwise(left, right) -> bool:
    if isinstance(left, dict):
        return bool(left.keys() == right.keys() and all(_all_bitwise(left[key], right[key]) for key in left))
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(left == right)


def _certificate():
    began = time.perf_counter()
    _validate_parent()
    metadata, driver, branch, events, heldout, conservation = _synthetic_bundle()
    with tempfile.TemporaryDirectory(prefix="imbh_cycle_input_bundle_") as directory:
        path = Path(directory) / "bundle"
        save_cycle_physical_input_bundle(
            path, metadata, driver, branch, events, heldout
        )
        loaded = load_cycle_physical_input_bundle(path)
        bitwise = all(
            _all_bitwise(left, right)
            for left, right in zip(
                (metadata, driver, branch, events, heldout), loaded, strict=True
            )
        )
        audit = validate_cycle_physical_input_bundle(
            *loaded,
            conservation_map=conservation,
            require_physical=False,
            checkpoint_roundtrip_bitwise=bitwise,
        )
        physical_rejected = False
        try:
            validate_cycle_physical_input_bundle(
                *loaded,
                conservation_map=conservation,
                require_physical=True,
                checkpoint_roundtrip_bitwise=bitwise,
            )
        except ValueError as error:
            physical_rejected = "structural/synthetic" in str(error)
    passed = bool(
        audit.structurally_passed
        and not audit.physically_usable
        and audit.synthetic_fixture
        and not audit.physical_model_complete
        and bitwise
        and physical_rejected
    )
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "audit": asdict(audit),
        "structurally_passed": audit.structurally_passed,
        "physically_usable": audit.physically_usable,
        "synthetic_fixture_rejected_when_physical_required": physical_rejected,
        "bundle_roundtrip_bitwise": bitwise,
        "validator_wall_seconds": time.perf_counter() - began,
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "input_schema_and_validator_certified": passed,
        "events_and_resets_physically_calibrated": False,
        "heldout_cycle_validation_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    fixture = {
        **{f"driver__{key}": value for key, value in driver.items()},
        **{f"branch__{key}": value for key, value in branch.items()},
        **{f"events__{key}": value for key, value in events.items()},
        **{f"heldout__{key}": value for key, value in heldout.items()},
        "conservation_map4x1232": conservation,
    }
    return metrics, metadata, fixture


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


def _canonicalize(metrics, metadata, fixture):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cycle-input validator certificate exists")
    hashes, _ = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "validator_metrics.json", metrics)
    utility._write_json(CANONICAL_DIRECTORY / "synthetic_fixture_metadata.json", metadata)
    np.savez_compressed(CANONICAL_DIRECTORY / "synthetic_fixture_arrays.npz", **fixture)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "input_schema_and_validator_certified": metrics[
            "input_schema_and_validator_certified"
        ],
        "synthetic_fixture_only": True,
        "physical_model_complete": False,
        "physical_payloads_acquired": False,
        "events_and_resets_physically_calibrated": False,
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
            "manifest_hashes": hashes,
            "reset_geometry_artifact": manifest.parent.ARTIFACT,
            "reset_geometry_checksum_manifest_sha256": utility._sha256(
                manifest.parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
            ),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Cycle physical-input bundle schema and validator certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        "A canonical metadata/driver/branch/events/heldout bundle now validates phase "
        "periodicity and period integration, positive phase rate, exact four-ledger closure "
        "of the 1,232-state forcing, all eleven outer incoming characteristics, branch port "
        "symmetry/dissipation/source nullity, retained-invariant closure, event reset closure, "
        "transverse guards, and prospective split disjointness. Serialization replays bitwise.\n\n"
        "The decisive fixture is explicitly synthetic and incomplete. The same validator "
        "rejects it when physical data are required, so this certificate cannot be mistaken "
        "for a calibrated driver, boundary, branch, or event bundle. No cycle-wide physical "
        "payload has been acquired and no cycle step is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, INPUT_SOURCE, INPUT_TEST, REPORT_RELATIVE)
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
    metrics, metadata, fixture = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0 if _canonicalize(metrics, metadata, fixture)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
