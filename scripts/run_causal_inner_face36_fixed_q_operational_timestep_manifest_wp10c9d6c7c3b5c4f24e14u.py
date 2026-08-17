#!/usr/bin/env python3
"""Freeze the first doubled-step fixed-Q operational-timestep rung."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14u"
ARTIFACT = (
    "causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PRIMARY_EVIDENCE_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_evidence_aggregation_"
    "wp10c9d6c7c3b5c4f24e14r"
)
HELDOUT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t"
)
SEED_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b"
)
SEED_PATH = SEED_DIRECTORY / "canonical_seed_continuation.npz"
FINE_REFERENCE_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    EXECUTION_RUNNER,
    EXECUTION_TEST,
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)
FINE_TIMESTEP_SECONDS = 1.0e-7
COARSE_TIMESTEP_SECONDS = 2.0e-7


CONTRACT = {
    "schema_version": 1,
    "work_package": WORK_PACKAGE,
    "definitions_only": True,
    "state": "primary_20ms",
    "purpose": "first_fail_fast_operational_timestep_rung",
    "common_start": {
        "source": "canonical_primary_BDF2_continuation_seed_e14b",
        "accepted_history_only": True,
        "previous_timestep_seconds": FINE_TIMESTEP_SECONDS,
    },
    "matched_endpoint": {
        "coarse_root": "coarse_2e7",
        "coarse_timestep_seconds": COARSE_TIMESTEP_SECONDS,
        "fine_reference_roots": ["cold_1", "warm_1"],
        "fine_timestep_seconds": FINE_TIMESTEP_SECONDS,
        "fine_reference_source": "certified_primary_evidence_e14r_from_e14l",
        "state_difference_relative_to_coarse_change_maximum": 0.1,
        "reaction_action_relative_difference_maximum": 0.1,
    },
    "solver_contract": {
        "BDF_order": 2,
        "variable_step_history": True,
        "initial_exact_assembly": True,
        "maximum_exact_assemblies": 2,
        "refresh_policy": "on_line_search_failure",
        "maximum_newton_iterations": 8,
        "maximum_line_search_iterations": 12,
        "bitwise_root_and_continuation_replay": True,
    },
    "binding_gates": {
        "maximum_scaled_residual": 1.0e-10,
        "maximum_Q3_relative_defect": 1.0e-12,
        "maximum_storage_parity_relative_defect": 1.0e-9,
        "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
        "maximum_reaction_ledger_relative_defect": 1.0e-12,
        "maximum_constraint_action_ledger_relative_defect": 1.0e-12,
        "maximum_raw_Schur_condition_number": 1.0e8,
        "maximum_H_over_R": 0.12,
        "minimum_scattering_optical_depth": 1.0,
        "maximum_scaled_primitive_change": 5.0e-3,
        "incoming_excision_characteristics": 0,
    },
    "decision": {
        "pass": "operational_timestep_rung_2e7_certified",
        "fail": "operational_timestep_rung_2e7_failed",
        "pass_authorizes_only": "definitions_only_operational_timestep_rung_4e7_manifest",
    },
    "hard_stops": {
        "no_4e7_execution": True,
        "no_fixed_Q_micro_solver": True,
        "no_physical_microburst": True,
        "no_fast_averaging": True,
        "no_reduced_slow_evolution": True,
    },
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(("git", "diff", "--cached", "--quiet"), cwd=ROOT).returncode == 0
    )


def _checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _validate_parents() -> dict:
    primary_hashes = _checksums(PRIMARY_EVIDENCE_DIRECTORY)
    heldout_hashes = _checksums(HELDOUT_DIRECTORY)
    seed_hashes = _checksums(SEED_DIRECTORY)
    fine_hashes = _checksums(FINE_REFERENCE_DIRECTORY)
    primary = _read(PRIMARY_EVIDENCE_DIRECTORY / "summary.json")
    heldout = _read(HELDOUT_DIRECTORY / "summary.json")
    fine = _read(FINE_REFERENCE_DIRECTORY / "metrics.json")
    if (
        primary["classification"] != "primary_bounded_continuation_evidence_certified"
        or not primary["passed"]
        or heldout["classification"] != "heldout_bounded_continuation_certified"
        or not heldout["passed"]
        or not heldout["operational_timestep_manifest_authorized"]
        or not fine["main_roots"]["cold_1"]["accepted"]
        or not fine["main_roots"]["warm_1"]["accepted"]
    ):
        raise RuntimeError("operational-timestep parent authorization changed")
    return {
        "primary_summary": primary,
        "heldout_summary": heldout,
        "fine_reference": {
            "cold_1": fine["main_roots"]["cold_1"],
            "warm_1": fine["main_roots"]["warm_1"],
        },
        "package_hashes": {
            "primary_evidence": primary_hashes,
            "heldout_continuation": heldout_hashes,
            "primary_seed": seed_hashes,
            "fine_reference": fine_hashes,
        },
    }


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(ARTIFACT_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "PROSPECTIVE",
            })
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(ARTIFACT_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    parents = _validate_parents()
    if not _clean():
        raise RuntimeError("operational-timestep manifest requires a clean tree")
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": "operational_timestep_rung_2e7_manifest_frozen_execution_authorized",
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "operational_timestep_rung_2e7_execution_authorized": True,
        "operational_timestep_rung_4e7_manifest_authorized": False,
        "operational_timestep_rung_4e7_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write(ARTIFACT_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(ARTIFACT_DIRECTORY / "parent_lock.json", parents)
    _write(ARTIFACT_DIRECTORY / "summary.json", summary)
    _write(ARTIFACT_DIRECTORY / "reference_lock.json", {
        "seed_path": str(SEED_PATH.relative_to(ROOT)),
        "seed_sha256": _sha(SEED_PATH),
        "fine_cold_result_sha256": _sha(FINE_REFERENCE_DIRECTORY / "result_cold_1.npz"),
        "fine_endpoint_result_sha256": _sha(FINE_REFERENCE_DIRECTORY / "result_warm_1.npz"),
    })
    _write(ARTIFACT_DIRECTORY / "provenance.json", {
        "schema_version": 1,
        "definition_commit": _git("rev-parse", "HEAD"),
        "definition_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "source_hashes": {relative: _sha(ROOT / relative) for relative in SOURCE_FILES},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    })
    files = ("execution_manifest.json", "parent_lock.json", "provenance.json", "reference_lock.json", "summary.json")
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("select --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
