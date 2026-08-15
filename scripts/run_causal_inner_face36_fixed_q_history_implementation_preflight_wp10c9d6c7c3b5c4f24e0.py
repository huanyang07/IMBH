#!/usr/bin/env python3
"""Certify the corrected fixed-Q history implementation before physics runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e0"
ARTIFACT = (
    "causal_inner_face36_fixed_q_history_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e0"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PARENT_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_constrained_history_contract_correction_"
    "wp10c9d6c7c3b5c4f24d1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_history_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e0.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_history_implementation_"
    "preflight_wp10c9d6c7c3b5c4f24e0.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
    "tests/test_causal_inner_fixed_q.py",
)
FOCUSED_TESTS = (
    "tests/test_causal_inner_fixed_q.py",
    "tests/test_causal_inner_monolithic_dae.py",
    "tests/test_causal_inner_monolithic_bdf.py",
    "tests/test_causal_inner_monolithic_discrete_tangent.py",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parent_contract() -> dict:
    summary = _read(PARENT_ARTIFACT / "summary.json")
    if (
        not summary["passed"]
        or not summary["implementation_preflight_authorized"]
        or summary["physical_history_execution_authorized"]
    ):
        raise RuntimeError("c4f24d1 contract changed")
    return summary


def _tracked_tree_is_clean() -> bool:
    unstaged = subprocess.run(("git", "diff", "--quiet"), cwd=ROOT)
    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet"),
        cwd=ROOT,
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def _run_focused_tests() -> dict:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *FOCUSED_TESTS,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    began = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - began
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_wall_seconds": elapsed,
        "passed": completed.returncode == 0,
    }


def _implementation_contract() -> dict:
    source = (ROOT / SOURCE_FILES[2]).read_text(encoding="utf-8")
    binding_residual_source = source.split(
        "    def residual(values: np.ndarray):",
        maxsplit=1,
    )[1].split("    values, evaluation = residual(unknown)", maxsplit=1)[0]
    required_tokens = {
        "increment_primary_binding": (
            "scaled_rate_per_s=scaled_interval_rate" in source
            and "scaled_rate_per_s=" not in binding_residual_source
        ),
        "fail_closed_acceptance": (
            "class CausalFiveFieldFixedQStepAcceptance" in source
            and "failure_reasons" in source
        ),
        "stable_schur_solve": (
            "def _stable_fixed_q_schur_inverse" in source
            and "np.linalg.solve(matrix, np.eye(3))" in source
        ),
        "normalized_kernel_rejected": (
            "requires raw or frozen-normalized " in source
            and '"reaction channels"' in source
        ),
        "accepted_history_only": (
            "rejected fixed-Q step cannot define BDF history" in source
        ),
        "restart_roundtrip": (
            "save_causal_five_field_fixed_q_bdf_restart" in source
            and "load_causal_five_field_fixed_q_bdf_restart" in source
        ),
        "state_local_reaction_on_restart": (
            "endpoint_reaction = causal_five_field_fixed_q_reaction" in source
        ),
        "solver_counters": (
            "exact_jacobian_assemblies" in source
            and "broyden_updates" in source
            and "linear_solves" in source
        ),
    }
    return {
        "schema_version": 1,
        "checks": required_tokens,
        "all_checks_passed": all(required_tokens.values()),
        "binding_temporal_form": "increment_primary_complete_BDF",
        "direct_rate_role": "post_root_parity_only",
        "binding_reaction_basis": "frozen_normalized",
        "maximum_complete_Jacobian_assemblies_per_root": 1,
        "physical_operator_changed": False,
    }


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "CERTIFIED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def main() -> None:
    _parent_contract()
    execution_commit = _git("rev-parse", "HEAD")
    execution_tree = _git("rev-parse", "HEAD^{tree}")
    tracked_clean = _tracked_tree_is_clean()
    untracked_files_at_start = _git(
        "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    if not tracked_clean:
        raise RuntimeError("implementation preflight requires a clean tracked tree")
    contract = _implementation_contract()
    tests = _run_focused_tests()
    passed = bool(contract["all_checks_passed"] and tests["passed"])
    classification = (
        "fixed_Q_history_implementation_preflight_certified_"
        "physical_history_ladder_authorized"
        if passed
        else "fixed_Q_history_implementation_preflight_rejected"
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "definitions_only": False,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "focused_tests_passed": tests["passed"],
        "implementation_contract_passed": contract["all_checks_passed"],
        "physical_history_execution_authorized": passed,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f24e1_authentic_BDF1_BDF2_history_ladder"
            if passed
            else "stop_and_repair_fixed_Q_history_implementation"
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CANONICAL_DIRECTORY / "implementation_contract.json", contract)
    _write(CANONICAL_DIRECTORY / "test_results.json", tests)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": execution_commit,
            "execution_tree": execution_tree,
            "tracked_worktree_clean_at_start": tracked_clean,
            "untracked_files_at_start": untracked_files_at_start,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in SOURCE_FILES
            },
            "parent_summary_sha256": _sha(PARENT_ARTIFACT / "summary.json"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "blas_thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    files = (
        "implementation_contract.json",
        "provenance.json",
        "summary.json",
        "test_results.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
