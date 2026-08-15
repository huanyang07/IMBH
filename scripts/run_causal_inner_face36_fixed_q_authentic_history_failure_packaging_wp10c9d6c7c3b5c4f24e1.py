#!/usr/bin/env python3
"""Package the fail-fast authentic fixed-Q history result."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e1"
ARTIFACT = (
    "causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1"
)
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
IMPLEMENTATION_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_history_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e0"
)
PHYSICAL_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_failure_"
    "packaging_wp10c9d6c7c3b5c4f24e1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_authentic_history_result_"
    "wp10c9d6c7c3b5c4f24e1.py"
)
CASE_ORDER = (
    "primary_coarse",
    "heldout_coarse",
    "primary_middle",
    "heldout_middle",
    "primary_fine",
    "heldout_fine",
)
GATES = {
    "maximum_scaled_residual": 1.0e-10,
    "maximum_Q3_relative_defect": 1.0e-12,
    "maximum_ledger_relative_defect": 1.0e-12,
    "maximum_storage_parity_relative_defect": 1.0e-9,
    "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
    "maximum_raw_Schur_condition_number": 1.0e8,
    "maximum_H_over_R": 0.12,
    "minimum_scattering_optical_depth": 1.0,
    "maximum_scaled_primitive_change": 5.0e-3,
    "maximum_complete_Jacobian_assemblies": 1,
}


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


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


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
                    "scientific_status": "REJECTED",
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
    if not _tracked_tree_is_clean():
        raise RuntimeError("failure packaging requires a clean tracked tree")
    implementation = _read(IMPLEMENTATION_ARTIFACT / "summary.json")
    if not implementation["passed"]:
        raise RuntimeError("fixed-Q history implementation certificate changed")
    identity = _read(CHECKPOINT_DIRECTORY / "execution_identity.json")
    case = _read(CHECKPOINT_DIRECTORY / "primary_coarse.json")
    if (
        case["passed"]
        or case["failed_stage"] != "BDF1"
        or case["BDF1"]["failure_reasons"]
        != ["nonlinear_root", "complete_residual"]
    ):
        raise RuntimeError("primary-coarse failure classification changed")
    later_cases_present = [
        name
        for name in CASE_ORDER[1:]
        if (CHECKPOINT_DIRECTORY / f"{name}.json").exists()
    ]
    if later_cases_present:
        raise RuntimeError("fail-fast ladder advanced after the first rejection")
    execution_runner = subprocess.run(
        (
            "git",
            "show",
            f"{identity['execution_commit']}:{PHYSICAL_RUNNER}",
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(execution_runner).hexdigest() != identity["runner_sha256"]:
        raise RuntimeError("physical runner provenance does not close")
    checkpoint_npz = CHECKPOINT_DIRECTORY / "primary_coarse_bdf1.npz"
    with np.load(checkpoint_npz, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    saved_metrics = json.loads(str(arrays.pop("metrics_json")))
    if saved_metrics != case["BDF1"]:
        raise RuntimeError("checkpoint metrics do not match the case record")
    residual_maximum = float(
        np.max(np.abs(arrays["augmented_scaled_residual"]))
    )
    if residual_maximum != case["BDF1"]["maximum_scaled_residual"]:
        raise RuntimeError("saved residual maximum does not match the case record")

    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "authentic_fixed_Q_history_ladder_rejected_at_primary_"
            "BDF1_solver_budget"
        ),
        "passed": False,
        "first_failed_case": "primary_coarse",
        "failed_stage": "BDF1",
        "failure_reasons": case["BDF1"]["failure_reasons"],
        "later_cases_executed": False,
        "physical_failure_detected": False,
        "continuous_KKT_or_reaction_architecture_rejected": False,
        "one_exact_Jacobian_plus_Broyden_policy_rejected": True,
        "diagnostic_exact_refresh_authorized": True,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c4f24e2_targeted_exact_refresh_diagnostic"
        ),
    }
    config = {
        "schema_version": 1,
        "case_order": list(CASE_ORDER),
        "executed_cases": ["primary_coarse"],
        "binding_temporal_form": "increment_primary_complete_BDF",
        "timestep_seconds": case["timestep_seconds"],
        "gates": GATES,
        "fail_fast": True,
        "diagnostic_refresh_may_not_convert_this_result_to_a_pass": True,
    }
    decisive = {
        "schema_version": 1,
        "case": case,
        "residual_gate_ratio": (
            case["BDF1"]["maximum_scaled_residual"]
            / GATES["maximum_scaled_residual"]
        ),
        "all_nonroot_acceptance_gates_passed": all(
            value
            for key, value in case["BDF1"]["acceptance"].items()
            if key
            not in {"accepted", "nonlinear_root_passed", "complete_residual_passed", "failure_reasons"}
        ),
        "checkpoint_residual_maximum": residual_maximum,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "config.json", config)
    _write(CANONICAL_DIRECTORY / "decisive_case.json", decisive)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "decisive_arrays.npz",
        **arrays,
    )
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "physical_execution": identity,
            "physical_checkpoint_hashes": {
                "execution_identity.json": _sha(
                    CHECKPOINT_DIRECTORY / "execution_identity.json"
                ),
                "primary_coarse.json": _sha(
                    CHECKPOINT_DIRECTORY / "primary_coarse.json"
                ),
                "primary_coarse_bdf1.npz": _sha(checkpoint_npz),
            },
            "packaging_commit": _git("rev-parse", "HEAD"),
            "packaging_tree": _git("rev-parse", "HEAD^{tree}"),
            "packaging_runner_sha256": _sha(ROOT / THIS_RUNNER),
            "canonical_test_sha256": _sha(ROOT / THIS_TEST),
            "tracked_worktree_clean_at_packaging_start": True,
            "untracked_files_at_packaging_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
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
        "config.json",
        "decisive_arrays.npz",
        "decisive_case.json",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in files
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
