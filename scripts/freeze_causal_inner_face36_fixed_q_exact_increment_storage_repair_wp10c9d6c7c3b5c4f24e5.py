#!/usr/bin/env python3
"""Freeze the prospective exact-increment storage repair contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e5"
ARTIFACT = (
    "causal_inner_face36_fixed_q_exact_increment_storage_repair_manifest_"
    "wp10c9d6c7c3b5c4f24e5"
)
DIRECTORY = ROOT / "results/canonical" / ARTIFACT
PARENT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_residual_resolution_audit_"
    "wp10c9d6c7c3b5c4f24e4"
)
CATALOG_CSV = ROOT / "results/manifests/canonical_artifacts.csv"
CATALOG_JSON = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/freeze_causal_inner_face36_fixed_q_exact_increment_storage_"
    "repair_wp10c9d6c7c3b5c4f24e5.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_exact_increment_storage_repair_"
    "wp10c9d6c7c3b5c4f24e5.py"
)
CONTRACT = {
    "schema_version": 1,
    "definitions_only": True,
    "physical_operator_changed": False,
    "binding_temporal_form": "increment_primary",
    "direct_rate_role": "independent_post_evaluation_parity_audit_only",
    "repair": (
        "carry_exact_solver_primitive_increment_through_affine_"
        "reconstruction_and_temporal_path_action"
    ),
    "forbid_endpoint_or_node_subtraction_as_binding_path_direction": True,
    "require_exact_endpoint_reconstruction": True,
    "require_inactive_affine_reconstruction": True,
    "maximum_increment_direct_storage_relative_defect": 1.0e-9,
    "maximum_mapped_endpoint_path_closure_defect": 2.0e-8,
    "maximum_saved_root_scaled_residual": 1.0e-10,
    "maximum_full_step_model_error_to_base_residual": 0.10,
    "minimum_first_three_model_error_orders": 1.5,
    "require_bitwise_restart_replay": True,
    "may_change_physical_equations": False,
    "may_change_bdf_coefficients_or_history": False,
    "may_change_row_scales_or_merit_norm": False,
    "may_relax_nonlinear_residual_gate": False,
    "physical_history_ladder_execution_authorized": False,
    "fixed_Q_micro_solver_authorized": False,
    "reduced_slow_evolution_authorized": False,
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
    with CATALOG_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "PROSPECTIVE",
                }
            )
    with CATALOG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CATALOG_JSON)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CATALOG_JSON, catalog)


def main() -> None:
    parent = _read(PARENT / "summary.json")
    if (
        parent["passed"]
        or parent["selected_next"] != "cancellation_prone_block_repair"
    ):
        raise RuntimeError("residual-resolution classification changed")
    if not _tracked_tree_is_clean():
        raise RuntimeError("repair manifest requires a clean tracked tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "fixed_Q_exact_increment_storage_repair_manifest_frozen_"
            "implementation_and_saved_endpoint_recertification_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "implementation_authorized": True,
        "saved_endpoint_recertification_authorized": True,
        "physical_history_ladder_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(DIRECTORY / "execution_manifest.json", CONTRACT)
    _write(DIRECTORY / "summary.json", summary)
    _write(
        DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "untracked_files_at_start": _git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
            "runner_sha256": _sha(ROOT / THIS_RUNNER),
            "test_sha256": _sha(ROOT / THIS_TEST),
            "parent_summary_sha256": _sha(PARENT / "summary.json"),
        },
    )
    names = ("execution_manifest.json", "provenance.json", "summary.json")
    (DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
