#!/usr/bin/env python3
"""Freeze and run the Schur-repaired bounded fixed-Q primary retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_fixed_q_primary_case_recovery_wp10c9d6c7c3b5c4f24e6 as e6  # noqa: E402


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e8"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e8"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_retry_wp10c9d6c7c3b5c4f24e8"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
SCHUR_AUDIT_DIRECTORY = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_schur_solve_audit_"
    "wp10c9d6c7c3b5c4f24e7"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e8.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e8.py"
)
CONTRACT = {
    **e6.CONTRACT,
    "required_schur_solve_method": (
        "row_column_equilibrated_LU_refined_1"
    ),
    "required_schur_audit_classification": (
        "fixed_Q_schur_solve_audit_passed_implementation_authorized"
    ),
}


def _schur_summary() -> dict:
    summary = e6._read(SCHUR_AUDIT_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["classification"]
        != CONTRACT["required_schur_audit_classification"]
        or summary["selected_method"]
        != CONTRACT["required_schur_solve_method"]
        or not summary["schur_solve_implementation_authorized"]
    ):
        raise RuntimeError("fixed-Q Schur solve authorization changed")
    return summary


def _configure() -> None:
    e6.WORK_PACKAGE = WORK_PACKAGE
    e6.MANIFEST_ARTIFACT = MANIFEST_ARTIFACT
    e6.RESULT_ARTIFACT = RESULT_ARTIFACT
    e6.MANIFEST_DIRECTORY = MANIFEST_DIRECTORY
    e6.RESULT_DIRECTORY = RESULT_DIRECTORY
    e6.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
    e6.THIS_RUNNER = THIS_RUNNER
    e6.THIS_TEST = THIS_TEST
    e6.CONTRACT = CONTRACT


def _refresh_checksums(directory: Path, names: tuple[str, ...]) -> None:
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{e6._sha(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _freeze() -> dict:
    _schur_summary()
    _configure()
    summary = e6._freeze()
    summary["classification"] = (
        "fixed_Q_schur_repaired_primary_retry_manifest_frozen_"
        "bounded_execution_authorized"
    )
    summary["schur_solve_method"] = CONTRACT["required_schur_solve_method"]
    e6._write(MANIFEST_DIRECTORY / "summary.json", summary)
    provenance = e6._read(MANIFEST_DIRECTORY / "provenance.json")
    provenance.update(
        {
            "schur_audit_summary_sha256": e6._sha(
                SCHUR_AUDIT_DIRECTORY / "summary.json"
            ),
            "schur_audit_metrics_sha256": e6._sha(
                SCHUR_AUDIT_DIRECTORY / "metrics.json"
            ),
            "selected_schur_solve_method": CONTRACT[
                "required_schur_solve_method"
            ],
        }
    )
    e6._write(MANIFEST_DIRECTORY / "provenance.json", provenance)
    _refresh_checksums(
        MANIFEST_DIRECTORY,
        ("execution_manifest.json", "provenance.json", "summary.json"),
    )
    e6._catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _execute() -> dict:
    _schur_summary()
    _configure()
    payload = e6._execute()
    payload["summary"]["schur_solve_method"] = CONTRACT[
        "required_schur_solve_method"
    ]
    payload["metrics"]["schur_audit_summary_sha256"] = e6._sha(
        SCHUR_AUDIT_DIRECTORY / "summary.json"
    )
    payload["metrics"]["schur_audit_metrics_sha256"] = e6._sha(
        SCHUR_AUDIT_DIRECTORY / "metrics.json"
    )
    e6._write(RESULT_DIRECTORY / "summary.json", payload["summary"])
    e6._write(RESULT_DIRECTORY / "metrics.json", payload["metrics"])
    provenance = e6._read(RESULT_DIRECTORY / "provenance.json")
    provenance.update(
        {
            "schur_audit_summary_sha256": payload["metrics"][
                "schur_audit_summary_sha256"
            ],
            "schur_audit_metrics_sha256": payload["metrics"][
                "schur_audit_metrics_sha256"
            ],
            "selected_schur_solve_method": CONTRACT[
                "required_schur_solve_method"
            ],
        }
    )
    e6._write(RESULT_DIRECTORY / "provenance.json", provenance)
    _refresh_checksums(
        RESULT_DIRECTORY,
        (
            "contract.json",
            "decisive_arrays.npz",
            "metrics.json",
            "provenance.json",
            "summary.json",
        ),
    )
    e6._catalog(
        RESULT_DIRECTORY,
        RESULT_ARTIFACT,
        payload["summary"],
        "SUPPORTED" if payload["summary"]["passed"] else "REJECTED",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.freeze == arguments.execute:
        raise SystemExit("select exactly one of --freeze or --execute")
    payload = _freeze() if arguments.freeze else _execute()
    print(json.dumps(e6._plain(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
