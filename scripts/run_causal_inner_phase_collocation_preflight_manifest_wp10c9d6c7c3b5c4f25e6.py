#!/usr/bin/env python3
"""Audit saved evidence and freeze the phase-collocation preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5 as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e6"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e7"
PARENT_COMMIT = "ed3ef4cfe8ad869b26e5dde107bf457c64cb56fb"
PARENT_TREE = "f0f23db0c6beafb76de6e499100e66a879f11a73"
CLASSIFICATION = "phase_collocation_evidence_audited_and_cold_preflight_frozen"

ARTIFACT = "causal_inner_phase_collocation_preflight_manifest_wp10c9d6c7c3b5c4f25e6"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_phase_collocation_preflight_manifest_wp10c9d6c7c3b5c4f25e6.py"
THIS_TEST = "tests/test_causal_inner_phase_collocation_preflight_manifest_wp10c9d6c7c3b5c4f25e6.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7.py"
EXECUTION_TEST = "tests/test_causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7.py"
CORE_SOURCE = "src/imri_qpe/layer3_minidisk_1d/phase_collocation.py"
CORE_TEST = "tests/test_phase_collocation.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PHASE_COLLOCATION_PREFLIGHT_"
    "MANIFEST_WP10C9D6C7C3B5C4F25E6_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CANDIDATE_ARRAYS = ROOT / (
    "results/canonical/causal_inner_hybrid_candidate_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25dc/candidate_geometry_arrays.npz"
)
COLD_RATE_ARRAYS = ROOT / (
    "results/canonical/causal_inner_cold_branch_anchor_preflight_"
    "wp10c9d6c7c3b5c4f25dy/cold_anchor_arrays.npz"
)
TANGENT_ARRAYS = ROOT / (
    "results/canonical/causal_inner_transition_hidden_tangent_"
    "wp10c9d6c7c3b5c4f25dk/transition_hidden_tangent_arrays.npz"
)
TRANSITION_GEOMETRY = ROOT / (
    "results/canonical/causal_inner_transition_tube_geometry_"
    "wp10c9d6c7c3b5c4f25ds/geometry_arrays.npz"
)

MAXIMUM_STATE_ERROR_OVER_LOCAL_PATH = 1.0e-4
MAXIMUM_RATE_RELATIVE_DEFECT = 1.0e-3
MINIMUM_RATE_DIRECTION_COSINE = 0.9999
MAXIMUM_ONE_VERSUS_TWO_WINDOW_STATE_DEFECT = 1.0e-4
MAXIMUM_ONE_VERSUS_TWO_WINDOW_RATE_DEFECT = 5.0e-4
MAXIMUM_INTERFACE_VALUE_DEFECT = 5.0e-12
MAXIMUM_CONSTRAINT_CONDITION_NUMBER = 1.0e4


def _helper():
    return parent.manifest.rejected.manifest.architecture.manifest.tube.manifest.geometry


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("phase-collocation parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("phase-collocation parent tree changed")
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["predictive_cycle_authorized"]
    ):
        raise RuntimeError("affine phase-engine decision changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("phase-collocation manifest requires a clean tracked tree")
    return {"parent_hashes": hashes}


def _audit_saved_evidence() -> dict:
    helper = _helper()
    candidates = helper._load_npz(CANDIDATE_ARRAYS)
    cold = helper._load_npz(COLD_RATE_ARRAYS)
    transition = helper._load_npz(TRANSITION_GEOMETRY)
    required_cold = []
    for milliseconds in (2, 5, 8, 12):
        prefix = f"candidate_{milliseconds:02d}ms"
        required_cold.extend(
            (
                f"{prefix}__coordinate_jacobian470x560",
                f"{prefix}__scaled_free_rate560_per_s",
            )
        )
    missing_cold = [name for name in required_cold if name not in cold]
    times = np.asarray(candidates["candidate_times_seconds"], dtype=float)
    return {
        "cold_state_count": int(len(times)),
        "cold_state_times_seconds": times.tolist(),
        "cold_exact_full_model_rate_times_seconds": [0.002, 0.005, 0.008, 0.012],
        "cold_exact_full_model_rate_witnesses_available": not missing_cold,
        "cold_missing_keys": missing_cold,
        "transition_state_count": int(len(transition["trajectory_times_seconds"])),
        "transition_interval_count": int(len(transition["trajectory_secants470_per_s"])),
        "transition_saved_continuous_rate_witnesses_available": False,
        "transition_saved_accepted_bdf_witnesses_available": True,
        "post_transition_accepted_state_available": False,
        "hot_exit_state_available": False,
        "complete_cycle_event_sequence_available": False,
    }


def _contract(audit: dict) -> dict:
    helper = _helper()
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "mathematical_architecture": {
            "state": "(q_R82,s_scalar,m_discrete)",
            "decoder": "D_m(q,s)=L*q+Z*(h_m0+U_m*a_m(s))",
            "phase_dynamics": "q_dot=g_m(q,s), s_dot=omega_m(q,s)>0",
            "full_coordinate_defect": "D_qD_m*g_m+D_sD_m*omega_m-F_m(D_m)",
            "offline_discretization": "multiple_shooting_with_8_primary_and_16_refined_Legendre_Gauss_Lobatto_nodes",
            "online_truth_calls": 0,
        },
        "stage_2_cold_replay": {
            "segments_seconds": [[0.002, 0.008], [0.008, 0.020]],
            "held_out_state_and_rate_times_seconds": [0.005, 0.012],
            "rate_witness": "saved_exact_unconstrained_full_model_continuous_rate_mapped_by_saved_exact_coordinate_Jacobian",
            "new_truth_calls": 0,
            "binding_gates": {
                "maximum_state_error_over_local_path": MAXIMUM_STATE_ERROR_OVER_LOCAL_PATH,
                "maximum_rate_relative_defect": MAXIMUM_RATE_RELATIVE_DEFECT,
                "minimum_rate_direction_cosine": MINIMUM_RATE_DIRECTION_COSINE,
                "maximum_one_versus_two_window_state_defect": MAXIMUM_ONE_VERSUS_TWO_WINDOW_STATE_DEFECT,
                "maximum_one_versus_two_window_rate_defect": MAXIMUM_ONE_VERSUS_TWO_WINDOW_RATE_DEFECT,
                "maximum_interface_value_defect": MAXIMUM_INTERFACE_VALUE_DEFECT,
                "maximum_constraint_condition_number": MAXIMUM_CONSTRAINT_CONDITION_NUMBER,
            },
        },
        "evidence_boundary": {
            "cold_full_vector_field_replay_authorized": True,
            "transition_full_vector_field_claim_from_secants_forbidden": True,
            "post_transition_extrapolation_as_truth_forbidden": True,
            "complete_cycle_calibration_missing": True,
            "predictive_cycle_authorized": False,
            "audit": audit,
        },
        "input_hashes": {
            "parent_summary": helper._sha(parent.CANONICAL_DIRECTORY / "summary.json"),
            "parent_metrics": helper._sha(parent.CANONICAL_DIRECTORY / "affine_engine_metrics.json"),
            "parent_arrays": helper._sha(parent.CANONICAL_DIRECTORY / "affine_engine_model_and_replay.npz"),
            "candidate_arrays": helper._sha(CANDIDATE_ARRAYS),
            "cold_rate_arrays": helper._sha(COLD_RATE_ARRAYS),
            "tangent_arrays": helper._sha(TANGENT_ARRAYS),
            "transition_geometry": helper._sha(TRANSITION_GEOMETRY),
        },
        "frozen_source_hashes": {
            name: helper._sha(ROOT / name)
            for name in (
                THIS_RUNNER,
                THIS_TEST,
                CORE_SOURCE,
                CORE_TEST,
                EXECUTION_RUNNER,
                EXECUTION_TEST,
            )
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
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
                    "sha256": helper._sha(path),
                    "scientific_status": "DEFINITIONS_ONLY",
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
    catalog = helper._read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": PARENT_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("phase-collocation manifest already exists")
    locked = _validate_parent(require_clean=True)
    audit = _audit_saved_evidence()
    if not audit["cold_exact_full_model_rate_witnesses_available"]:
        raise RuntimeError("saved cold vector-field evidence is incomplete")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "phase_collocation_contract.json", _contract(audit))
    helper._write_json(CANONICAL_DIRECTORY / "evidence_audit.json", audit)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "cold_full_vector_field_replay_authorized": True,
        "transition_full_vector_field_replay_authorized": False,
        "post_transition_segment_authorized": False,
        "predictive_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Phase-collocation preflight manifest WP10c9d6c7c3b5c4f25e6",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The saved cold database contains four independent exact full-model continuous-rate witnesses, so a no-new-truth phase-collocation replay is authorized. The transition database contains accepted BDF witnesses but no saved continuous-rate witnesses; secants may not be relabeled as a full vector-field defect.",
                "",
                "The post-transition segment, hot exit, and predictive cycle remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("use --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
