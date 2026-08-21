#!/usr/bin/env python3
"""Freeze the evidence-led hybrid phase-memory architecture synthesis."""

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

import run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy as cold  # noqa: E402
import run_causal_inner_conservative_transition_tube_wp10c9d6c7c3b5c4f25du as tube  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e0"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25e1"
PARENT_COMMIT = "6c3e60af5cbb816fe9bffb807390976f0329d979"
PARENT_TREE = "f241bfef944dc3b094d4af0c9aa1d4de3a0f6041"
CLASSIFICATION = "hybrid_conservative_phase_memory_architecture_synthesis_frozen"

HIDDEN_FRACTION_GATE = 0.25
DIRECTION_ENERGY_TARGET = 0.9999
MINIMUM_MODE_TURN_DEGREES = 45.0
ONLINE_CYCLE_SECONDS = 578_880.0
MAXIMUM_ONLINE_MACROSTEPS = 100_000

ARTIFACT = "causal_inner_hybrid_phase_memory_architecture_manifest_wp10c9d6c7c3b5c4f25e0"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_hybrid_phase_memory_architecture_manifest_wp10c9d6c7c3b5c4f25e0.py"
THIS_TEST = "tests/test_causal_inner_hybrid_phase_memory_architecture_manifest_wp10c9d6c7c3b5c4f25e0.py"
EXECUTION_RUNNER = "scripts/run_causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1.py"
EXECUTION_TEST = "tests/test_causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1.py"
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HYBRID_PHASE_MEMORY_ARCHITECTURE_"
    "MANIFEST_WP10C9D6C7C3B5C4F25E0_2026-08-21.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CANDIDATE_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_hybrid_candidate_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25dc"
)
TANGENT_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_transition_hidden_tangent_"
    "wp10c9d6c7c3b5c4f25dk"
)
GEOMETRY_DIRECTORY = tube.manifest.geometry.CANONICAL_DIRECTORY


def _validate_parent(*, require_clean: bool) -> dict:
    helper = tube.manifest.geometry
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("phase-memory synthesis parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("phase-memory synthesis parent tree changed")
    cold_hashes = helper._validate_checksums(cold.CANONICAL_DIRECTORY)
    tube_hashes = helper._validate_checksums(tube.CANONICAL_DIRECTORY)
    geometry_hashes = helper._validate_checksums(GEOMETRY_DIRECTORY)
    candidate_hashes = helper._validate_checksums(CANDIDATE_DIRECTORY)
    tangent_hashes = helper._validate_checksums(TANGENT_DIRECTORY)
    cold_summary = helper._read(cold.CANONICAL_DIRECTORY / "summary.json")
    tube_summary = helper._read(tube.CANONICAL_DIRECTORY / "summary.json")
    if (
        cold_summary["passed"]
        or cold_summary["classification"] != cold.FAIL_CLASSIFICATION
        or cold_summary["branch_root_executed"]
        or not tube_summary["passed"]
        or tube_summary["hot_exit_observed"]
    ):
        raise RuntimeError("phase-memory synthesis evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("phase-memory manifest requires a clean tracked tree")
    return {
        "cold_hashes": cold_hashes,
        "tube_hashes": tube_hashes,
        "geometry_hashes": geometry_hashes,
        "candidate_hashes": candidate_hashes,
        "tangent_hashes": tangent_hashes,
    }


def _contract() -> dict:
    helper = tube.manifest.geometry
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "hypotheses": {
            "memoryless_critical_graph": "H(q,h)=Q*F(Lq+Zh)=0",
            "dynamic_phase_tube": (
                "y=D_m(q,s)=Lq+Z*(h_m0+U_m*a_m(q,s))"
            ),
            "reduced_dynamics": {
                "macro": "dq_dt=g_m(q,s)=R*F(D_m(q,s))",
                "phase": (
                    "ds_dt=omega_m(q,s), fitted by tangent projection of "
                    "Q*F-D_qh*g_m"
                ),
                "normal_defect": (
                    "r_perp=Q*F-D_qh*g_m-D_sh*omega_m"
                ),
            },
            "hybrid_event": {
                "guard": "gamma_m(q,s)=0",
                "reset": "(q_plus,s_plus,m_plus)=Psi_m(q_minus,s_minus)",
                "macro_reset": "q_plus=q_minus+Delta_q_event_from_conservative_ledger",
            },
        },
        "decision_gates": {
            "memoryless_graph_rejected_if_all_cold_hidden_fractions_exceed": HIDDEN_FRACTION_GATE,
            "cold_direction_bundle_rank_at_energy": DIRECTION_ENERGY_TARGET,
            "combined_direction_bundle_rank_at_energy": DIRECTION_ENERGY_TARGET,
            "combined_rank_max": 2,
            "mode_switch_turn_degrees_min": MINIMUM_MODE_TURN_DEGREES,
            "transition_tube_parent_must_pass": True,
            "macro_conservation_identity": "R*D_m(q,s)=q",
        },
        "online_contract": {
            "state": "(q_in_R82,s_scalar,mode_discrete)",
            "online_truth_calls": 0,
            "online_470_roots": 0,
            "online_fixed_Q_microsteps": 0,
            "cycle_seconds": ONLINE_CYCLE_SECONDS,
            "maximum_macrosteps": MAXIMUM_ONLINE_MACROSTEPS,
            "minimum_average_macrostep_seconds": (
                ONLINE_CYCLE_SECONDS / MAXIMUM_ONLINE_MACROSTEPS
            ),
            "integrator": "event_detecting_conservative_IMEX_or_RK_with_dense_output",
            "error_control": "macro_ledger_plus_phase_plus_normal_tube_defect",
        },
        "evidence_scope": {
            "architecture_selection_only": True,
            "complete_cycle_truth_available": False,
            "hot_exit_truth_available": False,
            "complete_impulse_map_available": False,
            "full_cycle_predictive_validation_available": False,
            "reduced_cycle_authorized": False,
        },
        "input_hashes": {
            "cold_summary": helper._sha(cold.CANONICAL_DIRECTORY / "summary.json"),
            "cold_metrics": helper._sha(
                cold.CANONICAL_DIRECTORY / "cold_anchor_metrics.json"
            ),
            "cold_arrays": helper._sha(
                cold.CANONICAL_DIRECTORY / "cold_anchor_arrays.npz"
            ),
            "tube_summary": helper._sha(tube.CANONICAL_DIRECTORY / "summary.json"),
            "tube_metrics": helper._sha(
                tube.CANONICAL_DIRECTORY / "tube_metrics.json"
            ),
            "tube_arrays": helper._sha(
                tube.CANONICAL_DIRECTORY / "tube_model_and_validation.npz"
            ),
            "geometry_arrays": helper._sha(GEOMETRY_DIRECTORY / "geometry_arrays.npz"),
            "candidate_arrays": helper._sha(
                CANDIDATE_DIRECTORY / "candidate_geometry_arrays.npz"
            ),
            "tangent_arrays": helper._sha(
                TANGENT_DIRECTORY / "transition_hidden_tangent_arrays.npz"
            ),
        },
        "frozen_source_hashes": {
            name: helper._sha(ROOT / name)
            for name in (THIS_RUNNER, THIS_TEST, EXECUTION_RUNNER, EXECUTION_TEST)
        },
    }


def _update_catalog(summary: dict) -> None:
    helper = tube.manifest.geometry
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
    helper = tube.manifest.geometry
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("phase-memory architecture manifest already exists")
    inputs = _validate_parent(require_clean=True)
    contract = _contract()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "architecture_contract.json", contract)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {"parent_commit": PARENT_COMMIT, "parent_tree": PARENT_TREE, **inputs},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "new_truth_calls": 0,
        "memoryless_branch_graph_preserved_as_rejected": True,
        "dynamic_phase_architecture_test_authorized": True,
        "reduced_cycle_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "DEFINITIONS_ONLY",
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
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
                "# Hybrid phase-memory architecture manifest WP10c9d6c7c3b5c4f25e0",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "The exact cold preflight rejected a nearby memoryless hidden critical graph at every saved cold state. This package prospectively tests the evidence-supported alternative: an exactly conservative 82-coordinate macro ledger, one dynamic scalar phase, and a discrete mode for the sharp full-model/fixed-Q transition.",
                "",
                "The online model may call no 470-dimensional truth residual, root, or fixed-Q microstep. This is architecture selection only; missing hot-exit and complete-cycle truth remain explicit blockers.",
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
