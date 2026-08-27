#!/usr/bin/env python3
"""Partition checksum-locked legacy cycle evidence for the 11-field solver."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_legacy_cycle_evidence_compatibility_manifest_wp10c9d6c7c3b5c4f25fizzp as manifest  # noqa: E402


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "legacy_cycle_evidence_partitioned_no_direct_eleven_field_cycle_input"
FAIL_CLASSIFICATION = "legacy_cycle_evidence_compatibility_audit_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_legacy_cycle_evidence_compatibility_audit_"
    "wp10c9d6c7c3b5c4f25fizzp1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_LEGACY_CYCLE_EVIDENCE_"
    "COMPATIBILITY_AUDIT_WP10C9D6C7C3B5C4F25FIZZP1_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_legacy_cycle_evidence_compatibility_"
    "audit_wp10c9d6c7c3b5c4f25fizzp1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_legacy_cycle_evidence_compatibility_"
    "audit_wp10c9d6c7c3b5c4f25fizzp1.py"
)
PARENT_SHA256 = "d9b3cfa138db7f06ca802bf67f2cb0f93d6fe290ea4f4a5aa3dcafec9c8bb325"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _read(label: str, name: str = "summary.json"):
    specification = manifest.LEGACY_INPUTS[label]
    return _u()._read_json(manifest._legacy_directory(specification) / name)


def _shape(label: str, filename: str, array: str):
    specification = manifest.LEGACY_INPUTS[label]
    with np.load(manifest._legacy_directory(specification) / filename, allow_pickle=False) as payload:
        return tuple(int(value) for value in payload[array].shape)


def _validate_manifest(*, require_clean: bool = False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("legacy compatibility manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(manifest.CANONICAL_DIRECTORY / "compatibility_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["decision"]["pass_authorized_next"] != AUTHORIZED_NEXT
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("legacy compatibility manifest classification changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("legacy compatibility audit needs a clean tracked tree")
    return hashes, contract


def _audit():
    _, contract = _validate_manifest()
    complete_summary = _read("complete_cycle_attempt")
    complete_metrics = _read("complete_cycle_attempt", "cycle_execution_metrics.json")
    boundary = _read("five_field_fixed_exterior_boundary")
    geometry = _read("transition_geometry")
    tube = _read("conservative_transition_tube")
    terminal = _read("terminal_hot_exit_recovery")
    branch = _read("branch_first_impulse_architecture")
    cycle_map = _read("legacy_cycle_map")

    trajectory_shape = _shape("complete_cycle_attempt", "cycle_execution_arrays.npz", "trajectory_primitive_states")
    witness_shape = _shape("complete_cycle_attempt", "exact_witness_arrays.npz", "primitive_states")
    coordinate_shape = _shape("complete_cycle_attempt", "cycle_execution_arrays.npz", "trajectory_coordinates")
    macro_shape = _shape("complete_cycle_attempt", "cycle_execution_arrays.npz", "trajectory_macro82")
    boundary_trace_shape = _shape("five_field_fixed_exterior_boundary", "decisive_arrays.npz", "interface_trace_acoustic_shear")
    transition_shape = _shape("transition_geometry", "geometry_arrays.npz", "trajectory_coordinates470")
    terminal_shape = _shape("terminal_hot_exit_recovery", "checkpoint_step_12.npz", "current_primitive_charts")

    accepted_macrosteps = int(complete_metrics["gate_values"]["accepted_macrosteps"])
    initial_macrostep = float(complete_metrics["input_lock"]["contract"]["adaptive_acquisition"]["macro_step_seconds_initial"])
    observed_seconds = accepted_macrosteps * initial_macrostep
    cycle_seconds = float(contract["target_architecture"]["cycle_seconds"])

    inventory = [
        {
            "label": "legacy_five_field_profiles",
            "classification": "candidate_seed_after_deterministic_lift_and_new_audit",
            "payloads": ["65 accepted trajectory profiles", "192 exact-witness profiles"],
            "required_before_binding_use": [
                "conservative 112-node to 94-cell radial remap",
                "physical five-to-eleven-field entropy-port lift",
                "missing shear/vertical coordinate initialization audit",
                "new local port, source, trust-radius, and guard audits",
            ],
        },
        {
            "label": "legacy_fixed_exterior_characteristics",
            "classification": "candidate_seed_after_deterministic_lift_and_new_audit",
            "payloads": ["Kerr-Schild geometry", "inner all-outgoing/excision sign evidence"],
            "required_before_binding_use": [
                "eleven-field characteristic decomposition",
                "eleven-field entropy-stable SAT/port closure",
                "physical outer loading or outflow data",
            ],
        },
        {
            "label": "legacy_transition_tube_and_phase_path",
            "classification": "diagnostic_only",
            "payloads": ["470-coordinate local tube", "82-coordinate phase/mode prefix"],
            "reason": "only a 1.1 microsecond observed local segment; hot exit and full reset were not observed",
        },
        {
            "label": "legacy_hot_terminal_state",
            "classification": "candidate_seed_after_deterministic_lift_and_new_audit",
            "payloads": ["accepted 20.0017 ms five-field checkpoint"],
            "reason": "valid physical seed, but not an exit section or reset endpoint",
        },
        {
            "label": "legacy_branch_and_cycle_architectures",
            "classification": "diagnostic_only",
            "payloads": ["branch-first reset specification", "83-dimensional cycle-map prefix"],
            "reason": "definitions/prefix only; complete-cycle calibration and anchor-specific reset are missing",
        },
        {
            "label": "unobserved_hot_exit_and_complete_cycle",
            "classification": "binding_negative_evidence",
            "payloads": ["no hot exit", "no cycle return", "no complete impulse map"],
            "reason": "these absent observations may not be imputed or extrapolated",
        },
    ]

    facts = {
        "trajectory_primitive_shape": trajectory_shape,
        "exact_witness_primitive_shape": witness_shape,
        "trajectory_coordinate_shape": coordinate_shape,
        "trajectory_macro_shape": macro_shape,
        "boundary_trace_shape": boundary_trace_shape,
        "transition_coordinate_shape": transition_shape,
        "terminal_primitive_shape": terminal_shape,
        "accepted_macrosteps": accepted_macrosteps,
        "exact_free_field_witnesses": int(complete_metrics["gate_values"]["exact_free_field_witnesses"]),
        "observed_physical_seconds": observed_seconds,
        "target_cycle_seconds": cycle_seconds,
        "time_coverage_fraction": observed_seconds / cycle_seconds,
        "old_cycle_observed": bool(complete_metrics["gate_values"]["cycle_observed"]),
        "old_hot_exit_observed": bool(geometry["hot_exit_observed"] or tube["hot_exit_observed"] or terminal["hot_exit_reached"] or cycle_map["hot_exit_observed"]),
        "old_boundary_field_count": int(boundary_trace_shape[-1]),
        "old_boundary_maximum_closure_defect": float(boundary["maximum_characteristic_boundary_closure_defect"]),
        "old_boundary_inner_incoming_characteristics": int(boundary["per_reference"]["N769"]["incoming_inner_boundary_characteristic_count"]),
        "target_field_count": int(contract["target_architecture"]["fields_per_cell"]),
        "transition_duration_seconds": float(geometry["trajectory_duration_seconds"]),
        "terminal_elapsed_seconds": float(terminal["elapsed_time_seconds"]),
        "terminal_budget_exhausted": bool(terminal["budget_exhausted"]),
        "branch_truth_executed": bool(branch["branch_truth_execution_authorized"]),
        "cycle_map_complete_calibration_missing": bool(cycle_map["complete_cycle_calibration_missing"]),
        "legacy_complete_attempt_passed": bool(complete_summary["passed"]),
    }
    passed = bool(
        trajectory_shape == (65, 112, 5)
        and witness_shape == (192, 112, 5)
        and coordinate_shape == (65, 470)
        and macro_shape == (65, 82)
        and boundary_trace_shape[-1] == 5
        and transition_shape == (18, 470)
        and terminal_shape == (112, 5)
        and boundary["passed"]
        and boundary["maximum_characteristic_boundary_closure_defect"] <= 1.0e-10
        and not facts["old_cycle_observed"]
        and not facts["old_hot_exit_observed"]
        and facts["target_field_count"] == 11
        and not complete_summary["passed"]
        and geometry["passed"]
        and tube["passed"]
        and terminal["passed"]
        and terminal["budget_exhausted"]
        and branch["definitions_only"]
        and cycle_map["complete_cycle_calibration_missing"]
    )
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "input_package_count": len(manifest.LEGACY_INPUTS),
        "direct_binding_reuse_count": 0,
        "candidate_seed_classes_after_lift": 3,
        "diagnostic_only_classes": 2,
        "binding_negative_evidence_classes": 1,
        "facts": facts,
        "compatibility_inventory": inventory,
        "required_new_work": [
            "conservative radial remap and physical five-to-eleven-field lift",
            "cycle-wide anchor coverage and withheld-window plan",
            "eleven-field inner/outer physical boundary port certificate",
            "impact and hot-exit guard/reset acquisition",
            "held-out global phase-window validation",
        ],
        "new_truth_calls": 0,
        "new_trajectory_steps": 0,
        "cycle_wide_inputs_complete": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }


def _update_catalog(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("legacy compatibility audit already exists")
    hashes, _ = _validate_manifest(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "compatibility_audit.json", metrics)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "legacy_evidence_compatibility_audited": metrics["passed"], "direct_binding_reuse_count": metrics["direct_binding_reuse_count"], "new_lift_and_truth_acquisition_required": True, "cycle_wide_inputs_complete": False, "complete_cycle_execution_authorized": False, "complete_cycle_steps": 0, "authorized_next": metrics["authorized_next"]}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    facts = metrics["facts"]
    REPORT_PATH.write_text(f"# Legacy cycle-evidence compatibility audit\n\nClassification: `{metrics['classification']}`.\n\nAll seven legacy packages close their frozen checksums, but none is a direct binding input to the 94-cell, eleven-field production solver. The old attempt contains 65 accepted five-field trajectory profiles and 192 five-field exact witnesses, yet spans only `{facts['observed_physical_seconds']:.6f}` s of a `{facts['target_cycle_seconds']:.0f}` s cycle (`{facts['time_coverage_fraction']:.3e}` of the target duration). These profiles are candidate seeds only after conservative radial remapping and a physical five-to-eleven-field lift.\n\nThe certified old fixed-exterior characteristic closure is five-field evidence. Its geometry and inner-excision sign are useful seeds, but an eleven-field entropy-port boundary audit is still binding. The local transition tube is diagnostic because hot exit was never observed; no complete reset/impulse map exists. No cycle step is authorized.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}, "numpy": np.__version__})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run: parser.error("choose --run")
    metrics = _audit(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0 if _canonicalize(metrics)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
