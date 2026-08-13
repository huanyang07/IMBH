#!/usr/bin/env python3
"""Freeze the selected-time absolute coupling-flux localization contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_absolute_baseline_observable_memory_screen_wp10c9d6c7c3b5c4f1 as c4f1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f2"
ANALYZED_CERTIFICATE_COMMIT = c4f1.ANALYZED_CERTIFICATE_COMMIT
ARTIFACT = "causal_inner_absolute_coupling_localization_manifest_wp10c9d6c7c3b5c4f2"
THIS_RUNNER = "scripts/run_causal_inner_absolute_coupling_localization_manifest_wp10c9d6c7c3b5c4f2.py"
THIS_TEST = "tests/test_causal_inner_absolute_coupling_localization_manifest_wp10c9d6c7c3b5c4f2.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_ABSOLUTE_COUPLING_LOCALIZATION_MANIFEST_WP10C9D6C7C3B5C4F2_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "localization_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TIMES_MICROSECONDS = (5000, 10000, 16000, 20000)
PARENT_FACE_INDICES = (2, 40, 44, 46, 47, 48)
TRANSITION_FACE = 48
INTERIOR_CONTROL_FACES = (44, 46, 47)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "absolute_coupling_localization_manifest_frozen_existing_state_audit_authorized",
        "definitions_only": True,
        "new_trajectory": False,
        "operator_change": False,
        "selected_times_microseconds": TIMES_MICROSECONDS,
        "selected_parent_faces": PARENT_FACE_INDICES,
        "transition_parent_face": TRANSITION_FACE,
        "interior_control_faces": INTERIOR_CONTROL_FACES,
        "layouts": c4f1.LAYOUT_LABELS,
        "decomposition": {
            "actual": "evaluate_each_committed_base_state_on_its_native_layout",
            "shared_parent_reference": "restrict_fine_state_conservatively_then_repeat_parent_cell_average_on_each_layout",
            "native_state_part": "actual_native_flux_minus_shared_parent_reference_flux",
            "operator_part": "shared_parent_reference_flux_difference_across_layouts",
            "exact_sum_closure_required": True,
            "shared_parent_reference_is_diagnostic_not_a_physical_lift": True,
        },
        "prospective_gates": {
            "minimum_spatial_order": 0.75,
            "minimum_error_direction_cosine": 0.90,
            "maximum_decomposition_closure_defect": 1.0e-12,
            "maximum_flux_ledger_defect": 1.0e-12,
            "minimum_interior_faces_with_consistent_direction": 2,
            "maximum_fine_complement_fraction_of_middle_fine_difference": 0.10,
            "maximum_shared_parent_operator_fraction_of_middle_fine_difference_for_state_classification": 0.10,
            "minimum_shared_parent_operator_fraction_for_operator_classification": 0.50,
        },
        "decision_tree": {
            "transition_only_and_operator_fraction_at_least_half": "transition_operator_baseline_localized_static_followup_manifest_only",
            "transition_only_and_operator_fraction_at_most_point_one": "base_state_alignment_localized_fine_anchored_baseline_manifest_only",
            "multiple_interior_faces_reverse": "distributed_absolute_baseline_failure_reduction_blocked",
            "fine_complement_above_point_one": "fine_complement_exact_JVP_or_static_refinement_manifest_only",
            "inconclusive_mixed_decomposition": "no_memory_propagation_targeted_localization_only",
        },
        "response_certificate_preserved": True,
        "absolute_closure_fit_authorized": False,
        "observable_memory_propagation_authorized": False,
        "fixed_Q_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "hard_stops": (
            "do_not_relax_the_failed_absolute_direction_gate",
            "do_not_call_repeated_parent_averages_a_physical_lift",
            "do_not_run_a_new_trajectory_or_N1024_rescue",
            "do_not_start_memory_fixed_Q_or_reduced_evolution",
            "do_not_use_the_raw_excision_face_flux_as_slow_export",
        ),
        "authorized_next": "WP10c9d6c7c3b5c4f3_selected_time_absolute_coupling_localization",
    }


def _update_catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha256(path), "scientific_status": "PROSPECTIVE"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": True}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "latest_source_parent_commit": ANALYZED_CERTIFICATE_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write(CANONICAL_SUMMARY, catalog)


def main():
    parent = _read(c4f1.SUMMARY_PATH)
    if parent["classification"] != "absolute_extraction_baseline_direction_gate_failed_observable_memory_propagation_not_executed" or parent["physical_failure_detected"] or parent["observable_memory_propagation_executed"]:
        raise RuntimeError("c4f2 predecessor status changed")
    manifest = _manifest()
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": manifest["classification"], "passed": True, "definitions_only": True, "parent_negative_result_preserved": True, "response_certificate_preserved": True, "new_trajectory_authorized": False, "observable_memory_propagation_authorized": False, "fixed_Q_micro_solver_authorized": False, "reduced_slow_evolution_authorized": False, "physical_failure_detected": False, "authorized_next": manifest["authorized_next"]}
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "times_microseconds": TIMES_MICROSECONDS, "parent_faces": PARENT_FACE_INDICES, "layout_multipliers": (1, 2, 4)})
    _write(MANIFEST_PATH, manifest); _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Absolute coupling-localization manifest", "", f"Classification: `{summary['classification']}`.", "", "This definitions-only package freezes a four-time, six-face decomposition using committed states. It introduces no physical lift, no trajectory, and no relaxation of the failed absolute baseline gate.", "", "The next analysis distinguishes native-state and shared-parent operator contributions at and immediately inside the embedded transition. Memory and fixed-Q propagation remain blocked.", "")), encoding="utf-8")
    provenance = {"schema_version": SCHEMA_VERSION, "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT, "execution_head": _git("rev-parse", "HEAD"), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "input_summary_sha256": _sha256(c4f1.SUMMARY_PATH), "output_hashes": {}}
    provenance["output_hashes"] = {str(path.relative_to(ROOT)): _sha256(path) for path in (CONFIG_PATH, MANIFEST_PATH, SUMMARY_PATH, REPORT_PATH)}
    _write(PROVENANCE_PATH, provenance)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in (CONFIG_PATH, MANIFEST_PATH, SUMMARY_PATH, PROVENANCE_PATH)), encoding="utf-8")
    _update_catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
