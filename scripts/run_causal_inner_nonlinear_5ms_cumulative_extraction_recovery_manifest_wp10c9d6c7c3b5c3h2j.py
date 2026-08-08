#!/usr/bin/env python3
"""Freeze the cumulative 5 ms extraction-recovery scan."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_extraction_surface_certificate_wp10c9d6c7c3b5c3h2i1 as h2i1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2j"
ANALYZED_BASE_COMMIT = "4f3359a2070e90929a8002560c047bb3fa73c378"
ANALYZED_BASE_PARENT = "37a4fec66c6da5182202d467261cbcfa64093c11"
ANALYZED_BASE_TREE = "0e91799584eb9eb4d5cbbca8906fe5d0e835c7e3"

ARTIFACT = (
    "causal_inner_nonlinear_5ms_cumulative_extraction_recovery_manifest_"
    "wp10c9d6c7c3b5c3h2j"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_"
    "manifest_wp10c9d6c7c3b5c3h2j.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_"
    "manifest_wp10c9d6c7c3b5c3h2j.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_CUMULATIVE_"
    "EXTRACTION_RECOVERY_MANIFEST_WP10C9D6C7C3B5C3H2J_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "audit_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

CANDIDATE_COARSE_FACE_INDICES = (1, 2, 4, 8, 12, 16, 24, 32, 40)
LAYOUT_FACE_MULTIPLIERS = (1, 2, 4)
MINIMUM_CONSECUTIVE_PASSING_SURFACES = 2


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _validate_parent() -> dict:
    parent = _read_json(h2i1.SUMMARY_PATH)
    instant = parent["analysis"]["instantaneous_exterior_partition"]
    cumulative = parent["analysis"]["cumulative_exterior_partition"]
    if (
        parent["passed"]
        or not instant["binding_channel_passed"]
        or cumulative["binding_channel_passed"]
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3h2j_cumulative_extraction_recovery_manifest"
        or parent["fourth_duration_rung_manifest_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("h2j authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2j analyzed identity changed")
    return parent


def _manifest() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "cumulative_extraction_recovery_manifest_frozen_"
            "nested_common_surface_audit_authorized"
        ),
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "parent_fixed_surface_rejection_preserved": True,
        "candidate_coarse_face_indices": CANDIDATE_COARSE_FACE_INDICES,
        "layout_face_multipliers": LAYOUT_FACE_MULTIPLIERS,
        "minimum_consecutive_passing_surfaces": (
            MINIMUM_CONSECUTIVE_PASSING_SURFACES
        ),
        "selection_rule": (
            "select_the_innermost_surface_only_when_it_and_the_next_declared_"
            "surface_pass_both_full_window_instantaneous_and_cumulative_"
            "thirteen_export_contracts"
        ),
        "binding_window_seconds": (0.002, 0.005),
        "no_shortened_window_may_certify": True,
        "observable_names": h2i1.OBSERVABLE_NAMES,
        "spatial_gates": h2i1.SPATIAL_GATES,
        "temporal_gates": h2i1.TEMPORAL_GATES,
        "required_audits": {
            "same_physical_surface_across_layouts": True,
            "maximum_exterior_prefix_direct_identity_defect": 1.0e-12,
            "maximum_shared_conservative_face_defect": 1.0e-12,
            "incoming_excision_characteristics": 0,
            "all_exact_common_target_bits_required": True,
        },
        "decision_tree": (
            "two_consecutive_surfaces_pass__certify_innermost_selected_extraction_partition_and_authorize_fourth_duration_manifest",
            "instantaneous_passes_but_no_cumulative_recovery__localize_early_time_storage_and_source_buffer",
            "instantaneous_recovery_lost__reject_extraction_partition_route",
        ),
        "hard_stops": (
            "do_not_propagate_new_state",
            "do_not_change_operator_or_production_defaults",
            "do_not_select_a_surface_after_seeing_results_except_by_frozen_innermost_rule",
            "do_not_shorten_the_binding_two_to_five_ms_window",
            "do_not_relabel_extraction_flux_as_pointwise_horizon_flux",
            "do_not_start_fourth_duration_fixed_Q_or_reduction_before_a_pass",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2j1_cumulative_extraction_recovery_audit"
        ),
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha256(path), "scientific_status": "CERTIFIED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": ANALYZED_BASE_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "cumulative_recovery_audit_authorized": True,
        "new_propagation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "analyzed_base_commit": ANALYZED_BASE_COMMIT, "candidate_coarse_face_indices": CANDIDATE_COARSE_FACE_INDICES, "layout_face_multipliers": LAYOUT_FACE_MULTIPLIERS, "minimum_consecutive_passing_surfaces": MINIMUM_CONSECUTIVE_PASSING_SURFACES, "spatial_gates": h2i1.SPATIAL_GATES, "temporal_gates": h2i1.TEMPORAL_GATES})
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "source_parent_commit": ANALYZED_BASE_COMMIT, "scientific_status": "CERTIFIED", "working_head": _git_value("rev-parse", "HEAD"), "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "parent_summary_sha256": _sha256(h2i1.SUMMARY_PATH), "implementation_source_hashes": _source_identity(), "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}"})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Nonlinear 5 ms cumulative extraction-recovery manifest WP10c9d6c7c3b5c3h2j", "", "## Classification", "", f"`{manifest['classification']}`", "", "This definitions-only package preserves the failed fixed-surface cumulative certificate. It freezes a scan over nine already-declared nested common surfaces and selects the innermost surface only if it and the next surface pass both instantaneous and cumulative 13-export contracts over the complete 2-5 ms common window.", "", "No shorter window, new state propagation, operator change, or pointwise-horizon reinterpretation is allowed. Fourth duration, fixed-Q, and reduced slow evolution remain blocked pending h2j1.", "")), encoding="utf-8")
    names = ("audit_manifest.json", "config.json", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
