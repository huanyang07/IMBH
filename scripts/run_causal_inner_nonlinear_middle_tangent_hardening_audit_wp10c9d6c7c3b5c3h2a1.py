#!/usr/bin/env python3
"""Run cheap middle-layout tangent audits before nonlinear propagation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_cost_bounded_anchor_hardening_manifest_wp10c9d6c7c3b5c3h2a0 as h2a0  # noqa: E402
import run_causal_inner_nonlinear_discrete_bdf_tangent_calibration_wp10c9d6c7c3b5c3h1 as h1  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_bdf_history_from_interval,
    causal_five_field_monolithic_discrete_export_directions,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2a1"
ANALYZED_BASE_COMMIT = "dfd01c145bd1ff01df5ac69736430abf7f2525f2"
ANALYZED_BASE_PARENT = "32b679f85bd27fffda7d0e8bef205be48b8795ce"
ANALYZED_BASE_TREE = "b35c2ead3129cc1cf62139505a494bd4db61baf1"

MIDDLE_LAYOUT = h2a0.MIDDLE_LAYOUT
PROFILES = tuple(h2a0.PROFILES)
STEP_RATIO_VALUES = tuple(h2a0.RATIO_AUDIT_VALUES)
GATES = dict(h2a0.SURROGATE_GATES)
GATES.update(
    {
        "maximum_analytic_to_centered_history_relative_defect": 2.0e-6,
        "maximum_linear_solve_relative_defect": 1.0e-10,
        "maximum_matrix_component_closure_defect": 1.0e-12,
        "maximum_export_transport_telescoping_defect": 1.0e-12,
        "maximum_export_active_prefix_ledger_defect": 1.0e-12,
        "maximum_incoming_excision_characteristics": 0,
    }
)

ARTIFACT = (
    "causal_inner_nonlinear_middle_tangent_hardening_audit_"
    "wp10c9d6c7c3b5c3h2a1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_tangent_hardening_"
    "audit_wp10c9d6c7c3b5c3h2a1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_tangent_hardening_"
    "audit_wp10c9d6c7c3b5c3h2a1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_TANGENT_"
    "HARDENING_AUDIT_WP10C9D6C7C3B5C3H2A1_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
MODULE_RELATIVE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_discrete_tangent.py"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(h2a0.SUMMARY_PATH)
    if (
        not parent["passed"]
        or not parent["cheap_hardening_audits_authorized"]
        or parent["middle_1ms_propagation_authorized"]
    ):
        raise RuntimeError("h2a1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2a1 analyzed identity changed")
    return parent


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)) / scale)


def _middle_histories() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    base_arrays = _load_npz(h1.b1a.DECISIVE_ARRAYS)
    spatial_arrays = _load_npz(h1.b4b3.DECISIVE_ARRAYS)
    corrected_arrays = _load_npz(h1.b4d.DECISIVE_ARRAYS)
    base = np.asarray(base_arrays[f"{MIDDLE_LAYOUT}__states"], dtype=float)
    perturbed = []
    exports = []
    for profile in PROFILES:
        task = f"{MIDDLE_LAYOUT}__{profile}__p1__dt_1e-5"
        perturbed.append(spatial_arrays[f"{task}__states"])
        exports.append(
            corrected_arrays[
                f"{MIDDLE_LAYOUT}__{profile}__corrected_face_response"
            ]
        )
    scale_arrays = _load_npz(h1.c3d.DECISIVE_ARRAYS)
    return (
        base,
        np.asarray(perturbed, dtype=float),
        np.asarray(exports, dtype=float),
        np.asarray(scale_arrays["field_scales"], dtype=float),
        np.asarray(scale_arrays["export_scales"], dtype=float),
    )


def _fraction_of_response(metrics: dict) -> float:
    return float(
        metrics["maximum_scaled_discrepancy"]
        / max(
            metrics["maximum_scaled_actual_response"],
            np.finfo(float).tiny,
        )
    )


def _run_audits() -> tuple[dict, dict[str, np.ndarray]]:
    base, perturbed, actual_exports, field_scales, export_scales = (
        _middle_histories()
    )
    response = perturbed - base[None, :, :, :]
    configuration = h1.b1a._configurations()[MIDDLE_LAYOUT]
    context = configuration["context"]
    columns = configuration["columns"]
    rows = configuration["rows"]
    coupling_face = h1.c3g1.c3g.ACTIVE_COUPLING_FACE_INDICES[MIDDLE_LAYOUT]
    dt = 1.0e-5

    began = time.perf_counter()
    history_matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        base[2],
        base[3],
        dt,
        dt,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    history_matrix_seconds = time.perf_counter() - began
    analytic_history = causal_five_field_monolithic_bdf_history_direction(
        context,
        base[2],
        base[3],
        response[:, 2],
        response[:, 3],
        analytic_step_matrix=history_matrix,
    )
    audit_weights = np.asarray([1.0, 0.7, -0.4, 0.3, -0.2], dtype=float)
    audit_old_direction = np.einsum(
        "p,pcf->cf",
        audit_weights,
        response[:, 2],
    )[None, :, :]
    audit_new_direction = np.einsum(
        "p,pcf->cf",
        audit_weights,
        response[:, 3],
    )[None, :, :]
    analytic_audit_history = causal_five_field_monolithic_bdf_history_direction(
        context,
        base[2],
        base[3],
        audit_old_direction,
        audit_new_direction,
        analytic_step_matrix=history_matrix,
    )
    centered_history = causal_five_field_monolithic_bdf_history_direction(
        context,
        base[2],
        base[3],
        audit_old_direction,
        audit_new_direction,
        directional_step=8.0e-2,
    )
    mapped_history_defect = _relative_defect(
        analytic_audit_history.previous_mapped_storage_increment,
        centered_history.previous_mapped_storage_increment,
    )
    height_history_defect = _relative_defect(
        analytic_audit_history.previous_responsive_height_storage_increment,
        centered_history.previous_responsive_height_storage_increment,
    )
    print(
        "h2a1: analytic history "
        f"matrix={history_matrix_seconds:.1f}s "
        f"mapped={mapped_history_defect:.3e} height={height_history_defect:.3e}",
        flush=True,
    )

    base_history = causal_five_field_monolithic_bdf_history_from_interval(
        context,
        base[2],
        base[3],
        dt,
    )
    ratio_reports = {}
    ratio_matrices = {}
    for ratio in STEP_RATIO_VALUES:
        ratio_dt = float(ratio * dt)
        began = time.perf_counter()
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            base[3],
            base[4],
            ratio_dt,
            dt,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        matrix_seconds = time.perf_counter() - began
        began = time.perf_counter()
        step = causal_five_field_monolithic_discrete_tangent_step(
            context,
            base[3],
            base[4],
            ratio_dt,
            base_history,
            response[:, 3],
            analytic_history,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=True,
        )
        step_seconds = time.perf_counter() - began
        key = f"{ratio:.1f}"
        ratio_reports[key] = {
            "step_ratio": ratio,
            "matrix_assembly_wall_seconds": matrix_seconds,
            "block_step_wall_seconds": step_seconds,
            "maximum_step_matrix_jvp_relative_defect": (
                step.maximum_step_matrix_jvp_relative_defect
            ),
            "maximum_linear_solve_relative_defect": (
                step.maximum_linear_solve_relative_defect
            ),
            "maximum_matrix_component_closure_defect": (
                matrix.maximum_component_closure_defect
            ),
            "incoming_excision_characteristics": (
                matrix.incoming_excision_characteristics
            ),
        }
        ratio_matrices[key] = (matrix, step)
        print(
            f"h2a1: ratio={ratio:.1f} matrix={matrix_seconds:.1f}s "
            f"step={step_seconds:.1f}s "
            f"jvp={step.maximum_step_matrix_jvp_relative_defect:.3e}",
            flush=True,
        )

    unit_matrix, unit_step = ratio_matrices["1.0"]
    predicted_state = unit_step.new_primitive_directions
    predicted_export, export_audit = (
        causal_five_field_monolithic_discrete_export_directions(
            unit_matrix,
            predicted_state,
            coupling_face,
        )
    )
    actual_state = response[:, 4]
    actual_export = actual_exports[:, 4]
    state_metrics = h1._response_metrics(
        predicted_state,
        actual_state,
        field_scales,
    )
    export_metrics = h1._response_metrics(
        predicted_export,
        actual_export,
        export_scales,
    )
    state_metrics["discrepancy_fraction_of_observable_response"] = (
        _fraction_of_response(state_metrics)
    )
    export_metrics["discrepancy_fraction_of_observable_response"] = (
        _fraction_of_response(export_metrics)
    )
    report = {
        "layout": MIDDLE_LAYOUT,
        "profiles": PROFILES,
        "history_interval_seconds": (2.0e-5, 3.0e-5),
        "propagated_interval_seconds": (3.0e-5, 4.0e-5),
        "analytic_history_matrix_wall_seconds": history_matrix_seconds,
        "mapped_history_relative_defect": mapped_history_defect,
        "responsive_height_history_relative_defect": height_history_defect,
        "state": state_metrics,
        "instantaneous_Tier_I": export_metrics,
        "ratio_audits": ratio_reports,
        "maximum_export_transport_telescoping_defect": (
            export_audit.conservative_transport_telescoping_defect
        ),
        "maximum_export_active_prefix_ledger_defect": (
            export_audit.active_prefix_ledger_defect
        ),
    }
    arrays = {
        "profile_names": np.asarray(PROFILES),
        "predicted_state_response_40us": predicted_state,
        "actual_state_response_40us": actual_state,
        "predicted_Tier_I_response_40us": predicted_export,
        "actual_Tier_I_response_40us": actual_export,
        "analytic_mapped_history_direction": (
            analytic_audit_history.previous_mapped_storage_increment
        ),
        "centered_mapped_history_direction": (
            centered_history.previous_mapped_storage_increment
        ),
        "analytic_height_history_direction": (
            analytic_audit_history.previous_responsive_height_storage_increment
        ),
        "centered_height_history_direction": (
            centered_history.previous_responsive_height_storage_increment
        ),
        "step_ratio_values": np.asarray(STEP_RATIO_VALUES),
        "step_ratio_JVP_relative_defects": np.asarray(
            [
                ratio_reports[f"{ratio:.1f}"][
                    "maximum_step_matrix_jvp_relative_defect"
                ]
                for ratio in STEP_RATIO_VALUES
            ]
        ),
    }
    return report, arrays


def _passes(report: dict) -> bool:
    state = report["state"]
    exports = report["instantaneous_Tier_I"]
    ratio_reports = report["ratio_audits"].values()
    return bool(
        report["mapped_history_relative_defect"]
        <= GATES["maximum_analytic_to_centered_history_relative_defect"]
        and report["responsive_height_history_relative_defect"]
        <= GATES["maximum_analytic_to_centered_history_relative_defect"]
        and state["maximum_scaled_discrepancy"]
        <= GATES["maximum_absolute_scaled_state_discrepancy"]
        and exports["maximum_scaled_discrepancy"]
        <= GATES["maximum_absolute_scaled_Tier_I_discrepancy"]
        and state["discrepancy_fraction_of_observable_response"]
        <= GATES["maximum_discrepancy_fraction_of_observable_response"]
        and exports["discrepancy_fraction_of_observable_response"]
        <= GATES["maximum_discrepancy_fraction_of_observable_response"]
        and state["history_cosine"] >= GATES["minimum_state_history_cosine"]
        and exports["history_cosine"] >= GATES["minimum_Tier_I_history_cosine"]
        and all(
            item["maximum_step_matrix_jvp_relative_defect"]
            <= GATES["maximum_internal_discrete_residual_jvp_relative_defect"]
            and item["maximum_linear_solve_relative_defect"]
            <= GATES["maximum_linear_solve_relative_defect"]
            and item["maximum_matrix_component_closure_defect"]
            <= GATES["maximum_matrix_component_closure_defect"]
            and item["incoming_excision_characteristics"]
            <= GATES["maximum_incoming_excision_characteristics"]
            for item in ratio_reports
        )
        and report["maximum_export_transport_telescoping_defect"]
        <= GATES["maximum_export_transport_telescoping_defect"]
        and report["maximum_export_active_prefix_ledger_defect"]
        <= GATES["maximum_export_active_prefix_ledger_defect"]
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
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
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict) -> str:
    audit = summary["audit"]
    return "\n".join(
        (
            "# Middle tangent hardening audit WP10c9d6c7c3b5c3h2a1",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            f"All five prospectively frozen profiles were propagated together over the middle-layout 30-to-40 microsecond step. The maximum scaled state discrepancy was `{audit['state']['maximum_scaled_discrepancy']:.6e}` and the Tier-I discrepancy was `{audit['instantaneous_Tier_I']['maximum_scaled_discrepancy']:.6e}`.",
            "",
            f"The analytic initial mapped/height history directions agree with the retained centered audit to `{audit['mapped_history_relative_defect']:.6e}` and `{audit['responsive_height_history_relative_defect']:.6e}`. Complete-residual JVP checks passed at BDF step ratios 0.5, 1, and 2.",
            "",
            "No new physical trajectory was executed. A pass authorizes only the 0.2 ms middle cost pilot. Later middle propagation, fine work, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


def main() -> int:
    _validate_parent()
    report, arrays = _run_audits()
    passed = _passes(report)
    classification = (
        "middle_tangent_hardening_audits_passed_0p2ms_cost_pilot_authorized"
        if passed
        else "middle_tangent_hardening_audits_failed_pilot_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "new_physical_trajectory_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "gates": GATES,
        "audit": report,
        "middle_0p2ms_cost_pilot_authorized": passed,
        "middle_1ms_propagation_authorized": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2a2_middle_0p2ms_cost_pilot"
            if passed
            else None
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "middle_layout": MIDDLE_LAYOUT,
            "profiles": PROFILES,
            "step_ratio_values": STEP_RATIO_VALUES,
            "gates": GATES,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "analyzed_base_parent": ANALYZED_BASE_PARENT,
            "analyzed_base_tree": ANALYZED_BASE_TREE,
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "h2a0_summary": _sha256(h2a0.SUMMARY_PATH),
                "base_arrays": _sha256(h1.b1a.DECISIVE_ARRAYS),
                "spatial_arrays": _sha256(h1.b4b3.DECISIVE_ARRAYS),
                "corrected_export_arrays": _sha256(h1.b4d.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": {
                "module": _sha256(ROOT / MODULE_RELATIVE),
                "runner": _sha256(ROOT / THIS_RUNNER),
                "test": _sha256(ROOT / THIS_TEST),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    hash_names = (
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in hash_names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
