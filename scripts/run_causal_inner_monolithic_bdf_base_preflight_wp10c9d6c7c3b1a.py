#!/usr/bin/env python3
"""Run the physical-background monolithic BDF base-method preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a  # noqa: E402
import run_causal_inner_frozen_hardening_wp10c9d5a as d5a  # noqa: E402
import run_causal_inner_monolithic_four_level_wp10c9d6c2 as c6c2  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_frozen_tangent,
    evaluate_causal_five_field_monolithic_bdf,
    load_causal_five_field_monolithic_bdf_restart,
    save_causal_five_field_monolithic_bdf_restart,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_bdf import (  # noqa: E402
    causal_bdf_coefficients,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_reconstruct_face_charts,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    _step_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_frozen import (  # noqa: E402
    causal_five_field_radial_reduced_jacobian_pattern,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b1a"
ANALYZED_BASE_COMMIT = "276d693e648549427903182a51af84c5fb21a120"
ANALYZED_BASE_PARENT = "f89c368dc3154140352678ad1c3a98692d88b1aa"
ANALYZED_BASE_TREE = "b94d14442fe913a65c4cf7299015653bd4d7a3cf"

LAYOUTS = tuple(c7a.LAYOUTS[ratio] for ratio in c7a.REFINEMENT_RATIOS)
TIMESTEP_SECONDS = 1.0e-5
FIXED_STEPS = 4
MAXIMUM_SCALED_RESIDUAL = 1.0e-10
MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL = 1.0e-10
MAXIMUM_DISCRETE_LEDGER_DEFECT = 1.0e-12
MAXIMUM_DENSE_COLORED_DEFECT = 1.0e-10
MAXIMUM_INDEPENDENT_JVP_DEFECT = 1.0e-6
MAXIMUM_SCALED_PRIMITIVE_CHANGE = 5.0e-3
MAXIMUM_H_OVER_R = 0.25
MINIMUM_SCATTERING_OPTICAL_DEPTH = 1.0
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE = 2.0e-8
JVP_STEPS = (1.0e-6, 2.0e-6, 4.0e-6)
JVP_SELECTED_STEP = 2.0e-6

THIS_RUNNER = (
    "scripts/run_causal_inner_monolithic_bdf_base_preflight_"
    "wp10c9d6c7c3b1a.py"
)
THIS_MODULE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_bdf.py"
)
THIS_TEST = (
    "tests/test_causal_inner_monolithic_bdf_base_preflight_"
    "wp10c9d6c7c3b1a.py"
)
CORE_TEST = "tests/test_causal_inner_monolithic_bdf.py"
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_MONOLITHIC_BDF_BASE_PREFLIGHT_"
    "WP10C9D6C7C3B1A_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
C3A1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_physical_background_nonlinear_readiness_"
    "manifest_wp10c9d6c7c3a1"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a"
)
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
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
    summary = _read_json(C3A1_DIRECTORY / "summary.json")
    manifest = _read_json(
        C3A1_DIRECTORY / "physical_background_manifest.json"
    )
    if (
        summary["classification"]
        != "physical_embedded_background_nonlinear_ready_"
        "monolithic_bdf_method_preflight_authorized"
        or not summary["monolithic_bdf_method_preflight_authorized"]
        or manifest["method_preflight_contract"]["fixed_steps"]
        != FIXED_STEPS
        or manifest["method_preflight_contract"]["fixed_timestep_seconds"]
        != TIMESTEP_SECONDS
        or tuple(manifest["method_preflight_contract"]["layouts"])
        != LAYOUTS
    ):
        raise RuntimeError("c3a1 method authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b1a analyzed identity changed")
    return manifest


def _configurations():
    arrays = _load_npz(c7a.CANONICAL_DIRECTORY / "decisive_arrays.npz")
    replay_arrays = _load_npz(c7a.C0E_INPUTS)
    replay_contexts = _read_json(c7a.C0E_CONTEXTS)
    result = {}
    for ratio, label in zip(
        c7a.REFINEMENT_RATIOS,
        LAYOUTS,
        strict=True,
    ):
        context = d5a._context_from_payload(
            replay_contexts["contexts"][label],
            replay_arrays,
        )
        base = np.asarray(
            arrays[f"{label}__spliced_base_primitives"],
            dtype=float,
        )
        columns, rows = c6c2._scales_for(context, base)
        result[label] = {
            "context": context,
            "base": base,
            "columns": columns,
            "rows": rows,
        }
    return result


def _colored_recovery(matrix: np.ndarray, n_cells: int) -> tuple[float, int]:
    pattern = causal_five_field_radial_reduced_jacobian_pattern(n_cells)
    groups = causal_five_field_dae_jacobian_color_groups(pattern)
    dense_pattern = np.asarray(pattern.toarray(), dtype=bool)
    recovered = np.zeros_like(matrix)
    for group in groups:
        columns = np.asarray(group, dtype=int)
        seed = np.zeros(matrix.shape[1], dtype=float)
        seed[columns] = 1.0
        action = matrix @ seed
        for column in columns:
            rows = dense_pattern[:, column]
            recovered[rows, column] = action[rows]
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    defect = float(np.max(np.abs(recovered - matrix)) / scale)
    return defect, len(groups)


def _jvp_audit(configuration: dict, tangent) -> tuple[dict, dict]:
    context = configuration["context"]
    base = configuration["base"]
    columns = np.asarray(configuration["columns"], dtype=float)
    rows = np.asarray(configuration["rows"], dtype=float)
    predictor = (
        TIMESTEP_SECONDS
        * np.asarray(tangent.physical_base_rate_per_s, dtype=float)
    )
    scaled_predictor = predictor / columns
    matrix = _step_matrix(
        tangent,
        causal_bdf_coefficients(1, TIMESTEP_SECONDS),
    )
    direction = np.asarray(
        tangent.scaled_base_rate_per_s,
        dtype=float,
    )
    direction /= max(
        float(np.linalg.norm(direction)),
        np.finfo(float).tiny,
    )

    def function(values):
        increment = columns * np.asarray(values, dtype=float)
        evaluation = evaluate_causal_five_field_monolithic_bdf(
            base,
            base + increment.reshape(base.shape),
            TIMESTEP_SECONDS,
            context,
            order=1,
        )
        return evaluation.residual_rows.ravel() / rows

    actions = []
    matrix_action = matrix @ direction
    defects = []
    for step in JVP_STEPS:
        direct = (
            function(scaled_predictor + step * direction)
            - function(scaled_predictor - step * direction)
        ) / (2.0 * step)
        scale = max(
            float(np.linalg.norm(direct)),
            float(np.linalg.norm(matrix_action)),
            np.finfo(float).tiny,
        )
        actions.append(direct)
        defects.append(float(np.linalg.norm(direct - matrix_action) / scale))
    selected = JVP_STEPS.index(JVP_SELECTED_STEP)
    return (
        {
            "steps": list(JVP_STEPS),
            "selected_step": JVP_SELECTED_STEP,
            "selected_relative_defect": defects[selected],
            "minimum_relative_defect": min(defects),
            "passed": defects[selected] <= MAXIMUM_INDEPENDENT_JVP_DEFECT,
        },
        {
            "jvp_direction": direction,
            "jvp_direct_actions": np.asarray(actions),
            "jvp_matrix_action": matrix_action,
            "jvp_relative_defects": np.asarray(defects),
        },
    )


def _state_audit(context, charts: np.ndarray) -> dict:
    h_over_r = []
    optical_depth = []
    for radius, chart in zip(context.grid.centers, charts, strict=True):
        state = _cell_state(context, float(radius), chart)
        h_over_r.append(
            state.thermodynamics.proper_half_thickness / float(radius)
        )
        optical_depth.append(
            0.5 * context.kappa * state.primitive.surface_density
        )
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
        purpose="flux",
    )
    return {
        "maximum_h_over_r": float(max(h_over_r)),
        "minimum_scattering_optical_depth": float(min(optical_depth)),
        "minimum_reconstruction_factor": float(
            np.min(reconstruction.admissibility_factors)
        ),
    }


def _advance_layout(label: str, configuration: dict):
    context = configuration["context"]
    base = configuration["base"]
    started = time.perf_counter()
    print(f"c3b1a: build tangent {label}", flush=True)
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        base,
        primitive_column_scales=configuration["columns"],
        conservation_row_scales=configuration["rows"],
    )
    matrix = _step_matrix(
        tangent,
        causal_bdf_coefficients(1, TIMESTEP_SECONDS),
    )
    colored_defect, color_count = _colored_recovery(
        matrix,
        base.shape[0],
    )
    jvp_report = None
    jvp_arrays = {}
    if label == LAYOUTS[0]:
        print(f"c3b1a: independent JVP {label}", flush=True)
        jvp_report, jvp_arrays = _jvp_audit(configuration, tangent)

    current = np.array(base, copy=True)
    history = None
    steps = []
    states = [np.array(current, copy=True)]
    checkpoint = None
    replay_bitwise = True
    restart_roundtrip = True
    replay_step = None
    provenance = {
        "work_package": WORK_PACKAGE,
        "layout": label,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
    }
    for index in range(FIXED_STEPS):
        order = 1 if index == 0 else 2
        print(f"c3b1a: {label} step {index + 1}/{FIXED_STEPS}", flush=True)
        step = advance_causal_five_field_monolithic_bdf(
            context,
            current,
            TIMESTEP_SECONDS,
            tangent,
            order=order,
            history=history,
            residual_tolerance=MAXIMUM_SCALED_RESIDUAL,
            ledger_tolerance=MAXIMUM_DISCRETE_LEDGER_DEFECT,
            maximum_scaled_primitive_change=(
                MAXIMUM_SCALED_PRIMITIVE_CHANGE
            ),
        )
        steps.append(step)
        if not step.accepted or step.history is None:
            break
        current = step.primitive_charts
        history = step.history
        states.append(np.array(current, copy=True))
        if index == 1:
            checkpoint = CausalFiveFieldMonolithicBDFRestart(
                primitive_charts=np.array(current, copy=True),
                history=history,
                elapsed_time_seconds=2.0 * TIMESTEP_SECONDS,
                completed_steps=2,
                next_order=2,
                provenance=provenance,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "restart.npz"
                save_causal_five_field_monolithic_bdf_restart(
                    path,
                    context,
                    checkpoint,
                )
                restored = load_causal_five_field_monolithic_bdf_restart(
                    path,
                    context,
                    expected_provenance=provenance,
                )
            restart_roundtrip = (
                causal_five_field_monolithic_bdf_restarts_equal(
                    checkpoint,
                    restored,
                )
            )
        if index == 2 and checkpoint is not None:
            replay_step = advance_causal_five_field_monolithic_bdf(
                context,
                checkpoint.primitive_charts,
                TIMESTEP_SECONDS,
                tangent,
                order=2,
                history=checkpoint.history,
                residual_tolerance=MAXIMUM_SCALED_RESIDUAL,
                ledger_tolerance=MAXIMUM_DISCRETE_LEDGER_DEFECT,
                maximum_scaled_primitive_change=(
                    MAXIMUM_SCALED_PRIMITIVE_CHANGE
                ),
            )
            replay_bitwise = bool(
                replay_step.accepted
                and replay_step.history is not None
                and np.array_equal(
                    replay_step.primitive_charts,
                    step.primitive_charts,
                )
                and np.array_equal(
                    replay_step.history.previous_mapped_storage_increment,
                    step.history.previous_mapped_storage_increment,
                )
                and np.array_equal(
                    replay_step.history
                    .previous_responsive_height_storage_increment,
                    step.history
                    .previous_responsive_height_storage_increment,
                )
            )

    final_audit = _state_audit(context, current)
    residuals = [item.maximum_scaled_residual for item in steps]
    algebraic = [
        item.maximum_scaled_algebraic_residual for item in steps
    ]
    ledgers = [item.maximum_discrete_ledger_defect for item in steps]
    mapped_closures = [
        item.evaluation.maximum_mapped_endpoint_path_closure_defect
        for item in steps
    ]
    reconstruction = [
        item.minimum_path_reconstruction_factor for item in steps
    ]
    incoming = [
        item.incoming_excision_characteristics for item in steps
    ]
    completed = len(steps) == FIXED_STEPS and all(
        item.accepted for item in steps
    )
    passed = bool(
        completed
        and max(residuals, default=float("inf"))
        <= MAXIMUM_SCALED_RESIDUAL
        and max(algebraic, default=float("inf"))
        <= MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
        and max(ledgers, default=float("inf"))
        <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        and max(mapped_closures, default=float("inf"))
        <= MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
        and min(reconstruction, default=0.0)
        >= MINIMUM_RECONSTRUCTION_FACTOR
        and max(incoming, default=1) == 0
        and final_audit["maximum_h_over_r"] <= MAXIMUM_H_OVER_R
        and final_audit["minimum_scattering_optical_depth"]
        > MINIMUM_SCATTERING_OPTICAL_DEPTH
        and final_audit["minimum_reconstruction_factor"]
        >= MINIMUM_RECONSTRUCTION_FACTOR
        and colored_defect <= MAXIMUM_DENSE_COLORED_DEFECT
        and restart_roundtrip
        and replay_bitwise
        and (jvp_report is None or jvp_report["passed"])
    )
    report = {
        "label": label,
        "n_cells": int(base.shape[0]),
        "completed_steps": len(steps),
        "maximum_scaled_residual": max(residuals, default=None),
        "maximum_scaled_algebraic_residual": max(
            algebraic,
            default=None,
        ),
        "maximum_discrete_ledger_defect": max(ledgers, default=None),
        "maximum_mapped_endpoint_path_closure_defect": max(
            mapped_closures,
            default=None,
        ),
        "minimum_path_reconstruction_factor": min(
            reconstruction,
            default=None,
        ),
        "maximum_incoming_excision_characteristics": max(
            incoming,
            default=None,
        ),
        "maximum_dense_colored_jacobian_defect": colored_defect,
        "jacobian_color_count": color_count,
        "independent_jvp": jvp_report,
        "restart_roundtrip_bitwise": restart_roundtrip,
        "split_replay_bitwise": replay_bitwise,
        "final_state_audit": final_audit,
        "maximum_newton_iterations": max(
            (item.iterations for item in steps),
            default=None,
        ),
        "function_evaluations": sum(
            item.function_evaluations for item in steps
        )
        + (0 if replay_step is None else replay_step.function_evaluations),
        "elapsed_seconds": time.perf_counter() - started,
        "passed": passed,
    }
    arrays = {
        f"{label}__states": np.asarray(states),
        f"{label}__scaled_residuals": np.asarray(residuals),
        f"{label}__algebraic_residuals": np.asarray(algebraic),
        f"{label}__ledger_defects": np.asarray(ledgers),
        f"{label}__mapped_endpoint_path_closures": np.asarray(
            mapped_closures
        ),
        f"{label}__reconstruction_factors": np.asarray(reconstruction),
        f"{label}__incoming_excision": np.asarray(incoming),
        f"{label}__step_matrix": matrix,
        **{
            f"{label}__{name}": values
            for name, values in jvp_arrays.items()
        },
    }
    return report, arrays


def _update_manifests(summary: dict) -> None:
    artifact = (
        "causal_inner_monolithic_bdf_base_preflight_"
        "wp10c9d6c7c3b1a"
    )
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != artifact]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": artifact,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    compact = _read_json(CANONICAL_SUMMARY)
    compact.setdefault("artifacts", {})[artifact] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    compact.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, compact)


def _refresh_saved_packaging() -> int:
    summary = _read_json(SUMMARY_PATH)
    provenance = _read_json(PROVENANCE_PATH)
    provenance["implementation_source_hashes"] = {
        path: _sha256(ROOT / path)
        for path in (THIS_MODULE, THIS_RUNNER, THIS_TEST, CORE_TEST)
        if (ROOT / path).exists()
    }
    _write_json(PROVENANCE_PATH, provenance)
    checksums = {
        path.name: _sha256(path)
        for path in (
            CONFIG_PATH,
            SUMMARY_PATH,
            PROVENANCE_PATH,
            DECISIVE_ARRAYS,
        )
    }
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(checksums.items())
        ),
        encoding="utf-8",
    )
    _update_manifests(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def main() -> int:
    if "--package-only" in sys.argv:
        return _refresh_saved_packaging()
    parent_manifest = _validate_parent()
    configurations = _configurations()
    reports = []
    arrays = {}
    for label in LAYOUTS:
        report, values = _advance_layout(label, configurations[label])
        reports.append(report)
        arrays.update(values)
        if not report["passed"]:
            break

    all_layouts_completed = len(reports) == len(LAYOUTS)
    passed = bool(
        all_layouts_completed
        and all(report["passed"] for report in reports)
    )
    classification = (
        "monolithic_bdf_base_method_preflight_certified_"
        "full_profile_variant_preflight_authorized"
        if passed
        else "monolithic_bdf_base_method_preflight_failed_"
        "nonlinear_profile_propagation_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3b1b_full_profile_variant_method_preflight"
        if passed
        else "WP10c9d6c7c3b1a_failure_localization"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "fixed_timestep_seconds": TIMESTEP_SECONDS,
        "fixed_steps": FIXED_STEPS,
        "gates": {
            "maximum_scaled_residual": MAXIMUM_SCALED_RESIDUAL,
            "maximum_scaled_algebraic_residual": (
                MAXIMUM_SCALED_ALGEBRAIC_RESIDUAL
            ),
            "maximum_discrete_ledger_defect": (
                MAXIMUM_DISCRETE_LEDGER_DEFECT
            ),
            "maximum_dense_colored_jacobian_defect": (
                MAXIMUM_DENSE_COLORED_DEFECT
            ),
            "maximum_independent_jvp_defect": (
                MAXIMUM_INDEPENDENT_JVP_DEFECT
            ),
            "maximum_h_over_r": MAXIMUM_H_OVER_R,
            "minimum_scattering_optical_depth": (
                MINIMUM_SCATTERING_OPTICAL_DEPTH
            ),
            "minimum_reconstruction_factor": (
                MINIMUM_RECONSTRUCTION_FACTOR
            ),
            "maximum_mapped_endpoint_path_closure_defect": (
                MAXIMUM_MAPPED_ENDPOINT_PATH_CLOSURE
            ),
        },
        "mapped_storage_representation": (
            "stable analytic path integral of the exact mapped differential; "
            "endpoint subtraction retained as an independent closure audit"
        ),
        "full_profile_variant_matrix_deferred_to_c3b1b": True,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, config)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    decisive_hashes = {
        name: causal_array_sha256(values)
        for name, values in arrays.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "operator_changed": False,
        "production_defaults_changed": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "full_profile_variant_method_preflight_authorized": passed,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "c3a1_authorization_preserved": True,
        "c7c1b_strict_auxiliary_rejection_preserved": True,
        "base_layout_reports": reports,
        "maximum_scaled_residual": max(
            (
                report["maximum_scaled_residual"]
                for report in reports
                if report["maximum_scaled_residual"] is not None
            ),
            default=None,
        ),
        "maximum_discrete_ledger_defect": max(
            (
                report["maximum_discrete_ledger_defect"]
                for report in reports
                if report["maximum_discrete_ledger_defect"] is not None
            ),
            default=None,
        ),
        "maximum_dense_colored_jacobian_defect": max(
            (
                report["maximum_dense_colored_jacobian_defect"]
                for report in reports
            ),
            default=None,
        ),
        "maximum_independent_jvp_defect": max(
            (
                report["independent_jvp"]["selected_relative_defect"]
                for report in reports
                if report["independent_jvp"] is not None
            ),
            default=None,
        ),
        "all_restart_roundtrips_bitwise": all(
            report["restart_roundtrip_bitwise"] for report in reports
        ),
        "all_split_replays_bitwise": all(
            report["split_replay_bitwise"] for report in reports
        ),
        "config_sha256": causal_canonical_json_sha256(config),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": decisive_hashes,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
        "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
        "implementation_source_hashes": {
            path: _sha256(ROOT / path)
            for path in (THIS_MODULE, THIS_RUNNER, THIS_TEST, CORE_TEST)
            if (ROOT / path).exists()
        },
        "input_hashes": {
            "c3a1_summary": _sha256(C3A1_DIRECTORY / "summary.json"),
            "c3a1_manifest": _sha256(
                C3A1_DIRECTORY / "physical_background_manifest.json"
            ),
            "c7a_arrays": _sha256(
                c7a.CANONICAL_DIRECTORY / "decisive_arrays.npz"
            ),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": f"PYTHONPATH=src python3 {THIS_RUNNER}",
    }
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Monolithic BDF physical-background base preflight "
                "WP10c9d6c7c3b1a",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This package implements the complete path-increment "
                "BDF1/BDF2 method and tests the unperturbed committed "
                "physical background on every embedded layout. It is a "
                "method gate, not a long physical trajectory.",
                "",
                "## Result",
                "",
                *[
                    (
                        f"- `{report['label']}`: passed="
                        f"`{report['passed']}`, max residual="
                        f"`{report['maximum_scaled_residual']:.6e}`, "
                        f"steps=`{report['completed_steps']}`, replay="
                        f"`{report['split_replay_bitwise']}`"
                    )
                    for report in reports
                ],
                "",
                "The exact mapped storage differential is evaluated by its "
                "stable analytic path integral. Direct endpoint subtraction "
                "is retained as an independent closure audit because it "
                "suffers cancellation at small timesteps.",
                "",
                "## Authorized next",
                "",
                f"`{authorized_next}`",
                "",
                "The full frozen profile/sign/amplitude matrix remains "
                "unrun. Long nonlinear evolution, fixed-Q experiments, and "
                "slow reduction remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksums = {
        path.name: _sha256(path)
        for path in (
            CONFIG_PATH,
            SUMMARY_PATH,
            PROVENANCE_PATH,
            DECISIVE_ARRAYS,
        )
    }
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    _update_manifests(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
