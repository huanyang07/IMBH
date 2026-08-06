#!/usr/bin/env python3
"""Correct the 0.2 ms pilot breadth record with a true five-RHS tangent."""

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

import run_causal_inner_nonlinear_middle_tangent_hardening_audit_wp10c9d6c7c3b5c3h2a1 as h2a1  # noqa: E402
import run_causal_inner_nonlinear_middle_cost_pilot_wp10c9d6c7c3b5c3h2a2 as h2a2  # noqa: E402
import run_causal_inner_nonlinear_middle_1ms_continuation_manifest_wp10c9d6c7c3b5c3h2b0 as h2b0  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_discrete_export_directions,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2a3"
ANALYZED_BASE_COMMIT = "d3172f717a70a0721f43f451fd3137d9f934e54e"
ANALYZED_BASE_PARENT = "137c844506f369b0c526953385ed7a08c8faefda"
ANALYZED_BASE_TREE = "1cfdd35eaba2843831c828db927d570b66bf9fee"

PROFILES = tuple(h2a1.PROFILES)
MIDDLE_LAYOUT = h2a1.MIDDLE_LAYOUT
COUPLING_FACE = int(h2a2.COUPLING_FACE)
GENERIC_INDEX = PROFILES.index(h2a2.GENERIC_PROFILE)
AUDIT_INDICES = (0, 5)
GENERIC_CLOSURE_GATE = 1.0e-12

ARTIFACT = (
    "causal_inner_nonlinear_middle_pilot_breadth_correction_"
    "wp10c9d6c7c3b5c3h2a3"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_pilot_breadth_correction_"
    "wp10c9d6c7c3b5c3h2a3.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_pilot_breadth_correction_"
    "wp10c9d6c7c3b5c3h2a3.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_PILOT_BREADTH_"
    "CORRECTION_WP10C9D6C7C3B5C3H2A3_2026-08-06.md"
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


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)) / scale)


def _validate_parent() -> None:
    pilot = _read_json(h2a2.SUMMARY_PATH)
    manifest = _read_json(h2b0.SUMMARY_PATH)
    if not pilot["passed"] or not manifest["middle_1ms_propagation_authorized"]:
        raise RuntimeError("h2a3 parent authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2a3 analyzed identity changed")


def _pilot_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload = _load_npz(h2a2.DECISIVE_ARRAYS)
    base = {
        key.removeprefix("base__"): value
        for key, value in payload.items()
        if key.startswith("base__")
    }
    tangent = {
        key.removeprefix("tangent__"): value
        for key, value in payload.items()
        if key.startswith("tangent__")
    }
    return base, tangent


def _run() -> tuple[dict, dict[str, np.ndarray]]:
    short_base, short_perturbed, _exports, field_scales, export_scales = (
        h2a1._middle_histories()
    )
    response = short_perturbed - short_base[None, :, :, :]
    base_arrays, pilot_tangent = _pilot_arrays()
    configuration = h2a1.h1.b1a._configurations()[MIDDLE_LAYOUT]
    context = configuration["context"]
    columns = configuration["columns"]
    rows = configuration["rows"]
    base_states = base_arrays["accepted_states"]
    timesteps = base_arrays["accepted_timesteps"]
    previous_timesteps = base_arrays["accepted_previous_timesteps"]

    began = time.perf_counter()
    initial_matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        short_base[3],
        short_base[4],
        h2a2.INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        h2a2.INITIAL_PREVIOUS_TIMESTEP_SECONDS,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    initial_matrix_seconds = time.perf_counter() - began
    direction = np.asarray(response[:, 4], dtype=float)
    history_direction = causal_five_field_monolithic_bdf_history_direction(
        context,
        short_base[3],
        short_base[4],
        response[:, 3],
        response[:, 4],
        analytic_step_matrix=initial_matrix,
    )
    initial_export, initial_export_audit = (
        causal_five_field_monolithic_discrete_export_directions(
            initial_matrix,
            direction,
            COUPLING_FACE,
        )
    )

    state_directions = [np.array(direction, copy=True)]
    export_directions = [np.array(initial_export, copy=True)]
    primitive_history_directions = [
        np.array(history_direction.previous_primitive_increment, copy=True)
    ]
    mapped_history_directions = [
        np.array(history_direction.previous_mapped_storage_increment, copy=True)
    ]
    height_history_directions = [
        np.array(
            history_direction.previous_responsive_height_storage_increment,
            copy=True,
        )
    ]
    matrix_seconds = []
    step_seconds = []
    audit_flags = []
    jvp_defects = []
    linear_defects = []
    component_defects = []
    incoming = []
    export_ledgers = [initial_export_audit.active_prefix_ledger_defect]
    export_telescoping = [
        initial_export_audit.conservative_transport_telescoping_defect
    ]
    for index, dt in enumerate(timesteps):
        base_history = h2a2._history(
            base_arrays["accepted_primitive_histories"][index],
            base_arrays["accepted_mapped_histories"][index],
            base_arrays["accepted_height_histories"][index],
            previous_timesteps[index],
        )
        began = time.perf_counter()
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            base_states[index],
            base_states[index + 1],
            float(dt),
            float(previous_timesteps[index]),
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        matrix_seconds.append(time.perf_counter() - began)
        audited = index in AUDIT_INDICES
        began = time.perf_counter()
        step = causal_five_field_monolithic_discrete_tangent_step(
            context,
            base_states[index],
            base_states[index + 1],
            float(dt),
            base_history,
            direction,
            history_direction,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=audited,
        )
        step_seconds.append(time.perf_counter() - began)
        audit_flags.append(audited)
        direction = step.new_primitive_directions
        history_direction = step.new_history_directions
        export_direction, export_audit = (
            causal_five_field_monolithic_discrete_export_directions(
                matrix,
                direction,
                COUPLING_FACE,
            )
        )
        state_directions.append(np.array(direction, copy=True))
        export_directions.append(np.array(export_direction, copy=True))
        primitive_history_directions.append(
            np.array(history_direction.previous_primitive_increment, copy=True)
        )
        mapped_history_directions.append(
            np.array(history_direction.previous_mapped_storage_increment, copy=True)
        )
        height_history_directions.append(
            np.array(
                history_direction.previous_responsive_height_storage_increment,
                copy=True,
            )
        )
        if np.isfinite(step.maximum_step_matrix_jvp_relative_defect):
            jvp_defects.append(step.maximum_step_matrix_jvp_relative_defect)
        linear_defects.append(step.maximum_linear_solve_relative_defect)
        component_defects.append(matrix.maximum_component_closure_defect)
        incoming.append(matrix.incoming_excision_characteristics)
        export_ledgers.append(export_audit.active_prefix_ledger_defect)
        export_telescoping.append(
            export_audit.conservative_transport_telescoping_defect
        )
        print(
            f"h2a3: {index + 1}/{timesteps.size} "
            f"t={base_arrays['accepted_times'][index + 1]:.8e} "
            f"matrix={matrix_seconds[-1]:.1f}s step={step_seconds[-1]:.1f}s "
            f"audit={audited}",
            flush=True,
        )

    states = np.asarray(state_directions, dtype=float)
    exports = np.asarray(export_directions, dtype=float)
    pilot_states = np.asarray(pilot_tangent["state_directions"], dtype=float)
    pilot_exports = np.asarray(pilot_tangent["export_directions"], dtype=float)
    generic_state_defect = _relative_defect(
        states[:, GENERIC_INDEX],
        pilot_states,
    )
    generic_export_defect = _relative_defect(
        exports[:, GENERIC_INDEX],
        pilot_exports,
    )
    routine_steps = [
        value
        for value, audited in zip(step_seconds, audit_flags, strict=True)
        if not audited
    ]
    report = {
        "historical_pilot_tangent_direction_count": int(
            1 if pilot_states.ndim == 3 else pilot_states.shape[1]
        ),
        "corrected_profile_count": len(PROFILES),
        "profiles": PROFILES,
        "pilot_generic_cost_and_anchor_result_retained": True,
        "pilot_five_profile_wording_superseded": True,
        "generic_state_relative_closure_defect": generic_state_defect,
        "generic_Tier_I_relative_closure_defect": generic_export_defect,
        "initial_history_matrix_wall_seconds": initial_matrix_seconds,
        "matrix_assembly_wall_seconds": matrix_seconds,
        "block_step_wall_seconds": step_seconds,
        "routine_five_profile_block_step_median_wall_seconds": float(
            np.median(routine_steps)
        ),
        "audit_step_indices": list(AUDIT_INDICES),
        "maximum_step_matrix_jvp_relative_defect": max(jvp_defects, default=0.0),
        "maximum_linear_solve_relative_defect": max(
            linear_defects,
            default=0.0,
        ),
        "maximum_matrix_component_closure_defect": max(
            component_defects,
            default=0.0,
        ),
        "maximum_incoming_excision_characteristics": max(incoming, default=0),
        "maximum_export_active_prefix_ledger_defect": max(
            export_ledgers,
            default=0.0,
        ),
        "maximum_export_transport_telescoping_defect": max(
            export_telescoping,
            default=0.0,
        ),
    }
    gates = h2a1.GATES
    passed = bool(
        pilot_states.ndim == 3
        and len(PROFILES) == 5
        and generic_state_defect <= GENERIC_CLOSURE_GATE
        and generic_export_defect <= GENERIC_CLOSURE_GATE
        and report["maximum_step_matrix_jvp_relative_defect"]
        <= gates["maximum_internal_discrete_residual_jvp_relative_defect"]
        and report["maximum_linear_solve_relative_defect"]
        <= gates["maximum_linear_solve_relative_defect"]
        and report["maximum_matrix_component_closure_defect"]
        <= gates["maximum_matrix_component_closure_defect"]
        and report["maximum_incoming_excision_characteristics"] == 0
        and report["maximum_export_active_prefix_ledger_defect"]
        <= gates["maximum_export_active_prefix_ledger_defect"]
        and report["maximum_export_transport_telescoping_defect"]
        <= gates["maximum_export_transport_telescoping_defect"]
    )
    report["passed"] = passed
    arrays = {
        "profile_names": np.asarray(PROFILES),
        "accepted_times": base_arrays["accepted_times"],
        "accepted_timesteps": timesteps,
        "step_ratios": timesteps / previous_timesteps[:-1],
        "state_directions": states,
        "Tier_I_export_directions": exports,
        "primitive_history_directions": np.asarray(
            primitive_history_directions,
            dtype=float,
        ),
        "mapped_history_directions": np.asarray(
            mapped_history_directions,
            dtype=float,
        ),
        "height_history_directions": np.asarray(
            height_history_directions,
            dtype=float,
        ),
        "matrix_assembly_wall_seconds": np.asarray(matrix_seconds),
        "block_step_wall_seconds": np.asarray(step_seconds),
        "audit_flags": np.asarray(audit_flags, dtype=bool),
        "field_scales": field_scales,
        "export_scales": export_scales,
    }
    return report, arrays


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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
            "# Middle pilot breadth correction WP10c9d6c7c3b5c3h2a3",
            "",
            "## Classification",
            "",
            f"`{summary['classification']}`",
            "",
            "Inspection of the h2a2 decisive arrays found one tangent direction, not five. The nonlinear base, generic anchor, generic surrogate validation, bitwise replays, and measured cost projection remain valid. Only the five-profile wording was incorrect.",
            "",
            f"This correction propagates all five frozen profile directions over the exact committed 40-to-200 microsecond middle base schedule. The generic column closes against the historical pilot to `{audit['generic_state_relative_closure_defect']:.6e}` in state and `{audit['generic_Tier_I_relative_closure_defect']:.6e}` in Tier-I exports.",
            "",
            f"The routine five-profile block solve costs `{audit['routine_five_profile_block_step_median_wall_seconds']:.6e}` seconds after matrix assembly, so the pilot's cost conclusion is unchanged. The five profile and complete BDF-history directions at 0.2 ms are now committed for continuation.",
            "",
            "This prospective correction supersedes the breadth premise in h2a2/h2b0 without rewriting those historical artifacts. A pass authorizes the existing cost-bounded 1 ms execution contract using the corrected five-profile restart.",
            "",
            "Fine propagation, the 5 ms spatial certificate, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        )
    )


def main() -> int:
    _validate_parent()
    audit, arrays = _run()
    passed = bool(audit["passed"])
    classification = (
        "pilot_breadth_overclaim_corrected_five_profile_0p2ms_tangent_"
        "certified_1ms_execution_authorized"
        if passed
        else "pilot_breadth_correction_failed_1ms_execution_blocked"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "pilot_generic_science_and_cost_result_retained": True,
        "pilot_five_profile_wording_superseded": True,
        "audit": audit,
        "middle_1ms_propagation_authorized": passed,
        "middle_2ms_propagation_authorized": False,
        "middle_5ms_spatial_confirmation_certified": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2b1_middle_0p2_to_1ms_continuation"
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
            "layout": MIDDLE_LAYOUT,
            "profiles": PROFILES,
            "generic_index": GENERIC_INDEX,
            "coupling_face": COUPLING_FACE,
            "audit_indices": AUDIT_INDICES,
            "generic_closure_gate": GENERIC_CLOSURE_GATE,
            "tangent_gates": h2a1.GATES,
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
                "pilot_summary": _sha256(h2a2.SUMMARY_PATH),
                "pilot_decisive_arrays": _sha256(h2a2.DECISIVE_ARRAYS),
                "five_profile_short_audit": _sha256(h2a1.DECISIVE_ARRAYS),
                "continuation_manifest": _sha256(h2b0.MANIFEST_PATH),
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
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
