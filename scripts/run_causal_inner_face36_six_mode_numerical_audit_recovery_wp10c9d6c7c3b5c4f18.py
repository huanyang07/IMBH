#!/usr/bin/env python3
"""Recover the two failed c4f17 middle numerical audits.

Analysis only: reuse the saved middle state-direction history, reconstruct the
Petrov dual with stable linear algebra, and establish a directional face-36
finite-difference plateau.  No tangent or nonlinear trajectory is advanced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15 as c4f15  # noqa: E402
import run_causal_inner_face36_six_mode_coordinate_manifest_wp10c9d6c7c3b5c4f16 as c4f16  # noqa: E402
import run_causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17 as c4f17  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f18"
ARTIFACT = "causal_inner_face36_six_mode_numerical_audit_recovery_wp10c9d6c7c3b5c4f18"
THIS_RUNNER = "scripts/run_causal_inner_face36_six_mode_numerical_audit_recovery_wp10c9d6c7c3b5c4f18.py"
THIS_TEST = "tests/test_causal_inner_face36_six_mode_numerical_audit_recovery_wp10c9d6c7c3b5c4f18.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_SIX_MODE_NUMERICAL_AUDIT_RECOVERY_WP10C9D6C7C3B5C4F18_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

TIME_IDS_MICROSECONDS = (5000, 5400, 10000, 16000, 20000)
RELATIVE_STEPS = np.asarray((5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3))
DUAL_GATE = 1.0e-10
OUTPUT_GATE = 1.0e-8
DUAL_AGREEMENT_GATE = 1.0e-8


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


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _authorization() -> tuple[dict, dict]:
    parent = _read(c4f17.SUMMARY_PATH)
    recovery = _read(c4f17.CANONICAL_DIRECTORY / "recovery_manifest.json")
    expected = (
        "WP10c9d6c7c3b5c4f18_analysis_only_stable_dual_and_"
        "face36_directional_JVP_recovery"
    )
    if (
        parent["passed"]
        or not parent["middle_completed"]
        or parent["fine_executed"]
        or parent["failed_gates"]
        != ["dual_normalized_slow_annihilation", "face36_output_map"]
        or parent["authorized_next"] != expected
        or recovery["authorized_next"] != expected
        or not recovery["uses_saved_c4f17_middle_state_direction_history"]
        or recovery["reruns_middle_propagation"]
        or recovery["runs_fine_propagation"]
        or recovery["fixed_Q_micro_solver_authorized"]
    ):
        raise RuntimeError("c4f18 authorization changed")
    return parent, recovery


def _stable_duals(layout, configuration, trajectory, basis):
    reaction = c4f15._reaction_preflight(
        "middle", 0, layout, configuration, trajectory
    )
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        trajectory["states"][0].shape
    )
    directions = c4f17.c4f1._initial_directions(
        configuration,
        trajectory,
        c4f13.PARENT_CORE_FACE * int(layout.refinement_ratio),
        trajectory["states"].shape[1],
    )["current"]
    scaled = c4f13._scaled_directions(directions, columns)
    state_lifts = scaled.T @ basis
    descriptor = reaction["descriptor"]
    reaction_lift = reaction["reaction_lift"]
    reaction_scale = np.linalg.norm(descriptor @ reaction_lift, axis=0)
    normalized_reaction_lift = reaction_lift / reaction_scale[None, :]
    trial = np.column_stack((scaled.T, normalized_reaction_lift))
    target = np.column_stack(
        (basis.T, np.zeros((basis.shape[1], 3), dtype=float))
    )
    descriptor_trial = descriptor @ trial

    q, r = np.linalg.qr(descriptor_trial, mode="reduced")
    dual_qr = target @ np.linalg.solve(r, q.T @ descriptor)
    u, singular, vt = np.linalg.svd(descriptor_trial, full_matrices=False)
    dual_svd = target @ ((vt.T / singular) @ (u.T @ descriptor))

    def metrics(dual):
        return {
            "biorthogonality_defect": float(
                np.max(np.abs(dual @ state_lifts - np.eye(basis.shape[1])))
            ),
            "normalized_slow_lift_annihilation_defect": float(
                np.max(np.abs(dual @ normalized_reaction_lift))
            ),
            "initial_consensus_coefficient_defect": float(
                np.max(np.abs(dual @ scaled.T - basis.T))
            ),
        }

    scale = max(
        float(np.linalg.norm(dual_qr)),
        float(np.linalg.norm(dual_svd)),
        np.finfo(float).tiny,
    )
    return {
        "qr": dual_qr,
        "svd": dual_svd,
        "qr_metrics": metrics(dual_qr),
        "svd_metrics": metrics(dual_svd),
        "relative_QR_SVD_difference": float(
            np.linalg.norm(dual_qr - dual_svd) / scale
        ),
        "descriptor_trial_condition_number": float(
            singular[0] / singular[-1]
        ),
    }


def _relative_defect(analytic, reference) -> float:
    scale = max(
        float(np.linalg.norm(analytic)),
        float(np.linalg.norm(reference)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(analytic - reference) / scale)


def _directional_plateau(layout, configuration, trajectory, saved):
    context = configuration["context"]
    time_ids = np.rint(saved["times"] * 1.0e6).astype(np.int64)
    central = np.empty((len(TIME_IDS_MICROSECONDS), c4f17.MODE_DIMENSION, RELATIVE_STEPS.size))
    five_point = np.empty_like(central)
    for time_slot, time_id in enumerate(TIME_IDS_MICROSECONDS):
        match = np.flatnonzero(time_ids == time_id)
        if match.size != 1:
            raise RuntimeError(f"c4f18 time id {time_id} is not unique")
        index = int(match[0])
        state = trajectory["states"][index]
        for mode in range(c4f17.MODE_DIMENSION):
            direction = saved["middle_state_directions"][index, mode]
            analytic = saved["middle_face36_outputs"][index, mode]
            for step_slot, step in enumerate(RELATIVE_STEPS):
                plus = c4f13._face36_flux(
                    context, state + step * direction, layout
                )
                minus = c4f13._face36_flux(
                    context, state - step * direction, layout
                )
                plus_two = c4f13._face36_flux(
                    context, state + 2.0 * step * direction, layout
                )
                minus_two = c4f13._face36_flux(
                    context, state - 2.0 * step * direction, layout
                )
                central_reference = (plus - minus) / (2.0 * step)
                five_point_reference = (
                    -plus_two + 8.0 * plus - 8.0 * minus + minus_two
                ) / (12.0 * step)
                central[time_slot, mode, step_slot] = _relative_defect(
                    analytic, central_reference
                )
                five_point[time_slot, mode, step_slot] = _relative_defect(
                    analytic, five_point_reference
                )
        print(f"c4f18: face36 audit {time_id / 1000.0:.1f} ms complete", flush=True)

    maximum_by_step = np.max(five_point, axis=(0, 1))
    eligible_pairs = [
        (index, index + 1)
        for index in range(RELATIVE_STEPS.size - 1)
        if max(maximum_by_step[index], maximum_by_step[index + 1])
        <= OUTPUT_GATE
    ]
    selected = eligible_pairs[0] if eligible_pairs else None
    return {
        "central": central,
        "five_point": five_point,
        "maximum_central_by_step": np.max(central, axis=(0, 1)),
        "maximum_five_point_by_step": maximum_by_step,
        "eligible_adjacent_pairs": eligible_pairs,
        "selected_adjacent_pair": selected,
        "passed": bool(selected is not None),
    }


def _catalog(summary):
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
                    "sha256": _sha(path),
                    "scientific_status": (
                        "SUPPORTED BUT NOT FULLY CERTIFIED"
                        if summary["passed"]
                        else "REJECTED CANDIDATE"
                    ),
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
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def main() -> None:
    began = time.perf_counter()
    parent, recovery = _authorization()
    layout, configuration, trajectory = c4f13._layout_data("middle")
    saved = c4f17._load(c4f17.DECISIVE_ARRAYS)
    basis = c4f17._basis()
    dual = _stable_duals(layout, configuration, trajectory, basis)
    plateau = _directional_plateau(layout, configuration, trajectory, saved)

    qr_pass = bool(
        dual["qr_metrics"]["biorthogonality_defect"] <= DUAL_GATE
        and dual["qr_metrics"]["normalized_slow_lift_annihilation_defect"]
        <= DUAL_GATE
    )
    svd_pass = bool(
        dual["svd_metrics"]["biorthogonality_defect"] <= DUAL_GATE
        and dual["svd_metrics"]["normalized_slow_lift_annihilation_defect"]
        <= DUAL_GATE
    )
    dual_pass = bool(
        qr_pass
        and svd_pass
        and dual["relative_QR_SVD_difference"] <= DUAL_AGREEMENT_GATE
    )
    passed = bool(dual_pass and plateau["passed"])
    if passed:
        classification = (
            "middle_six_mode_numerical_audits_recovered_saved_dynamic_"
            "history_reclassified_fine_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4f19_definitions_only_fine_six_mode_"
            "dynamic_coordinate_replay_manifest"
        )
    elif not dual_pass:
        classification = "six_mode_Petrov_coordinate_rejected_stable_dual_failed"
        authorized_next = "definitions_only_retained_coordinate_redesign_manifest"
    else:
        classification = "face36_directional_JVP_audit_failed_derivative_localization_required"
        authorized_next = "definitions_only_face36_derivative_localization_manifest"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "saved_middle_history_reclassified": passed,
        "dual_QR_passed": qr_pass,
        "dual_SVD_passed": svd_pass,
        "dual_recovery_passed": dual_pass,
        "face36_directional_JVP_plateau_passed": plateau["passed"],
        "QR_metrics": dual["qr_metrics"],
        "SVD_metrics": dual["svd_metrics"],
        "relative_QR_SVD_dual_difference": dual["relative_QR_SVD_difference"],
        "descriptor_trial_condition_number": dual[
            "descriptor_trial_condition_number"
        ],
        "maximum_central_defect_by_step": plateau[
            "maximum_central_by_step"
        ],
        "maximum_five_point_defect_by_step": plateau[
            "maximum_five_point_by_step"
        ],
        "eligible_adjacent_step_pairs": [
            [float(RELATIVE_STEPS[left]), float(RELATIVE_STEPS[right])]
            for left, right in plateau["eligible_adjacent_pairs"]
        ],
        "selected_adjacent_step_pair": (
            None
            if plateau["selected_adjacent_pair"] is None
            else [
                float(RELATIVE_STEPS[index])
                for index in plateau["selected_adjacent_pair"]
            ]
        ),
        "wall_seconds": float(time.perf_counter() - began),
        "new_tangent_trajectory": False,
        "new_nonlinear_trajectory": False,
        "fine_executed": False,
        "fixed_Q_reaction_applied": False,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "guard_complement_retained": True,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _save(
        DECISIVE_ARRAYS,
        relative_steps=RELATIVE_STEPS,
        time_ids_microseconds=np.asarray(TIME_IDS_MICROSECONDS),
        dual_QR=dual["qr"],
        dual_SVD=dual["svd"],
        central_relative_defects=plateau["central"],
        five_point_relative_defects=plateau["five_point"],
        maximum_central_defect_by_step=plateau["maximum_central_by_step"],
        maximum_five_point_defect_by_step=plateau[
            "maximum_five_point_by_step"
        ],
    )
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "time_ids_microseconds": TIME_IDS_MICROSECONDS,
            "relative_steps": RELATIVE_STEPS,
            "dual_gate": DUAL_GATE,
            "dual_QR_SVD_agreement_gate": DUAL_AGREEMENT_GATE,
            "face36_output_map_gate": OUTPUT_GATE,
            "plateau_rule": (
                "first_common_adjacent_relative_step_pair_whose_maximum_"
                "five_point_defect_over_all_five_times_and_six_directions_"
                "is_at_most_1e-8"
            ),
            "tolerance_relaxation": False,
        },
    )
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(
        "# Face-36 six-mode numerical-audit recovery\n\n"
        f"Classification: `{classification}`.\n\n"
        f"Stable reduced-QR/SVD dual recovery: `{dual_pass}`. The QR/SVD "
        f"normalized slow-annihilation defects are `{dual['qr_metrics']['normalized_slow_lift_annihilation_defect']:.3e}` / `{dual['svd_metrics']['normalized_slow_lift_annihilation_defect']:.3e}` and their relative dual difference is `{dual['relative_QR_SVD_difference']:.3e}`.\n\n"
        f"Face-36 five-point plateau recovery: `{plateau['passed']}`. The maximum defects over all five times and six saved directions across the frozen step sweep are `{np.array2string(plateau['maximum_five_point_by_step'], precision=3)}`. The selected adjacent pair is `{summary['selected_adjacent_step_pair']}`.\n\n"
        "No tangent/nonlinear trajectory, fine replay, fixed-Q reaction, 50 ms run, or reduced slow evolution was executed. The frozen tolerances were not relaxed.\n",
        encoding="utf-8",
    )
    source_paths = (
        THIS_RUNNER,
        THIS_TEST,
        c4f17.THIS_RUNNER,
        c4f15.THIS_RUNNER,
        c4f13.THIS_RUNNER,
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _read(CANONICAL_SUMMARY)[
                "latest_source_parent_commit"
            ],
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "parent_summary_sha256": _sha(c4f17.SUMMARY_PATH),
            "recovery_manifest_sha256": _sha(
                c4f17.CANONICAL_DIRECTORY / "recovery_manifest.json"
            ),
            "saved_middle_arrays_sha256": _sha(c4f17.DECISIVE_ARRAYS),
            "source_hashes": {
                path: _sha(ROOT / path)
                for path in source_paths
                if (ROOT / path).exists()
            },
        },
    )
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
