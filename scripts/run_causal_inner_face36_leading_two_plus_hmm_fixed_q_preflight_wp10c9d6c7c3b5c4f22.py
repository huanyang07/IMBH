#!/usr/bin/env python3
"""Run the analysis-only leading-two plus HMM fixed-Q preflight.

The package assembles one frozen, self-consistent monolithic tangent at the
committed middle and fine 20 ms endpoints.  The macro constraint is enforced
with the ledger-derived reaction lift from c4f15.  No nonlinear state, BDF
history, tangent history, or physical trajectory is advanced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.linalg import expm, lu_factor, lu_solve


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15 as c4f15  # noqa: E402
import run_causal_inner_face36_six_mode_fine_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f20 as c4f20  # noqa: E402
import run_causal_inner_face36_leading_two_plus_hmm_manifest_wp10c9d6c7c3b5c4f21 as c4f21  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f22"
ARTIFACT = (
    "causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_"
    "wp10c9d6c7c3b5c4f22"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_"
    "wp10c9d6c7c3b5c4f22.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_"
    "wp10c9d6c7c3b5c4f22.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_LEADING_TWO_PLUS_HMM_FIXED_Q_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F22_2026-08-13.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LIFT_COUNT = 24
LEADING_DIMENSION = 2
SELECTED_RELATIVE_STEPS = np.asarray((5.0e-5, 1.0e-4), dtype=float)
TRANSIENT_WINDOWS_SECONDS = np.asarray((4.0e-4, 2.0e-3, 5.0e-3, 1.0e-2))
GATES = {
    "maximum_DQ_M_inverse_BQ_identity_defect": 1.0e-10,
    "maximum_KKT_linear_solve_relative_defect": 1.0e-10,
    "maximum_reaction_ledger_relative_defect": 1.0e-12,
    "maximum_reaction_support_relative_defect": 1.0e-12,
    "maximum_a2_dual_reaction_annihilation_defect": 1.0e-10,
    "maximum_a2_dual_biorthogonality_defect": 1.0e-10,
    "maximum_projected_block_solve_relative_defect": 1.0e-10,
    "maximum_face36_directional_JVP_relative_defect": 1.0e-8,
    "incoming_excision_characteristics": 0,
}


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
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authorization() -> dict:
    summary = _read(c4f21.SUMMARY_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f22_analysis_only_leading_two_plus_HMM_"
        "fixed_Q_constraint_preflight"
    )
    if (
        not summary["passed"]
        or not summary["fixed_Q_constraint_preflight_authorized"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["nonlinear_retained_mode_pilot_authorized"]
        or summary["authorized_next"] != expected
    ):
        raise RuntimeError("c4f22 authorization changed")
    return summary


def _saved_directions(label: str) -> np.ndarray:
    if label == "middle":
        return np.asarray(
            c4f20._middle_with_recovered_amplitudes()["state_directions"][-1]
        )
    with np.load(c4f20.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        return np.asarray(arrays["fine_state_directions"][-1])


def _physical_reaction_projection(
    values: np.ndarray, constraint: np.ndarray, reaction_lift: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project columns with the ledger-derived reaction, never Euclideanly."""

    schur = constraint @ reaction_lift
    coefficients = np.linalg.solve(schur, constraint @ values)
    projected = values - reaction_lift @ coefficients
    return projected, schur


def _stable_a2_duals(
    descriptor: np.ndarray,
    constraint: np.ndarray,
    reaction_lift: np.ndarray,
    leading_lifts: np.ndarray,
) -> dict:
    reaction_scale = np.linalg.norm(descriptor @ reaction_lift, axis=0)
    normalized_reaction = reaction_lift / reaction_scale[None, :]
    trial = np.column_stack((leading_lifts, normalized_reaction))
    target = np.column_stack((np.eye(LEADING_DIMENSION), np.zeros((2, 3))))
    descriptor_trial = descriptor @ trial

    q, r = np.linalg.qr(descriptor_trial, mode="reduced")
    dual_qr = target @ np.linalg.solve(r, q.T @ descriptor)
    u, singular, vt = np.linalg.svd(descriptor_trial, full_matrices=False)
    dual_svd = target @ ((vt.T / singular) @ (u.T @ descriptor))

    def metrics(dual: np.ndarray) -> dict:
        return {
            "biorthogonality_defect": float(
                np.max(np.abs(dual @ leading_lifts - np.eye(2)))
            ),
            "normalized_reaction_annihilation_defect": float(
                np.max(np.abs(dual @ normalized_reaction))
            ),
        }

    scale = max(
        float(np.linalg.norm(dual_qr)),
        float(np.linalg.norm(dual_svd)),
        np.finfo(float).tiny,
    )
    return {
        "dual_qr": dual_qr,
        "dual_svd": dual_svd,
        "qr_metrics": metrics(dual_qr),
        "svd_metrics": metrics(dual_svd),
        "relative_QR_SVD_difference": float(
            np.linalg.norm(dual_qr - dual_svd) / scale
        ),
        "descriptor_trial_condition_number": float(singular[0] / singular[-1]),
        "leading_Q3_defect": float(np.max(np.abs(constraint @ leading_lifts))),
    }


def _smooth_random_scaled_directions(shape: tuple[int, int], count: int) -> np.ndarray:
    rng = np.random.default_rng(422024)
    values = rng.normal(size=(count, *shape))
    for _ in range(5):
        interior = 0.25 * values[:, :-2] + 0.5 * values[:, 1:-1] + 0.25 * values[:, 2:]
        values[:, 1:-1] = interior
        values[:, 0] = 0.75 * values[:, 0] + 0.25 * values[:, 1]
        values[:, -1] = 0.75 * values[:, -1] + 0.25 * values[:, -2]
    return values.reshape(count, -1)


def _equal_q_lifts(
    saved_physical: np.ndarray,
    columns: np.ndarray,
    layout,
    constraint: np.ndarray,
    reaction_lift: np.ndarray,
) -> tuple[np.ndarray, list[str], float]:
    scaled_saved = saved_physical.reshape(6, -1) / columns[None, :]
    restricted = np.asarray(
        [
            restrict_causal_embedded_patch_cell_averages(direction, layout)
            for direction in saved_physical
        ]
    )
    complement_physical = saved_physical - restricted[:, layout.parent_cell_indices]
    scaled_complement = complement_physical.reshape(6, -1) / columns[None, :]
    random_scaled = _smooth_random_scaled_directions(saved_physical.shape[1:], 12)
    raw = np.vstack((scaled_saved, scaled_complement, random_scaled)).T
    projected, _schur = _physical_reaction_projection(raw, constraint, reaction_lift)
    q, _r = np.linalg.qr(projected, mode="reduced")
    if q.shape[1] != LIFT_COUNT:
        raise RuntimeError("c4f22 equal-Q lift block lost rank")
    leakage = float(np.max(np.abs(constraint @ q)))
    labels = (
        ["leading_explicit"] * 2
        + ["weak_guard_enrichment"] * 4
        + ["refinement_complement"] * 6
        + ["smooth_random_guard"] * 12
    )
    return q, labels, leakage


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _face36_directional_audit(
    context,
    state: np.ndarray,
    columns: np.ndarray,
    layout,
    output_map: np.ndarray,
    lifts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    central = np.empty((LIFT_COUNT, SELECTED_RELATIVE_STEPS.size))
    five_point = np.empty_like(central)
    for slot in range(LIFT_COUNT):
        scaled = lifts[:, slot]
        physical = (columns * scaled).reshape(state.shape)
        analytic = output_map @ scaled
        for step_slot, step in enumerate(SELECTED_RELATIVE_STEPS):
            plus = c4f13._face36_flux(context, state + step * physical, layout)
            minus = c4f13._face36_flux(context, state - step * physical, layout)
            plus_two = c4f13._face36_flux(
                context, state + 2.0 * step * physical, layout
            )
            minus_two = c4f13._face36_flux(
                context, state - 2.0 * step * physical, layout
            )
            central[slot, step_slot] = _relative_defect(
                analytic, (plus - minus) / (2.0 * step)
            )
            five_point[slot, step_slot] = _relative_defect(
                analytic,
                (-plus_two + 8.0 * plus - 8.0 * minus + minus_two)
                / (12.0 * step),
            )
    return central, five_point


def _transient_diagnostics(
    projected_generator: np.ndarray,
    lifts: np.ndarray,
    output_map: np.ndarray,
    constraint: np.ndarray,
    a2_dual: np.ndarray,
) -> dict[str, np.ndarray]:
    # These are deliberately screened-subspace diagnostics, not a claim about
    # the full frozen propagator.  A full 560/1040-dimensional expm action is
    # disproportionate to this nonbinding preflight and would not establish
    # nonlinear guard mixing in any case.
    reduced_generator = lifts.T @ projected_generator @ lifts
    reduced_output = output_map @ lifts
    reduced_a2 = a2_dual @ lifts
    state_singular = []
    output_singular = []
    a2_singular = []
    q_leakage = []
    state_gain = []
    output_gain = []
    for window in TRANSIENT_WINDOWS_SECONDS:
        reduced_propagator = expm(reduced_generator * window)
        propagated = lifts @ reduced_propagator
        state_values = np.linalg.svd(reduced_propagator, compute_uv=False)
        output_values = np.linalg.svd(
            reduced_output @ reduced_propagator, compute_uv=False
        )
        a2_values = np.linalg.svd(
            reduced_a2 @ reduced_propagator, compute_uv=False
        )
        state_singular.append(state_values)
        output_singular.append(output_values)
        a2_singular.append(a2_values)
        q_leakage.append(np.max(np.abs(constraint @ propagated)))
        state_gain.append(state_values[0])
        output_gain.append(output_values[0])
    return {
        "transient_windows_seconds": TRANSIENT_WINDOWS_SECONDS,
        "state_singular_values": np.asarray(state_singular),
        "face36_output_singular_values": np.asarray(output_singular),
        "a2_singular_values": np.asarray(a2_singular),
        "maximum_Q3_leakage_by_window": np.asarray(q_leakage),
        "maximum_state_gain_by_window": np.asarray(state_gain),
        "maximum_face36_gain_by_window": np.asarray(output_gain),
        "reduced_generator": reduced_generator,
    }


def _endpoint(label: str) -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    layout, configuration, trajectory = c4f20.c4f13._layout_data(label)
    context = configuration["context"]
    state = np.asarray(trajectory["states"][-1])
    columns = np.asarray(configuration["columns"], dtype=float).ravel()
    rows = np.asarray(configuration["rows"], dtype=float).ravel()
    saved = _saved_directions(label)

    reaction = c4f15._reaction_preflight(
        label, -1, layout, configuration, trajectory
    )
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    descriptor = np.asarray(reaction["descriptor"])
    constraint = np.asarray(reaction["q_scaled"])
    reaction_lift = np.asarray(reaction["reaction_lift"])
    generator = np.asarray(tangent.scaled_generator_per_s)
    descriptor_defect = _relative_defect(
        descriptor, tangent.descriptor_scaled_matrix
    )

    projected_generator, schur = _physical_reaction_projection(
        generator, constraint, reaction_lift
    )
    projected_generator_defect = float(
        np.linalg.norm(constraint @ projected_generator)
        / max(float(np.linalg.norm(projected_generator)), np.finfo(float).tiny)
    )
    projector_idempotence = _relative_defect(
        projected_generator,
        _physical_reaction_projection(
            projected_generator, constraint, reaction_lift
        )[0],
    )

    scaled_saved = saved.reshape(6, -1) / columns[None, :]
    leading, _ = _physical_reaction_projection(
        scaled_saved[:2].T, constraint, reaction_lift
    )
    duals = _stable_a2_duals(
        descriptor, constraint, reaction_lift, leading
    )
    lifts, lift_labels, lift_q_defect = _equal_q_lifts(
        saved, columns, layout, constraint, reaction_lift
    )

    factor = lu_factor(descriptor)
    rng = np.random.default_rng(2200 + (0 if label == "middle" else 1))
    forcing = rng.normal(size=(state.size, LIFT_COUNT))
    minimum = lu_solve(factor, forcing)
    multiplier = -np.linalg.solve(schur, constraint @ minimum)
    constrained = minimum + reaction_lift @ multiplier
    reaction_rows = descriptor @ reaction_lift
    upper = descriptor @ constrained - reaction_rows @ multiplier - forcing
    lower = constraint @ constrained
    solve_scale = max(
        float(np.linalg.norm(descriptor @ constrained)),
        float(np.linalg.norm(reaction_rows @ multiplier)),
        float(np.linalg.norm(forcing)),
        1.0,
    )
    kkt_defect = float(max(np.linalg.norm(upper), np.linalg.norm(lower)) / solve_scale)
    projected_minimum, _ = _physical_reaction_projection(
        minimum, constraint, reaction_lift
    )
    projected_block_defect = _relative_defect(constrained, projected_minimum)

    output_map = c4f13._face36_output_map(tangent, layout)
    central, five_point = _face36_directional_audit(
        context, state, columns, layout, output_map, lifts
    )
    transient = _transient_diagnostics(
        projected_generator,
        lifts,
        output_map,
        constraint,
        duals["dual_qr"],
    )
    base_fixed_rate, _ = _physical_reaction_projection(
        tangent.scaled_base_rate_per_s[:, None], constraint, reaction_lift
    )

    dual_biorth = max(
        duals["qr_metrics"]["biorthogonality_defect"],
        duals["svd_metrics"]["biorthogonality_defect"],
    )
    dual_reaction = max(
        duals["qr_metrics"]["normalized_reaction_annihilation_defect"],
        duals["svd_metrics"]["normalized_reaction_annihilation_defect"],
    )
    metrics = {
        "time_seconds": float(trajectory["times"][-1]),
        "dimensions": int(state.size),
        "wall_seconds": float(time.perf_counter() - began),
        "descriptor_cross_implementation_relative_defect": descriptor_defect,
        "DQ_M_inverse_BQ_identity_defect": reaction[
            "DQ_M_inverse_BQ_identity_defect"
        ],
        "KKT_linear_solve_relative_defect": max(
            reaction["KKT_linear_solve_relative_defect"], kkt_defect
        ),
        "reaction_ledger_relative_defect": reaction[
            "reaction_ledger_relative_defect"
        ],
        "reaction_support_relative_defect": reaction[
            "reaction_support_relative_defect"
        ],
        "a2_dual_biorthogonality_defect": dual_biorth,
        "a2_dual_reaction_annihilation_defect": dual_reaction,
        "a2_dual_relative_QR_SVD_difference": duals[
            "relative_QR_SVD_difference"
        ],
        "a2_descriptor_trial_condition_number": duals[
            "descriptor_trial_condition_number"
        ],
        "a2_leading_Q3_defect": duals["leading_Q3_defect"],
        "equal_Q_lift_block_Q3_defect": lift_q_defect,
        "projected_generator_Q3_relative_defect": projected_generator_defect,
        "projected_generator_idempotence_relative_defect": projector_idempotence,
        "projected_block_solve_relative_defect": projected_block_defect,
        "maximum_face36_central_JVP_relative_defect": float(np.max(central)),
        "maximum_face36_five_point_JVP_relative_defect": float(
            np.max(five_point)
        ),
        "maximum_base_fixed_Q_rate_defect": float(
            np.max(np.abs(constraint @ base_fixed_rate))
        ),
        "maximum_finite_time_Q3_leakage": float(
            np.max(transient["maximum_Q3_leakage_by_window"])
        ),
        "maximum_frozen_state_transient_gain": float(
            np.max(transient["maximum_state_gain_by_window"])
        ),
        "maximum_frozen_face36_output_gain": float(
            np.max(transient["maximum_face36_gain_by_window"])
        ),
        "incoming_excision_characteristics": int(
            tangent.incoming_excision_characteristics
        ),
        "frozen_projection_omits_state_derivative_of_reaction": True,
        "transient_diagnostic_is_24_lift_Galerkin_only": True,
        "lift_labels": lift_labels,
    }
    passed = bool(
        metrics["DQ_M_inverse_BQ_identity_defect"]
        <= GATES["maximum_DQ_M_inverse_BQ_identity_defect"]
        and metrics["KKT_linear_solve_relative_defect"]
        <= GATES["maximum_KKT_linear_solve_relative_defect"]
        and metrics["reaction_ledger_relative_defect"]
        <= GATES["maximum_reaction_ledger_relative_defect"]
        and metrics["reaction_support_relative_defect"]
        <= GATES["maximum_reaction_support_relative_defect"]
        and metrics["a2_dual_reaction_annihilation_defect"]
        <= GATES["maximum_a2_dual_reaction_annihilation_defect"]
        and metrics["a2_dual_biorthogonality_defect"]
        <= GATES["maximum_a2_dual_biorthogonality_defect"]
        and metrics["projected_block_solve_relative_defect"]
        <= GATES["maximum_projected_block_solve_relative_defect"]
        and metrics["maximum_face36_five_point_JVP_relative_defect"]
        <= GATES["maximum_face36_directional_JVP_relative_defect"]
        and metrics["incoming_excision_characteristics"]
        == GATES["incoming_excision_characteristics"]
    )
    metrics["passed"] = passed
    arrays = {
        f"{label}_constraint": constraint,
        f"{label}_reaction_lift": reaction_lift,
        f"{label}_leading_state_lifts": leading,
        f"{label}_a2_dual_QR": duals["dual_qr"],
        f"{label}_a2_dual_SVD": duals["dual_svd"],
        f"{label}_equal_Q_lifts": lifts,
        f"{label}_projected_leading_generator": (
            duals["dual_qr"] @ projected_generator @ leading
        ),
        f"{label}_leading_face36_output_map": output_map @ leading,
        f"{label}_face36_central_defects": central,
        f"{label}_face36_five_point_defects": five_point,
        f"{label}_transient_windows_seconds": transient[
            "transient_windows_seconds"
        ],
        f"{label}_state_singular_values": transient["state_singular_values"],
        f"{label}_face36_output_singular_values": transient[
            "face36_output_singular_values"
        ],
        f"{label}_a2_singular_values": transient["a2_singular_values"],
        f"{label}_transient_Q3_leakage": transient[
            "maximum_Q3_leakage_by_window"
        ],
        f"{label}_screened_reduced_generator": transient["reduced_generator"],
    }
    return metrics, arrays


def _catalog(summary: dict) -> None:
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
                    "scientific_status": "DIAGNOSTIC ONLY",
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


def _provenance() -> None:
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _git("rev-parse", "HEAD"),
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "parent_summary_sha256": _sha(c4f21.SUMMARY_PATH),
            "middle_dynamic_arrays_sha256": _sha(c4f20.c4f17.DECISIVE_ARRAYS),
            "fine_dynamic_arrays_sha256": _sha(c4f20.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py": _sha(
                    ROOT
                    / "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py"
                ),
            },
        },
    )


def _refresh_metadata() -> None:
    summary = _read(SUMMARY_PATH)
    if not summary["passed"] or not DECISIVE_ARRAYS.exists():
        raise RuntimeError("c4f22 decisive result is unavailable")
    _provenance()
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)


def main() -> None:
    _authorization()
    began = time.perf_counter()
    middle, middle_arrays = _endpoint("middle")
    print(json.dumps({"middle": middle}, indent=2, sort_keys=True), flush=True)
    fine, fine_arrays = _endpoint("fine")
    passed = bool(middle["passed"] and fine["passed"])
    classification = (
        "leading_two_plus_HMM_fixed_Q_constraint_preflight_passed_"
        "one_Q_nonlinear_manifest_authorized"
        if passed
        else "leading_two_plus_HMM_fixed_Q_constraint_preflight_failed"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c4f23_definitions_only_one_Q_leading_two_plus_HMM_"
        "nonlinear_pilot_manifest"
        if passed
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "middle": middle,
        "fine": fine,
        "gates": GATES,
        "fixed_Q_KKT_algebra_certified": passed,
        "frozen_projected_local_tangent_certified": passed,
        "state_dependent_constrained_tangent_certified": False,
        "guard_mixing_or_decay_claimed": False,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "one_Q_nonlinear_pilot_manifest_authorized": passed,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": authorized_next,
        "total_wall_seconds": float(time.perf_counter() - began),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "representative_time_seconds": 0.020,
            "layouts": ["middle", "fine"],
            "screened_equal_Q_lifts_per_layout": LIFT_COUNT,
            "lift_classes": middle["lift_labels"],
            "selected_relative_steps": SELECTED_RELATIVE_STEPS,
            "transient_windows_seconds": TRANSIENT_WINDOWS_SECONDS,
            "shared_exchange_parent_face": 36,
            "raw_face48_exchange_forbidden": True,
            "projection": "ledger_reaction_P=I-L*(DQ*L)^-1*DQ",
            "projected_generator": "G_Q=P*G_at_frozen_20ms_state",
            "transient_diagnostic": (
                "24_lift_Galerkin_exponential_nonbinding_not_full_propagator"
            ),
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **middle_arrays, **fine_arrays)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 leading-two plus HMM fixed-Q preflight\n\n"
        f"Classification: `{classification}`.\n\n"
        "No trajectory was advanced. At the committed middle and fine 20 ms "
        "states, the macro constraint is imposed with the ledger-derived "
        "reaction projection `P = I - L (DQ L)^-1 DQ`; no Euclidean primitive "
        "projection is used.\n\n"
        "## Binding results\n\n"
        "| layout | DQ M^-1 BQ | KKT | a2 biorth | a2 reaction | block | "
        "face-36 five-point | incoming |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| middle | {middle['DQ_M_inverse_BQ_identity_defect']:.3e} | "
        f"{middle['KKT_linear_solve_relative_defect']:.3e} | "
        f"{middle['a2_dual_biorthogonality_defect']:.3e} | "
        f"{middle['a2_dual_reaction_annihilation_defect']:.3e} | "
        f"{middle['projected_block_solve_relative_defect']:.3e} | "
        f"{middle['maximum_face36_five_point_JVP_relative_defect']:.3e} | "
        f"{middle['incoming_excision_characteristics']} |\n"
        f"| fine | {fine['DQ_M_inverse_BQ_identity_defect']:.3e} | "
        f"{fine['KKT_linear_solve_relative_defect']:.3e} | "
        f"{fine['a2_dual_biorthogonality_defect']:.3e} | "
        f"{fine['a2_dual_reaction_annihilation_defect']:.3e} | "
        f"{fine['projected_block_solve_relative_defect']:.3e} | "
        f"{fine['maximum_face36_five_point_JVP_relative_defect']:.3e} | "
        f"{fine['incoming_excision_characteristics']} |\n\n"
        "The 24-direction finite-time screens use the frozen Galerkin "
        "generator on the screened lift span. They are local diagnostics. "
        "They do not prove guard mixing, attraction, or decay. In particular, "
        "`P G` omits the state derivative of the reaction projection; the next "
        "definitions-only nonlinear-pilot manifest must require the complete "
        "state-dependent constrained residual and its JVP before any microburst.\n\n"
        "A pass authorizes only that definitions-only one-Q pilot manifest. "
        "The fixed-Q micro-solver, nonlinear pilot propagation, 50 ms run, and "
        "reduced slow evolution remain blocked.\n",
        encoding="utf-8",
    )
    _provenance()
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh_metadata:
        _refresh_metadata()
    else:
        main()
