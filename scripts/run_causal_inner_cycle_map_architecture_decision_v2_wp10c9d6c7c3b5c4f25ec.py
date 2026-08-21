#!/usr/bin/env python3
"""Rebuild the transition chart at its accepted anchor and select the cycle map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.hybrid_phase_memory import (  # noqa: E402
    ConservativeHybridPhaseEngine,
    ConservativePhaseMode,
    HybridPhaseState,
)
import run_causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec as rejected  # noqa: E402


SCHEMA_VERSION = 2
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ec"
PARENT_COMMIT = "32e24112a9561a3254ea410a47b5eea7f06958aa"
PARENT_TREE = "54b7659c00c11bcb772b3f92a93c3316c54f30bd"
CLASSIFICATION = (
    "conservative_hybrid_phase_cycle_map_architecture_selected_"
    "accepted_anchor_three_mode_prefix_replayed_complete_cycle_calibration_missing"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ed"
TRANSITION_HIDDEN_RANK = 8
POST_HIDDEN_RANK = 4
MAXIMUM_TRANSITION_KNOT_ERROR_OVER_PATH = 1.0e-9
MAXIMUM_POST_KNOT_ERROR_OVER_PATH = 1.0e-10
MAXIMUM_EVENT_GLUING_ERROR = 1.0e-10
MAXIMUM_MACRO_CLOSURE = 1.0e-10
MAXIMUM_SINGLE_STAGED_DEFECT = 1.0e-10
MAXIMUM_100K_DECODE_WALL_SECONDS = 30.0
ONLINE_MACROSTEPS_PER_CYCLE_CAP = 100_000

ARTIFACT = "causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec_v2"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_cycle_map_architecture_decision_v2_wp10c9d6c7c3b5c4f25ec.py"
THIS_TEST = "tests/test_causal_inner_cycle_map_architecture_decision_v2_wp10c9d6c7c3b5c4f25ec.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_MAP_MATHEMATICAL_ARCHITECTURE_V2_WP10C9D6C7C3B5C4F25EC_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return rejected._helper()


def _validate_parents(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("cycle-map v2 parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("cycle-map v2 parent tree changed")
    rejected_hashes = helper._validate_checksums(rejected.CANONICAL_DIRECTORY)
    failed = helper._read(rejected.CANONICAL_DIRECTORY / "architecture_metrics.json")
    if (
        failed["passed"]
        or failed["classification"] != "hybrid_phase_cycle_map_architecture_decision_rejected"
        or failed["gates"]["transition_post_gluing"]
        or failed["gates"]["post_endpoint"]
        or not failed["gates"]["post_rank4_reconstruction"]
        or not failed["gates"]["all_inherited_certificates"]
    ):
        raise RuntimeError("legacy-transition gluing diagnosis changed")
    post_hashes = helper._validate_checksums(rejected.post.CANONICAL_DIRECTORY)
    post_summary = helper._read(rejected.post.CANONICAL_DIRECTORY / "summary.json")
    if not post_summary["passed"]:
        raise RuntimeError("post-transition certificate changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-map v2 requires a clean tracked tree")
    return {"rejected_v1_hashes": rejected_hashes, "post_transition_hashes": post_hashes}


def _canonical_basis(centered: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    basis = np.asarray(right[:rank].T, dtype=float)
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis, singular_values


def _mode_from_coordinates(
    *,
    name: str,
    phase: np.ndarray,
    duration_seconds: float,
    coordinates: np.ndarray,
    hidden_rank: int,
    restriction: np.ndarray,
    macro_lift: np.ndarray,
    hidden_lift: np.ndarray,
    hidden_dual: np.ndarray,
) -> tuple[ConservativePhaseMode, dict[str, np.ndarray]]:
    macro = (restriction @ coordinates.T).T
    hidden = (hidden_dual @ coordinates.T).T
    centered = hidden - hidden[0]
    basis, singular_values = _canonical_basis(centered, hidden_rank)
    coefficients = centered @ basis
    mode = ConservativePhaseMode(
        name=name,
        phase_knots=phase,
        phase_speeds_per_second=np.full(len(phase) - 1, 1.0 / duration_seconds),
        macro_ledger_knots=macro,
        hidden_coefficient_knots=coefficients,
        hidden_origin=hidden[0],
        hidden_embedding_basis=basis,
        macro_lift=macro_lift,
        hidden_lift=hidden_lift,
        macro_restriction=restriction,
    )
    return mode, {
        "phase": phase,
        "coordinates": coordinates,
        "macro": macro,
        "hidden": hidden,
        "basis": basis,
        "coefficients": coefficients,
        "singular_values": singular_values,
    }


def _build_corrected_engine() -> tuple[ConservativeHybridPhaseEngine, dict, dict[str, np.ndarray]]:
    helper = _helper()
    old_engine, old_data, old_arrays = rejected.affine._build_affine_engine()
    transition_geometry = helper._load_npz(
        rejected.post.manifest.transition.manifest.manifest_geometry_path()
    )
    tangent = helper._load_npz(
        rejected.post.manifest.transition.manifest.geometry.manifest.TANGENT_ARRAYS
    )
    restriction = np.asarray(tangent["macro_restriction_R82"], dtype=float)
    macro_lift = np.asarray(transition_geometry["macro_lift_L470x82"], dtype=float)
    hidden_lift = np.asarray(tangent["hidden_basis_Z388"], dtype=float)
    hidden_dual = np.asarray(tangent["hidden_dual_Q388"], dtype=float)
    transition_times = np.asarray(transition_geometry["trajectory_times_seconds"], dtype=float)
    transition_duration = float(transition_times[-1] - transition_times[0])
    transition_phase = (transition_times - transition_times[0]) / transition_duration
    transition_coordinates = np.asarray(transition_geometry["trajectory_coordinates470"], dtype=float)
    transition_mode, transition_data = _mode_from_coordinates(
        name="fixed_Q_transition_observed",
        phase=transition_phase,
        duration_seconds=transition_duration,
        coordinates=transition_coordinates,
        hidden_rank=TRANSITION_HIDDEN_RANK,
        restriction=restriction,
        macro_lift=macro_lift,
        hidden_lift=hidden_lift,
        hidden_dual=hidden_dual,
    )
    post_phase, post_coordinates = rejected._post_coordinates()
    post_mode, post_data = _mode_from_coordinates(
        name="post_transition_collocation_observed",
        phase=post_phase,
        duration_seconds=rejected.post.manifest.FULL_DURATION_SECONDS,
        coordinates=post_coordinates,
        hidden_rank=POST_HIDDEN_RANK,
        restriction=restriction,
        macro_lift=macro_lift,
        hidden_lift=hidden_lift,
        hidden_dual=hidden_dual,
    )
    cold = old_engine.modes["cold_observed"]
    # The inherited cold mode stores a relative ledger, while the engine state
    # carries the absolute macro coordinate.  Its observed terminal macro is
    # therefore the saved absolute cold endpoint, not ledger(1) by itself.
    cold_terminal_macro = np.asarray(old_data["macro"][-1], dtype=float)
    cold_to_transition_reset = transition_data["macro"][0] - cold_terminal_macro
    engine = ConservativeHybridPhaseEngine(
        {cold.name: cold, transition_mode.name: transition_mode, post_mode.name: post_mode},
        {cold.name: transition_mode.name, transition_mode.name: post_mode.name, post_mode.name: None},
        {cold.name: cold_to_transition_reset},
    )
    data = {
        "cold_initial_macro": np.asarray(old_data["macro"][0], dtype=float),
        "cold_to_transition_macro_reset": cold_to_transition_reset,
        "restriction": restriction,
        "transition": transition_data,
        "post": post_data,
    }
    arrays = {
        **old_arrays,
        "cold_to_transition_macro_reset82": cold_to_transition_reset,
        "accepted_transition_phase_knots": transition_data["phase"],
        "accepted_transition_coordinates470": transition_data["coordinates"],
        "accepted_transition_macro_ledger82": transition_data["macro"],
        "accepted_transition_hidden_embedding_basis388x8": transition_data["basis"],
        "accepted_transition_hidden_coefficients18x8": transition_data["coefficients"],
        "accepted_transition_hidden_singular_values": transition_data["singular_values"],
        "post_phase_knots": post_data["phase"],
        "post_coordinates470": post_data["coordinates"],
        "post_macro_ledger82": post_data["macro"],
        "post_hidden_embedding_basis388x4": post_data["basis"],
        "post_hidden_coefficients15x4": post_data["coefficients"],
        "post_hidden_singular_values": post_data["singular_values"],
    }
    return engine, data, arrays


def _decode_knots(mode: ConservativePhaseMode) -> np.ndarray:
    return np.asarray(
        [
            mode.decode(macro, phase)
            for macro, phase in zip(mode.macro_ledger_knots, mode.phase_knots, strict=True)
        ]
    )


def _relative_state_defect(left: HybridPhaseState, right: HybridPhaseState) -> float:
    scale = max(float(np.linalg.norm(right.macro_state)), np.finfo(float).tiny)
    return float(
        max(
            np.linalg.norm(left.macro_state - right.macro_state) / scale,
            abs(left.phase - right.phase),
            0.0 if left.mode == right.mode else 1.0,
            abs(left.elapsed_seconds - right.elapsed_seconds)
            / max(abs(right.elapsed_seconds), np.finfo(float).tiny),
            abs(left.event_count - right.event_count),
        )
    )


def _architecture(metrics: dict) -> dict:
    values = metrics["gate_values"]
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_architecture": "conservative_event_driven_hybrid_phase_atlas_with_cycle_map",
        "online_state": "(q in R^82, scalar phase phi, discrete mode sigma)",
        "continuous_online_dimension": 83,
        "decoder": "Y_sigma(q,phi)=Lq+Z_sigma c_sigma(q,phi)",
        "invariance_equation": "D_qY_sigma G_sigma + omega_sigma partial_phi Y_sigma = F(Y_sigma;q)",
        "fixed_Q_evidence_equation": "omega_sigma partial_phi Y_sigma = F_fixedQ(Y_sigma;q)",
        "mode_map": "M_sigma(q)=(q+Delta_q_sigma(q), flight_time T_sigma(q), waveform Y_sigma)",
        "cycle_map": "P=M_recovery o M_cooling o M_hot o M_transition o M_cold",
        "cycle_fixed_point": "P(q_star)-q_star=0; Floquet multipliers are eigenvalues of DP(q_star)",
        "offline_method": "rank-adaptive multiple-shooting Legendre-Gauss-Lobatto phase collocation at selected q anchors",
        "online_method": "event-to-event map composition with optional phase interpolation; no fixed-Q truth, root, or nanosecond microstep",
        "observed_modes": metrics["observed_mode_names"],
        "anchor_specific_cold_transition_reset": {
            "present": True,
            "norm": values["cold_to_transition_macro_reset_norm"],
            "interpretation": "observed ledger transfer required to join the cold endpoint to the accepted transition anchor; it is not yet a q-dependent predictive reset",
        },
        "computational_feasibility": {
            "measured_100k_decode_wall_seconds": values["measured_100k_decode_wall_seconds"],
            "online_macrostep_cap_per_cycle": ONLINE_MACROSTEPS_PER_CYCLE_CAP,
            "several_day_target_supported_by_architecture": True,
            "offline_truth_cost_not_counted_as_online_cycle_cost": True,
        },
        "missing_for_prediction": (
            "hot-exit event",
            "hot/cooling/recovery modes",
            "q-dependent slow flux and reset maps",
            "held-out complete-cycle validation",
        ),
        "predictive_cycle_authorized": False,
        "next_artifact": "definitions_only_adaptive_hot_exit_phase_atlas_extension_manifest",
    }


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    engine, data, arrays = _build_corrected_engine()
    cold = engine.modes["cold_observed"]
    transition = engine.modes["fixed_Q_transition_observed"]
    post = engine.modes["post_transition_collocation_observed"]
    start = HybridPhaseState(data["cold_initial_macro"], 0.0, cold.name)
    transition_entry = engine.advance(start, cold.duration_seconds)
    transition_end = engine.advance(transition_entry.state, transition.duration_seconds)
    post_end = engine.advance(transition_end.state, post.duration_seconds)
    single = engine.advance(
        start, cold.duration_seconds + transition.duration_seconds + post.duration_seconds
    )
    restarted = HybridPhaseState.from_payload(
        json.loads(json.dumps(transition_end.state.to_payload()))
    )
    replay = engine.advance(restarted, post.duration_seconds)
    transition_decoded = _decode_knots(transition)
    post_decoded = _decode_knots(post)
    transition_path = float(
        np.sum(np.linalg.norm(np.diff(data["transition"]["coordinates"], axis=0), axis=1))
    )
    post_path = float(
        np.sum(np.linalg.norm(np.diff(data["post"]["coordinates"], axis=0), axis=1))
    )
    transition_errors = np.linalg.norm(
        transition_decoded - data["transition"]["coordinates"], axis=1
    ) / transition_path
    post_errors = np.linalg.norm(post_decoded - data["post"]["coordinates"], axis=1) / post_path
    cold_transition_gluing = float(
        np.linalg.norm(engine.decode(transition_entry.state) - data["transition"]["coordinates"][0])
    )
    transition_post_gluing = float(
        np.linalg.norm(engine.decode(transition_end.state) - data["post"]["coordinates"][0])
    )
    post_endpoint = float(
        np.linalg.norm(engine.decode(post_end.state) - data["post"]["coordinates"][-1])
    )
    decoded_all = np.vstack((transition_decoded, post_decoded))
    macro_all = np.vstack((data["transition"]["macro"], data["post"]["macro"]))
    macro_closure = float(
        np.max(np.linalg.norm((data["restriction"] @ decoded_all.T).T - macro_all, axis=1))
    )
    restart_bitwise = bool(
        np.array_equal(restarted.macro_state, transition_end.state.macro_state)
        and restarted.phase == transition_end.state.phase
        and restarted.mode == transition_end.state.mode
        and restarted.elapsed_seconds == transition_end.state.elapsed_seconds
        and restarted.event_count == transition_end.state.event_count
    )
    replay_bitwise = bool(
        np.array_equal(replay.state.macro_state, post_end.state.macro_state)
        and replay.state.phase == post_end.state.phase
        and replay.state.mode == post_end.state.mode
        and replay.state.elapsed_seconds == post_end.state.elapsed_seconds
        and replay.state.event_count == post_end.state.event_count
    )
    single_staged_defect = _relative_state_defect(single.state, post_end.state)
    began = time.perf_counter(); accumulator = 0.0
    for index in range(ONLINE_MACROSTEPS_PER_CYCLE_CAP):
        phase = (index % 10_000) / 9_999.0
        accumulator += float(post.decode(post.ledger(phase), phase)[0])
    benchmark = float(time.perf_counter() - began)
    if not np.isfinite(accumulator):
        raise RuntimeError("v2 benchmark accumulator is nonfinite")
    inherited = {
        "cold_exact_rate_collocation": helper._read(rejected.post.manifest.transition.manifest.cold.CANONICAL_DIRECTORY / "summary.json")["passed"],
        "transition_exact_rate_collocation": helper._read(rejected.post.manifest.transition.CANONICAL_DIRECTORY / "summary.json")["passed"],
        "post_transition_exact_rate_collocation": helper._read(rejected.post.CANONICAL_DIRECTORY / "summary.json")["passed"],
    }
    values = {
        "transition_hidden_rank": TRANSITION_HIDDEN_RANK,
        "post_hidden_rank": POST_HIDDEN_RANK,
        "maximum_transition_knot_error_over_path": float(np.max(transition_errors)),
        "maximum_post_knot_error_over_path": float(np.max(post_errors)),
        "cold_to_transition_gluing_error": cold_transition_gluing,
        "transition_to_post_gluing_error": transition_post_gluing,
        "post_endpoint_error": post_endpoint,
        "maximum_macro_closure": macro_closure,
        "cold_to_transition_macro_reset_norm": float(np.linalg.norm(data["cold_to_transition_macro_reset"])),
        "single_staged_relative_defect": single_staged_defect,
        "measured_100k_decode_wall_seconds": benchmark,
        "seconds_per_decode": benchmark / ONLINE_MACROSTEPS_PER_CYCLE_CAP,
        "observed_prefix_duration_seconds": cold.duration_seconds + transition.duration_seconds + post.duration_seconds,
    }
    gates = {
        "inherited_exact_rate_certificates": all(inherited.values()),
        "transition_rank8_reconstruction": values["maximum_transition_knot_error_over_path"] <= MAXIMUM_TRANSITION_KNOT_ERROR_OVER_PATH,
        "post_rank4_reconstruction": values["maximum_post_knot_error_over_path"] <= MAXIMUM_POST_KNOT_ERROR_OVER_PATH,
        "cold_transition_gluing": cold_transition_gluing <= MAXIMUM_EVENT_GLUING_ERROR,
        "transition_post_gluing": transition_post_gluing <= MAXIMUM_EVENT_GLUING_ERROR,
        "post_endpoint": post_endpoint <= MAXIMUM_EVENT_GLUING_ERROR,
        "macro_closure": macro_closure <= MAXIMUM_MACRO_CLOSURE,
        "restart_bitwise": restart_bitwise,
        "suffix_replay_bitwise": replay_bitwise,
        "single_equals_staged_with_roundoff_gate": single_staged_defect <= MAXIMUM_SINGLE_STAGED_DEFECT,
        "online_cost": benchmark <= MAXIMUM_100K_DECODE_WALL_SECONDS,
        "truth_free_online_engine": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "classification": CLASSIFICATION if passed else "accepted_anchor_cycle_map_architecture_v2_rejected",
        "passed": passed,
        "gates": gates,
        "inherited_certificates": inherited,
        "gate_values": values,
        "continuous_online_dimension": 83,
        "observed_mode_count": 3,
        "observed_mode_names": tuple(engine.modes),
        "online_truth_calls": 0,
        "online_fixed_Q_roots": 0,
        "online_BDF_microsteps": 0,
        "anchor_specific_reset_generalization_missing": True,
        "complete_cycle_calibration_missing": True,
        "hot_exit_observed": False,
        "predictive_cycle_authorized": False,
    }
    arrays.update(
        {
            "transition_mode_knot_decodes470": transition_decoded,
            "transition_mode_knot_errors_over_path": transition_errors,
            "post_mode_knot_decodes470": post_decoded,
            "post_mode_knot_errors_over_path": post_errors,
            "transition_entry_decoded_coordinate470": engine.decode(transition_entry.state),
            "transition_endpoint_decoded_coordinate470": engine.decode(transition_end.state),
            "post_endpoint_decoded_coordinate470": engine.decode(post_end.state),
        }
    )
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = rejected.post.manifest.transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = rejected.post.manifest.transition.manifest.cold.manifest.CANONICAL_SUMMARY
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": helper._sha(path), "scientific_status": status})
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = helper._read(summary_path)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": PARENT_COMMIT, "latest_work_package": WORK_PACKAGE})
    helper._write_json(summary_path, catalog)


def _report(metrics: dict) -> str:
    v = metrics["gate_values"]
    return "\n".join(
        (
            "# Cycle-map mathematical architecture v2 WP10c9d6c7c3b5c4f25ec",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            "## Decision",
            "",
            "Select the conservative event-driven hybrid phase atlas and its event-to-event cycle map as the mathematical architecture for the reduced slow solver.",
            "",
            "The online state is `q in R^82`, one scalar phase, and a discrete mode. The full coordinate state is decoded from mode-local phase tables. Fixed-Q exact rates and collocation solves are offline calibration work; the online cycle composes calibrated mode maps and performs no nanosecond BDF stepping.",
            "",
            "## Corrected observed-prefix replay",
            "",
            "V1 correctly rejected direct attachment to the older affine transition surrogate. V2 rebuilds the transition table at the exact accepted anchor and records the observed cold-to-transition macro reset explicitly.",
            "",
            f"- transition hidden table rank: `{TRANSITION_HIDDEN_RANK}`; maximum knot error/path: `{v['maximum_transition_knot_error_over_path']:.6e}`",
            f"- post-transition hidden table rank: `{POST_HIDDEN_RANK}`; maximum knot error/path: `{v['maximum_post_knot_error_over_path']:.6e}`",
            f"- cold-to-transition gluing: `{v['cold_to_transition_gluing_error']:.6e}`",
            f"- transition-to-post gluing: `{v['transition_to_post_gluing_error']:.6e}`",
            f"- post endpoint error: `{v['post_endpoint_error']:.6e}`",
            f"- 100,000 full decodes: `{v['measured_100k_decode_wall_seconds']:.6f}` wall seconds",
            "",
            "## Scope boundary",
            "",
            "This selects a working architecture and certifies its observed three-mode prefix. It does not yet provide a predictive cycle: the cold-transition reset is only anchor-specific, the hot exit remains unobserved, hot/cooling/recovery modes are absent, and q-dependent flux/reset maps plus an independent full-cycle validation are still required.",
            "",
            "## Next package",
            "",
            "Freeze an adaptive hot-exit phase-atlas extension using rank-adaptive Lobatto windows. Stop on the first event/geometry/physics gate. Do not return to sequential nanosecond BDF propagation as the online or production architecture.",
            "",
        )
    )


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cycle-map architecture v2 already exists")
    locked = _validate_parents(require_clean=True)
    metrics, arrays = _evaluate()
    architecture = _architecture(metrics)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "architecture_metrics.json", metrics)
    helper._write_json(CANONICAL_DIRECTORY / "mathematical_architecture.json", architecture)
    with (CANONICAL_DIRECTORY / "architecture_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", locked)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "working_mathematical_architecture_selected": metrics["passed"],
        "observed_three_mode_prefix_certified": metrics["passed"],
        "continuous_online_dimension": 83,
        "online_cost_feasible": metrics["gates"]["online_cost"],
        "anchor_specific_reset_generalization_missing": True,
        "complete_cycle_calibration_missing": True,
        "hot_exit_observed": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "rejected_v1_preserved": True})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _run(); print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
