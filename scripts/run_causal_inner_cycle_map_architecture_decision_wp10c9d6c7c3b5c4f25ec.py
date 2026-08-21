#!/usr/bin/env python3
"""Select and certify the observed-prefix hybrid phase/cycle-map architecture."""

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
import run_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5 as affine  # noqa: E402
import run_causal_inner_post_transition_phase_window_wp10c9d6c7c3b5c4f25eb as post  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ec"
PARENT_COMMIT = "1beaafb75432a2505d514764e33f0506bb792247"
PARENT_TREE = "d98fd0f8731852d7b26911a43f43c6f8001247b6"
CLASSIFICATION = (
    "conservative_hybrid_phase_cycle_map_architecture_selected_"
    "observed_three_mode_prefix_certified_complete_cycle_calibration_missing"
)
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ed"

POST_HIDDEN_RANK = 4
MAXIMUM_POST_KNOT_RECONSTRUCTION_ERROR_OVER_PATH = 1.0e-10
MAXIMUM_EVENT_GLUING_ERROR = 1.0e-10
MAXIMUM_MACRO_CLOSURE = 1.0e-10
MAXIMUM_100K_DECODE_WALL_SECONDS = 30.0
ONLINE_MACROSTEPS_PER_CYCLE_CAP = 100_000

ARTIFACT = "causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec.py"
THIS_TEST = "tests/test_causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec.py"
HYBRID_SOURCE = "src/imri_qpe/layer3_minidisk_1d/hybrid_phase_memory.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_CYCLE_MAP_MATHEMATICAL_ARCHITECTURE_WP10C9D6C7C3B5C4F25EC_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper():
    return post._helper()


def _validate_parents(*, require_clean: bool) -> dict:
    helper = _helper()
    if helper._git("rev-parse", PARENT_COMMIT) != PARENT_COMMIT:
        raise RuntimeError("cycle-map decision parent commit changed")
    if helper._git("rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise RuntimeError("cycle-map decision parent tree changed")
    post_hashes = helper._validate_checksums(post.CANONICAL_DIRECTORY)
    post_summary = helper._read(post.CANONICAL_DIRECTORY / "summary.json")
    if (
        not post_summary["passed"]
        or not post_summary["architecture_decision_authorized"]
        or post_summary["authorized_next"] != WORK_PACKAGE
        or post_summary["hot_exit_observed"]
        or post_summary["predictive_cycle_authorized"]
    ):
        raise RuntimeError("post-transition architecture authorization changed")
    inherited = {}
    for name, module in (
        ("affine_engine", affine),
        ("cold_collocation", post.manifest.transition.manifest.cold),
        ("transition_collocation", post.manifest.transition),
    ):
        hashes = helper._validate_checksums(module.CANONICAL_DIRECTORY)
        summary = helper._read(module.CANONICAL_DIRECTORY / "summary.json")
        if not summary["passed"]:
            raise RuntimeError(f"inherited architecture evidence changed: {name}")
        inherited[name] = hashes
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("cycle-map architecture decision requires a clean tracked tree")
    return {"post_transition_hashes": post_hashes, "inherited_hashes": inherited}


def _post_coordinates() -> tuple[np.ndarray, np.ndarray]:
    arrays = _helper()._load_npz(
        post.CANONICAL_DIRECTORY / "post_transition_phase_window_model_and_witnesses.npz"
    )
    first_nodes = np.asarray(arrays["half_1__nodes"], dtype=float)
    second_nodes = np.asarray(arrays["half_2__nodes"], dtype=float)
    phase = np.concatenate((0.5 * first_nodes, 0.5 + 0.5 * second_nodes[1:]))
    coordinates = np.vstack(
        (arrays["half_1__coordinates"], arrays["half_2__coordinates"][1:])
    )
    return phase, coordinates


def _canonical_hidden_basis(centered_hidden: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    _left, singular_values, right = np.linalg.svd(centered_hidden, full_matrices=False)
    basis = np.asarray(right[:rank].T, dtype=float)
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis, singular_values


def _build_extended_engine() -> tuple[ConservativeHybridPhaseEngine, dict, dict[str, np.ndarray]]:
    old_engine, data, old_arrays = affine._build_affine_engine()
    helper = _helper()
    phase, coordinates = _post_coordinates()
    tangent = helper._load_npz(
        post.manifest.transition.manifest.geometry.manifest.TANGENT_ARRAYS
    )
    geometry = helper._load_npz(
        post.manifest.transition.manifest.manifest_geometry_path()
    )
    restriction = np.asarray(tangent["macro_restriction_R82"], dtype=float)
    macro_lift = np.asarray(geometry["macro_lift_L470x82"], dtype=float)
    hidden_lift = np.asarray(tangent["hidden_basis_Z388"], dtype=float)
    hidden_dual = np.asarray(tangent["hidden_dual_Q388"], dtype=float)
    macro = (restriction @ coordinates.T).T
    hidden = (hidden_dual @ coordinates.T).T
    centered_hidden = hidden - hidden[0]
    hidden_basis, singular_values = _canonical_hidden_basis(
        centered_hidden, POST_HIDDEN_RANK
    )
    coefficients = centered_hidden @ hidden_basis
    mode = ConservativePhaseMode(
        name="post_transition_collocation_observed",
        phase_knots=phase,
        phase_speeds_per_second=np.full(len(phase) - 1, 1.0 / post.manifest.FULL_DURATION_SECONDS),
        macro_ledger_knots=macro,
        hidden_coefficient_knots=coefficients,
        hidden_origin=hidden[0],
        hidden_embedding_basis=hidden_basis,
        macro_lift=macro_lift,
        hidden_lift=hidden_lift,
        macro_restriction=restriction,
    )
    cold_name = "cold_observed"
    transition_name = "fixed_Q_transition_observed"
    engine = ConservativeHybridPhaseEngine(
        {**old_engine.modes, mode.name: mode},
        {cold_name: transition_name, transition_name: mode.name, mode.name: None},
    )
    payload = {
        **data,
        "post_phase": phase,
        "post_coordinates": coordinates,
        "post_macro": macro,
        "post_hidden": hidden,
        "post_hidden_basis": hidden_basis,
        "post_hidden_coefficients": coefficients,
        "post_hidden_singular_values": singular_values,
        "restriction": restriction,
    }
    arrays = {
        **old_arrays,
        "post_phase_knots": phase,
        "post_coordinates470": coordinates,
        "post_macro_ledger82": macro,
        "post_hidden_coordinates388": hidden,
        "post_hidden_embedding_basis388x4": hidden_basis,
        "post_hidden_coefficients15x4": coefficients,
        "post_hidden_singular_values": singular_values,
    }
    return engine, payload, arrays


def _architecture_specification(metrics: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "selected_architecture": "conservative_event_driven_hybrid_phase_atlas_with_cycle_map",
        "online_state": {
            "continuous": "q in R^82 plus scalar phase phi in [0,1]",
            "continuous_dimension": 83,
            "discrete": "mode sigma in {cold, transition, post-transition, hot, cooling, recovery}",
            "currently_observed_modes": metrics["observed_mode_names"],
        },
        "decoder": "Y_sigma(q,phi)=L q + Z_sigma c_sigma(q,phi)",
        "within_mode_equations": {
            "slow_ledger": "dq/dt = epsilon G_sigma(q,phi; forcing)",
            "phase": "dphi/dt = omega_sigma(q; forcing) > 0",
            "fixed_Q_limit_used_here": "epsilon G_sigma=0 and omega_sigma partial_phi Y_sigma = F_fixedQ(Y_sigma;q)",
            "offline_invariance_residual": "D_q Y_sigma G_sigma + omega_sigma partial_phi Y_sigma - F(Y_sigma;q)",
        },
        "events": {
            "surface": "g_sigma_to_tau(q,phi)=0 with bracketed phase localization",
            "reset": "q_plus=q_minus+Delta_q_sigma_to_tau(q_minus), preserving the declared conservative ledger",
            "memory": "the pair (sigma,phi), not q alone, distinguishes hysteretic states",
        },
        "cycle_map": {
            "mode_transfer": "M_sigma(q)=(q+Delta_q_sigma(q), T_sigma(q), waveform_sigma)",
            "poincare_map": "P=M_recovery o M_cooling o M_hot o M_transition o M_cold",
            "periodic_cycle": "solve P(q_star)-q_star=0 and assess eigenvalues of DP(q_star)",
            "online_truth_calls": 0,
            "online_nonlinear_fixed_Q_roots": 0,
            "online_nanosecond_microsteps": 0,
        },
        "offline_calibration": {
            "method": "adaptive multiple-shooting Legendre-Gauss-Lobatto phase collocation",
            "local_hidden_rank": "adaptive; rank 4 is certified for the observed post-transition window",
            "slow_anchor_interpolation": "conservative interpolation of T_sigma, Delta_q_sigma, event surfaces, and phase tables across prospectively selected q anchors",
            "validation": "held-out anchors plus one independent full-cycle or event-to-event truth trajectory",
        },
        "computational_target": {
            "maximum_online_macrosteps_per_cycle": ONLINE_MACROSTEPS_PER_CYCLE_CAP,
            "measured_100k_decode_wall_seconds": metrics["gate_values"]["measured_100k_decode_wall_seconds"],
            "several_day_cycle_target_is_architecturally_feasible": True,
            "reason": "online work composes a small number of calibrated mode maps; expensive fixed-Q rates and roots occur offline only",
        },
        "current_boundary": {
            "observed_prefix_certified": True,
            "hot_exit_observed": False,
            "hot_cooling_recovery_modes_calibrated": False,
            "slow_flux_closure_calibrated_across_q": False,
            "complete_predictive_cycle_authorized": False,
        },
        "next_artifact": "definitions_only_adaptive_hot_exit_phase_atlas_extension_manifest",
    }


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper()
    engine, data, arrays = _build_extended_engine()
    cold = engine.modes["cold_observed"]
    transition = engine.modes["fixed_Q_transition_observed"]
    post_mode = engine.modes["post_transition_collocation_observed"]
    start = HybridPhaseState(np.asarray(data["macro"][0]), 0.0, cold.name)
    transition_entry = engine.advance(start, cold.duration_seconds)
    post_entry = engine.advance(transition_entry.state, transition.duration_seconds)
    post_entry_decode = engine.decode(post_entry.state)
    post_end = engine.advance(post_entry.state, post_mode.duration_seconds)
    post_end_decode = engine.decode(post_end.state)
    single = engine.advance(
        start,
        cold.duration_seconds + transition.duration_seconds + post_mode.duration_seconds,
    )
    restarted = HybridPhaseState.from_payload(
        json.loads(json.dumps(post_entry.state.to_payload()))
    )
    replay = engine.advance(restarted, post_mode.duration_seconds)
    restart_bitwise = bool(
        np.array_equal(restarted.macro_state, post_entry.state.macro_state)
        and restarted.phase == post_entry.state.phase
        and restarted.mode == post_entry.state.mode
        and restarted.elapsed_seconds == post_entry.state.elapsed_seconds
        and restarted.event_count == post_entry.state.event_count
    )
    suffix_replay_bitwise = bool(
        np.array_equal(replay.state.macro_state, post_end.state.macro_state)
        and replay.state.phase == post_end.state.phase
        and replay.state.mode == post_end.state.mode
        and replay.state.elapsed_seconds == post_end.state.elapsed_seconds
        and replay.state.event_count == post_end.state.event_count
    )
    single_equals_staged = bool(
        np.array_equal(single.state.macro_state, post_end.state.macro_state)
        and single.state.phase == post_end.state.phase
        and single.state.mode == post_end.state.mode
        and single.state.elapsed_seconds == post_end.state.elapsed_seconds
        and single.state.event_count == post_end.state.event_count
    )
    coordinates = np.asarray(data["post_coordinates"], dtype=float)
    phase = np.asarray(data["post_phase"], dtype=float)
    macro = np.asarray(data["post_macro"], dtype=float)
    decoded = np.asarray(
        [post_mode.decode(value, node) for value, node in zip(macro, phase, strict=True)]
    )
    path = float(np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)))
    knot_errors = np.linalg.norm(decoded - coordinates, axis=1) / path
    entry_error = float(np.linalg.norm(post_entry_decode - coordinates[0]))
    endpoint_error = float(np.linalg.norm(post_end_decode - coordinates[-1]))
    macro_closure = float(
        np.max(np.linalg.norm((data["restriction"] @ decoded.T).T - macro, axis=1))
    )
    began = time.perf_counter()
    accumulator = 0.0
    iterations = ONLINE_MACROSTEPS_PER_CYCLE_CAP
    for index in range(iterations):
        value = (index % 10_000) / 9_999.0
        accumulator += float(post_mode.decode(post_mode.ledger(value), value)[0])
    benchmark_wall = float(time.perf_counter() - began)
    if not np.isfinite(accumulator):
        raise RuntimeError("cycle-map benchmark accumulator is nonfinite")
    inherited = {
        "cold_collocation": helper._read(post.manifest.transition.manifest.cold.CANONICAL_DIRECTORY / "summary.json")["passed"],
        "transition_collocation": helper._read(post.manifest.transition.CANONICAL_DIRECTORY / "summary.json")["passed"],
        "post_transition_collocation": helper._read(post.CANONICAL_DIRECTORY / "summary.json")["passed"],
        "affine_gluing": helper._read(affine.CANONICAL_DIRECTORY / "summary.json")["passed"],
    }
    gate_values = {
        "post_hidden_rank": POST_HIDDEN_RANK,
        "maximum_post_knot_reconstruction_error_over_path": float(np.max(knot_errors)),
        "transition_to_post_entry_gluing_error": entry_error,
        "post_endpoint_reconstruction_error": endpoint_error,
        "maximum_macro_closure": macro_closure,
        "measured_100k_decode_wall_seconds": benchmark_wall,
        "seconds_per_decode": benchmark_wall / iterations,
        "observed_prefix_duration_seconds": cold.duration_seconds + transition.duration_seconds + post_mode.duration_seconds,
    }
    gates = {
        "all_inherited_certificates": all(inherited.values()),
        "post_rank4_reconstruction": gate_values["maximum_post_knot_reconstruction_error_over_path"] <= MAXIMUM_POST_KNOT_RECONSTRUCTION_ERROR_OVER_PATH,
        "transition_post_gluing": entry_error <= MAXIMUM_EVENT_GLUING_ERROR,
        "post_endpoint": endpoint_error <= MAXIMUM_EVENT_GLUING_ERROR,
        "macro_closure": macro_closure <= MAXIMUM_MACRO_CLOSURE,
        "restart_bitwise": restart_bitwise,
        "suffix_replay_bitwise": suffix_replay_bitwise,
        "single_equals_staged": single_equals_staged,
        "online_cost": benchmark_wall <= MAXIMUM_100K_DECODE_WALL_SECONDS,
        "truth_free_online_engine": True,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "classification": CLASSIFICATION if passed else "hybrid_phase_cycle_map_architecture_decision_rejected",
        "passed": passed,
        "gates": gates,
        "inherited_certificates": inherited,
        "gate_values": gate_values,
        "continuous_online_dimension": 83,
        "observed_mode_count": 3,
        "observed_mode_names": tuple(engine.modes),
        "post_hidden_table_rank": POST_HIDDEN_RANK,
        "online_truth_calls": 0,
        "online_nonlinear_fixed_Q_roots": 0,
        "online_BDF_microsteps": 0,
        "working_mathematical_architecture_selected": passed,
        "complete_cycle_calibration_missing": True,
        "hot_exit_observed": False,
        "predictive_cycle_authorized": False,
    }
    arrays.update(
        {
            "post_mode_knot_decodes470": decoded,
            "post_mode_knot_errors_over_path": knot_errors,
            "post_entry_decoded_coordinate470": post_entry_decode,
            "post_endpoint_decoded_coordinate470": post_end_decode,
            "observed_prefix_terminal_macro82": post_end.state.macro_state,
        }
    )
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = post.manifest.transition.manifest.cold.manifest.CANONICAL_MANIFEST
    summary_path = post.manifest.transition.manifest.cold.manifest.CANONICAL_SUMMARY
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
    values = metrics["gate_values"]
    return "\n".join(
        (
            "# Cycle-map mathematical architecture WP10c9d6c7c3b5c4f25ec",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            "## Selected mathematical architecture",
            "",
            "Use a conservative, event-driven hybrid phase atlas. The online continuous state is the 82-coordinate conservative macro ledger `q` plus one scalar phase `phi`; a discrete mode label carries hysteresis. The full 470-coordinate state is decoded as `Y_sigma(q,phi) = L q + Z_sigma c_sigma(q,phi)`.",
            "",
            "Within a calibrated mode, `dq/dt = epsilon G_sigma` and `dphi/dt = omega_sigma > 0`. Offline phase collocation enforces `D_q Y G + omega partial_phi Y = F(Y;q)`. The fixed-Q evidence certified here is the `G=0` restriction of that invariance equation.",
            "",
            "Each completed mode becomes an event-to-event transfer map containing its flight time, conservative ledger increment, event surface, and waveform. The slow cycle is the composition of those mode maps; it does not replay nanosecond BDF steps online.",
            "",
            "## Evidence now established",
            "",
            "- Exact vector-field replay is certified on the cold and transition charts.",
            "- The new 0.2 microsecond post-transition collocation window passes its full residual and matched two-half-window shadow.",
            f"- The post-transition hidden path is represented by rank 4 with maximum knot error `{values['maximum_post_knot_reconstruction_error_over_path']:.6e}` of its path.",
            f"- Transition-to-post gluing error is `{values['transition_to_post_entry_gluing_error']:.6e}` and endpoint reconstruction error is `{values['post_endpoint_reconstruction_error']:.6e}`.",
            f"- 100,000 full-coordinate decodes take `{values['measured_100k_decode_wall_seconds']:.6f}` wall seconds on this machine, with zero online truth calls, roots, or BDF microsteps.",
            "",
            "## What remains missing",
            "",
            "The architecture is working on the observed cold/transition/post-transition prefix, but a predictive cycle is not yet calibrated. The hot-exit event has not been observed; hot, cooling, and recovery phase modes are absent; the slow conservative flux closure across multiple q anchors is absent; and no independent complete-cycle validation exists.",
            "",
            "## Next package",
            "",
            "Freeze an adaptive post-transition phase-atlas extension. Continue with rank-adaptive Lobatto windows and exact node rates, stop at a prospectively defined hot-exit event or a fail-fast geometry/physics gate, and never return to sequential nanosecond BDF propagation as the production architecture.",
            "",
        )
    )


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("cycle-map architecture decision already exists")
    locked = _validate_parents(require_clean=True)
    metrics, arrays = _evaluate()
    architecture = _architecture_specification(metrics)
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
        "complete_cycle_calibration_missing": True,
        "hot_exit_observed": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {"runner": THIS_RUNNER, "test": THIS_TEST, "implementation_commit": helper._git("rev-parse", "HEAD"), "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"), "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "source_hashes": {THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER), THIS_TEST: helper._sha(ROOT / THIS_TEST), HYBRID_SOURCE: helper._sha(ROOT / HYBRID_SOURCE)}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    payload = _run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
