#!/usr/bin/env python3
"""Replay and benchmark the truth-free conservative hot-mode engine."""

from __future__ import annotations

import argparse
import csv
import io
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

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import (  # noqa: E402
    ConservativeCoordinateSplit,
    ConservativeHeunEngine,
    ConservativeHiddenAmplitudeModel,
    HiddenAmplitudeState,
    HystereticModeSelector,
    LocalAffineReducedPatch,
)
import run_causal_inner_truth_free_hot_mode_engine_manifest_wp10c9d6c7c3b5c4f25fb as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fc"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fd"
PASS_CLASSIFICATION = "truth_free_conservative_hot_mode_engine_verified"
FAIL_CLASSIFICATION = "truth_free_conservative_hot_mode_engine_rejected"
ARTIFACT = "causal_inner_truth_free_hot_mode_engine_wp10c9d6c7c3b5c4f25fc"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TRUTH_FREE_HOT_MODE_ENGINE_"
    "WP10C9D6C7C3B5C4F25FC_2026-08-22.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = manifest.EXECUTION_RUNNER
THIS_TEST = manifest.EXECUTION_TEST


def _helper():
    return manifest._helper()


def _relative_increment(
    candidate: np.ndarray,
    reference: np.ndarray,
    anchor: np.ndarray,
) -> float:
    return float(
        np.linalg.norm((candidate - anchor) - (reference - anchor))
        / max(float(np.linalg.norm(reference - anchor)), np.finfo(float).tiny)
    )


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "hot_engine_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["truth_free_hot_mode_engine_replay_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["work_package"] != manifest.WORK_PACKAGE
    ):
        raise RuntimeError("truth-free hot engine manifest changed")
    current = {
        name: helper._sha(path) for name, path in manifest._decisive_inputs().items()
    }
    if current != contract["decisive_input_hashes"]:
        raise RuntimeError("truth-free hot engine decisive input changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"truth-free hot engine source changed: {relative}")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("truth-free hot engine replay requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _build_engine() -> tuple[
    ConservativeHeunEngine,
    HiddenAmplitudeState,
    dict[str, np.ndarray],
]:
    helper = _helper()
    parent = manifest.parent
    hot_arrays = helper._load_npz(
        parent.hot.CANONICAL_DIRECTORY / "hot_free_field_arrays.npz"
    )
    off_arrays = helper._load_npz(
        parent.CANONICAL_DIRECTORY / "hot_mode_off_axis_arrays.npz"
    )
    base = parent.arclength._source()._base_inputs()
    geometry = base["geometry"]
    split = ConservativeCoordinateSplit(
        macro_restriction=geometry["R"],
        macro_lift=geometry["L"],
        hidden_dual=geometry["Q"],
        hidden_lift=geometry["Z"],
        tolerance=5.0e-11,
    )
    center = manifest.HOT_CENTER_INDEX
    coordinate = np.asarray(hot_arrays["coordinates5x470"])[center]
    coordinate_rate = np.asarray(
        hot_arrays["coordinate_free_rates5x470_per_s"]
    )[center]
    macro, hidden = split.split(coordinate)
    macro_rate, hidden_rate = split.split_rate(coordinate_rate)
    hidden_basis = np.asarray(off_arrays["extended_hidden_rate_basis388xr"])
    model = ConservativeHiddenAmplitudeModel(
        split=split,
        hidden_origin=hidden,
        hidden_basis=hidden_basis,
    )
    anchor_reduced_rate = np.concatenate(
        (macro_rate, hidden_basis.T @ hidden_rate)
    )
    full_reduced_rate = np.concatenate((
        np.asarray(off_arrays["off_axis_macro_free_rates3x82_per_s"])[1],
        np.asarray(off_arrays["off_axis_hidden_free_rates3x388_per_s"])[1]
        @ hidden_basis,
    ))
    patch = LocalAffineReducedPatch(
        anchor_macro=macro,
        anchor_amplitudes=np.zeros(hidden_basis.shape[1]),
        anchor_reduced_rate=anchor_reduced_rate,
        physical_rate_delta=full_reduced_rate - anchor_reduced_rate,
        macro_step_seconds=manifest.MACRO_STEP_SECONDS,
        mode="hot",
        anchor_id="hot_center_02",
        maximum_absolute_eta=manifest.MAXIMUM_ABSOLUTE_PATCH_COORDINATE,
    )
    engine = ConservativeHeunEngine(
        model=model,
        patch=patch,
        forcing_angular_frequency=(
            2.0 * np.pi / manifest.REFERENCE_PHASE_PERIOD_SECONDS
        ),
        maximum_embedded_error_fraction=manifest.MAXIMUM_EMBEDDED_ERROR_FRACTION,
    )
    state = HiddenAmplitudeState(
        macro=macro,
        amplitudes=np.zeros(hidden_basis.shape[1]),
        forcing_phase=0.0,
        mode="hot",
        elapsed_seconds=0.0,
    )
    evidence = {
        "anchor_coordinate470": coordinate,
        "anchor_coordinate_rate470_per_s": coordinate_rate,
        "anchor_hidden_rate388_per_s": hidden_rate,
        "certified_heun_coordinate470": np.asarray(off_arrays["heun_coordinate470"]),
        "hidden_basis388x2": hidden_basis,
    }
    return engine, state, evidence


def _state_roundtrip(state: HiddenAmplitudeState) -> HiddenAmplitudeState:
    stream = io.BytesIO()
    np.savez(
        stream,
        macro=state.macro,
        amplitudes=state.amplitudes,
        forcing_phase=np.asarray(state.forcing_phase),
        elapsed_seconds=np.asarray(state.elapsed_seconds),
        mode=np.asarray(state.mode),
    )
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as payload:
        return HiddenAmplitudeState(
            macro=payload["macro"],
            amplitudes=payload["amplitudes"],
            forcing_phase=float(payload["forcing_phase"]),
            mode=str(payload["mode"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
        )


def _bitwise_state(left: HiddenAmplitudeState, right: HiddenAmplitudeState) -> bool:
    return bool(
        np.array_equal(left.macro, right.macro)
        and np.array_equal(left.amplitudes, right.amplitudes)
        and left.forcing_phase == right.forcing_phase
        and left.elapsed_seconds == right.elapsed_seconds
        and left.mode == right.mode
    )


def _evaluate(locked: dict) -> tuple[dict, dict[str, np.ndarray]]:
    engine, initial, evidence = _build_engine()
    timestep = manifest.MACRO_STEP_SECONDS
    result = engine.step(initial, timestep)
    decoded = engine.model.decode(result.candidate)
    reference = evidence["certified_heun_coordinate470"]
    anchor = evidence["anchor_coordinate470"]
    endpoint_increment = _relative_increment(decoded, reference, anchor)
    macro_reference = engine.model.split.macro_restriction @ reference
    macro_anchor = engine.model.split.macro_restriction @ anchor
    macro_increment = _relative_increment(
        result.candidate.macro, macro_reference, macro_anchor
    )
    decoder_macro = engine.model.split.macro_restriction @ decoded
    decoder_closure = float(
        np.linalg.norm(decoder_macro - result.candidate.macro)
        / max(float(np.linalg.norm(result.candidate.macro)), np.finfo(float).tiny)
    )
    _macro_rate, _amplitude_rate, hidden_projection = engine.model.project_rate(
        evidence["anchor_coordinate_rate470_per_s"]
    )

    restored_initial = _state_roundtrip(initial)
    replay = engine.step(restored_initial, timestep)
    restart_bitwise = _bitwise_state(initial, restored_initial)
    replay_bitwise = _bitwise_state(result.candidate, replay.candidate)
    oversized = engine.step(initial, manifest.OVERSIZE_REJECTION_FACTOR * timestep)

    selector = HystereticModeSelector(
        relative_switch_margin=manifest.MODE_SWITCH_MARGIN,
        persistence_steps=manifest.MODE_SWITCH_PERSISTENCE,
    )
    first = selector.update(
        current_mode="cold", normalized_distances={"cold": 1.0, "hot": 0.5}
    )
    second = selector.update(
        current_mode=first.mode,
        normalized_distances={"cold": 1.0, "hot": 0.5},
        pending_mode=first.pending_mode,
        pending_count=first.pending_count,
    )
    no_chatter = selector.update(
        current_mode="hot", normalized_distances={"cold": 0.95, "hot": 1.0}
    )
    mode_policy_passed = bool(
        first.mode == "cold"
        and first.pending_mode == "hot"
        and second.mode == "hot"
        and second.switched
        and no_chatter.mode == "hot"
        and not no_chatter.switched
    )

    began = time.perf_counter()
    benchmark_checksum = 0.0
    for _index in range(manifest.BENCHMARK_STEPS):
        benchmark_result = engine.step(initial, timestep)
        benchmark_checksum += float(engine.model.decode(benchmark_result.candidate)[0])
    benchmark_wall = float(time.perf_counter() - began)

    values = {
        "endpoint_increment_defect": endpoint_increment,
        "macro_increment_defect": macro_increment,
        "macro_ledger_defect": result.macro_ledger_defect,
        "decoder_macro_closure": decoder_closure,
        "anchor_hidden_rate_projection_defect": hidden_projection,
        "embedded_error_fraction": result.embedded_error_fraction,
        "start_patch_coordinate": result.start_eta,
        "predictor_patch_coordinate": result.predictor_eta,
        "endpoint_patch_coordinate": result.endpoint_eta,
        "accepted_step": result.accepted,
        "oversize_step_rejected": not oversized.accepted,
        "oversize_failure_reasons": list(oversized.failure_reasons),
        "restart_bitwise": restart_bitwise,
        "replay_bitwise": replay_bitwise,
        "mode_policy_passed": mode_policy_passed,
        "benchmark_steps": manifest.BENCHMARK_STEPS,
        "benchmark_wall_seconds": benchmark_wall,
        "benchmark_steps_per_second": manifest.BENCHMARK_STEPS / benchmark_wall,
        "benchmark_checksum": benchmark_checksum,
        "online_truth_calls": 0,
        "online_fixed_Q_reaction_calls": 0,
        "online_coordinate_retractions": 0,
        "online_nonlinear_roots": 0,
        "online_BDF_microsteps": 0,
    }
    gates = {
        "accepted_macro_step": values["accepted_step"],
        "endpoint_increment": endpoint_increment <= manifest.MAXIMUM_ENDPOINT_INCREMENT_DEFECT,
        "macro_increment": macro_increment <= manifest.MAXIMUM_MACRO_INCREMENT_DEFECT,
        "macro_ledger": result.macro_ledger_defect <= manifest.MAXIMUM_MACRO_LEDGER_DEFECT,
        "decoder_macro_closure": decoder_closure <= manifest.MAXIMUM_DECODER_MACRO_CLOSURE,
        "hidden_rate_projection": hidden_projection <= manifest.MAXIMUM_ANCHOR_HIDDEN_RATE_PROJECTION_DEFECT,
        "embedded_error": result.embedded_error_fraction <= manifest.MAXIMUM_EMBEDDED_ERROR_FRACTION,
        "patch_trust": max(abs(result.start_eta), abs(result.predictor_eta), abs(result.endpoint_eta)) <= manifest.MAXIMUM_ABSOLUTE_PATCH_COORDINATE,
        "oversize_fail_closed": values["oversize_step_rejected"],
        "restart_bitwise": restart_bitwise,
        "replay_bitwise": replay_bitwise,
        "hysteretic_mode_policy": mode_policy_passed,
        "online_cost": benchmark_wall <= manifest.MAXIMUM_BENCHMARK_WALL_SECONDS,
        "online_forbidden_work": all(values[name] == 0 for name in (
            "online_truth_calls",
            "online_fixed_Q_reaction_calls",
            "online_coordinate_retractions",
            "online_nonlinear_roots",
            "online_BDF_microsteps",
        )),
    }
    passed = bool(all(gates.values()))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "gates": gates,
        "gate_values": values,
        "input_lock": locked,
    }
    arrays = {
        **evidence,
        "candidate_coordinate470": decoded,
        "candidate_macro82": result.candidate.macro,
        "candidate_amplitudes2": result.candidate.amplitudes,
        "anchor_macro82": initial.macro,
        "anchor_amplitudes2": initial.amplitudes,
        "patch_eta_dual82": engine.patch.eta_dual,
        "patch_anchor_reduced_rate84_per_s": engine.patch.anchor_reduced_rate,
        "patch_physical_rate_delta84_per_s": engine.patch.physical_rate_delta,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = manifest.parent.arclength._source()._post().manifest.transition.manifest.cold.manifest
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": status,
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "case", "path", "bytes", "sha256", "scientific_status"
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("truth-free hot engine result already exists")
    locked = _validate_manifest(require_clean=True)
    metrics, arrays = _evaluate(locked)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "hot_engine_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "hot_engine_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with (CANONICAL_DIRECTORY / "hot_engine_checkpoint.npz").open("wb") as handle:
        np.savez(
            handle,
            macro=arrays["candidate_macro82"],
            amplitudes=arrays["candidate_amplitudes2"],
            forcing_phase=np.asarray(
                2.0 * np.pi * manifest.MACRO_STEP_SECONDS
                / manifest.REFERENCE_PHASE_PERIOD_SECONDS
            ),
            elapsed_seconds=np.asarray(manifest.MACRO_STEP_SECONDS),
            mode=np.asarray("hot"),
            anchor_id=np.asarray("hot_center_02"),
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "adaptive_complete_cycle_acquisition_manifest_authorized": metrics["passed"],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT if metrics["passed"] else None,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    values = metrics["gate_values"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Truth-free conservative hot-mode engine",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"One 0.25 ms Heun macro step has endpoint-increment defect `{values['endpoint_increment_defect']:.6e}`, macro-increment defect `{values['macro_increment_defect']:.6e}`, and embedded correction `{values['embedded_error_fraction']:.6e}`.",
            "",
            f"The 100,000-step update-plus-full-decode benchmark took `{values['benchmark_wall_seconds']:.6f}` s (`{values['benchmark_steps_per_second']:.3f}` steps/s). Restart and replay are bitwise; a 2x oversized step rejects fail-closed.",
            "",
            "Every online truth, fixed-Q reaction, retraction, nonlinear-root, and BDF counter is zero. A pass authorizes only the final adaptive complete-cycle acquisition manifest, not complete-cycle execution itself.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    print(json.dumps(_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
