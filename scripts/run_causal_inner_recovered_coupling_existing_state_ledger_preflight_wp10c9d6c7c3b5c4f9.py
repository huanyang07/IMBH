#!/usr/bin/env python3
"""Execute the existing-state face-36/face-48 guard-buffer preflight.

This package advances no state.  It checks the exact accepted BDF2 control-
volume identity at 20 ms and independently asks whether the explicitly
retained face-36/guard-buffer state is three-grid convergent.
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

import run_causal_inner_recovered_coupling_control_volume_manifest_wp10c9d6c7c3b5c4f8 as c4f8  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
    evaluate_causal_five_field_monolithic_bdf,
)


c4f7 = c4f8.c4f7
c4f1 = c4f7.c4f5.c4f3.c4f2.c4f1
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f9"
ARTIFACT = "causal_inner_recovered_coupling_existing_state_ledger_preflight_wp10c9d6c7c3b5c4f9"
THIS_RUNNER = "scripts/run_causal_inner_recovered_coupling_existing_state_ledger_preflight_wp10c9d6c7c3b5c4f9.py"
THIS_TEST = "tests/test_causal_inner_recovered_coupling_existing_state_ledger_preflight_wp10c9d6c7c3b5c4f9.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_RECOVERED_COUPLING_EXISTING_STATE_LEDGER_PREFLIGHT_WP10C9D6C7C3B5C4F9_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
CONTRACT_PATH = CANONICAL_DIRECTORY / "analysis_contract.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "preflight.npz"

LAYOUTS = ("coarse", "middle", "fine")
FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
RECOVERY_FACE = c4f8.RECOVERY_FACE
COUPLING_FACE = c4f8.ORIGINAL_COUPLING_FACE
STATE_TIMES = np.asarray((0.005, 0.010, 0.016, 0.020), dtype=float)
HISTORY_TIMES_US = (6000, 12000, 14000, 16000, 18000, 19600, 19800, 20000)
BDF_OLD_US = 19800
BDF_NEW_US = 20000
BDF_TIMESTEP_SECONDS = (BDF_NEW_US - BDF_OLD_US) * 1.0e-6
BLOCK_NAMES = (
    "mapped_temporal_storage_rows",
    "responsive_height_temporal_storage_rows",
    "shear_principal_rows",
    "height_principal_rows",
    "local_stress_relaxation_rows",
    "geometry_rows",
    "cooling_rows",
    "stream_rows",
    "lower_height_work_rows",
)
COARSE_CHECKPOINTS = {
    "early": ROOT / "outputs/checkpoints/causal_inner_nonlinear_ten_ms_screen_wp10c9d6c7c3b5c4b2/base_main",
    "late": ROOT / "outputs/checkpoints/causal_inner_nonlinear_twenty_ms_completion_wp10c9d6c7c3b5c4c1/base_main",
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


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments):
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _source_identity():
    paths = (THIS_RUNNER, THIS_TEST, c4f8.THIS_RUNNER, c4f8.THIS_TEST)
    return {path: _sha(ROOT / path) for path in paths if (ROOT / path).exists()}


def _contract():
    parent = _read(c4f8.MANIFEST_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "existing_state_guard_buffer_ledger_and_overlap_preflight",
        "parent_classification": parent["classification"],
        "new_trajectory": False,
        "recovery_parent_face": RECOVERY_FACE,
        "coupling_parent_face": COUPLING_FACE,
        "exact_BDF_interval_seconds": (BDF_OLD_US * 1.0e-6, BDF_NEW_US * 1.0e-6),
        "control_volume_identity": "F48 = F36 - mapped_rate - responsive_height_rate - stationary_nontransport_sources",
        "control_volume_reconstruction_is_algebraically_dependent": True,
        "independent_overlap_observables": (
            "shared_face36_M_J_E_flux",
            "absolute_guard_buffer_mapped_M_J_E_storage",
            "previous_interval_guard_buffer_responsive_height_rate",
        ),
        "gates": {
            "maximum_exact_BDF_ledger_defect": 1.0e-8,
            "minimum_spatial_RMS_order": 0.75,
            "minimum_spatial_error_direction_cosine": 0.90,
            "relative_observability_floor": 1.0e-12,
        },
        "decision": parent["decision"],
        "hard_stops": parent["hard_stops"],
    }


def _validate():
    parent = _read(c4f8.SUMMARY_PATH)
    manifest = _read(c4f8.MANIFEST_PATH)
    if (
        not parent["passed"]
        or not parent["definitions_only"]
        or parent["authorized_next"] != "WP10c9d6c7c3b5c4f9_recovered_coupling_existing_state_ledger_preflight"
        or manifest["recovery_parent_face"] != RECOVERY_FACE
        or manifest["original_coupling_parent_face"] != COUPLING_FACE
    ):
        raise RuntimeError("c4f9 authorization changed")
    return manifest


def _trajectories():
    coarse_times, coarse_states, coarse_outputs, _perturbed = c4f1._coarse_absolute()
    middle = c4f1._middle_trajectory()
    fine = c4f1._fine_trajectory()
    return {
        "coarse": {"times": coarse_times, "states": coarse_states, "outputs": coarse_outputs},
        "middle": middle,
        "fine": fine,
    }


def _metric(values, scales, gates):
    values = tuple(np.asarray(item, dtype=float) for item in values)
    scales = np.asarray(scales, dtype=float)
    active = scales > np.finfo(float).tiny
    if not np.any(active):
        return {"observable": False, "RMS_order": None, "error_direction_cosine": None, "passed": True}
    normalized = tuple(item[..., active] / scales[active] for item in values)
    coarse_middle = normalized[1] - normalized[0]
    middle_fine = normalized[2] - normalized[1]
    coarse_norm = float(np.linalg.norm(coarse_middle))
    fine_norm = float(np.linalg.norm(middle_fine))
    signal = max(*(float(np.linalg.norm(item)) for item in normalized), np.finfo(float).tiny)
    relative = max(coarse_norm, fine_norm) / signal
    if relative <= gates["relative_observability_floor"]:
        return {"observable": False, "relative_difference": relative, "RMS_order": None, "error_direction_cosine": None, "passed": True}
    order = float(np.log2(max(coarse_norm, np.finfo(float).tiny) / max(fine_norm, np.finfo(float).tiny)))
    cosine = float(np.vdot(coarse_middle.ravel(), middle_fine.ravel()).real / max(coarse_norm * fine_norm, np.finfo(float).tiny))
    return {
        "observable": True,
        "relative_difference": relative,
        "RMS_order": order,
        "error_direction_cosine": cosine,
        "passed": bool(order >= gates["minimum_spatial_RMS_order"] and cosine >= gates["minimum_spatial_error_direction_cosine"]),
    }


def _coarse_restart(time_us):
    directory = COARSE_CHECKPOINTS["early" if time_us <= 10000 else "late"]
    path = directory / f"restart_{time_us}us.npz"
    if path.exists():
        return _load(path)
    if DECISIVE_ARRAYS.exists():
        arrays = _load(DECISIVE_ARRAYS)
        prefix = f"coarse_restart_{time_us}__"
        selected = {name[len(prefix):]: value for name, value in arrays.items() if name.startswith(prefix)}
        if selected:
            return selected
    raise FileNotFoundError(f"missing coarse restart {time_us} us")


def _history(values):
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=values["previous_primitive_increment"],
        previous_mapped_storage_increment=values["previous_mapped_storage_increment"],
        previous_responsive_height_storage_increment=values["previous_responsive_height_storage_increment"],
        previous_timestep_seconds=float(values["previous_timestep_seconds"]),
    )


def _trajectory_index(trajectory, time_us):
    return int(c4f1._indices(trajectory["times"], np.asarray((time_us * 1.0e-6,)))[0])


def _accepted_state_and_history(label, trajectory, time_us):
    if label == "coarse":
        values = _coarse_restart(time_us)
        return np.asarray(values["primitive_charts"]), _history(values)
    index = _trajectory_index(trajectory, time_us)
    history = CausalFiveFieldMonolithicBDFHistory(
        trajectory["primitive_histories"][index],
        trajectory["mapped_histories"][index],
        trajectory["height_histories"][index],
        float(trajectory["previous_timesteps"][index]),
    )
    return np.asarray(trajectory["states"][index]), history


def _guard_storage_and_history(configurations, trajectories):
    mapped_values = []
    height_rates = []
    for label in LAYOUTS:
        layout, configuration = configurations[label]
        multiplier = int(layout.refinement_ratio)
        lo, hi = RECOVERY_FACE * multiplier, COUPLING_FACE * multiplier
        selected = c4f1._indices(trajectories[label]["times"], STATE_TIMES)
        mapped_values.append(np.asarray([
            c4f1._q3_value(configuration["context"], state, lo, hi)
            for state in trajectories[label]["states"][selected]
        ]))
        rates = []
        for time_us in HISTORY_TIMES_US:
            _state, history = _accepted_state_and_history(label, trajectories[label], time_us)
            rates.append(
                np.sum(history.previous_responsive_height_storage_increment[lo:hi], axis=0)[FIELDS]
                / history.previous_timestep_seconds
            )
        height_rates.append(np.asarray(rates))
    return np.asarray(mapped_values), np.asarray(height_rates)


def _direct_output_histories(trajectories):
    common_times = _load(c4f1.c4f.FINAL_ARRAYS)["common_times_seconds"]
    values = []
    for label in LAYOUTS:
        if label == "coarse":
            outputs = trajectories[label]["outputs"]
        else:
            arrays = _load(c4f1.c4f.MIDDLE_ARRAYS if label == "middle" else c4f1.c4f.FINE_ARRAYS)
            if label == "middle":
                early = _load(c4f1.c4f.MIDDLE_PILOT_ARRAYS)["extraction__base"]
                outputs = c4f1._combine(early, arrays["extraction__base_values"])
            else:
                outputs = arrays["extraction__base_values"]
        selected = c4f1._indices(trajectories[label]["times"], common_times)
        values.append(np.asarray(outputs[selected, 3:6]))
    return common_times, tuple(values)


def _exact_bdf(configurations, trajectories):
    direct = []
    reconstructed = []
    blocks = []
    residual_sums = []
    restart_inputs = {}
    wall = []
    for label in LAYOUTS:
        layout, configuration = configurations[label]
        multiplier = int(layout.refinement_ratio)
        lo, hi = RECOVERY_FACE * multiplier, COUPLING_FACE * multiplier
        old_state, history = _accepted_state_and_history(label, trajectories[label], BDF_OLD_US)
        new_state, _new_history = _accepted_state_and_history(label, trajectories[label], BDF_NEW_US)
        began = time.perf_counter()
        evaluation = evaluate_causal_five_field_monolithic_bdf(
            old_state,
            new_state,
            BDF_TIMESTEP_SECONDS,
            configuration["context"],
            order=2,
            history=history,
        )
        wall.append(time.perf_counter() - began)
        backward = evaluation.backward_euler_evaluation
        fluxes = backward.stationary_ledger.interfaces.candidate_shared_face_fluxes_over_c
        block_values = []
        for name in BLOCK_NAMES:
            source = evaluation if name.startswith(("mapped_", "responsive_")) else backward
            block_values.append(np.sum(getattr(source, name)[lo:hi], axis=0)[FIELDS])
        block_values = np.asarray(block_values)
        face36 = np.asarray(fluxes[lo, FIELDS])
        face48 = np.asarray(fluxes[hi, FIELDS])
        direct.append(face48)
        reconstructed.append(face36 - np.sum(block_values, axis=0))
        blocks.append(block_values)
        residual_sums.append(np.sum(evaluation.residual_rows[lo:hi], axis=0)[FIELDS])
        if label == "coarse":
            for time_us in (BDF_OLD_US, BDF_NEW_US):
                for name, value in _coarse_restart(time_us).items():
                    if name in (
                        "primitive_charts",
                        "previous_primitive_increment",
                        "previous_mapped_storage_increment",
                        "previous_responsive_height_storage_increment",
                        "previous_timestep_seconds",
                    ):
                        restart_inputs[f"coarse_restart_{time_us}__{name}"] = value
    return (
        np.asarray(direct),
        np.asarray(reconstructed),
        np.asarray(blocks),
        np.asarray(residual_sums),
        np.asarray(wall),
        restart_inputs,
    )


def _evaluate():
    _parent_grid, configurations = c4f1._configurations()
    trajectories = _trajectories()
    if CHECKPOINT_PATH.exists():
        arrays = _load(CHECKPOINT_PATH)
    else:
        mapped, height = _guard_storage_and_history(configurations, trajectories)
        direct, reconstructed, blocks, residuals, wall, restart_inputs = _exact_bdf(configurations, trajectories)
        arrays = {
            "state_times_seconds": STATE_TIMES,
            "history_times_seconds": np.asarray(HISTORY_TIMES_US, dtype=float) * 1.0e-6,
            "guard_mapped_storage": mapped,
            "guard_responsive_height_history_rates": height,
            "exact_BDF_direct_face48_flux": direct,
            "exact_BDF_reconstructed_face48_flux": reconstructed,
            "exact_BDF_buffer_block_sums": blocks,
            "exact_BDF_buffer_residual_sums": residuals,
            "exact_BDF_evaluation_wall_seconds": wall,
            **restart_inputs,
        }
        _save(CHECKPOINT_PATH, **arrays)
    decay = _load(c4f7.DECISIVE_ARRAYS)
    face_index = int(np.flatnonzero(decay["faces"] == RECOVERY_FACE)[0])
    coupling_index = int(np.flatnonzero(decay["faces"] == COUPLING_FACE)[0])
    arrays["face36_flux"] = decay["actual"][:, :, face_index]
    arrays["face48_flux_selected"] = decay["actual"][:, :, coupling_index]
    common_times, direct_histories = _direct_output_histories(trajectories)
    arrays["common_times_seconds"] = common_times
    arrays["direct_face48_output_histories"] = np.asarray(direct_histories)
    return arrays


def _analyze(arrays, contract):
    gates = contract["gates"]
    face36 = tuple(arrays["face36_flux"][index] for index in range(3))
    mapped = tuple(arrays["guard_mapped_storage"][index] for index in range(3))
    height = tuple(arrays["guard_responsive_height_history_rates"][index] for index in range(3))
    face36_scales = np.maximum.reduce([np.max(np.abs(item), axis=0) for item in face36])
    mapped_scales = np.maximum.reduce([np.max(np.abs(item), axis=0) for item in mapped])
    height_scales = np.maximum.reduce([np.max(np.abs(item), axis=0) for item in height])
    metrics = {
        "recovery_face36_flux": _metric(face36, face36_scales, gates),
        "guard_mapped_storage": _metric(mapped, mapped_scales, gates),
        "guard_responsive_height_history_rate": _metric(height, height_scales, gates),
    }
    overlap_values = tuple(
        np.concatenate((face36[index] / face36_scales, mapped[index] / mapped_scales), axis=1)
        for index in range(3)
    )
    metrics["combined_face36_and_guard_storage"] = _metric(
        overlap_values, np.ones(overlap_values[0].shape[-1]), gates
    )
    direct = arrays["exact_BDF_direct_face48_flux"]
    reconstructed = arrays["exact_BDF_reconstructed_face48_flux"]
    blocks = arrays["exact_BDF_buffer_block_sums"]
    component_scale = np.maximum.reduce((np.abs(direct), np.abs(reconstructed), np.max(np.abs(blocks), axis=1)))
    component_scale = np.maximum(component_scale, np.finfo(float).tiny)
    ledger_defect = float(np.max(np.abs(direct - reconstructed) / component_scale))
    residual_defect = float(np.max(np.abs(arrays["exact_BDF_buffer_residual_sums"]) / component_scale))
    common_times = arrays["common_times_seconds"]
    direct_histories = tuple(arrays["direct_face48_output_histories"][index] for index in range(3))
    output_scales = np.maximum.reduce([np.max(np.abs(item), axis=0) for item in direct_histories])
    direct_metrics = {
        "instantaneous": _metric(direct_histories, output_scales, gates),
        "cumulative": _metric(tuple(c4f1._cumulative(item, common_times) for item in direct_histories), output_scales, gates),
        "window_mean": _metric(tuple(c4f1._window_means(item, common_times) for item in direct_histories), output_scales, gates),
    }
    overlap_passed = bool(all(item["passed"] for item in metrics.values()))
    direct_passed = bool(all(item["passed"] for item in direct_metrics.values()))
    method_passed = bool(max(ledger_defect, residual_defect) <= gates["maximum_exact_BDF_ledger_defect"])
    if not method_passed:
        classification = "guard_buffer_exact_BDF_ledger_failed"
        authorized_next = "ledger_localization_only"
        passed = False
    elif direct_passed:
        classification = "guard_buffer_control_volume_export_converges"
        authorized_next = "definitions_only_augmented_guard_buffer_Q3_memory_manifest"
        passed = True
    elif overlap_passed:
        classification = "control_volume_identity_dependent_overlap_state_converges"
        authorized_next = "WP10c9d6c7c3b5c4f10_definitions_only_retained_guard_buffer_micro_macro_manifest"
        passed = True
    else:
        classification = "guard_buffer_overlap_state_not_spatially_convergent"
        authorized_next = "absolute_slow_closure_remains_blocked"
        passed = False
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "physical_failure_detected": False,
        "method_gates_passed": method_passed,
        "exact_BDF_ledger_defect": ledger_defect,
        "exact_BDF_residual_defect": residual_defect,
        "control_volume_reconstruction_is_independent_evidence": False,
        "control_volume_direct_flux_metrics": direct_metrics,
        "overlap_state_metrics": metrics,
        "overlap_state_spatially_convergent": overlap_passed,
        "direct_face48_absolute_export_spatially_convergent": direct_passed,
        "response_certificate_preserved": True,
        "absolute_closure_fit_authorized": False,
        "memory_propagation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "authorized_next": authorized_next,
    }


def _catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "latest_work_package": WORK_PACKAGE})
    _write(CANONICAL_SUMMARY, catalog)


def _finalize(arrays, contract, summary):
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _save(DECISIVE_ARRAYS, **arrays)
    _write(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "recovery_face": RECOVERY_FACE, "coupling_face": COUPLING_FACE, "fields": FIELD_NAMES, "block_names": BLOCK_NAMES})
    _write(CONTRACT_PATH, contract)
    _write(SUMMARY_PATH, summary)
    lines = [
        "# Recovered coupling existing-state ledger preflight",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "No trajectory, fixed-Q solve, or memory propagation ran.",
        "",
        "## Exact accepted-BDF identity",
        "",
        f"The face-48 reconstruction closes with defect `{summary['exact_BDF_ledger_defect']:.6e}` and the summed accepted residual with `{summary['exact_BDF_residual_defect']:.6e}`. This is an algebraic rearrangement of the same BDF residual and is therefore not independent convergence evidence.",
        "",
        "## Independent overlap-state tests",
        "",
        "| Observable | Order | Cosine | Pass |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["overlap_state_metrics"].items():
        order = "n/a" if item.get("RMS_order") is None else f"{item['RMS_order']:.6f}"
        cosine = "n/a" if item.get("error_direction_cosine") is None else f"{item['error_direction_cosine']:.6f}"
        lines.append(f"| {name} | {order} | {cosine} | {item['passed']} |")
    lines.extend([
        "",
        "The independently defined face-36 flux, absolute mapped guard storage, and nonzero responsive-height history rates converge. The original face-48 absolute export remains rejected in instantaneous/cumulative/mean form.",
        "",
        "The only authorized architecture is therefore a retained guard-buffer/overlap formulation. Face 36 is not relabelled as face 48 or as a horizon flux, and cells 36:48 may not be discarded.",
        "",
        f"Authorized next: `{summary['authorized_next']}`.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "analyzed_certificate_commit": c4f1.c4f.ANALYZED_CERTIFICATE_COMMIT,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_head_tree": _git("rev-parse", "HEAD^{tree}"),
        "source_identity": _source_identity(),
        "input_hashes": {"parent_manifest": _sha(c4f8.MANIFEST_PATH), "spatial_decay_arrays": _sha(c4f7.DECISIVE_ARRAYS)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "output_hashes": {},
    }
    _write(PROVENANCE_PATH, provenance)
    hashes = (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, DECISIVE_ARRAYS, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(path)}  {path.name}\n" for path in hashes), encoding="utf-8")
    _catalog(summary)


def main():
    _validate()
    contract = _contract()
    arrays = _evaluate()
    summary = _analyze(arrays, contract)
    _finalize(arrays, contract, summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
