#!/usr/bin/env python3
"""Scan declared common surfaces for full-window cumulative recovery."""

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

import run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_manifest_wp10c9d6c7c3b5c3h2j as h2j  # noqa: E402
import run_causal_inner_nonlinear_5ms_extraction_surface_certificate_wp10c9d6c7c3b5c3h2i1 as h2i1  # noqa: E402
import run_causal_inner_nonlinear_5ms_inner_face_half_cell_audit_wp10c9d6c7c3b5c3h2h1 as h2h1  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2j1"
ANALYZED_BASE_COMMIT = "ef491daea69fa0f92075b5de768e7b84b9de839d"
ANALYZED_BASE_PARENT = "4f3359a2070e90929a8002560c047bb3fa73c378"
ANALYZED_BASE_TREE = "2a2abe4b46f857d0348eba615716d4310ccb1552"

ARTIFACT = (
    "causal_inner_nonlinear_5ms_cumulative_extraction_recovery_audit_"
    "wp10c9d6c7c3b5c3h2j1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_"
    "audit_wp10c9d6c7c3b5c3h2j1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_cumulative_extraction_recovery_"
    "audit_wp10c9d6c7c3b5c3h2j1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_CUMULATIVE_"
    "EXTRACTION_RECOVERY_AUDIT_WP10C9D6C7C3B5C3H2J1_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"

LAYOUTS = h2h1.LAYOUTS
FACES = h2j.CANDIDATE_COARSE_FACE_INDICES
MULTIPLIERS = h2j.LAYOUT_FACE_MULTIPLIERS
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)


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
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _source_identity() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in (THIS_RUNNER, THIS_TEST) if (ROOT / path).exists()}


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(h2j.SUMMARY_PATH)
    manifest = _read_json(h2j.MANIFEST_PATH)
    if (
        not summary["passed"]
        or not summary["cumulative_recovery_audit_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c3h2j1_cumulative_extraction_recovery_audit"
        or tuple(manifest["candidate_coarse_face_indices"]) != FACES
        or tuple(manifest["layout_face_multipliers"]) != MULTIPLIERS
        or manifest["minimum_consecutive_passing_surfaces"] != 2
        or tuple(manifest["binding_window_seconds"]) != (0.002, 0.005)
    ):
        raise RuntimeError("h2j1 frozen contract changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^") != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}") != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2j1 analyzed identity changed")
    return summary, manifest


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    identity = _source_identity()
    if not CHECKPOINT_JSON.exists():
        return ({"work_package": WORK_PACKAGE, "analyzed_base_commit": ANALYZED_BASE_COMMIT, "source_identity": identity, "completed": []}, {})
    progress = _read_json(CHECKPOINT_JSON)
    if progress.get("work_package") != WORK_PACKAGE or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT or progress.get("source_identity") != identity:
        raise RuntimeError("saved h2j1 progress belongs to different code")
    arrays = _load_npz(CHECKPOINT_ARRAYS) if CHECKPOINT_ARRAYS.exists() else {}
    return progress, arrays


def _observable_from_ledger(ledger, extraction_face: int, coupling_face: int):
    fluxes = np.asarray(ledger.interfaces.candidate_shared_face_fluxes_over_c, dtype=float)
    residual = np.asarray(ledger.residual_rows, dtype=float)
    cooling = np.asarray(ledger.cooling_rows, dtype=float)
    height = np.asarray(ledger.lower_height_work_rows, dtype=float)
    transport = np.asarray(ledger.conservative_transport_rows, dtype=float)
    region = slice(extraction_face, coupling_face)
    value = np.concatenate((fluxes[extraction_face, CONSERVATIVE_FIELDS], fluxes[coupling_face, CONSERVATIVE_FIELDS], -np.sum(residual[region][:, CONSERVATIVE_FIELDS], axis=0), -np.sum(cooling[region][:, CONSERVATIVE_FIELDS[1:]], axis=0), -np.sum(height[region][:, CONSERVATIVE_FIELDS[1:]], axis=0)))
    face_difference = fluxes[extraction_face, CONSERVATIVE_FIELDS] - fluxes[coupling_face, CONSERVATIVE_FIELDS]
    source_remainder = residual - transport
    reconstructed = face_difference - np.sum(source_remainder[region][:, CONSERVATIVE_FIELDS], axis=0)
    scale = max(float(np.linalg.norm(reconstructed)), float(np.linalg.norm(value[6:9])), np.finfo(float).tiny)
    return value, float(np.linalg.norm(reconstructed - value[6:9]) / scale)


def _evaluate(times: np.ndarray, inputs: dict):
    progress, cached = _load_progress()
    completed = set(progress["completed"])
    started = time.monotonic()
    for layout_name, multiplier in zip(LAYOUTS, MULTIPLIERS, strict=True):
        payload = inputs[layout_name]
        context = payload["configuration"]["context"]
        coupling = h2i1.h2i.COUPLING_COARSE_FACE_INDEX * multiplier
        layout_faces = np.asarray(FACES, dtype=int) * multiplier
        radii = context.grid.edges[layout_faces] / context.grid.gravitational_radius
        if layout_name != LAYOUTS[0]:
            coarse = inputs[LAYOUTS[0]]["configuration"]["context"].grid
            coarse_radii = coarse.edges[np.asarray(FACES)] / coarse.gravitational_radius
            if not np.array_equal(radii, coarse_radii):
                raise RuntimeError("common surface bit patterns changed")
        for branch in ("base", "anchor"):
            for common_index in range(times.size):
                key = f"{layout_name}__{branch}__t{common_index}"
                if key in completed:
                    continue
                print(f"h2j1: evaluate {key}", flush=True)
                state_index = int(payload["accepted_indices"][common_index])
                ledger = causal_five_field_radial_candidate_ledger(context, payload[branch][state_index])
                values = []
                identities = []
                for face in layout_faces:
                    value, defect = _observable_from_ledger(ledger, int(face), coupling)
                    values.append(value)
                    identities.append(defect)
                cached[f"{key}__observables"] = np.asarray(values)
                cached[f"{key}__identities"] = np.asarray(identities)
                cached[f"{key}__ledger_audit"] = np.asarray((ledger.interfaces.shared_conservative_face_defect, ledger.local_block_ledger_defect, ledger.source_double_count_defect, ledger.interfaces.incoming_excision_characteristics), dtype=float)
                completed.add(key)
                progress["completed"] = sorted(completed)
                progress["elapsed_wall_seconds"] = time.monotonic() - started
                _save_progress(progress, cached)
    histories = {}
    identities = {}
    ledger_audits = {}
    for layout_name in LAYOUTS:
        for branch in ("base", "anchor"):
            histories[(layout_name, branch)] = np.asarray([cached[f"{layout_name}__{branch}__t{i}__observables"] for i in range(times.size)])
            identities[(layout_name, branch)] = np.asarray([cached[f"{layout_name}__{branch}__t{i}__identities"] for i in range(times.size)])
            ledger_audits[(layout_name, branch)] = np.asarray([cached[f"{layout_name}__{branch}__t{i}__ledger_audit"] for i in range(times.size)])
    coarse_grid = inputs[LAYOUTS[0]]["configuration"]["context"].grid
    radii = coarse_grid.edges[np.asarray(FACES)] / coarse_grid.gravitational_radius
    return histories, identities, ledger_audits, np.asarray(radii), time.monotonic() - started


def _analyze(histories: dict, identities: dict, ledger_audits: dict, radii: np.ndarray, times: np.ndarray):
    parent_arrays = _load_npz(h2i1.DECISIVE_ARRAYS)
    scales = np.asarray(parent_arrays["export_scales"], dtype=float)
    envelope = float(parent_arrays["temporal_uncertainty_envelope"])
    responses = {name: histories[(name, "anchor")] - histories[(name, "base")] for name in LAYOUTS}
    faces = []
    for position, coarse_face in enumerate(FACES):
        instant_histories = tuple(responses[name][:, position, :] for name in LAYOUTS)
        cumulative_histories = tuple(h2i1._cumulative(values, times) for values in instant_histories)
        instant = h2i1._metric(instant_histories, scales)
        cumulative = h2i1._metric(cumulative_histories, scales * float(times[-1]))
        instant_temporal = h2i1._temporal(instant, envelope)
        cumulative_temporal = h2i1._temporal(cumulative, envelope)
        audit_defect = max(float(np.max(values[:, position])) for values in identities.values())
        passed = bool(instant_temporal["passed"] and cumulative_temporal["passed"] and audit_defect <= 1.0e-12)
        faces.append({"coarse_face_index": coarse_face, "radius_rg": radii[position], "passed": passed, "instantaneous": {**instant, "temporal_classification": instant_temporal}, "cumulative": {**cumulative, "temporal_classification": cumulative_temporal}, "maximum_exterior_prefix_direct_identity_defect": audit_defect})
    required = h2j.MINIMUM_CONSECUTIVE_PASSING_SURFACES
    selected = None
    for index in range(len(faces) - required + 1):
        if all(faces[index + offset]["passed"] for offset in range(required)):
            selected = index
            break
    ledger_max = np.max(np.concatenate(tuple(ledger_audits.values()), axis=0), axis=0)
    audits = {"maximum_shared_conservative_face_defect": ledger_max[0], "maximum_local_block_ledger_defect": ledger_max[1], "maximum_source_double_count_defect": ledger_max[2], "maximum_incoming_excision_characteristics": int(ledger_max[3]), "passed": bool(ledger_max[0] <= 1.0e-12 and ledger_max[1] <= 1.0e-11 and ledger_max[2] <= 1.0e-12 and int(ledger_max[3]) == 0)}
    recovery = selected is not None and audits["passed"]
    analysis = {"binding_window_seconds": (float(times[0]), float(times[-1])), "faces": faces, "minimum_consecutive_passing_surfaces": required, "recovery_selected": recovery, "selected_coarse_face_index": faces[selected]["coarse_face_index"] if recovery else None, "selected_radius_rg": faces[selected]["radius_rg"] if recovery else None, "selected_layout_face_indices": tuple(faces[selected]["coarse_face_index"] * m for m in MULTIPLIERS) if recovery else None, "ledger_audits": audits, "raw_inner_face_rejection_preserved": True, "pointwise_horizon_flux_convergence_claimed": False}
    decisive = {"times_seconds": times, "candidate_coarse_face_indices": np.asarray(FACES), "candidate_radii_rg": radii, "export_scales": scales, "temporal_uncertainty_envelope": np.asarray(envelope)}
    for name in LAYOUTS:
        decisive[f"{name}__base_observables"] = histories[(name, "base")]
        decisive[f"{name}__anchor_observables"] = histories[(name, "anchor")]
        decisive[f"{name}__response"] = responses[name]
        decisive[f"{name}__base_identity_defects"] = identities[(name, "base")]
        decisive[f"{name}__anchor_identity_defects"] = identities[(name, "anchor")]
    return analysis, decisive


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": ANALYZED_BASE_COMMIT, "latest_work_package": WORK_PACKAGE})
    _write_json(CANONICAL_SUMMARY, catalog)


def main() -> int:
    _validate_parent()
    _certificate, times, inputs = h2h1._standardized_inputs()
    histories, identities, ledgers, radii, elapsed = _evaluate(times, inputs)
    analysis, decisive = _analyze(histories, identities, ledgers, radii, times)
    passed = bool(analysis["recovery_selected"])
    classification = "five_ms_cumulative_extraction_recovery_certified_fourth_duration_manifest_authorized" if passed else "five_ms_cumulative_extraction_recovery_failed_later_duration_blocked"
    authorized_next = "WP10c9d6c7c3b5c4a_fourth_duration_rung_manifest" if passed else "WP10c9d6c7c3b5c3h2k_early_time_inner_buffer_localization_manifest"
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": classification, "passed": passed, "analysis": analysis, "middle_fine_5ms_extraction_partition_spatial_certificate_issued": passed, "third_duration_rung_extraction_partition_spatial_convergence_certified": passed, "raw_inner_face_spatial_convergence_certified": False, "pointwise_horizon_flux_convergence_certified": False, "fourth_duration_rung_manifest_authorized": passed, "physical_failure_detected": False, "fixed_q_micro_solver_authorized": False, "reduced_slow_evolution_authorized": False, "authorized_next": authorized_next}
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "analyzed_base_commit": ANALYZED_BASE_COMMIT, "candidate_coarse_face_indices": FACES, "candidate_radii_rg": radii, "layout_face_multipliers": MULTIPLIERS, "observable_names": h2i1.OBSERVABLE_NAMES, "spatial_gates": h2i1.SPATIAL_GATES, "temporal_gates": h2i1.TEMPORAL_GATES})
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "analyzed_base_commit": ANALYZED_BASE_COMMIT, "source_parent_commit": ANALYZED_BASE_COMMIT, "scientific_status": "CERTIFIED" if passed else "REJECTED", "working_head": _git_value("rev-parse", "HEAD"), "runner": THIS_RUNNER, "test": THIS_TEST, "report": REPORT_RELATIVE, "elapsed_wall_seconds": elapsed, "input_hashes": {"manifest": _sha256(h2j.MANIFEST_PATH), "fixed_surface_certificate": _sha256(h2i1.SUMMARY_PATH)}, "implementation_source_hashes": _source_identity(), "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}"})
    selected = next((face for face in analysis["faces"] if face["coarse_face_index"] == analysis["selected_coarse_face_index"]), None)
    if selected is None:
        measurements = "No two consecutive declared surfaces pass both binding channels."
    else:
        measurements = f"The innermost prospectively selected surface is coarse face `{selected['coarse_face_index']}` at `R={selected['radius_rg']:.8f} r_g`. Instantaneous RMS/min-component orders are `{selected['instantaneous']['observed_rms_order']:.6f}/{selected['instantaneous']['minimum_significant_component_order']:.6f}`; cumulative values are `{selected['cumulative']['observed_rms_order']:.6f}/{selected['cumulative']['minimum_significant_component_order']:.6f}`."
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Nonlinear 5 ms cumulative extraction-recovery audit WP10c9d6c7c3b5c3h2j1", "", "## Classification", "", f"`{classification}`", "", measurements, "", "The scan uses the complete 2-5 ms common window and the frozen innermost-two-consecutive-surface rule. No new state was propagated and no operator or production default changed.", "", "The certificate applies to the conservative exterior domain partition. The raw excision-face and pointwise horizon-flux rejection remains preserved. No physical failure was detected.", "", f"Only `{authorized_next}` is authorized. Fixed-Q experiments and reduced slow evolution remain blocked.", "")), encoding="utf-8")
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
