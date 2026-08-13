#!/usr/bin/env python3
"""Localize the absolute coupling-flux refinement-direction reversal."""

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

import run_causal_inner_absolute_coupling_localization_manifest_wp10c9d6c7c3b5c4f2 as c4f2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)


c4f1 = c4f2.c4f1
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f3"
ANALYZED_CERTIFICATE_COMMIT = c4f2.ANALYZED_CERTIFICATE_COMMIT
ARTIFACT = "causal_inner_selected_time_absolute_coupling_localization_wp10c9d6c7c3b5c4f3"
THIS_RUNNER = "scripts/run_causal_inner_selected_time_absolute_coupling_localization_wp10c9d6c7c3b5c4f3.py"
THIS_TEST = "tests/test_causal_inner_selected_time_absolute_coupling_localization_wp10c9d6c7c3b5c4f3.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_SELECTED_TIME_ABSOLUTE_COUPLING_LOCALIZATION_WP10C9D6C7C3B5C4F3_2026-08-13.md"
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
CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "face_fluxes.npz"
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"
FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
LAYOUTS = c4f1.LAYOUT_LABELS
TIMES = np.asarray(c4f2.TIMES_MICROSECONDS, dtype=float) * 1.0e-6
FACES = np.asarray(c4f2.PARENT_FACE_INDICES, dtype=int)


def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value); return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)): return int(value)
    return value


def _read(path): return json.loads(path.read_text(encoding="utf-8"))


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays); os.replace(temporary, path)


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args): return subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _source_identity():
    return {path: _sha(ROOT / path) for path in (THIS_RUNNER, THIS_TEST, c4f2.THIS_RUNNER, c4f2.THIS_TEST) if (ROOT / path).exists()}


def _validate():
    parent = _read(c4f2.SUMMARY_PATH); manifest = _read(c4f2.MANIFEST_PATH)
    if not parent["passed"] or parent["authorized_next"] != "WP10c9d6c7c3b5c4f3_selected_time_absolute_coupling_localization" or not manifest["definitions_only"]:
        raise RuntimeError("c4f3 authorization changed")
    return manifest


def _trajectories():
    coarse_times, coarse_states, _outputs, _perturbed = c4f1._coarse_absolute()
    middle = c4f1._middle_trajectory(); fine = c4f1._fine_trajectory()
    return {
        "coarse": (coarse_times, coarse_states),
        "middle": (middle["times"], middle["states"]),
        "fine": (fine["times"], fine["states"]),
    }


def _selected_states():
    return {
        label: states[c4f1._indices(times, TIMES)]
        for label, (times, states) in _trajectories().items()
    }


def _face_fluxes(context, state, multiplier):
    ledger = causal_five_field_radial_candidate_ledger(context, state)
    face_indices = FACES * int(multiplier)
    fluxes = np.asarray(ledger.interfaces.candidate_shared_face_fluxes_over_c)[face_indices][:, FIELDS]
    audit = np.asarray((ledger.interfaces.shared_conservative_face_defect, ledger.local_block_ledger_defect, ledger.source_double_count_defect, ledger.interfaces.incoming_excision_characteristics), dtype=float)
    return fluxes, audit


def _evaluate():
    parent_grid, configurations = c4f1._configurations()
    del parent_grid
    selected = _selected_states()
    source = _source_identity()
    shape = (len(LAYOUTS), TIMES.size, FACES.size, FIELDS.size)
    if CHECKPOINT_PATH.exists() and PROGRESS_PATH.exists():
        progress = _read(PROGRESS_PATH); arrays = _load(CHECKPOINT_PATH)
        if progress.get("source_identity") != source: raise RuntimeError("c4f3 checkpoint source changed")
    else:
        arrays = {"actual_fluxes": np.full(shape, np.nan), "reference_fluxes": np.full(shape, np.nan), "actual_audits": np.full((len(LAYOUTS), TIMES.size, 4), np.nan), "reference_audits": np.full((len(LAYOUTS), TIMES.size, 4), np.nan), "evaluation_wall_seconds": np.full((len(LAYOUTS), TIMES.size, 2), np.nan)}
        progress = {"source_identity": source, "completed": []}
    completed = set(progress["completed"])
    fine_layout, _fine_configuration = configurations["fine"]
    fine_parent = c4f1.c4f.c4e12.c4e9.c4e4._restrict(
        selected["fine"], fine_layout
    )
    for layout_index, label in enumerate(LAYOUTS):
        layout, configuration = configurations[label]
        multiplier = int(layout.refinement_ratio)
        for time_index, time_value in enumerate(TIMES):
            for kind in ("actual", "reference"):
                key = f"{label}:{time_index}:{kind}"
                if key in completed: continue
                state = selected[label][time_index] if kind == "actual" else fine_parent[time_index][layout.parent_cell_indices]
                began = time.perf_counter()
                fluxes, audit = _face_fluxes(configuration["context"], state, multiplier)
                arrays[f"{kind}_fluxes"][layout_index, time_index] = fluxes
                arrays[f"{kind}_audits"][layout_index, time_index] = audit
                arrays["evaluation_wall_seconds"][layout_index, time_index, 0 if kind == "actual" else 1] = time.perf_counter() - began
                completed.add(key); progress["completed"] = sorted(completed)
                _save(CHECKPOINT_PATH, **arrays); _write(PROGRESS_PATH, progress)
                print(f"c4f3: {key} wall={arrays['evaluation_wall_seconds'][layout_index,time_index,0 if kind == 'actual' else 1]:.1f}s", flush=True)
    arrays["selected_times_seconds"] = TIMES
    arrays["parent_face_indices"] = FACES
    arrays["fine_parent_states"] = fine_parent
    return arrays, configurations


def _metric(values, scales, gates):
    normalized = values / scales[None, None, :]
    cm = normalized[1] - normalized[0]; mf = normalized[2] - normalized[1]
    cm_norm = float(np.linalg.norm(cm)); mf_norm = float(np.linalg.norm(mf))
    order = float(np.log2(max(cm_norm, np.finfo(float).tiny) / max(mf_norm, np.finfo(float).tiny)))
    cosine = float(np.vdot(cm.ravel(), mf.ravel()).real / max(cm_norm * mf_norm, np.finfo(float).tiny))
    return {"order": order, "cosine": cosine, "coarse_middle_norm": cm_norm, "middle_fine_norm": mf_norm, "passed": bool(order >= gates["minimum_spatial_order"] and cosine >= gates["minimum_error_direction_cosine"])}


def _analyze(arrays, manifest):
    gates = manifest["prospective_gates"]
    scales = _load(c4f1.c4f.MIDDLE_ARRAYS)["tangent__export_scales"][:3]
    actual = arrays["actual_fluxes"]; reference = arrays["reference_fluxes"]
    face_metrics = {str(face): _metric(actual[:, :, index], scales, gates) for index, face in enumerate(FACES)}
    reference_metrics = {str(face): _metric(reference[:, :, index], scales, gates) for index, face in enumerate(FACES)}
    transition_index = int(np.flatnonzero(FACES == c4f2.TRANSITION_FACE)[0])
    interior_passes = sum(face_metrics[str(face)]["passed"] for face in c4f2.INTERIOR_CONTROL_FACES)
    transition = face_metrics[str(c4f2.TRANSITION_FACE)]
    actual_cm = actual[1] - actual[0]; actual_mf = actual[2] - actual[1]
    reference_cm = reference[1] - reference[0]; reference_mf = reference[2] - reference[1]
    native = actual - reference
    native_cm = native[1] - native[0]; native_mf = native[2] - native[1]
    closure_cm = actual_cm - reference_cm - native_cm; closure_mf = actual_mf - reference_mf - native_mf
    scale = max(float(np.linalg.norm(actual_cm)), float(np.linalg.norm(actual_mf)), np.finfo(float).tiny)
    closure = float(max(np.linalg.norm(closure_cm), np.linalg.norm(closure_mf)) / scale)
    actual_mf_transition = actual_mf[:, transition_index] / scales[None, :]
    reference_mf_transition = reference_mf[:, transition_index] / scales[None, :]
    fine_complement_transition = native[2, :, transition_index] / scales[None, :]
    denominator = max(float(np.linalg.norm(actual_mf_transition)), np.finfo(float).tiny)
    operator_fraction = float(np.linalg.norm(reference_mf_transition) / denominator)
    complement_fraction = float(np.linalg.norm(fine_complement_transition) / denominator)
    max_audit = float(max(np.nanmax(np.abs(arrays["actual_audits"][..., :3])), np.nanmax(np.abs(arrays["reference_audits"][..., :3]))))
    incoming = int(max(np.nanmax(arrays["actual_audits"][..., 3]), np.nanmax(arrays["reference_audits"][..., 3])))
    transition_only = bool(not transition["passed"] and interior_passes >= gates["minimum_interior_faces_with_consistent_direction"])
    if closure > gates["maximum_decomposition_closure_defect"] or max_audit > gates["maximum_flux_ledger_defect"] or incoming != 0:
        classification = "coupling_localization_method_gate_failed"
        authorized_next = "localization_repair_only"; passed = False
    elif complement_fraction > gates["maximum_fine_complement_fraction_of_middle_fine_difference"]:
        classification = "fine_complement_dominates_absolute_coupling_difference"
        authorized_next = "definitions_only_fine_complement_exact_JVP_manifest"; passed = True
    elif transition_only and operator_fraction >= gates["minimum_shared_parent_operator_fraction_for_operator_classification"]:
        classification = "transition_operator_absolute_baseline_defect_localized"
        authorized_next = "definitions_only_static_transition_operator_followup_manifest"; passed = True
    elif transition_only and operator_fraction <= gates["maximum_shared_parent_operator_fraction_of_middle_fine_difference_for_state_classification"]:
        classification = "base_state_alignment_absolute_coupling_defect_localized"
        authorized_next = "definitions_only_fine_anchored_baseline_representation_manifest"; passed = True
    elif not transition_only:
        classification = "distributed_absolute_baseline_direction_failure"
        authorized_next = "absolute_baseline_localization_only"; passed = False
    else:
        classification = "mixed_absolute_coupling_decomposition_inconclusive"
        authorized_next = "targeted_static_coupling_localization_manifest"; passed = False
    return {
        "schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": classification, "passed": passed, "physical_failure_detected": False,
        "face_metrics": face_metrics, "shared_parent_reference_face_metrics": reference_metrics,
        "transition_only_failure": transition_only, "passing_interior_control_faces": int(interior_passes),
        "decomposition_closure_defect": closure, "transition_shared_parent_operator_fraction_of_middle_fine_difference": operator_fraction,
        "transition_fine_complement_fraction_of_middle_fine_difference": complement_fraction,
        "maximum_ledger_audit_defect": max_audit, "maximum_incoming_excision_characteristics": incoming,
        "response_certificate_preserved": True, "absolute_closure_fit_authorized": False, "observable_memory_propagation_authorized": False,
        "fixed_Q_micro_solver_authorized": False, "reduced_slow_evolution_authorized": False, "fifty_ms_propagation_authorized": False,
        "authorized_next": authorized_next,
    }


def _catalog(summary):
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": _sha(path), "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "latest_source_parent_commit": ANALYZED_CERTIFICATE_COMMIT, "latest_work_package": WORK_PACKAGE}); _write(CANONICAL_SUMMARY, catalog)


def _finalize(summary, arrays, manifest):
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _save(DECISIVE_ARRAYS, **arrays)
    _write(CONFIG_PATH, {"schema_version": SCHEMA_VERSION, "times_seconds": TIMES, "parent_faces": FACES, "fields": FIELD_NAMES})
    _write(CONTRACT_PATH, manifest); _write(SUMMARY_PATH, summary)
    lines = ["# Selected-time absolute coupling localization", "", f"Classification: `{summary['classification']}`.", "", "No trajectory or fixed-Q propagation ran.", "", "| Parent face | Order | Cosine | Pass |", "|---:|---:|---:|---:|"]
    for face in FACES:
        item = summary["face_metrics"][str(face)]; lines.append(f"| {face} | {item['order']:.6f} | {item['cosine']:.6f} | {item['passed']} |")
    lines.extend(["", "## Decomposition", "", f"Shared-parent operator fraction of the transition middle-fine difference: `{summary['transition_shared_parent_operator_fraction_of_middle_fine_difference']:.6e}`.", "", f"Fine-complement fraction: `{summary['transition_fine_complement_fraction_of_middle_fine_difference']:.6e}`.", "", f"Exact decomposition closure defect: `{summary['decomposition_closure_defect']:.6e}`.", "", "The shared-parent construction is a diagnostic repeated-cell-average state, not a physical lift. The failed absolute gate is not relaxed, and the certified differential response remains preserved.", "", f"Authorized next: `{summary['authorized_next']}`.", ""])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    provenance = {"schema_version": SCHEMA_VERSION, "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT, "execution_head": _git("rev-parse", "HEAD"), "source_identity": _source_identity(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "input_hashes": {str(path.relative_to(ROOT)): _sha(path) for path in (c4f1.c4f.COARSE_EARLY_ARRAYS, c4f1.c4f.COARSE_ARRAYS, c4f1.c4f.MIDDLE_PILOT_ARRAYS, c4f1.c4f.MIDDLE_ARRAYS, c4f1.c4f.FINE_ARRAYS)}, "output_hashes": {}}
    provenance["output_hashes"] = {str(path.relative_to(ROOT)): _sha(path) for path in (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, DECISIVE_ARRAYS, REPORT_PATH)}; _write(PROVENANCE_PATH, provenance)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{_sha(path)}  {path.name}\n" for path in (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, PROVENANCE_PATH, DECISIVE_ARRAYS)), encoding="utf-8")
    _catalog(summary)


def main():
    manifest = _validate(); arrays, _configurations = _evaluate(); summary = _analyze(arrays, manifest); _finalize(summary, arrays, manifest); print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__": main()
