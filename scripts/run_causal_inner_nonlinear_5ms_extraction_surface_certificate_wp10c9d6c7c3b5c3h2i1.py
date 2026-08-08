#!/usr/bin/env python3
"""Certify the conservative 5 ms extraction-surface domain partition."""

from __future__ import annotations

import argparse
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

import run_causal_inner_nonlinear_5ms_extraction_surface_manifest_wp10c9d6c7c3b5c3h2i as h2i  # noqa: E402
import run_causal_inner_nonlinear_5ms_inner_face_half_cell_audit_wp10c9d6c7c3b5c3h2h1 as h2h1  # noqa: E402
import run_causal_inner_nonlinear_5ms_spatial_certificate_wp10c9d6c7c3b5c3h2f as h2f  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2i1"
ANALYZED_BASE_COMMIT = "37a4fec66c6da5182202d467261cbcfa64093c11"
ANALYZED_BASE_PARENT = "aafe96d55a1137810066c3333cb868efdef79f42"
ANALYZED_BASE_TREE = "312551195a080d51e4b2dcbc325b74d35009f350"

ARTIFACT = (
    "causal_inner_nonlinear_5ms_extraction_surface_certificate_"
    "wp10c9d6c7c3b5c3h2i1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_extraction_surface_certificate_"
    "wp10c9d6c7c3b5c3h2i1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_extraction_surface_certificate_"
    "wp10c9d6c7c3b5c3h2i1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_EXTRACTION_"
    "SURFACE_CERTIFICATE_WP10C9D6C7C3B5C3H2I1_2026-08-08.md"
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
OBSERVABLE_NAMES = tuple(h2i._manifest()["observable_names"])
SPATIAL_GATES = dict(h2i.SPATIAL_GATES)
TEMPORAL_GATES = dict(h2i.TEMPORAL_GATES)
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
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _source_identity() -> dict[str, str]:
    return {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }


def _validate_parent() -> tuple[dict, dict]:
    summary = _read_json(h2i.SUMMARY_PATH)
    manifest = _read_json(h2i.MANIFEST_PATH)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["extraction_partition_certificate_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c3h2i1_conservative_extraction_surface_certificate"
        or summary["fourth_duration_rung_manifest_authorized"]
        or summary["fixed_q_micro_solver_authorized"]
        or summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("h2i1 authorization changed")
    if (
        manifest["extraction_surface"]["coarse_face_index"]
        != h2i.EXTRACTION_COARSE_FACE_INDEX
        or tuple(manifest["extraction_surface"]["layout_face_indices"].values())
        != h2i.EXTRACTION_LAYOUT_FACE_INDICES
        or tuple(manifest["observable_names"]) != OBSERVABLE_NAMES
        or dict(manifest["spatial_gates"]) != SPATIAL_GATES
        or dict(manifest["temporal_gates"]) != TEMPORAL_GATES
    ):
        raise RuntimeError("h2i1 frozen contract changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2i1 analyzed identity changed")
    return summary, manifest


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    identity = _source_identity()
    if not CHECKPOINT_JSON.exists():
        return (
            {
                "work_package": WORK_PACKAGE,
                "analyzed_base_commit": ANALYZED_BASE_COMMIT,
                "source_identity": identity,
                "completed": [],
            },
            {},
        )
    progress = _read_json(CHECKPOINT_JSON)
    if (
        progress.get("work_package") != WORK_PACKAGE
        or progress.get("analyzed_base_commit") != ANALYZED_BASE_COMMIT
        or progress.get("source_identity") != identity
    ):
        raise RuntimeError("saved h2i1 progress belongs to different code")
    arrays = _load_npz(CHECKPOINT_ARRAYS) if CHECKPOINT_ARRAYS.exists() else {}
    return progress, arrays


def _exterior_observable(context, state, extraction_face: int, coupling_face: int):
    ledger = causal_five_field_radial_candidate_ledger(context, state)
    fluxes = np.asarray(
        ledger.interfaces.candidate_shared_face_fluxes_over_c, dtype=float
    )
    residual = np.asarray(ledger.residual_rows, dtype=float)
    cooling = np.asarray(ledger.cooling_rows, dtype=float)
    height = np.asarray(ledger.lower_height_work_rows, dtype=float)
    transport = np.asarray(ledger.conservative_transport_rows, dtype=float)
    if not 0 < extraction_face < coupling_face <= residual.shape[0]:
        raise RuntimeError("extraction partition indices are invalid")
    region = slice(extraction_face, coupling_face)
    value = np.concatenate(
        (
            fluxes[extraction_face, CONSERVATIVE_FIELDS],
            fluxes[coupling_face, CONSERVATIVE_FIELDS],
            -np.sum(residual[region][:, CONSERVATIVE_FIELDS], axis=0),
            -np.sum(cooling[region][:, CONSERVATIVE_FIELDS[1:]], axis=0),
            -np.sum(height[region][:, CONSERVATIVE_FIELDS[1:]], axis=0),
        )
    )
    face_difference = (
        fluxes[extraction_face, CONSERVATIVE_FIELDS]
        - fluxes[coupling_face, CONSERVATIVE_FIELDS]
    )
    transport_sum = np.sum(transport[region][:, CONSERVATIVE_FIELDS], axis=0)
    transport_scale = max(
        float(np.linalg.norm(face_difference)),
        float(np.linalg.norm(transport_sum)),
        np.finfo(float).tiny,
    )
    transport_defect = float(
        np.linalg.norm(face_difference - transport_sum) / transport_scale
    )
    source_remainder = residual - transport
    reconstructed_net = face_difference - np.sum(
        source_remainder[region][:, CONSERVATIVE_FIELDS], axis=0
    )
    net_scale = max(
        float(np.linalg.norm(reconstructed_net)),
        float(np.linalg.norm(value[6:9])),
        np.finfo(float).tiny,
    )
    direct_defect = float(
        np.linalg.norm(reconstructed_net - value[6:9]) / net_scale
    )
    audit = np.asarray(
        (
            ledger.interfaces.shared_conservative_face_defect,
            ledger.local_block_ledger_defect,
            ledger.source_double_count_defect,
            ledger.interfaces.maximum_split_closure_defect,
            ledger.interfaces.incoming_excision_characteristics,
            transport_defect,
            direct_defect,
        ),
        dtype=float,
    )
    return value, audit


def _evaluate_histories(times: np.ndarray, inputs: dict):
    progress, cached = _load_progress()
    completed = set(progress["completed"])
    started = time.monotonic()
    for layout_name, face in zip(
        LAYOUTS, h2i.EXTRACTION_LAYOUT_FACE_INDICES, strict=True
    ):
        payload = inputs[layout_name]
        ratio = int(payload["layout"].refinement_ratio)
        coupling_face = h2i.COUPLING_COARSE_FACE_INDEX * ratio
        context = payload["configuration"]["context"]
        physical_radius_rg = float(
            context.grid.edges[face] / context.grid.gravitational_radius
        )
        if physical_radius_rg != h2i.EXTRACTION_RADIUS_RG:
            raise RuntimeError(f"{layout_name} extraction radius changed")
        for branch in ("base", "anchor"):
            history = payload[branch]
            for common_index in range(times.size):
                state_index = int(payload["accepted_indices"][common_index])
                key = f"{layout_name}__{branch}__t{common_index}"
                if key in completed:
                    continue
                print(f"h2i1: evaluate {key}", flush=True)
                value, audit = _exterior_observable(
                    context,
                    history[state_index],
                    face,
                    coupling_face,
                )
                cached[f"{key}__observable"] = value
                cached[f"{key}__audit"] = audit
                completed.add(key)
                progress["completed"] = sorted(completed)
                progress["elapsed_wall_seconds"] = time.monotonic() - started
                _save_progress(progress, cached)
    histories = {}
    audits = {}
    for layout_name in LAYOUTS:
        for branch in ("base", "anchor"):
            histories[(layout_name, branch)] = np.asarray(
                [
                    cached[
                        f"{layout_name}__{branch}__t{index}__observable"
                    ]
                    for index in range(times.size)
                ]
            )
            audits[(layout_name, branch)] = np.asarray(
                [
                    cached[f"{layout_name}__{branch}__t{index}__audit"]
                    for index in range(times.size)
                ]
            )
    return histories, audits, time.monotonic() - started


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(times)[:, None],
        axis=0,
    )
    return result


def _metric(histories, scales: np.ndarray) -> dict:
    metrics = causal_packet_history_metrics(
        *histories,
        physical_scales=scales,
        relative_activity=SPATIAL_GATES["minimum_relative_activity"],
        minimum_rms_order=SPATIAL_GATES["minimum_rms_order"],
        minimum_maximum_order=SPATIAL_GATES["minimum_maximum_order"],
        minimum_significant_component_order=SPATIAL_GATES[
            "minimum_significant_component_order"
        ],
        maximum_fine_normalized_difference=SPATIAL_GATES[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=SPATIAL_GATES["minimum_history_cosine"],
        minimum_refinement_error_cosine=SPATIAL_GATES[
            "minimum_refinement_error_cosine"
        ],
    )
    indices = np.asarray(metrics.significant_components, dtype=int)
    component_orders = {
        OBSERVABLE_NAMES[index]: float(metrics.component_orders[position])
        for position, index in enumerate(indices)
    }
    return {
        "raw_spatial_contract_passed": bool(metrics.passed),
        "significant_components": tuple(OBSERVABLE_NAMES[index] for index in indices),
        "failed_component_orders": tuple(
            name
            for name, order in component_orders.items()
            if order < SPATIAL_GATES["minimum_significant_component_order"]
        ),
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "component_orders": component_orders,
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "coarse_medium_rms_difference": metrics.coarse_medium_rms_difference,
        "medium_fine_rms_difference": metrics.medium_fine_rms_difference,
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _temporal(metric: dict, envelope: float) -> dict:
    fine = float(metric["maximum_fine_normalized_difference"])
    observable = bool(
        fine > TEMPORAL_GATES["observability_factor"] * float(envelope)
    )
    ratio = float(envelope) / max(fine, np.finfo(float).tiny)
    upper = fine + float(envelope)
    if observable:
        passed = bool(
            metric["raw_spatial_contract_passed"]
            and ratio
            <= TEMPORAL_GATES["maximum_temporal_to_observable_spatial_ratio"]
        )
        route = "observable_spatial_order_contract"
    else:
        passed = bool(
            upper <= SPATIAL_GATES["maximum_fine_normalized_difference"]
        )
        route = "unobservable_upper_bound_without_order_or_direction_claim"
    return {
        "passed": passed,
        "route": route,
        "spatial_difference_observable": observable,
        "spatial_orders_and_error_direction_certifying": bool(
            observable
            and ratio
            <= TEMPORAL_GATES["maximum_temporal_to_observable_spatial_ratio"]
        ),
        "temporal_uncertainty_envelope": float(envelope),
        "temporal_uncertainty_to_medium_fine_difference_ratio": ratio,
        "observability_threshold": (
            TEMPORAL_GATES["observability_factor"] * float(envelope)
        ),
        "conservative_fine_difference_upper_bound": upper,
    }


def _analyze(histories: dict, audits: dict, times: np.ndarray):
    certificate = _read_json(h2f.SUMMARY_PATH)
    h2f_arrays = _load_npz(h2f.DECISIVE_ARRAYS)
    scales = np.asarray(h2f_arrays["export_scales"], dtype=float)
    responses = tuple(
        histories[(name, "anchor")] - histories[(name, "base")]
        for name in LAYOUTS
    )
    instantaneous = _metric(responses, scales)
    cumulative_histories = tuple(_cumulative(values, times) for values in responses)
    cumulative = _metric(cumulative_histories, scales * float(times[-1]))
    envelope = float(
        certificate["analysis"]["instantaneous_Tier_I"]
        ["temporal_classification"]["temporal_uncertainty_envelope"]
    )
    instant_temporal = _temporal(instantaneous, envelope)
    cumulative_temporal = _temporal(cumulative, envelope)
    state_passed = bool(
        certificate["analysis"]["state"]["binding_channel_passed"]
    )
    maximum_audit = np.max(
        np.concatenate(tuple(values for values in audits.values()), axis=0),
        axis=0,
    )
    audit_report = {
        "maximum_shared_conservative_face_defect": maximum_audit[0],
        "maximum_local_block_ledger_defect": maximum_audit[1],
        "maximum_source_double_count_defect": maximum_audit[2],
        "maximum_split_closure_defect": maximum_audit[3],
        "maximum_incoming_excision_characteristics": int(maximum_audit[4]),
        "maximum_exterior_prefix_direct_identity_defect": maximum_audit[6],
    }
    manifest = _read_json(h2i.MANIFEST_PATH)
    required = manifest["required_audits"]
    audit_passed = bool(
        audit_report["maximum_shared_conservative_face_defect"]
        <= required["shared_conservative_face_defect_maximum"]
        and audit_report["maximum_local_block_ledger_defect"]
        <= required["local_block_ledger_defect_maximum"]
        and audit_report["maximum_source_double_count_defect"]
        <= required["source_double_count_defect_maximum"]
        and audit_report["maximum_incoming_excision_characteristics"]
        == required["incoming_excision_characteristics"]
        and audit_report["maximum_exterior_prefix_direct_identity_defect"]
        <= required["exterior_prefix_direct_identity_defect_maximum"]
    )
    overall = bool(
        state_passed
        and instant_temporal["passed"]
        and cumulative_temporal["passed"]
        and audit_passed
    )
    analysis = {
        "extraction_radius_rg": h2i.EXTRACTION_RADIUS_RG,
        "layout_extraction_face_indices": dict(
            zip(LAYOUTS, h2i.EXTRACTION_LAYOUT_FACE_INDICES, strict=True)
        ),
        "state_channel_inherited_from_h2f_passed": state_passed,
        "instantaneous_exterior_partition": {
            **instantaneous,
            "temporal_classification": instant_temporal,
            "binding_channel_passed": bool(instant_temporal["passed"]),
        },
        "cumulative_exterior_partition": {
            **cumulative,
            "temporal_classification": cumulative_temporal,
            "binding_channel_passed": bool(cumulative_temporal["passed"]),
        },
        "ledger_audits": {**audit_report, "passed": audit_passed},
        "raw_inner_face_rejection_preserved": True,
        "pointwise_horizon_flux_convergence_claimed": False,
        "extraction_partition_spatial_certificate_passed": overall,
    }
    decisive = {
        "times_seconds": times,
        "export_scales": scales,
        "temporal_uncertainty_envelope": np.asarray(envelope),
    }
    for layout_name, response, cumulative_response in zip(
        LAYOUTS, responses, cumulative_histories, strict=True
    ):
        decisive[f"{layout_name}__base_exterior_observables"] = histories[
            (layout_name, "base")
        ]
        decisive[f"{layout_name}__anchor_exterior_observables"] = histories[
            (layout_name, "anchor")
        ]
        decisive[f"{layout_name}__exterior_response"] = response
        decisive[f"{layout_name}__cumulative_exterior_response"] = cumulative_response
        decisive[f"{layout_name}__base_audits"] = audits[(layout_name, "base")]
        decisive[f"{layout_name}__anchor_audits"] = audits[(layout_name, "anchor")]
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
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": status,
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


def _authorized_after_failure(analysis: dict) -> str:
    instant = analysis["instantaneous_exterior_partition"]
    cumulative = analysis["cumulative_exterior_partition"]
    if instant["binding_channel_passed"] and not cumulative["binding_channel_passed"]:
        return (
            "WP10c9d6c7c3b5c3h2j_cumulative_extraction_recovery_manifest"
        )
    if any(
        name.startswith("extraction_flux")
        for name in instant["failed_component_orders"]
    ):
        return "WP10c9d6c7c3b5c3h2j_near_horizon_partition_redesign_manifest"
    return "WP10c9d6c7c3b5c3h2j_exterior_partition_source_localization_manifest"


def _refresh_metadata(summary: dict) -> None:
    provenance = _read_json(PROVENANCE_PATH)
    provenance["implementation_source_hashes"] = _source_identity()
    provenance["metadata_refreshed_at_head"] = _git_value("rev-parse", "HEAD")
    provenance["command"] = f"PYTHONPATH=src:scripts python {THIS_RUNNER} --reuse-decisive"
    _write_json(PROVENANCE_PATH, provenance)
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)


def main(*, reuse_decisive: bool = False) -> int:
    _validate_parent()
    if reuse_decisive:
        if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
            raise RuntimeError("h2i1 decisive evidence is absent")
        summary = _read_json(SUMMARY_PATH)
        summary["analysis"]["ledger_audits"].pop(
            "maximum_transport_telescoping_defect", None
        )
        if not summary["passed"]:
            old_next = summary["authorized_next"]
            summary["authorized_next"] = _authorized_after_failure(
                summary["analysis"]
            )
            _write_json(SUMMARY_PATH, summary)
            if REPORT_PATH.exists():
                REPORT_PATH.write_text(
                    REPORT_PATH.read_text(encoding="utf-8").replace(
                        old_next, summary["authorized_next"]
                    ),
                    encoding="utf-8",
                )
        _refresh_metadata(summary)
        print(json.dumps(_plain(summary), indent=2, sort_keys=True))
        return 0 if summary["passed"] else 2

    _certificate, times, inputs = h2h1._standardized_inputs()
    histories, audits, elapsed = _evaluate_histories(times, inputs)
    analysis, decisive = _analyze(histories, audits, times)
    passed = bool(analysis["extraction_partition_spatial_certificate_passed"])
    if passed:
        classification = (
            "five_ms_extraction_partition_spatial_certificate_passed_"
            "fourth_duration_manifest_authorized"
        )
        authorized_next = "WP10c9d6c7c3b5c4a_fourth_duration_rung_manifest"
    else:
        classification = (
            "five_ms_extraction_partition_spatial_certificate_rejected_"
            "later_duration_blocked"
        )
        authorized_next = _authorized_after_failure(analysis)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "analysis": analysis,
        "middle_fine_5ms_extraction_partition_spatial_certificate_issued": True,
        "third_duration_rung_extraction_partition_spatial_convergence_certified": passed,
        "raw_inner_face_spatial_convergence_certified": False,
        "pointwise_horizon_flux_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": passed,
        "physical_failure_detected": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "layouts": LAYOUTS,
            "extraction_radius_rg": h2i.EXTRACTION_RADIUS_RG,
            "extraction_layout_face_indices": h2i.EXTRACTION_LAYOUT_FACE_INDICES,
            "observable_names": OBSERVABLE_NAMES,
            "spatial_gates": SPATIAL_GATES,
            "temporal_gates": TEMPORAL_GATES,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "elapsed_wall_seconds": elapsed,
            "input_hashes": {
                "manifest": _sha256(h2i.MANIFEST_PATH),
                "h2f_summary": _sha256(h2f.SUMMARY_PATH),
                "h2f_arrays": _sha256(h2f.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    instant = analysis["instantaneous_exterior_partition"]
    cumulative = analysis["cumulative_exterior_partition"]
    audit = analysis["ledger_audits"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms extraction-surface spatial certificate WP10c9d6c7c3b5c3h2i1",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"The fixed `R={h2i.EXTRACTION_RADIUS_RG:.14f} r_g` surface was evaluated on the committed coarse/middle/fine nonlinear 5 ms trajectories without propagating a new state or changing the operator.",
                "",
                "## Binding measurements",
                "",
                f"- Instantaneous exterior-partition RMS/max/min-component orders: `{instant['observed_rms_order']:.6f}` / `{instant['observed_maximum_order']:.6f}` / `{instant['minimum_significant_component_order']:.6f}`.",
                f"- Instantaneous refinement-error cosine: `{instant['refinement_error_cosine']:.6f}`; fine fixed-scale difference: `{instant['maximum_fine_normalized_difference']:.6e}`; temporal/spatial ratio: `{instant['temporal_classification']['temporal_uncertainty_to_medium_fine_difference_ratio']:.6f}`.",
                f"- Cumulative exterior-partition RMS/max/min-component orders: `{cumulative['observed_rms_order']:.6f}` / `{cumulative['observed_maximum_order']:.6f}` / `{cumulative['minimum_significant_component_order']:.6f}`.",
                f"- Maximum exterior prefix identity defect: `{audit['maximum_exterior_prefix_direct_identity_defect']:.3e}`; incoming excision characteristics: `{audit['maximum_incoming_excision_characteristics']}`.",
                "",
                "## Interpretation",
                "",
                "This certificate is for the conservative domain partition consumed by the slow exterior: extraction-surface M/J/E flux, coupling flux, exterior net drive, cooling, and vertical work. The localized excision-to-extraction buffer remains part of the inner microdomain with explicit storage and sources.",
                "",
                "The historical raw excision-face and pointwise horizon-flux rejection is preserved. The extraction flux is not relabeled as the instantaneous horizon flux. No physical failure was detected.",
                "",
                f"Only `{authorized_next}` is authorized. Fixed-Q experiments and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-decisive", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(main(reuse_decisive=arguments.reuse_decisive))
