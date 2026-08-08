#!/usr/bin/env python3
"""Freeze the conservative 5 ms extraction-surface certificate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_inner_face_half_cell_audit_wp10c9d6c7c3b5c3h2h1 as h2h1  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2i"
ANALYZED_BASE_COMMIT = "aafe96d55a1137810066c3333cb868efdef79f42"
ANALYZED_BASE_PARENT = "78cbe736c79aa44950c6c7266409bea2957a2aaa"
ANALYZED_BASE_TREE = "593fed09e0a993e20f435bdd480406cc5f757e99"

ARTIFACT = (
    "causal_inner_nonlinear_5ms_extraction_surface_manifest_"
    "wp10c9d6c7c3b5c3h2i"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_5ms_extraction_surface_manifest_"
    "wp10c9d6c7c3b5c3h2i.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_5ms_extraction_surface_manifest_"
    "wp10c9d6c7c3b5c3h2i.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_5MS_EXTRACTION_"
    "SURFACE_MANIFEST_WP10C9D6C7C3B5C3H2I_2026-08-08.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "certificate_manifest.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

LAYOUTS = h2h1.LAYOUTS
EXTRACTION_COARSE_FACE_INDEX = 1
EXTRACTION_LAYOUT_FACE_INDICES = (1, 2, 4)
EXTRACTION_RADIUS_RG = 1.8750165318355323
COUPLING_COARSE_FACE_INDEX = 48
SPATIAL_GATES = {
    "minimum_rms_order": 0.75,
    "minimum_maximum_order": 0.75,
    "minimum_significant_component_order": 0.75,
    "maximum_fine_normalized_difference": 0.05,
    "minimum_history_cosine": 0.90,
    "minimum_refinement_error_cosine": 0.90,
    "minimum_relative_activity": 1.0e-8,
}
TEMPORAL_GATES = {
    "maximum_temporal_to_observable_spatial_ratio": 0.10,
    "observability_factor": 5.0,
    "unobservable_route": (
        "report_upper_bound_only_without_order_or_direction_claim"
    ),
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _validate_parent() -> dict:
    parent = _read_json(h2h1.SUMMARY_PATH)
    recovery = parent["audit"]["common_face_recovery"]
    if (
        not parent["passed"]
        or parent["classification"]
        != "five_ms_inner_export_recovers_on_common_surface_"
        "extraction_surface_manifest_authorized"
        or parent["authorized_next"]
        != "WP10c9d6c7c3b5c3h2i_conservative_extraction_surface_manifest"
        or not recovery["compact_recovery_selected"]
        or recovery["recovery_face_index"] != EXTRACTION_COARSE_FACE_INDEX
        or parent["fourth_duration_rung_manifest_authorized"]
        or parent["fixed_q_micro_solver_authorized"]
        or parent["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("h2i authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h2i analyzed identity changed")
    return parent


def _manifest() -> dict:
    observable_names = (
        "extraction_flux_mass",
        "extraction_flux_angular_momentum",
        "extraction_flux_killing_energy",
        "coupling_flux_mass",
        "coupling_flux_angular_momentum",
        "coupling_flux_killing_energy",
        "exterior_net_drive_mass",
        "exterior_net_drive_angular_momentum",
        "exterior_net_drive_killing_energy",
        "exterior_cooling_angular_momentum",
        "exterior_cooling_killing_energy",
        "exterior_vertical_work_angular_momentum",
        "exterior_vertical_work_killing_energy",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "conservative_extraction_surface_manifest_frozen_"
            "five_ms_partition_certificate_authorized"
        ),
        "definitions_only": True,
        "propagation_executed": False,
        "operator_changed": False,
        "production_defaults_changed": False,
        "raw_inner_face_rejection_preserved": True,
        "extraction_surface": {
            "radius_rg": EXTRACTION_RADIUS_RG,
            "coarse_face_index": EXTRACTION_COARSE_FACE_INDEX,
            "layout_face_indices": dict(
                zip(LAYOUTS, EXTRACTION_LAYOUT_FACE_INDICES, strict=True)
            ),
            "coupling_coarse_face_index": COUPLING_COARSE_FACE_INDEX,
            "same_physical_surface_required": True,
        },
        "domain_partition": {
            "inner_buffer": "excision_to_extraction_surface",
            "slow_exterior": "extraction_surface_to_coupling_surface",
            "slow_exterior_consumes_extraction_surface_flux": True,
            "inner_buffer_storage_and_sources_remain_explicit": True,
            "extraction_flux_is_not_relabeled_as_instantaneous_horizon_flux": True,
            "pointwise_horizon_flux_convergence_not_claimed": True,
        },
        "observable_names": observable_names,
        "observable_definition": {
            "first_three": "shared_face_flux_at_fixed_extraction_surface",
            "next_three": "shared_face_flux_at_fixed_coupling_surface",
            "exterior_net_drive": (
                "minus_sum_of_complete_stationary_residual_rows_between_"
                "extraction_and_coupling_faces"
            ),
            "exterior_cooling_and_vertical_work": (
                "minus_sum_of_the_same_ledger_rows_on_the_exterior_partition"
            ),
        },
        "binding_channels": (
            "instantaneous_extraction_partition_response",
            "cumulative_extraction_partition_response",
            "state_response_inherited_from_h2f",
        ),
        "spatial_gates": SPATIAL_GATES,
        "temporal_gates": TEMPORAL_GATES,
        "required_audits": {
            "shared_conservative_face_defect_maximum": 1.0e-12,
            "local_block_ledger_defect_maximum": 1.0e-11,
            "source_double_count_defect_maximum": 1.0e-12,
            "incoming_excision_characteristics": 0,
            "exterior_prefix_direct_identity_defect_maximum": 1.0e-12,
            "use_all_committed_common_targets": True,
            "no_time_interpolation": True,
            "cumulative_rule": "trapezoid_on_exact_common_target_bits",
        },
        "decision_tree": (
            "all_state_instantaneous_and_cumulative_partition_gates_pass__issue_five_ms_extraction_partition_spatial_certificate",
            "extraction_flux_passes_but_exterior_net_drive_fails__localize_exterior_source_partition",
            "instantaneous_passes_but_cumulative_fails__audit_time_window_and_storage_buffer",
            "extraction_flux_fails__reject_partition_route_and_return_to_near_horizon_redesign",
        ),
        "hard_stops": (
            "do_not_propagate_new_state",
            "do_not_change_operator_or_production_defaults",
            "do_not_call_extraction_flux_the_pointwise_horizon_flux",
            "do_not_hide_inner_buffer_storage_or_sources",
            "do_not_relax_the_rejected_raw_inner_face_contract",
            "do_not_start_fourth_duration_fixed_Q_or_slow_reduction",
        ),
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2i1_conservative_extraction_surface_certificate"
        ),
    }


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
                    "scientific_status": "CERTIFIED",
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


def main() -> int:
    parent = _validate_parent()
    manifest = _manifest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": manifest["classification"],
        "passed": True,
        "definitions_only": True,
        "parent_classification_preserved": parent["classification"],
        "raw_inner_face_rejection_preserved": True,
        "extraction_partition_certificate_authorized": True,
        "new_propagation_authorized": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": manifest["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "extraction_surface": manifest["extraction_surface"],
            "observable_names": manifest["observable_names"],
            "spatial_gates": SPATIAL_GATES,
            "temporal_gates": TEMPORAL_GATES,
        },
    )
    _write_json(MANIFEST_PATH, manifest)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "scientific_status": "CERTIFIED",
            "working_head": _git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "parent_summary_sha256": _sha256(h2h1.SUMMARY_PATH),
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Nonlinear 5 ms conservative extraction-surface manifest WP10c9d6c7c3b5c3h2i",
                "",
                "## Classification",
                "",
                f"`{manifest['classification']}`",
                "",
                "This definitions-only package freezes the first common physical surface at `R=1.8750165318355323 r_g` (layout face indices `1/2/4`) as the boundary between the localized inner buffer and the slow exterior.",
                "",
                "The certificate will test all thirteen exterior-partition observables instantaneously and cumulatively on the committed 5 ms trajectories. The extraction flux is the conservative boundary flux consumed by the slow exterior; it is not renamed as the pointwise horizon flux. Inner-buffer storage and source terms remain explicit in the complete ledger.",
                "",
                "No state is propagated and no operator, production default, or historical raw-inner-face verdict is changed. Only the operator-neutral h2i1 certificate is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = (
        "certificate_manifest.json",
        "config.json",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
