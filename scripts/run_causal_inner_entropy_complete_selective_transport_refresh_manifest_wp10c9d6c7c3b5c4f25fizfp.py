#!/usr/bin/env python3
"""Freeze a saved-evidence-selected partial refresh after transport failure."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_transported_third_macro_patch_execution_wp10c9d6c7c3b5c4f25fizfo as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    unpack_macro_outputs,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas_transport import (  # noqa: E402
    transport_thermodynamic_affine_macro_atlas,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    ThermodynamicAffineMacroAtlas,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizfp_"
    "entropy_complete_selective_transport_refresh_manifest"
)
CLASSIFICATION = (
    "entropy_complete_lnSigma_betaPhi_selective_refresh_manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizfq_"
    "entropy_complete_selectively_refreshed_third_patch_execution"
)
ARTIFACT = (
    "causal_inner_entropy_complete_selective_transport_refresh_manifest_"
    "wp10c9d6c7c3b5c4f25fizfp"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_SELECTIVE_"
    "TRANSPORT_REFRESH_MANIFEST_WP10C9D6C7C3B5C4F25FIZFP_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_selective_transport_refresh_"
    "manifest_wp10c9d6c7c3b5c4f25fizfp.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_selective_transport_refresh_"
    "manifest_wp10c9d6c7c3b5c4f25fizfp.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b4a21f11447208004346eb298f7a0dc5637eb73c68f2a0dcffb1a1e3dc8b793d"
)
PATCH_1_ARRAYS = parent.parent.PATCH_1_ARRAYS
PATCH_2_ARRAYS = parent.PATCH_2_ARRAYS
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FIELD_NAMES = ("lnSigma", "beta_phi", "lnT", "beta_r", "chi")
DIAGNOSTIC_SEED = 20_260_829


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("transport rejection checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = utils._read_json(
        parent.CANONICAL_DIRECTORY / "transported_patch_metrics.json"
    )
    if (
        summary["classification"] != parent.FAIL_CLASSIFICATION
        or summary["passed"]
        or summary["transported_third_patch_certified"]
        or summary["accepted_absolute_horizon_seconds"] != 8.0e-3
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] is not None
        or metrics["new_truth_operator_calls"] != 9
        or not metrics["all_truth_physical_gates_passed"]
        or metrics["maximum_independent_transport_JVP_relative_defect"] <= 5.0e-2
        or metrics["endpoint_maximum_macro_rate_relative_defect"] <= 5.0e-2
    ):
        raise RuntimeError("unchanged transport rejection classification changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"transport rejection source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("selective refresh manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _patch_1_atlas(values: dict[str, np.ndarray]) -> ThermodynamicAffineMacroAtlas:
    return ThermodynamicAffineMacroAtlas(
        anchor_macro_state=values["atlas_anchor_macro_state"],
        macro_coordinate_scales=values["atlas_macro_coordinate_scales"],
        base_normalized_output=values["atlas_base_normalized_output"],
        normalized_output_jacobian=values[
            "atlas_normalized_output_chart_jacobian"
        ],
        output_component_scales=values["atlas_output_component_scales"],
        trust_coordinate_infinity=1.5e-1,
        macro_coordinate_pullback=values["atlas_macro_coordinate_pullbacks"],
    )


def _evidence_diagnostics() -> dict:
    with np.load(PATCH_1_ARRAYS) as archive:
        patch_1_values = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(PATCH_2_ARRAYS) as archive:
        patch_2_values = {name: np.asarray(archive[name]) for name in archive.files}
    patch_1 = _patch_1_atlas(patch_1_values)
    patch_2_base = unpack_macro_outputs(
        patch_2_values["patch_2_base_normalized_output"]
        * patch_2_values["patch_2_output_component_scales"]
    )
    transported = transport_thermodynamic_affine_macro_atlas(
        patch_1,
        new_anchor_macro_state=patch_2_values["patch_2_anchor_macro_state"],
        new_macro_coordinate_scales=patch_2_values[
            "patch_2_macro_coordinate_scales"
        ],
        new_macro_chart_tangents=patch_2_values["patch_2_macro_chart_tangents"],
        new_macro_coordinate_pullbacks=patch_2_values[
            "patch_2_macro_coordinate_pullbacks"
        ],
        new_base_outputs=patch_2_base,
        trust_coordinate_infinity=1.5e-1,
    ).atlas
    transported_jacobian = np.asarray(transported.normalized_output_jacobian)
    rebuilt_jacobian = np.asarray(
        patch_2_values["patch_2_normalized_output_chart_jacobian"]
    )
    difference = rebuilt_jacobian - transported_jacobian
    per_field = {}
    for field, name in enumerate(FIELD_NAMES):
        columns = np.arange(field, 80, 5)
        per_field[name] = {
            "relative_frobenius_defect": float(
                np.linalg.norm(difference[:, columns])
                / max(
                    np.linalg.norm(rebuilt_jacobian[:, columns]),
                    np.finfo(float).tiny,
                )
            ),
            "absolute_error_fraction_of_full_tangent": float(
                np.linalg.norm(difference[:, columns])
                / max(np.linalg.norm(rebuilt_jacobian), np.finfo(float).tiny)
            ),
        }
    generator = np.random.default_rng(DIAGNOSTIC_SEED)
    directions = generator.normal(size=(128, 80))
    directions /= np.max(np.abs(directions), axis=1)[:, None]
    truth_products = directions @ rebuilt_jacobian.T

    def probe(jacobian: np.ndarray) -> np.ndarray:
        predicted = directions @ jacobian.T
        return np.linalg.norm(predicted - truth_products, axis=1) / np.maximum(
            np.linalg.norm(truth_products, axis=1), np.finfo(float).tiny
        )

    subset_records = []
    selected = None
    for cardinality in range(6):
        for subset in itertools.combinations(range(5), cardinality):
            hybrid = np.array(transported_jacobian, copy=True)
            for field in subset:
                columns = np.arange(field, 80, 5)
                hybrid[:, columns] = rebuilt_jacobian[:, columns]
            defects = probe(hybrid)
            record = {
                "field_indices": list(subset),
                "field_names": [FIELD_NAMES[field] for field in subset],
                "maximum_probe_relative_defect": float(np.max(defects)),
                "p95_probe_relative_defect": float(np.quantile(defects, 0.95)),
                "mean_probe_relative_defect": float(np.mean(defects)),
            }
            subset_records.append(record)
            if selected is None and record["maximum_probe_relative_defect"] <= 5.0e-2:
                selected = record
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("no saved-evidence selective refresh meets the probe gate")
    return {
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "probe_directions": 128,
        "transported_full_tangent_relative_frobenius_defect": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(rebuilt_jacobian), np.finfo(float).tiny)
        ),
        "per_input_field": per_field,
        "minimal_selected_field_indices": selected["field_indices"],
        "minimal_selected_field_names": selected["field_names"],
        "selected_maximum_probe_relative_defect": selected[
            "maximum_probe_relative_defect"
        ],
        "selected_p95_probe_relative_defect": selected[
            "p95_probe_relative_defect"
        ],
        "evaluated_subset_records": subset_records,
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "preserved_rejection": {
            "unchanged_transport_rejected": True,
            "failed_12ms_candidate_not_accepted_or_propagated": True,
            "physical_and_stability_diagnostics_remain_passing": True,
            "no_gate_relaxed": True,
        },
        "saved_evidence_selection": {
            "comparison": "full_patch_1_tangent_transported_to_full_patch_2_tangent",
            "probe_directions": 128,
            "probe_seed": DIAGNOSTIC_SEED,
            "maximum_selected_probe_relative_defect": 5.0e-2,
            "minimal_cardinality_selection_required": True,
            "selected_input_field_indices": (0, 1),
            "selected_input_fields": ("lnSigma", "beta_phi"),
            "unselected_input_fields": ("lnT", "beta_r", "chi"),
        },
        "selective_colored_refresh": {
            "source_tangent": "patch_2_chain_rule_transported_to_8ms_anchor",
            "cell_colors_per_selected_field": 3,
            "central_colored_chart_step": 2.0e-2,
            "selected_fields": 2,
            "new_colored_truth_calls": 12,
            "maximum_colored_support_leakage_ratio": 1.0e-12,
            "replace_only_selected_field_columns": True,
            "all_truth_physical_gates_binding": True,
        },
        "blind_validation": {
            "independent_JVP_directions": 4,
            "central_JVP_chart_step": 1.0e-2,
            "new_JVP_truth_calls": 8,
            "maximum_independent_JVP_relative_defect": 5.0e-2,
            "validation_calls_must_not_refit_the_Jacobian": True,
        },
        "dynamic_validation": {
            "overlap_witness": "certified_patch_2_7ms_state",
            "maximum_interpatch_output_relative_defect_per_block": 1.0e-1,
            "maximum_interpatch_macro_rate_relative_defect_per_field": 1.0e-1,
            "fixed_macrostep_seconds": 1.0e-3,
            "macrosteps": 4,
            "absolute_elapsed_endpoint_seconds": 1.2e-2,
            "atlas_trust_coordinate_infinity": 1.5e-1,
            "reserved_trust_coordinate_infinity": 1.2e-1,
            "one_new_dynamic_endpoint_truth_call": True,
            "maximum_endpoint_truth_output_relative_defect_per_block": 5.0e-2,
            "maximum_endpoint_truth_macro_rate_relative_defect_per_field": 5.0e-2,
            "maximum_endpoint_macro_roundtrip_relative_defect": 1.0e-10,
            "maximum_local_spectral_abscissa_per_second": 0.0,
            "exact_integrated_ledger_relative_defect_max": 1.0e-12,
        },
        "acquisition_cost": {
            "full_patch_truth_calls": 39,
            "selectively_refreshed_patch_truth_calls": 21,
            "maximum_selective_to_full_truth_call_fraction": 5.5e-1,
            "minimum_full_to_selective_speedup": 1.8,
            "new_global_roots": 0,
            "online_truth_calls_per_macrostep": 0,
        },
        "decision": {
            "pass": "authorize_definitions_only_adaptive_selective_refresh_cadence_manifest",
            "derivative_or_endpoint_failure": "require_full_patch_rebuilds",
            "physical_or_stability_failure": "stop_the_seven_field_cycle_path",
            "no_retrospective_gate_change": True,
        },
        "claim_boundary": {
            "one_selective_refresh_execution_authorized": True,
            "unbounded_atlas_growth_authorized": False,
            "complete_cycle_execution_authorized": False,
        },
        "authorized_next": AUTHORIZED_NEXT,
    }


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": "DEFINITIONS_ONLY"})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("selective refresh manifest already exists")
    validated = _validate_parent(require_clean=True)
    diagnostics = _evidence_diagnostics()
    selected = diagnostics["minimal_selected_field_indices"]
    if (
        selected != [0, 1]
        or diagnostics["selected_maximum_probe_relative_defect"] > 5.0e-2
    ):
        raise RuntimeError("saved evidence does not support the frozen field selection")
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "selective_refresh_contract.json", _contract())
    utils._write_json(CANONICAL_DIRECTORY / "saved_selection_diagnostics.json", diagnostics)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": CLASSIFICATION, "passed": True, "definitions_only": True, "unchanged_transport_rejection_preserved": True, "selected_input_fields": ["lnSigma", "beta_phi"], "selectively_refreshed_third_patch_execution_authorized": True, "complete_cycle_execution_authorized": False, "authorized_next": AUTHORIZED_NEXT}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "patch_1_arrays_sha256": utils._sha256(PATCH_1_ARRAYS), "patch_2_arrays_sha256": utils._sha256(PATCH_2_ARRAYS)})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Entropy-complete selective transport-refresh manifest", "", f"Classification: `{CLASSIFICATION}`.", "", "The unchanged transported tangent rejection is preserved. Saved full-patch evidence selects only `lnSigma` and `beta_phi` for a fresh colored refresh; the other three input-field columns remain exact chain-rule transports.", "", f"The selected two-field hybrid has maximum 128-direction saved-Jacobian probe defect `{diagnostics['selected_maximum_probe_relative_defect']:.6e}`. The prospective package uses 21 truth calls versus 39 for a full rebuild.", "", "No unbounded atlas growth or complete-cycle execution is authorized.", "", f"Authorized next: `{AUTHORIZED_NEXT}` only.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "scientific_status": "DEFINITIONS_ONLY", "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        parser.error("choose --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
