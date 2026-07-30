#!/usr/bin/env python3
"""Freeze a prospective proof-style signed-band export contract.

This definitions-only package inherits five ordinary profiles that were
frozen in WP10c9d6c6e0 and certified spectrally eligible, but never
propagated, in WP10c9d6c6e1.  It changes no operator and propagates no state.
"""

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

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6f0"
ANALYZED_BASE_COMMIT = "595a200bd2218eb0dfdfc2478f2706f917bc561b"
ANALYZED_BASE_PARENT = "9f4f4b3a720a404619663206878ffc475228eb3f"
ANALYZED_BASE_TREE = "0555e83c0e9464ecf844f659351984d871d40710"
THIS_RUNNER = (
    "scripts/run_causal_inner_band_envelope_manifest_"
    "wp10c9d6c6f0.py"
)

SELECTED_BASE_PROFILES = (
    "p3__inward_shear",
    "p3__outward_shear",
    "p5__inward_shear",
    "p5__outward_shear",
    "p3__material",
)

E0_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_manifest_wp10c9d6c6e0"
)
E0_MANIFEST = E0_DIRECTORY / "conditioning_manifest.json"
E0_SUMMARY = E0_DIRECTORY / "summary.json"
E0_PROVENANCE = E0_DIRECTORY / "provenance.json"

E1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1"
)
E1_CONFIG = E1_DIRECTORY / "config.json"
E1_SUMMARY = E1_DIRECTORY / "summary.json"
E1_ARRAYS = E1_DIRECTORY / "decisive_arrays.npz"
E1_PROVENANCE = E1_DIRECTORY / "provenance.json"

D_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_height_localization_wp10c9d6c6d"
)
D_CONFIG = D_DIRECTORY / "config.json"
D_SUMMARY = D_DIRECTORY / "summary.json"
D_ARRAYS = D_DIRECTORY / "decisive_arrays.npz"
D_PROVENANCE = D_DIRECTORY / "provenance.json"

E2B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_bandlimited_balance_feasibility_wp10c9d6c6e2b"
)
E2B_CONFIG = E2B_DIRECTORY / "config.json"
E2B_SUMMARY = E2B_DIRECTORY / "summary.json"
E2B_SELECTED = E2B_DIRECTORY / "selected_profile_manifest.json"
E2B_PROVENANCE = E2B_DIRECTORY / "provenance.json"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_manifest_wp10c9d6c6f0"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "band_envelope_manifest.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_integral_conditioning.py",
    "tests/test_causal_inner_integral_conditioning.py",
    "tests/test_causal_inner_band_envelope_manifest_wp10c9d6c6f0.py",
)

PHYSICAL_BAND_CONTRACT = {
    "selection_grid": "uniform_N128",
    "selection_rule": "nearest_grid_edge_to_each_target",
    "target_edges_over_rg": (1.8, 3.0, 5.0, 8.0, 10.5),
    "always_include_outer_domain_edge": True,
    "restriction_rule": "exact_sum_of_nested_cell_integrals",
    "time_quadrature": "trapezoid_on_frozen_65_samples",
    "time_horizon_s": 0.125,
}

COMPONENT_ROUTE_CONTRACT = {
    "historical_direct_component_route_preserved": True,
    "standard_route_applies_to_every_significant_component": True,
    "alternate_route_scope": {
        "physical_block": "lower_height_work",
        "conservative_channel": "angular_momentum",
        "history_types": ("instantaneous", "cumulative"),
    },
    "alternate_route_forbidden_for_all_other_components": True,
    "alternate_route_requires_direct_order_failure_only": True,
    "alternate_route_requires_all_other_parent_gates": True,
    "minimum_relative_activity": 1.0e-8,
    "active_cell_rule": (
        "maximum_absolute_coarse_or_restricted_medium_or_restricted_fine_"
        "cell_history_over_fixed_physical_scale_at_least_1e-8"
    ),
    "active_band_rule": (
        "maximum_absolute_coarse_or_restricted_medium_or_restricted_fine_"
        "band_history_over_fixed_physical_scale_at_least_1e-8"
    ),
    "minimum_direct_or_band_rms_order": 0.75,
    "minimum_direct_or_band_maximum_order": 0.75,
    "minimum_active_cell_rms_order": 0.75,
    "minimum_active_band_refinement_error_cosine": 0.90,
    "maximum_global_fine_normalized_difference": 0.05,
    "maximum_absolute_band_error_envelope": 0.05,
    "maximum_cancellation_ratio_each_grid_pair": 0.25,
    "maximum_direct_sum_defect": 1.0e-12,
    "maximum_signed_gram_closure_defect": 1.0e-12,
    "maximum_continuum_uncertainty_to_fine_difference": 0.10,
    "fine_band_error_envelope_definition": (
        "sum_over_active_bands_of_maximum_over_time_absolute_"
        "restricted_N512_minus_restricted_N256_band_error_divided_by_"
        "fixed_physical_angular_momentum_scale"
    ),
    "cancellation_ratio_definition": (
        "time_weighted_norm_of_signed_sum_of_band_errors_divided_by_"
        "sum_of_time_weighted_band_error_norms"
    ),
    "triangle_inequality_is_binding_not_a_fitted_model": True,
    "no_fitted_coefficients": True,
    "minimum_profiles_required_to_use_alternate_route": 0,
    "route_usage_must_be_reported": True,
    "no_retroactive_application_to_wp10c9d6c6c": True,
}

PROPAGATION_CONTRACT = {
    "binding_grids": ("uniform_N128", "uniform_N256", "uniform_N512"),
    "time_horizon_s": 0.125,
    "time_sample_count": 65,
    "exact_profile_projections_inherited_from_wp10c9d6c6e1": True,
    "exact_boundary_semigroup_integral_required": True,
    "state_and_aggregate_export_gates_unchanged_from_wp10c9d6c6c": True,
    "minimum_rms_order": 0.75,
    "minimum_maximum_order": 0.75,
    "maximum_fine_normalized_difference": 0.05,
    "minimum_history_cosine": 0.90,
    "minimum_refinement_error_cosine": 0.90,
    "maximum_state_reference_uncertainty_to_fine_difference": 0.10,
    "all_profile_variants_binding": True,
    "sign_and_amplitude_scaling_must_replay": True,
    "no_definition_or_threshold_changes_after_this_manifest": True,
    "classification_if_all_variants_pass": (
        "uniform_operator_certified_for_declared_resolved_profile_class"
    ),
    "classification_if_any_variant_fails": (
        "prospective_band_envelope_validation_failed"
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
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


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c6f0 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {path: _sha256(ROOT / path) for path in IMPLEMENTATION_SOURCES}
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_canonical_catalog() -> None:
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CANONICAL_MANIFEST.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _parent_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _build_manifest() -> tuple[dict, dict]:
    e0_manifest = json.loads(E0_MANIFEST.read_text(encoding="utf-8"))
    e0_summary = json.loads(E0_SUMMARY.read_text(encoding="utf-8"))
    e1_summary = json.loads(E1_SUMMARY.read_text(encoding="utf-8"))
    d_summary = json.loads(D_SUMMARY.read_text(encoding="utf-8"))
    e2b_summary = json.loads(E2B_SUMMARY.read_text(encoding="utf-8"))

    if e0_manifest["manifest_sha256"] != e0_summary["manifest_sha256"]:
        raise RuntimeError("WP10c9d6c6e0 manifest identity changed")
    if d_summary["classification"] != (
        "convergent_bands_noncontracting_cancellation_remainder"
    ):
        raise RuntimeError("WP10c9d6c6d diagnosis changed")
    if e1_summary["classification"] != "frozen_integral_profiles_ineligible":
        raise RuntimeError("WP10c9d6c6e1 classification changed")
    if e2b_summary["classification"] != (
        "no_eligible_bandlimited_balance_profile"
    ):
        raise RuntimeError("WP10c9d6c6e2b classification changed")

    definitions = {
        name: e0_manifest["base_profile_definitions"][name]
        for name in SELECTED_BASE_PROFILES
    }
    eligibility_reports = {
        name: e1_summary["eligibility_report"]["profile_reports"][name]
        for name in SELECTED_BASE_PROFILES
    }
    if not all(report["passed"] for report in eligibility_reports.values()):
        raise RuntimeError("an inherited ordinary profile is not eligible")
    variants = [
        item
        for item in e0_manifest["profile_variants"]
        if item["base_profile"] in SELECTED_BASE_PROFILES
    ]
    if len(variants) != 20 or not all(item["binding"] for item in variants):
        raise RuntimeError("inherited profile variants changed")

    prefixes = tuple(f"{name}__" for name in SELECTED_BASE_PROFILES)
    projection_hashes = {
        name: value
        for name, value in e1_summary["decisive_array_hashes"].items()
        if name.startswith(prefixes)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "eligibility_reused_not_reoptimized": True,
        "inherited_e0_manifest_sha256": e0_summary["manifest_sha256"],
        "inherited_e1_decisive_arrays_sha256": (
            e1_summary["decisive_arrays_sha256"]
        ),
        "closed_e2b_selected_profile_manifest_sha256": (
            e2b_summary["selected_profile_manifest_sha256"]
        ),
        "base_profile_definitions": definitions,
        "profile_eligibility_reports": eligibility_reports,
        "profile_projection_hashes": projection_hashes,
        "profile_variants": variants,
        "fixed_physical_scale_array_sha256": (
            d_summary["decisive_array_hashes"][
                "fixed_physical_observable_scales"
            ]
        ),
        "physical_band_contract": PHYSICAL_BAND_CONTRACT,
        "component_route_contract": COMPONENT_ROUTE_CONTRACT,
        "prospective_propagation_contract": PROPAGATION_CONTRACT,
        "historical_classifications_preserved": (
            "prospective_uniform_packet_validation_failed",
            "convergent_bands_noncontracting_cancellation_remainder",
            "frozen_integral_profiles_ineligible",
            "no_eligible_bandlimited_balance_profile",
        ),
    }
    manifest = {
        **payload,
        "manifest_sha256": causal_canonical_json_sha256(payload),
    }
    inherited = {
        "e0": e0_summary,
        "e1": e1_summary,
        "d": d_summary,
        "e2b": e2b_summary,
    }
    return manifest, inherited


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "propagation_executed": False,
        "base_profiles": SELECTED_BASE_PROFILES,
        "base_profile_count": len(SELECTED_BASE_PROFILES),
        "variant_count": len(manifest["profile_variants"]),
        "physical_band_contract": PHYSICAL_BAND_CONTRACT,
        "component_route_contract": COMPONENT_ROUTE_CONTRACT,
        "prospective_propagation_contract": PROPAGATION_CONTRACT,
    }


def run() -> dict:
    identity = _validate_analyzed_git_identity()
    manifest, inherited = _build_manifest()
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, _config(manifest))
    _write_json(MANIFEST_PATH, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": (
            "band_envelope_contract_and_heldout_profiles_frozen_"
            "uniform_propagation_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c6f1_prospective_band_envelope_propagation"
        ),
        "passed": True,
        "operator_changed": False,
        "propagation_executed": False,
        "eligibility_reused_not_reoptimized": True,
        "base_profile_count": len(SELECTED_BASE_PROFILES),
        "profile_variant_count": len(manifest["profile_variants"]),
        "all_inherited_profiles_eligible": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "inherited_e0_manifest_sha256": (
            inherited["e0"]["manifest_sha256"]
        ),
        "inherited_e1_decisive_arrays_sha256": (
            inherited["e1"]["decisive_arrays_sha256"]
        ),
        "closed_e2b_selected_profile_manifest_sha256": (
            inherited["e2b"]["selected_profile_manifest_sha256"]
        ),
        "historical_classifications_preserved": True,
        "c6c_rejection_preserved": True,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "PROSPECTIVE MANIFEST ONLY",
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value(
            "rev-parse", "HEAD^{tree}"
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src python scripts/"
            "run_causal_inner_band_envelope_manifest_wp10c9d6c6f0.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": _parent_hashes(
            (
                E0_MANIFEST,
                E0_SUMMARY,
                E0_PROVENANCE,
                E1_CONFIG,
                E1_SUMMARY,
                E1_ARRAYS,
                E1_PROVENANCE,
                D_CONFIG,
                D_SUMMARY,
                D_ARRAYS,
                D_PROVENANCE,
                E2B_CONFIG,
                E2B_SUMMARY,
                E2B_SELECTED,
                E2B_PROVENANCE,
            )
        ),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "authorized_next": summary["authorized_next"],
                "base_profile_count": summary["base_profile_count"],
                "profile_variant_count": summary["profile_variant_count"],
                "manifest_sha256": summary["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
