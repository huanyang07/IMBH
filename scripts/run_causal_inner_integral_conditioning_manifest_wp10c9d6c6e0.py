#!/usr/bin/env python3
"""Freeze the prospective signed-integral conditioning contract.

This package performs no propagation and changes no operator.  It freezes
the alternate component gate and unseen analytic profile definitions before
their eligibility or time-history outcomes are evaluated.
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
WORK_PACKAGE = "WP10c9d6c6e0"
ANALYZED_BASE_COMMIT = "c3acf82390a6f4fca1efd891bc4823d3b5ee318b"
ANALYZED_BASE_PARENT = "80bdb60674d8a3afaf3e35a61edcae5934bc1a1f"
ANALYZED_BASE_TREE = "66d0353eca8f45875749168d54b9bda629f4ded7"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_integral_conditioning_manifest_wp10c9d6c6e0.py"
)

AMPLITUDE_FACTORS = (0.5, 1.0)
SIGNS = (-1, 1)
PROFILE_DEFINITIONS = {
    "p3__inward_shear": {
        "role": "unseen_standard_shear",
        "family": "inward_shear",
        "envelope": {"kind": "full_domain_sine_power", "power": 3},
    },
    "p3__outward_shear": {
        "role": "unseen_standard_shear",
        "family": "outward_shear",
        "envelope": {"kind": "full_domain_sine_power", "power": 3},
    },
    "p5__inward_shear": {
        "role": "unseen_standard_shear",
        "family": "inward_shear",
        "envelope": {"kind": "full_domain_sine_power", "power": 5},
    },
    "p5__outward_shear": {
        "role": "unseen_standard_shear",
        "family": "outward_shear",
        "envelope": {"kind": "full_domain_sine_power", "power": 5},
    },
    "balanced_p2_p4__inward_shear": {
        "role": "unseen_cancellation_stress",
        "family": "inward_shear",
        "envelope": {
            "kind": "continuum_balanced_sine_combination",
            "first_power": 2,
            "second_power": 4,
            "coefficient_rule": (
                "minus_primary_continuum_global_lower_height_angular_"
                "action_power2_divided_by_power4"
            ),
            "primary_continuum_nodes": 769,
            "secondary_continuum_nodes": 513,
        },
    },
    "balanced_p2_p4__outward_shear": {
        "role": "unseen_cancellation_stress",
        "family": "outward_shear",
        "envelope": {
            "kind": "continuum_balanced_sine_combination",
            "first_power": 2,
            "second_power": 4,
            "coefficient_rule": (
                "minus_primary_continuum_global_lower_height_angular_"
                "action_power2_divided_by_power4"
            ),
            "primary_continuum_nodes": 769,
            "secondary_continuum_nodes": 513,
        },
    },
    "p3__material": {
        "role": "unseen_standard_control",
        "family": "material",
        "envelope": {"kind": "full_domain_sine_power", "power": 3},
    },
}

PHYSICAL_BAND_CONTRACT = {
    "selection_grid": "uniform_N128",
    "selection_rule": "nearest_grid_edge_to_each_target",
    "target_edges_over_rg": (1.8, 3.0, 5.0, 8.0, 10.5),
    "always_include_outer_domain_edge": True,
    "restriction_rule": "exact_sum_of_nested_cell_integrals",
}

INTEGRAL_CONDITIONING_CONTRACT = {
    "historical_direct_component_route_preserved": True,
    "historical_minimum_relative_activity": 1.0e-8,
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
    "inactive_band_rule": (
        "same_fixed_physical_1e-8_activity_floor_as_component_gate"
    ),
    "alternate_route_requires_direct_order_failure": True,
    "alternate_route_requires_all_other_parent_gates": True,
    "minimum_unseen_cancellation_stress_profiles_using_alternate_route": 2,
    "no_retroactive_application_to_wp10c9d6c6c": True,
}

PROPAGATION_CONTRACT = {
    "binding_grids": ("uniform_N128", "uniform_N256", "uniform_N512"),
    "time_horizon_s": 0.125,
    "time_sample_count": 65,
    "state_and_aggregate_export_gates_unchanged_from_wp10c9d6c6c": True,
    "minimum_rms_order": 0.75,
    "minimum_maximum_order": 0.75,
    "maximum_fine_normalized_difference": 0.05,
    "minimum_history_cosine": 0.90,
    "minimum_refinement_error_cosine": 0.90,
    "maximum_state_reference_uncertainty_to_fine_difference": 0.10,
    "exact_boundary_semigroup_integral_required": True,
    "all_profile_variants_binding": True,
    "all_base_profiles_must_pass_spectral_projection_and_purity_gates": True,
    "fail_before_propagation_when_any_base_profile_is_ineligible": True,
    "no_definition_or_threshold_changes_after_this_manifest": True,
}

ELIGIBILITY_CONTRACT = {
    "spectral_energy_quantile": 0.99,
    "minimum_theta_99": 0.0,
    "maximum_theta_99": 0.30,
    "maximum_nyquist_alias_fraction": 1.0e-3,
    "maximum_endpoint_cell_fraction": 5.0e-3,
    "minimum_global_family_purity": 0.995,
    "minimum_active_cell_family_purity": 0.98,
    "maximum_projection_replay_defect": 2.0e-12,
    "maximum_balance_coefficient_relative_769_513_difference": 1.0e-6,
    "maximum_secondary_continuum_initial_cancellation_ratio": 1.0e-6,
}

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_height_localization_wp10c9d6c6d"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_manifest_wp10c9d6c6e0"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "conditioning_manifest.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_integral_conditioning.py",
    "tests/test_causal_inner_integral_conditioning.py",
    "tests/"
    "test_causal_inner_integral_conditioning_manifest_wp10c9d6c6e0.py",
)


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
        raise RuntimeError("WP10c9d6c6e0 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
    }
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
        "w",
        newline="",
        encoding="utf-8",
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


def _build_manifest() -> dict:
    variants = []
    for base_name, definition in PROFILE_DEFINITIONS.items():
        for factor in AMPLITUDE_FACTORS:
            for sign in SIGNS:
                variants.append(
                    {
                        "profile_id": (
                            f"{base_name}::a{factor:.2f}::"
                            f"{'plus' if sign > 0 else 'minus'}"
                        ),
                        "base_profile": base_name,
                        "role": definition["role"],
                        "amplitude_factor": float(factor),
                        "sign": int(sign),
                        "binding": True,
                    }
                )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "eligibility_evaluated": False,
        "base_profile_definitions": PROFILE_DEFINITIONS,
        "profile_variants": variants,
        "physical_band_contract": PHYSICAL_BAND_CONTRACT,
        "eligibility_contract": ELIGIBILITY_CONTRACT,
        "integral_conditioning_contract": (
            INTEGRAL_CONDITIONING_CONTRACT
        ),
        "prospective_propagation_contract": PROPAGATION_CONTRACT,
    }
    return {
        **payload,
        "manifest_sha256": causal_canonical_json_sha256(payload),
    }


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "propagation_executed": False,
        "amplitude_factors": AMPLITUDE_FACTORS,
        "signs": SIGNS,
        "profile_count": len(PROFILE_DEFINITIONS),
        "variant_count": (
            len(PROFILE_DEFINITIONS)
            * len(AMPLITUDE_FACTORS)
            * len(SIGNS)
        ),
        "physical_band_contract": PHYSICAL_BAND_CONTRACT,
        "eligibility_contract": ELIGIBILITY_CONTRACT,
        "integral_conditioning_contract": (
            INTEGRAL_CONDITIONING_CONTRACT
        ),
        "prospective_propagation_contract": PROPAGATION_CONTRACT,
    }


def run() -> dict:
    identity = _validate_analyzed_git_identity()
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != "convergent_bands_noncontracting_cancellation_remainder"
        or parent["authorized_next"]
        != "prospective_integral_conditioning_audit"
        or not parent["passed"]
    ):
        raise RuntimeError("WP10c9d6c6e0 authorization changed")
    manifest = _build_manifest()
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, _config())
    _write_json(MANIFEST_PATH, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": (
            "integral_conditioning_contract_and_profiles_frozen_"
            "eligibility_audit_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c6e1_profile_eligibility_and_propagation"
        ),
        "passed": True,
        "operator_changed": False,
        "propagation_executed": False,
        "eligibility_evaluated": False,
        "parent_classification": parent["classification"],
        "parent_classification_preserved": True,
        "c6c_rejection_preserved": True,
        "base_profile_count": len(PROFILE_DEFINITIONS),
        "profile_variant_count": len(manifest["profile_variants"]),
        "cancellation_stress_profile_count": sum(
            definition["role"] == "unseen_cancellation_stress"
            for definition in PROFILE_DEFINITIONS.values()
        ),
        "manifest_sha256": manifest["manifest_sha256"],
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
            "rev-parse",
            "HEAD^{tree}",
        ),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src python "
            "scripts/"
            "run_causal_inner_integral_conditioning_manifest_"
            "wp10c9d6c6e0.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                PARENT_CONFIG,
                PARENT_SUMMARY,
                PARENT_ARRAYS,
                PARENT_PROVENANCE,
            )
        },
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
                "profile_variant_count": summary[
                    "profile_variant_count"
                ],
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
