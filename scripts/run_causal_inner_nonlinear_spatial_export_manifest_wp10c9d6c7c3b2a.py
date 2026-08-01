#!/usr/bin/env python3
"""Freeze the short-horizon nonlinear spatial/export pilot contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b2a"
ANALYZED_BASE_COMMIT = "bb7eac431f4f12fd03d27f2937a515e5f5993eb1"
ANALYZED_BASE_PARENT = "ebca18c23571361b0fb1d9c1ecbd0aa20f226df7"
ANALYZED_BASE_TREE = "3767c3bc19f50fc36bfa9fc719ab65fb8efea9c0"

ARTIFACT = (
    "causal_inner_nonlinear_spatial_export_manifest_"
    "wp10c9d6c7c3b2a"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_spatial_export_manifest_"
    "wp10c9d6c7c3b2a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_spatial_export_manifest_"
    "wp10c9d6c7c3b2a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_NONLINEAR_SPATIAL_EXPORT_MANIFEST_"
    "WP10C9D6C7C3B2A_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

LAYOUTS = (
    "N128_exterior_N128_inner_c48",
    "N128_exterior_N256_inner_c48",
    "N128_exterior_N512_inner_c48",
)
INNER_REFINEMENT_RATIOS = (1, 2, 4)
LAYOUT_CELL_COUNTS = (64, 112, 208)
PROFILES = (
    "p4__inward_shear",
    "p4__outward_shear",
    "p3_buffer45__inward_shear",
    "p3_buffer45__outward_shear",
)
VARIANT_MULTIPLIERS = (1.0, -1.0, 0.5, -0.5)
TIMESTEP_SECONDS = 1.0e-5
STEP_COUNT = 4
OUTPUT_TIMES_SECONDS = np.arange(STEP_COUNT + 1) * TIMESTEP_SECONDS
COUPLING_PARENT_FACE = 48

OBSERVABLE_NAMES = (
    "inner_flux_mass",
    "inner_flux_angular_momentum",
    "inner_flux_killing_energy",
    "interface_flux_mass",
    "interface_flux_angular_momentum",
    "interface_flux_killing_energy",
    "net_drive_mass",
    "net_drive_angular_momentum",
    "net_drive_killing_energy",
    "cooling_angular_momentum",
    "cooling_killing_energy",
    "vertical_work_angular_momentum",
    "vertical_work_killing_energy",
)

MINIMUM_RMS_ORDER = 0.75
MINIMUM_MAXIMUM_ORDER = 0.75
MINIMUM_SIGNIFICANT_COMPONENT_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_REFINEMENT_ERROR_COSINE = 0.90
MINIMUM_RELATIVE_ACTIVITY = 1.0e-8

BASE_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a"
)
STEP1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_profile_screen_wp10c9d6c7c3b1b1"
)
STEP2_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_step2_screen_wp10c9d6c7c3b1b2a"
)
STEP3_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_step3_replay_wp10c9d6c7c3b1b2b2"
)
STEP4_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_fourth_step_wp10c9d6c7c3b1b3a"
)
C3A1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_physical_background_nonlinear_readiness_"
    "manifest_wp10c9d6c7c3a1"
)
C7A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)

CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "nonlinear_spatial_export_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _case_suffix(multiplier: float) -> str:
    return {
        1.0: "p1",
        -1.0: "m1",
        0.5: "p0p5",
        -0.5: "m0p5",
    }[float(multiplier)]


def _case_prefix(layout: str, profile: str, multiplier: float) -> str:
    return f"{layout}__{profile}__{_case_suffix(multiplier)}"


def _validate_parent() -> dict:
    parent = _read_json(STEP4_DIRECTORY / "summary.json")
    if (
        not parent["passed"]
        or parent["completed_case_count"] != 48
        or not parent["full_four_step_method_preflight_certified"]
        or not parent["nonlinear_spatial_export_manifest_authorized"]
        or parent["classification"]
        != "full_profile_variant_four_step_monolithic_bdf_method_"
        "preflight_certified_nonlinear_spatial_export_manifest_"
        "authorized"
        or parent["authorized_next"]
        != "WP10c9d6c7c3b2a_nonlinear_spatial_export_manifest"
    ):
        raise RuntimeError("four-step nonlinear parent certificate changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3b2a analyzed identity changed")
    return parent


def _audit_input_trajectory() -> tuple[dict, dict[str, np.ndarray]]:
    readiness = _read_json(C3A1_DIRECTORY / "summary.json")
    readiness_config = _read_json(C3A1_DIRECTORY / "config.json")
    if (
        not readiness["passed"]
        or not readiness["monolithic_bdf_method_preflight_authorized"]
        or tuple(readiness_config["profiles"]) != PROFILES
        or tuple(readiness_config["variant_multipliers"])
        != VARIANT_MULTIPLIERS
        or tuple(readiness_config["layouts"]) != LAYOUTS
        or readiness_config["short_preflight_timestep_seconds"]
        != TIMESTEP_SECONDS
        or readiness_config["short_preflight_steps"] != STEP_COUNT
    ):
        raise RuntimeError("physical nonlinear readiness contract changed")

    base = _load_npz(BASE_DIRECTORY / "decisive_arrays.npz")
    step1 = _load_npz(STEP1_DIRECTORY / "decisive_arrays.npz")
    step2 = _load_npz(STEP2_DIRECTORY / "decisive_arrays.npz")
    step3 = _load_npz(STEP3_DIRECTORY / "decisive_arrays.npz")
    step4 = _load_npz(STEP4_DIRECTORY / "decisive_arrays.npz")
    c7a = _load_npz(C7A_DIRECTORY / "decisive_arrays.npz")

    continuity = np.zeros(
        (len(LAYOUTS), len(PROFILES), len(VARIANT_MULTIPLIERS), 3)
    )
    for layout_index, (layout, cells) in enumerate(
        zip(LAYOUTS, LAYOUT_CELL_COUNTS, strict=True)
    ):
        states = np.asarray(base[f"{layout}__states"], dtype=float)
        if states.shape != (STEP_COUNT + 1, cells, 5):
            raise RuntimeError("unperturbed base history shape changed")
        for profile_index, profile in enumerate(PROFILES):
            for variant_index, multiplier in enumerate(
                VARIANT_MULTIPLIERS
            ):
                prefix = _case_prefix(layout, profile, multiplier)
                pairs = (
                    (
                        step1[f"{prefix}__final_state"],
                        step2[f"{prefix}__step2_old_state"],
                    ),
                    (
                        step2[f"{prefix}__step2_final_state"],
                        step3[f"{prefix}__step3_old_state"],
                    ),
                    (
                        step3[f"{prefix}__step3_final_state"],
                        step4[f"{prefix}__step4_old_state"],
                    ),
                )
                for pair_index, (left, right) in enumerate(pairs):
                    left_array = np.asarray(left, dtype=float)
                    right_array = np.asarray(right, dtype=float)
                    if left_array.shape != (cells, 5):
                        raise RuntimeError("perturbed history shape changed")
                    continuity[
                        layout_index,
                        profile_index,
                        variant_index,
                        pair_index,
                    ] = float(np.max(np.abs(left_array - right_array)))

    field_scales = np.asarray(c7a["field_scales"], dtype=float)
    observable_scales = np.asarray(
        c7a["fixed_physical_observable_scales"],
        dtype=float,
    )
    if (
        field_scales.shape != (5,)
        or observable_scales.shape != (len(OBSERVABLE_NAMES),)
        or np.any(field_scales <= 0.0)
        or np.any(observable_scales <= 0.0)
    ):
        raise RuntimeError("fixed physical scales changed")

    maximum_continuity = float(np.max(continuity))
    audit = {
        "passed": maximum_continuity == 0.0,
        "layout_count": len(LAYOUTS),
        "profile_count": len(PROFILES),
        "variant_count": len(VARIANT_MULTIPLIERS),
        "case_count": (
            len(LAYOUTS) * len(PROFILES) * len(VARIANT_MULTIPLIERS)
        ),
        "saved_time_count": STEP_COUNT + 1,
        "maximum_step_continuity_defect": maximum_continuity,
        "all_step_boundaries_bitwise_continuous": (
            maximum_continuity == 0.0
        ),
    }
    decisive = {
        "output_times_seconds": OUTPUT_TIMES_SECONDS,
        "inner_refinement_ratios": np.asarray(
            INNER_REFINEMENT_RATIOS,
            dtype=np.int64,
        ),
        "layout_cell_counts": np.asarray(
            LAYOUT_CELL_COUNTS,
            dtype=np.int64,
        ),
        "variant_multipliers": np.asarray(VARIANT_MULTIPLIERS),
        "field_scales": field_scales,
        "fixed_physical_observable_scales": observable_scales,
        "step_continuity_defects": continuity,
    }
    return audit, decisive


def _manifest(audit: dict) -> dict:
    gates = {
        "minimum_rms_order": MINIMUM_RMS_ORDER,
        "minimum_maximum_order": MINIMUM_MAXIMUM_ORDER,
        "minimum_significant_component_order": (
            MINIMUM_SIGNIFICANT_COMPONENT_ORDER
        ),
        "maximum_fine_normalized_difference": (
            MAXIMUM_FINE_NORMALIZED_DIFFERENCE
        ),
        "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
        "minimum_refinement_error_cosine": (
            MINIMUM_REFINEMENT_ERROR_COSINE
        ),
        "minimum_relative_activity": MINIMUM_RELATIVE_ACTIVITY,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "nonlinear_short_horizon_spatial_export_manifest_frozen_"
            "canonical_response_pilot_authorized"
        ),
        "authorized_next": (
            "WP10c9d6c7c3b2b_nonlinear_short_horizon_"
            "spatial_export_pilot"
        ),
        "operator_changed": False,
        "propagation_executed": False,
        "input_audit": audit,
        "scope": {
            "purpose": (
                "reuse the certified four-step nonlinear histories to test "
                "short-time embedded spatial contraction before any new "
                "trajectory is launched"
            ),
            "horizon_seconds": float(OUTPUT_TIMES_SECONDS[-1]),
            "timestep_seconds": TIMESTEP_SECONDS,
            "output_times_seconds": OUTPUT_TIMES_SECONDS.tolist(),
            "layouts": list(LAYOUTS),
            "inner_refinement_ratios": list(INNER_REFINEMENT_RATIOS),
            "profiles": list(PROFILES),
            "variant_multipliers": list(VARIANT_MULTIPLIERS),
            "all_profile_variants_binding": True,
            "variants_are_sign_and_amplitude_controls_not_independent_"
            "physical_profiles": True,
        },
        "nonlinear_response": {
            "definition": (
                "perturbed nonlinear trajectory minus the independently "
                "evolved unperturbed trajectory on the same layout"
            ),
            "state_restriction": (
                "exact conservative restriction to the common 64-cell "
                "parent grid at every saved time"
            ),
            "state_normalization": "fixed c7a five-field physical scales",
            "instantaneous_export_definition": (
                "evaluate the complete monolithic stationary ledger at "
                "each saved state with zero temporal increment, then apply "
                "causal_embedded_active_direct_observables at parent face 48"
            ),
            "export_response": (
                "perturbed direct observable minus unperturbed direct "
                "observable on the same layout and saved time"
            ),
            "cumulative_export_rule": (
                "trapezoidal integration of the instantaneous nonlinear "
                "export response on the five frozen saved times"
            ),
            "observable_names": list(OBSERVABLE_NAMES),
            "fixed_observable_scales": (
                "c7a fixed physical 13-export scales"
            ),
        },
        "tier_I_binding_contract": {
            "state_response": True,
            "instantaneous_13_export_response": True,
            "cumulative_13_export_response": True,
            "gates": gates,
            "spatial_triplet": list(LAYOUTS),
            "refinement_ratio": 2.0,
            "fail_fast_order": [
                "state_response",
                "instantaneous_13_export_response",
                "cumulative_13_export_response",
            ],
            "significant_component_activity_uses_fixed_physical_scales": True,
            "no_gate_changes_after_manifest": True,
        },
        "tier_II_contract": {
            "status": "diagnostic_only_nonpromoted",
            "report_total_positive_and_characteristic_energy_if_available": (
                True
            ),
            "may_not_rescue_or_fail_tier_I": True,
            "c7c1b_auxiliary_observability_remains_unresolved": True,
        },
        "method_gates_inherited": {
            "full_four_step_preflight_must_remain_passed": True,
            "maximum_scaled_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "BDF2_step3_split_restart_replay": "bitwise",
            "all_final_checkpoint_roundtrips": "bitwise",
            "incoming_excision_characteristics": 0,
        },
        "interpretation_limits": {
            "spatial_pilot_only": True,
            "temporal_convergence_certified": False,
            "long_time_physical_convergence_certified": False,
            "interface_scattering_certified": False,
            "nonlinear_physical_ladder_authorized": False,
            "fixed_q_micro_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        },
        "decision": {
            "all_tier_I_channels_pass": (
                "authorize a definitions-only temporally refined bounded "
                "pilot; do not authorize the long ladder"
            ),
            "state_response_fails": (
                "stop and localize the nonlinear spatial response; no "
                "export interpretation"
            ),
            "state_passes_but_export_fails": (
                "freeze the failed direct observable and audit its exact "
                "nonlinear ledger; no operator redesign without a stable "
                "mechanism"
            ),
            "tier_II_only_fails": (
                "retain Tier-II as unresolved and do not alter the Tier-I "
                "pilot classification"
            ),
        },
        "hard_stops": [
            "do not relabel any c7c1b or c3b1 classification",
            "do not change production defaults",
            "do not use N1024",
            "do not infer temporal convergence from one timestep",
            "do not launch the 0.125-second nonlinear ladder",
            "do not begin fixed-Q or reduced slow-time evolution",
        ],
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
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    global_summary = _read_json(CANONICAL_SUMMARY)
    global_summary.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    global_summary["latest_work_package"] = WORK_PACKAGE
    global_summary["latest_source_parent_commit"] = ANALYZED_BASE_COMMIT
    global_summary["case_count"] = len(global_summary["artifacts"])
    _write_json(CANONICAL_SUMMARY, global_summary)


def _report(manifest: dict) -> str:
    audit = manifest["input_audit"]
    gates = manifest["tier_I_binding_contract"]["gates"]
    return "\n".join(
        [
            "# Nonlinear spatial/export manifest WP10c9d6c7c3b2a",
            "",
            "## Classification",
            "",
            "`nonlinear_short_horizon_spatial_export_manifest_frozen_"
            "canonical_response_pilot_authorized`",
            "",
            "This definitions-only package changes no operator and runs no "
            "new trajectory. It freezes a fail-fast analysis of the "
            "already-certified four-step nonlinear histories.",
            "",
            "## Frozen pilot",
            "",
            f"- layouts: `{len(LAYOUTS)}`",
            f"- physical profiles: `{len(PROFILES)}`",
            f"- sign/amplitude variants per profile: "
            f"`{len(VARIANT_MULTIPLIERS)}`",
            f"- saved nonlinear cases: `{audit['case_count']}`",
            f"- saved times: `{audit['saved_time_count']}` through "
            f"`{OUTPUT_TIMES_SECONDS[-1]:.1e} s`",
            "- maximum step-boundary continuity defect: "
            f"`{audit['maximum_step_continuity_defect']:.3e}`",
            "",
            "The response is the perturbed nonlinear trajectory minus the "
            "independently evolved unperturbed trajectory on the same "
            "layout. State is conservatively restricted to the common "
            "64-cell parent grid. Tier I binds the state response and the "
            "instantaneous and cumulative 13-export responses.",
            "",
            "## Frozen Tier-I gates",
            "",
            f"- minimum RMS/max/component order: "
            f"`{gates['minimum_rms_order']:.2f}` / "
            f"`{gates['minimum_maximum_order']:.2f}` / "
            f"`{gates['minimum_significant_component_order']:.2f}`",
            "- maximum fine normalized difference: "
            f"`{gates['maximum_fine_normalized_difference']:.2f}`",
            f"- minimum history/error cosine: "
            f"`{gates['minimum_history_cosine']:.2f}` / "
            f"`{gates['minimum_refinement_error_cosine']:.2f}`",
            "",
            "Tier II remains diagnostic and cannot rescue or fail Tier I. "
            "This short pilot does not certify time convergence, long-time "
            "physics, interface scattering, fixed-Q averaging, or reduced "
            "slow evolution.",
            "",
            "## Authorized next",
            "",
            "`WP10c9d6c7c3b2b_nonlinear_short_horizon_"
            "spatial_export_pilot`",
            "",
            "If all Tier-I channels pass, the next package may freeze a "
            "small temporally refined bounded pilot. The 0.125-second "
            "nonlinear ladder remains blocked.",
            "",
        ]
    )


def main() -> None:
    parent = _validate_parent()
    audit, decisive = _audit_input_trajectory()
    manifest = _manifest(audit)
    passed = bool(audit["passed"])
    if not passed:
        manifest["classification"] = (
            "nonlinear_short_horizon_spatial_export_manifest_failed"
        )
        manifest["authorized_next"] = None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": list(LAYOUTS),
        "inner_refinement_ratios": list(INNER_REFINEMENT_RATIOS),
        "profiles": list(PROFILES),
        "variant_multipliers": list(VARIANT_MULTIPLIERS),
        "timestep_seconds": TIMESTEP_SECONDS,
        "step_count": STEP_COUNT,
        "output_times_seconds": OUTPUT_TIMES_SECONDS.tolist(),
        "coupling_parent_face": COUPLING_PARENT_FACE,
        "observable_names": list(OBSERVABLE_NAMES),
        "tier_I_gates": manifest["tier_I_binding_contract"]["gates"],
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)

    input_paths = {
        "physical_readiness_summary": C3A1_DIRECTORY / "summary.json",
        "embedded_manifest_arrays": C7A_DIRECTORY / "decisive_arrays.npz",
        "base_history_arrays": BASE_DIRECTORY / "decisive_arrays.npz",
        "step1_arrays": STEP1_DIRECTORY / "decisive_arrays.npz",
        "step2_arrays": STEP2_DIRECTORY / "decisive_arrays.npz",
        "step3_arrays": STEP3_DIRECTORY / "decisive_arrays.npz",
        "step4_arrays": STEP4_DIRECTORY / "decisive_arrays.npz",
        "step4_summary": STEP4_DIRECTORY / "summary.json",
    }
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
        "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
        "passed": passed,
        "classification": manifest["classification"],
        "authorized_next": manifest["authorized_next"],
        "operator_changed": False,
        "propagation_executed": False,
        "input_audit": audit,
        "parent_classification": parent["classification"],
        "c7c1b_strict_auxiliary_classification_preserved": True,
        "four_step_method_preflight_preserved": True,
        "short_horizon_spatial_export_pilot_authorized": passed,
        "temporal_convergence_certified": False,
        "long_nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "manifest_file_sha256": _sha256(MANIFEST_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: hashlib.sha256(
                np.ascontiguousarray(values).view(np.uint8)
            ).hexdigest()
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "input_hashes": {
            name: _sha256(path) for name, path in input_paths.items()
        },
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src "
                "/Users/huanyang/.cache/codex-runtimes/"
                "codex-primary-runtime/dependencies/python/bin/python3 "
                + THIS_RUNNER
            ),
            "analyzed_base_commit": ANALYZED_BASE_COMMIT,
            "implementation_parent_commit": _git_value(
                "rev-parse", "HEAD"
            ),
            "implementation_parent_tree_sha": _git_value(
                "rev-parse", "HEAD^{tree}"
            ),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation_source_hashes": source_hashes,
            "input_hashes": summary["input_hashes"],
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    names = (
        "config.json",
        "nonlinear_spatial_export_manifest.json",
        "decisive_arrays.npz",
        "summary.json",
        "provenance.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}"
            for name in names
        )
        + "\n",
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
