#!/usr/bin/env python3
"""Freeze the bounded nonlinear monolithic-DAE certification contract."""

from __future__ import annotations

from dataclasses import replace
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
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_direct_continuum_embedded_manifest_wp10c9d6c7c2c1 as c2c1  # noqa: E402
import run_causal_inner_direct_continuum_embedded_recertification_manifest_wp10c9d6c7c2c5 as c2c5  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_characteristic_dissipation import (  # noqa: E402
    causal_five_field_coordinate_principal_basis,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
    causal_five_field_reconstruct_face_charts,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3a"
ANALYZED_BASE_COMMIT = "2ba5049414c2cc5b5e88f00f36562d78c593c640"
ANALYZED_BASE_PARENT = "9ec5997536a5c497b95a15500bd2f8f533234031"
ANALYZED_BASE_TREE = "a2990bc1173668e504ea382fbc85ec8c83ad95c2"

LAYOUT_LABELS = c2c1.LAYOUT_LABELS
REFINEMENT_RATIOS = (1, 2, 4)
PROFILE_NAMES = c2c5.PROFILES
PROFILE_ANGLES_DEGREES = c2c5.ANGLES_DEGREES
NONLINEAR_AMPLITUDES = (0.05, -0.05, 0.025, -0.025)
METHOD_PREFLIGHT_PROFILE_INDICES = (0, 3, 7)
PHYSICAL_LADDER_PROFILE_INDICES = (0, 2, 4, 6)
SHORT_PREFLIGHT_TIMESTEP_SECONDS = 1.0e-5
SHORT_PREFLIGHT_STEPS = 4
PHYSICAL_HORIZON_SECONDS = 11.82686804912109
OUTPUT_SAMPLE_COUNT = 513

MAXIMUM_INITIAL_H_OVER_R = 0.25
MINIMUM_INITIAL_SCATTERING_OPTICAL_DEPTH = 1.0
MINIMUM_INITIAL_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
REQUIRED_INCOMING_EXCISION_CHARACTERISTICS = 0
MAXIMUM_NONLINEAR_SCALED_RESIDUAL = 1.0e-10
MAXIMUM_ALGEBRAIC_SCALED_RESIDUAL = 1.0e-10
MAXIMUM_DISCRETE_LEDGER_DEFECT = 1.0e-12
MAXIMUM_DENSE_COLORED_JACOBIAN_DEFECT = 1.0e-10
MAXIMUM_INDEPENDENT_JVP_DEFECT = 1.0e-6
MAXIMUM_TEMPORAL_TO_SPATIAL_ERROR_FRACTION = 0.10
MINIMUM_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_REFINEMENT_ERROR_COSINE = 0.90

THIS_RUNNER = (
    "scripts/run_causal_inner_bounded_nonlinear_manifest_"
    "wp10c9d6c7c3a.py"
)
THIS_TEST = (
    "tests/test_causal_inner_bounded_nonlinear_manifest_"
    "wp10c9d6c7c3a.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_BOUNDED_NONLINEAR_MANIFEST_"
    "WP10C9D6C7C3A_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
C2C6_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_recertification_"
    "wp10c9d6c7c2c6"
)
C2C5_DIRECTORY = c2c5.CANONICAL_DIRECTORY
C2C1_DIRECTORY = c2c1.CANONICAL_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_bounded_nonlinear_manifest_wp10c9d6c7c3a"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
MANIFEST_PATH = CANONICAL_DIRECTORY / "bounded_nonlinear_manifest.json"
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


def _validate_parent() -> tuple[dict, dict]:
    c2c6 = _read_json(C2C6_DIRECTORY / "summary.json")
    c2c5 = _read_json(C2C5_DIRECTORY / "summary.json")
    if (
        not c2c6["passed"]
        or c2c6["classification"]
        != "direct_continuum_embedded_linear_class_certified_"
        "bounded_nonlinear_manifest_authorized"
        or c2c6["authorized_next"]
        != "WP10c9d6c7c3a_bounded_nonlinear_contract_manifest"
        or not c2c5["passed"]
    ):
        raise RuntimeError("c2c5/c2c6 nonlinear-manifest authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3a analyzed identity changed")
    return c2c6, c2c5


def _initial_state_audit() -> tuple[dict, dict[str, np.ndarray]]:
    arrays = _load_npz(C2C5_DIRECTORY / "decisive_arrays.npz")
    c2c1_arrays = _load_npz(C2C1_DIRECTORY / "decisive_arrays.npz")
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        _parent_base,
        _field_scales,
    ) = c2a2._load_inputs()
    shape = (len(REFINEMENT_RATIOS), len(PROFILE_NAMES), len(NONLINEAR_AMPLITUDES))
    maximum_h = np.zeros(shape, dtype=float)
    minimum_tau = np.zeros(shape, dtype=float)
    minimum_factor = np.zeros(shape, dtype=float)
    incoming = np.zeros(shape, dtype=np.int64)
    minimum_sigma = np.zeros(shape, dtype=float)
    minimum_temperature = np.zeros(shape, dtype=float)

    for layout_index, ratio in enumerate(REFINEMENT_RATIOS):
        label = LAYOUT_LABELS[ratio]
        grid = make_kerr_schild_column_grid_from_edges(
            np.asarray(c2c1_arrays[f"{label}__grid_edges"], dtype=float),
            parent_context.grid.gravitational_radius,
        )
        context = replace(
            parent_context,
            grid=grid,
            stream_sources=None,
        ).validated()
        base = np.asarray(
            c2c1_arrays[f"{label}__base_primitive_charts"],
            dtype=float,
        )
        packets = np.asarray(arrays[f"{label}__packets"], dtype=float)
        base_inner_basis = causal_five_field_coordinate_principal_basis(
            context,
            float(grid.edges[0]),
            base[0],
        )
        for profile_index in range(len(PROFILE_NAMES)):
            for amplitude_index, amplitude in enumerate(
                NONLINEAR_AMPLITUDES
            ):
                state = base + float(amplitude) * packets[profile_index]
                reconstruction = causal_five_field_reconstruct_face_charts(
                    context,
                    state,
                    purpose="flux",
                )
                local_h = []
                local_tau = []
                local_sigma = []
                local_temperature = []
                for radius, chart in zip(
                    grid.centers,
                    state,
                    strict=True,
                ):
                    cell = _cell_state(context, float(radius), chart)
                    sigma = float(cell.primitive.surface_density)
                    temperature = float(cell.thermodynamics.temperature)
                    height = float(cell.thermodynamics.proper_half_thickness)
                    local_h.append(height / float(radius))
                    local_tau.append(0.5 * context.kappa * sigma)
                    local_sigma.append(sigma)
                    local_temperature.append(temperature)
                maximum_h[layout_index, profile_index, amplitude_index] = max(
                    local_h
                )
                minimum_tau[layout_index, profile_index, amplitude_index] = min(
                    local_tau
                )
                minimum_sigma[
                    layout_index, profile_index, amplitude_index
                ] = min(local_sigma)
                minimum_temperature[
                    layout_index, profile_index, amplitude_index
                ] = min(local_temperature)
                minimum_factor[
                    layout_index, profile_index, amplitude_index
                ] = float(
                    np.min(reconstruction.admissibility_factors)
                )
                incoming[
                    layout_index, profile_index, amplitude_index
                ] = int(base_inner_basis.incoming_inner_characteristics)

    passed = bool(
        np.max(maximum_h) <= MAXIMUM_INITIAL_H_OVER_R
        and np.min(minimum_tau)
        > MINIMUM_INITIAL_SCATTERING_OPTICAL_DEPTH
        and np.min(minimum_factor)
        >= MINIMUM_INITIAL_RECONSTRUCTION_FACTOR
        and np.max(incoming)
        == REQUIRED_INCOMING_EXCISION_CHARACTERISTICS
        and np.min(minimum_sigma) > 0.0
        and np.min(minimum_temperature) > 0.0
    )
    report = {
        "passed": passed,
        "maximum_h_over_r": float(np.max(maximum_h)),
        "minimum_scattering_optical_depth": float(np.min(minimum_tau)),
        "minimum_reconstruction_admissibility_factor": float(
            np.min(minimum_factor)
        ),
        "maximum_incoming_excision_characteristics": int(np.max(incoming)),
        "minimum_surface_density": float(np.min(minimum_sigma)),
        "minimum_temperature": float(np.min(minimum_temperature)),
        "variant_count": int(np.prod(shape)),
    }
    decisive = {
        "profile_angles_degrees": np.asarray(
            PROFILE_ANGLES_DEGREES, dtype=float
        ),
        "nonlinear_amplitudes": np.asarray(
            NONLINEAR_AMPLITUDES, dtype=float
        ),
        "method_preflight_profile_indices": np.asarray(
            METHOD_PREFLIGHT_PROFILE_INDICES, dtype=np.int64
        ),
        "physical_ladder_profile_indices": np.asarray(
            PHYSICAL_LADDER_PROFILE_INDICES, dtype=np.int64
        ),
        "maximum_initial_h_over_r": maximum_h,
        "minimum_initial_scattering_optical_depth": minimum_tau,
        "minimum_initial_reconstruction_factor": minimum_factor,
        "incoming_excision_characteristics": incoming,
        "minimum_initial_surface_density": minimum_sigma,
        "minimum_initial_temperature": minimum_temperature,
        "physical_output_times_seconds": np.linspace(
            0.0,
            PHYSICAL_HORIZON_SECONDS,
            OUTPUT_SAMPLE_COUNT,
        ),
    }
    return report, decisive


def _manifest(initial: dict) -> dict:
    selected_preflight = [
        PROFILE_NAMES[index] for index in METHOD_PREFLIGHT_PROFILE_INDICES
    ]
    selected_physical = [
        PROFILE_NAMES[index] for index in PHYSICAL_LADDER_PROFILE_INDICES
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "bounded_nonlinear_contract_frozen_"
            "monolithic_bdf_method_preflight_authorized"
        ),
        "operator_changed": False,
        "propagation_executed": False,
        "historical_c2c3_rejection_preserved": True,
        "c2c6_linear_certification_preserved": True,
        "nonlinear_architecture": {
            "residual": (
                "production-neutral primitive-only monolithic "
                "descriptor-path DAE"
            ),
            "implementation": (
                "causal_inner_monolithic_dae."
                "evaluate_causal_five_field_monolithic_backward_euler"
            ),
            "stationary_operator": (
                "center-broken complete radial fluctuation ledger"
            ),
            "temporal_operator": (
                "mapped endpoint increment plus path-integrated "
                "responsive-height one-form"
            ),
            "shared_MJE_face_flux": True,
            "uses_production_generator": False,
            "uses_production_anchor_storage_derivative": False,
            "production_defaults_changed": False,
            "audit_only_finite_difference_characteristic_split": True,
        },
        "initial_state_contract": {
            "profiles": list(PROFILE_NAMES),
            "amplitudes": list(NONLINEAR_AMPLITUDES),
            "layouts": [
                LAYOUT_LABELS[ratio] for ratio in REFINEMENT_RATIOS
            ],
            "base_plus_amplitude_times_frozen_packet": True,
            "initial_audit": initial,
            "gates": {
                "maximum_h_over_r": MAXIMUM_INITIAL_H_OVER_R,
                "minimum_scattering_optical_depth": (
                    MINIMUM_INITIAL_SCATTERING_OPTICAL_DEPTH
                ),
                "minimum_reconstruction_factor": (
                    MINIMUM_INITIAL_RECONSTRUCTION_FACTOR
                ),
                "incoming_excision_characteristics": (
                    REQUIRED_INCOMING_EXCISION_CHARACTERISTICS
                ),
            },
        },
        "method_preflight_contract": {
            "authorized_next": (
                "WP10c9d6c7c3b1_monolithic_bdf_method_preflight"
            ),
            "profiles": selected_preflight,
            "amplitudes": list(NONLINEAR_AMPLITUDES),
            "layouts": [
                LAYOUT_LABELS[ratio] for ratio in REFINEMENT_RATIOS
            ],
            "fixed_timestep_seconds": SHORT_PREFLIGHT_TIMESTEP_SECONDS,
            "fixed_steps": SHORT_PREFLIGHT_STEPS,
            "requirements": {
                "increment_primary_BDF1_and_variable_step_BDF2": True,
                "previous_complete_monolithic_storage_increment_stored": True,
                "maximum_scaled_residual": (
                    MAXIMUM_NONLINEAR_SCALED_RESIDUAL
                ),
                "maximum_scaled_algebraic_residual": (
                    MAXIMUM_ALGEBRAIC_SCALED_RESIDUAL
                ),
                "maximum_discrete_ledger_defect": (
                    MAXIMUM_DISCRETE_LEDGER_DEFECT
                ),
                "maximum_dense_colored_jacobian_defect": (
                    MAXIMUM_DENSE_COLORED_JACOBIAN_DEFECT
                ),
                "maximum_independent_JVP_defect": (
                    MAXIMUM_INDEPENDENT_JVP_DEFECT
                ),
                "BDF2_split_restart_replay": "bitwise",
                "incoming_excision_characteristics": 0,
                "unchanged_admissibility_branch": True,
            },
            "stop_condition": (
                "any method, residual, Jacobian, ledger, replay, "
                "admissibility, or causality failure"
            ),
        },
        "conditional_physical_ladder_contract": {
            "not_authorized_until_method_preflight_passes": True,
            "profiles": selected_physical,
            "amplitudes": list(NONLINEAR_AMPLITUDES),
            "layouts": [
                LAYOUT_LABELS[ratio] for ratio in REFINEMENT_RATIOS
            ],
            "horizon_seconds": PHYSICAL_HORIZON_SECONDS,
            "output_sample_count": OUTPUT_SAMPLE_COUNT,
            "comparison": (
                "perturbed nonlinear trajectory minus independently evolved "
                "unperturbed nonlinear background on the same layout"
            ),
            "state_restriction": (
                "conservative restriction to the common N98 parent grid"
            ),
            "tier_I_observables": (
                "state, inner and coupling M/J/E flux, net M/J/E drive, "
                "cooling, and responsive-height work"
            ),
            "temporal_refinement": {
                "independent_of_spatial_refinement": True,
                "maximum_temporal_to_spatial_error_fraction": (
                    MAXIMUM_TEMPORAL_TO_SPATIAL_ERROR_FRACTION
                ),
            },
            "spatial_gates": {
                "minimum_RMS_order": MINIMUM_ORDER,
                "minimum_maximum_order": MINIMUM_ORDER,
                "minimum_significant_component_order": MINIMUM_ORDER,
                "maximum_fine_normalized_difference": (
                    MAXIMUM_FINE_NORMALIZED_DIFFERENCE
                ),
                "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
                "minimum_refinement_error_cosine": (
                    MINIMUM_REFINEMENT_ERROR_COSINE
                ),
            },
            "amplitude_diagnostics": {
                "sign_pair_odd_response": True,
                "sign_pair_even_response": True,
                "half_amplitude_control": True,
                "no_linearity_assumption_used_to_pass_spatial_gates": True,
            },
        },
        "hard_stops": [
            "no nonlinear physical ladder before c3b1 passes",
            "no production-default change",
            "no fixed-Q experiment",
            "no reduced slow-time evolution",
            "no N1024 rescue",
            "no tide wind hot-state S-curve or QPE-cycle physics",
        ],
    }


def _report(manifest: dict) -> str:
    initial = manifest["initial_state_contract"]["initial_audit"]
    preflight = manifest["method_preflight_contract"]
    physical = manifest["conditional_physical_ladder_contract"]
    return "\n".join(
        [
            "# Bounded nonlinear contract manifest WP10c9d6c7c3a",
            "",
            "## Classification",
            "",
            "`bounded_nonlinear_contract_frozen_"
            "monolithic_bdf_method_preflight_authorized`",
            "",
            "No nonlinear propagation or operator change occurs in this "
            "package. The c2c3 rejection and c2c6 frozen-linear "
            "certification remain unchanged.",
            "",
            "## Frozen nonlinear architecture",
            "",
            "The next method package must evolve the production-neutral "
            "primitive-only monolithic descriptor-path DAE. Its stationary "
            "part is the center-broken complete radial fluctuation ledger; "
            "its temporal part is the exact mapped endpoint increment plus "
            "the declared responsive-height path product. It may not reuse "
            "the production generator or production-anchor storage action.",
            "",
            "Every perturbed trajectory is compared with a separately "
            "evolved unperturbed background on the same layout. This makes "
            "the certified object the nonlinear response, not the unrelated "
            "slow drift of the base state.",
            "",
            "## Frozen states",
            "",
            f"- profiles: `{len(PROFILE_NAMES)}`",
            f"- amplitudes: `{list(NONLINEAR_AMPLITUDES)}`",
            f"- layouts: `{list(preflight['layouts'])}`",
            f"- maximum initial H/R: `{initial['maximum_h_over_r']:.6e}`",
            "- minimum initial scattering optical depth: "
            f"`{initial['minimum_scattering_optical_depth']:.6e}`",
            "- minimum reconstruction factor: "
            f"`{initial['minimum_reconstruction_admissibility_factor']:.16g}`",
            "- maximum incoming excision characteristics: "
            f"`{initial['maximum_incoming_excision_characteristics']}`",
            "",
            (
                "All 96 frozen initial states are admissible under the "
                "declared physical gates."
                if initial["passed"]
                else "The frozen nonlinear class is rejected before "
                "propagation. Its manufactured base reaches "
                f"H/R={initial['maximum_h_over_r']:.6g}, above the "
                f"{MAXIMUM_INITIAL_H_OVER_R:.2f} physical gate. The "
                "failure is present in the base construction and must not "
                "be repaired by reducing packet amplitude."
            ),
            "",
            "## Authorized next package",
            "",
            f"`{preflight['authorized_next']}`",
            "",
            (
                "It must implement and certify monolithic increment-primary "
                "BDF1/BDF2, preserve the complete prior storage increment, "
                "reach the unchanged `1e-10` residual, close the physical "
                "ledger, match dense/colored and independent Jacobian "
                "actions, retain zero incoming excision modes, and replay a "
                "split BDF2 run bitwise."
                if initial["passed"]
                else "The next package is definitions-only. It must select "
                "an already committed physical embedded background, verify "
                "H/R, optical depth, causality, reconstruction, exact "
                "restriction, and Tier-I linear eligibility, and freeze "
                "bounded nonlinear profiles without propagating them."
            ),
            "",
            (
                "The long nonlinear ladder remains conditional. If method "
                "preflight passes, it uses four frozen directions, four "
                "signed amplitudes, the `98/147/245` layouts, and the "
                f"`{physical['horizon_seconds']:.12g} s` horizon. Tier-I "
                "state/export responses retain all order, difference, and "
                "direction gates from the linear certification."
                if initial["passed"]
                else "No monolithic BDF method or physical trajectory is "
                "authorized from this manufactured background."
            ),
            "",
            "## Stop gates",
            "",
            "- Do not run the long nonlinear ladder before a physical "
            "background and c3b1 pass.",
            "- Do not change production defaults or tune a profile.",
            "- Do not start fixed-Q or reduced evolution.",
            "",
        ]
    )


def _update_global_manifests(summary: dict) -> None:
    artifact_id = "causal_inner_bounded_nonlinear_manifest_wp10c9d6c7c3a"
    relative_directory = str(CANONICAL_DIRECTORY.relative_to(ROOT))
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [
        existing
        for existing in rows
        if existing.get("case") != artifact_id
    ]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": artifact_id,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha256(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    fieldnames = ["case", "path", "bytes", "sha256", "scientific_status"]
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    global_summary = (
        _read_json(CANONICAL_SUMMARY)
        if CANONICAL_SUMMARY.exists()
        else {}
    )
    entries = global_summary.setdefault("artifacts", {})
    entries[artifact_id] = {
        "path": relative_directory,
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    _write_json(CANONICAL_SUMMARY, global_summary)


def main() -> None:
    c2c6_summary, c2c5_summary = _validate_parent()
    initial, decisive = _initial_state_audit()
    manifest = _manifest(initial)
    passed = bool(initial["passed"])
    if not passed:
        manifest["classification"] = (
            "bounded_nonlinear_manufactured_background_rejected_"
            "physical_background_readiness_manifest_authorized"
        )
        manifest["method_preflight_contract"]["authorized_next"] = (
            "WP10c9d6c7c3a1_physical_background_"
            "nonlinear_readiness_manifest"
        )
        manifest["method_preflight_contract"][
            "method_preflight_authorized"
        ] = False
        manifest["conditional_physical_ladder_contract"][
            "manufactured_background_rejected"
        ] = True

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profile_names": list(PROFILE_NAMES),
        "profile_angles_degrees": list(PROFILE_ANGLES_DEGREES),
        "nonlinear_amplitudes": list(NONLINEAR_AMPLITUDES),
        "method_preflight_profile_indices": list(
            METHOD_PREFLIGHT_PROFILE_INDICES
        ),
        "physical_ladder_profile_indices": list(
            PHYSICAL_LADDER_PROFILE_INDICES
        ),
        "short_preflight_timestep_seconds": (
            SHORT_PREFLIGHT_TIMESTEP_SECONDS
        ),
        "short_preflight_steps": SHORT_PREFLIGHT_STEPS,
        "physical_horizon_seconds": PHYSICAL_HORIZON_SECONDS,
        "output_sample_count": OUTPUT_SAMPLE_COUNT,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)

    source_paths = (THIS_RUNNER, THIS_TEST)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in source_paths
        if (ROOT / path).exists()
    }
    input_hashes = {
        "c2c6_summary": _sha256(C2C6_DIRECTORY / "summary.json"),
        "c2c6_arrays": _sha256(C2C6_DIRECTORY / "decisive_arrays.npz"),
        "c2c5_manifest": _sha256(
            C2C5_DIRECTORY / "recertification_manifest.json"
        ),
        "c2c5_arrays": _sha256(C2C5_DIRECTORY / "decisive_arrays.npz"),
        "c2c1_arrays": _sha256(C2C1_DIRECTORY / "decisive_arrays.npz"),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent_commit": ANALYZED_BASE_PARENT,
        "analyzed_base_tree_sha": ANALYZED_BASE_TREE,
        "passed": passed,
        "classification": manifest["classification"],
        "authorized_next": manifest["method_preflight_contract"][
            "authorized_next"
        ],
        "operator_changed": False,
        "propagation_executed": False,
        "historical_c2c3_classification_preserved": True,
        "c2c6_classification": c2c6_summary["classification"],
        "c2c5_classification": c2c5_summary["classification"],
        "initial_state_audit": initial,
        "profile_count": len(PROFILE_NAMES),
        "amplitude_count": len(NONLINEAR_AMPLITUDES),
        "initial_variant_count": initial["variant_count"],
        "manifest_sha256": causal_canonical_json_sha256(manifest),
        "config_sha256": _sha256(CONFIG_PATH),
        "manifest_file_sha256": _sha256(MANIFEST_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "nonlinear_physical_ladder_authorized": False,
        "monolithic_bdf_method_preflight_authorized": passed,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "command": (
            "PYTHONPATH=src python "
            "scripts/run_causal_inner_bounded_nonlinear_manifest_"
            "wp10c9d6c7c3a.py"
        ),
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "implementation_parent_commit": _git_value("rev-parse", "HEAD"),
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
        "input_hashes": input_hashes,
    }
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")

    checksums = []
    for path in (
        CONFIG_PATH,
        MANIFEST_PATH,
        DECISIVE_ARRAYS,
        SUMMARY_PATH,
        PROVENANCE_PATH,
    ):
        checksums.append(f"{_sha256(path)}  {path.name}")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    _update_global_manifests(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
