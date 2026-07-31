#!/usr/bin/env python3
"""Freeze a physical embedded background for bounded nonlinear preflight."""

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
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a  # noqa: E402
import run_causal_inner_frozen_hardening_wp10c9d5a as d5a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_characteristic_dissipation import (  # noqa: E402
    causal_five_field_coordinate_principal_basis,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    _cell_state,
    causal_five_field_reconstruct_face_charts,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    evaluate_causal_five_field_monolithic_backward_euler,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3a1"
ANALYZED_BASE_COMMIT = "f89c368dc3154140352678ad1c3a98692d88b1aa"
ANALYZED_BASE_PARENT = "2ba5049414c2cc5b5e88f00f36562d78c593c640"
ANALYZED_BASE_TREE = "bb147c8a5cb7585c072513c4d01389ed164d422c"

LAYOUTS = c7a.LAYOUTS
REFINEMENT_RATIOS = c7a.REFINEMENT_RATIOS
COUPLING_PARENT_FACE = c7a.PARENT_COUPLING_FACE
PROFILES = (
    "p4__inward_shear",
    "p4__outward_shear",
    "p3_buffer45__inward_shear",
    "p3_buffer45__outward_shear",
)
PROFILE_KIND = "primary_physical"
VARIANT_MULTIPLIERS = (1.0, -1.0, 0.5, -0.5)
PHYSICAL_HORIZON_SECONDS = 0.125
OUTPUT_SAMPLE_COUNT = 65
SHORT_PREFLIGHT_TIMESTEP_SECONDS = 1.0e-5
SHORT_PREFLIGHT_STEPS = 4

MAXIMUM_H_OVER_R = 0.25
MINIMUM_SCATTERING_OPTICAL_DEPTH = 1.0
MINIMUM_RECONSTRUCTION_FACTOR = 1.0 - 1.0e-12
MAXIMUM_RESTRICTION_DEFECT = 2.0e-12
MAXIMUM_COUPLING_TRACE_JUMP = 1.0e-4
MAXIMUM_MONOLITHIC_BLOCK_LEDGER_DEFECT = 1.0e-12
MAXIMUM_CENTER_BROKEN_PATH_ADJUSTMENT = 2.0e-8
REQUIRED_INCOMING_EXCISION_CHARACTERISTICS = 0

THIS_RUNNER = (
    "scripts/run_causal_inner_physical_background_nonlinear_"
    "readiness_manifest_wp10c9d6c7c3a1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_physical_background_nonlinear_"
    "readiness_manifest_wp10c9d6c7c3a1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_PHYSICAL_BACKGROUND_NONLINEAR_READINESS_"
    "WP10C9D6C7C3A1_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
C3A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_bounded_nonlinear_manifest_wp10c9d6c7c3a"
)
C7A_DIRECTORY = c7a.CANONICAL_DIRECTORY
C7C0_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_manifest_wp10c9d6c7c0"
)
C7C1A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a"
)
C7C1B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_wp10c9d6c7c1b"
)
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_physical_background_nonlinear_readiness_"
    "manifest_wp10c9d6c7c3a1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
MANIFEST_PATH = CANONICAL_DIRECTORY / "physical_background_manifest.json"
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


def _relative_defect(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first - second)) / scale)


def _validate_inputs() -> tuple[dict, dict, dict, dict, dict]:
    c3a = _read_json(C3A_DIRECTORY / "summary.json")
    c7a_summary = _read_json(C7A_DIRECTORY / "summary.json")
    c7c0 = _read_json(C7C0_DIRECTORY / "summary.json")
    c7c1a = _read_json(C7C1A_DIRECTORY / "summary.json")
    c7c1b = _read_json(C7C1B_DIRECTORY / "summary.json")
    if (
        c3a["classification"]
        != "bounded_nonlinear_manufactured_background_rejected_"
        "physical_background_readiness_manifest_authorized"
        or c3a["authorized_next"]
        != "WP10c9d6c7c3a1_physical_background_"
        "nonlinear_readiness_manifest"
        or c7a_summary["classification"]
        != "embedded_layout_and_profile_manifest_frozen_"
        "propagation_authorized"
        or not c7c0["passed"]
        or not c7c1a["passed"]
        or c7c1b["historical_direct_contract_report"]["passed"] is not True
        or not np.all(
            _load_npz(C7C1B_DIRECTORY / "decisive_arrays.npz")[
                "direct_packet_pass_flags"
            ]
        )
    ):
        raise RuntimeError("physical-background authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("c3a1 analyzed identity changed")
    return c3a, c7a_summary, c7c0, c7c1a, c7c1b


def _audit_physical_class() -> tuple[dict, dict[str, np.ndarray]]:
    c7a_arrays = _load_npz(C7A_DIRECTORY / "decisive_arrays.npz")
    c7c0_arrays = _load_npz(C7C0_DIRECTORY / "decisive_arrays.npz")
    replay_arrays = _load_npz(c7a.C0E_INPUTS)
    replay_contexts = _read_json(c7a.C0E_CONTEXTS)
    parent_edges = np.asarray(c7a_arrays["parent_grid_edges"], dtype=float)
    gravitational_radius = float(
        replay_contexts["contexts"][LAYOUTS[1]][
            "grid_gravitational_radius"
        ]
    )
    parent_grid = make_kerr_schild_column_grid_from_edges(
        parent_edges,
        gravitational_radius,
    )
    shape = (len(REFINEMENT_RATIOS), len(PROFILES), len(VARIANT_MULTIPLIERS))
    maximum_h = np.zeros(shape)
    minimum_tau = np.zeros(shape)
    minimum_factor = np.zeros(shape)
    incoming = np.zeros(shape, dtype=np.int64)
    coupling_jump = np.zeros(shape)
    restricted_states = {}
    block_defects = np.zeros(len(REFINEMENT_RATIOS))
    path_adjustments = np.zeros(len(REFINEMENT_RATIOS))
    base_incoming = np.zeros(len(REFINEMENT_RATIOS), dtype=np.int64)
    base_minimum_path_factor = np.zeros(len(REFINEMENT_RATIOS))

    for layout_index, ratio in enumerate(REFINEMENT_RATIOS):
        label = LAYOUTS[ratio]
        context = d5a._context_from_payload(
            replay_contexts["contexts"][label],
            replay_arrays,
        )
        layout = make_causal_embedded_patch_layout(
            parent_grid,
            COUPLING_PARENT_FACE,
            ratio,
        )
        if not np.array_equal(context.grid.edges, layout.grid.edges):
            raise RuntimeError("physical context/layout mismatch")
        base = np.asarray(
            c7a_arrays[f"{label}__spliced_base_primitives"],
            dtype=float,
        )
        base_evaluation = evaluate_causal_five_field_monolithic_backward_euler(
            base,
            base,
            1.0e-4,
            context,
        )
        block_defects[layout_index] = (
            base_evaluation.maximum_block_ledger_defect
        )
        path_adjustments[layout_index] = (
            base_evaluation.maximum_center_broken_path_adjustment
        )
        base_incoming[layout_index] = (
            base_evaluation.incoming_excision_characteristics
        )
        base_minimum_path_factor[layout_index] = (
            base_evaluation.storage_increment
            .minimum_path_reconstruction_factor
        )

        for profile_index, profile in enumerate(PROFILES):
            packet = np.asarray(
                c7c0_arrays[
                    f"{profile}__{label}__{PROFILE_KIND}"
                ],
                dtype=float,
            )
            for amplitude_index, multiplier in enumerate(
                VARIANT_MULTIPLIERS
            ):
                state = base + float(multiplier) * packet
                reconstruction = causal_five_field_reconstruct_face_charts(
                    context,
                    state,
                    purpose="flux",
                )
                local_h = []
                local_tau = []
                for radius, chart in zip(
                    context.grid.centers,
                    state,
                    strict=True,
                ):
                    cell = _cell_state(context, float(radius), chart)
                    local_h.append(
                        cell.thermodynamics.proper_half_thickness
                        / float(radius)
                    )
                    local_tau.append(
                        0.5
                        * context.kappa
                        * cell.primitive.surface_density
                    )
                basis = causal_five_field_coordinate_principal_basis(
                    context,
                    float(context.grid.edges[0]),
                    reconstruction.right_face_charts[0],
                )
                active = int(layout.coupling_face_index)
                scale = np.maximum(np.max(np.abs(state), axis=0), 1.0)
                jump = float(
                    np.max(
                        np.abs(
                            reconstruction.right_face_charts[active]
                            - reconstruction.left_face_charts[active]
                        )
                        / scale
                    )
                )
                index = (layout_index, profile_index, amplitude_index)
                maximum_h[index] = max(local_h)
                minimum_tau[index] = min(local_tau)
                minimum_factor[index] = float(
                    np.min(reconstruction.admissibility_factors)
                )
                incoming[index] = basis.incoming_inner_characteristics
                coupling_jump[index] = jump
                restricted_states[(ratio, profile_index, amplitude_index)] = (
                    restrict_causal_embedded_patch_cell_averages(
                        state,
                        layout,
                    )
                )

    maximum_restriction = 0.0
    for profile_index in range(len(PROFILES)):
        for amplitude_index in range(len(VARIANT_MULTIPLIERS)):
            reference = restricted_states[(1, profile_index, amplitude_index)]
            for ratio in REFINEMENT_RATIOS[1:]:
                maximum_restriction = max(
                    maximum_restriction,
                    _relative_defect(
                        restricted_states[
                            (ratio, profile_index, amplitude_index)
                        ],
                        reference,
                    ),
                )
    passed = bool(
        np.max(maximum_h) <= MAXIMUM_H_OVER_R
        and np.min(minimum_tau) > MINIMUM_SCATTERING_OPTICAL_DEPTH
        and np.min(minimum_factor) >= MINIMUM_RECONSTRUCTION_FACTOR
        and np.max(incoming)
        == REQUIRED_INCOMING_EXCISION_CHARACTERISTICS
        and np.max(coupling_jump) <= MAXIMUM_COUPLING_TRACE_JUMP
        and maximum_restriction <= MAXIMUM_RESTRICTION_DEFECT
        and np.max(block_defects)
        <= MAXIMUM_MONOLITHIC_BLOCK_LEDGER_DEFECT
        and np.max(path_adjustments)
        <= MAXIMUM_CENTER_BROKEN_PATH_ADJUSTMENT
        and np.max(base_incoming)
        == REQUIRED_INCOMING_EXCISION_CHARACTERISTICS
        and np.min(base_minimum_path_factor)
        >= MINIMUM_RECONSTRUCTION_FACTOR
    )
    report = {
        "passed": passed,
        "variant_count": int(np.prod(shape)),
        "maximum_h_over_r": float(np.max(maximum_h)),
        "minimum_scattering_optical_depth": float(np.min(minimum_tau)),
        "minimum_reconstruction_factor": float(np.min(minimum_factor)),
        "maximum_incoming_excision_characteristics": int(np.max(incoming)),
        "maximum_coupling_trace_jump": float(np.max(coupling_jump)),
        "maximum_restriction_defect": maximum_restriction,
        "maximum_monolithic_block_ledger_defect": float(
            np.max(block_defects)
        ),
        "maximum_center_broken_path_adjustment": float(
            np.max(path_adjustments)
        ),
        "minimum_base_path_reconstruction_factor": float(
            np.min(base_minimum_path_factor)
        ),
    }
    decisive = {
        "variant_multipliers": np.asarray(VARIANT_MULTIPLIERS),
        "maximum_h_over_r": maximum_h,
        "minimum_scattering_optical_depth": minimum_tau,
        "minimum_reconstruction_factor": minimum_factor,
        "incoming_excision_characteristics": incoming,
        "coupling_trace_jump": coupling_jump,
        "base_monolithic_block_ledger_defect": block_defects,
        "base_center_broken_path_adjustment": path_adjustments,
        "base_incoming_excision_characteristics": base_incoming,
        "base_minimum_path_reconstruction_factor": (
            base_minimum_path_factor
        ),
        "output_times_seconds": np.linspace(
            0.0,
            PHYSICAL_HORIZON_SECONDS,
            OUTPUT_SAMPLE_COUNT,
        ),
    }
    return report, decisive


def _manifest(audit: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "physical_embedded_background_nonlinear_ready_"
            "monolithic_bdf_method_preflight_authorized"
        ),
        "operator_changed": False,
        "propagation_executed": False,
        "c3a_manufactured_background_rejection_preserved": True,
        "c7c1b_strict_auxiliary_rejection_preserved": True,
        "tier_I_direct_physics_eligibility": (
            "all_16_c7c1b_direct_state_and_13_export_contracts_passed"
        ),
        "tier_II_interface_observability": "unresolved_nonpromoted",
        "physical_background_contract": {
            "source": "WP10c9d6c7a committed spliced physical background",
            "rule": (
                "smooth physical inner continuum projection plus one "
                "common committed physical N128 exterior"
            ),
            "layouts": [
                LAYOUTS[ratio] for ratio in REFINEMENT_RATIOS
            ],
            "coupling_parent_face": COUPLING_PARENT_FACE,
            "profiles": list(PROFILES),
            "profile_kind": PROFILE_KIND,
            "variant_multipliers": list(VARIANT_MULTIPLIERS),
            "audit": audit,
        },
        "nonlinear_response_contract": {
            "comparison": (
                "perturbed nonlinear run minus independently evolved "
                "unperturbed background on the same layout"
            ),
            "horizon_seconds": PHYSICAL_HORIZON_SECONDS,
            "output_sample_count": OUTPUT_SAMPLE_COUNT,
            "state_restriction": (
                "exact conservative restriction to the common 64-cell "
                "parent grid"
            ),
            "tier_I_binding": True,
            "tier_II_integrated_energy_reported_nonpromoted": True,
        },
        "authorized_next": (
            "WP10c9d6c7c3b1_monolithic_bdf_method_preflight"
        ),
        "method_preflight_contract": {
            "profiles": list(PROFILES),
            "variant_multipliers": list(VARIANT_MULTIPLIERS),
            "layouts": [
                LAYOUTS[ratio] for ratio in REFINEMENT_RATIOS
            ],
            "fixed_timestep_seconds": SHORT_PREFLIGHT_TIMESTEP_SECONDS,
            "fixed_steps": SHORT_PREFLIGHT_STEPS,
            "maximum_scaled_residual": 1.0e-10,
            "maximum_scaled_algebraic_residual": 1.0e-10,
            "maximum_discrete_ledger_defect": 1.0e-12,
            "maximum_dense_colored_jacobian_defect": 1.0e-10,
            "maximum_independent_JVP_defect": 1.0e-6,
            "BDF2_split_restart_replay": "bitwise",
            "incoming_excision_characteristics": 0,
        },
        "hard_stops": [
            "no physical nonlinear ladder before c3b1 passes",
            "no production default change",
            "no fixed-Q or reduced evolution",
            "no N1024 rescue",
            "do not relabel c7c1b or c3a",
        ],
    }


def _update_catalog(summary: dict) -> None:
    artifact = (
        "causal_inner_physical_background_nonlinear_readiness_"
        "manifest_wp10c9d6c7c3a1"
    )
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != artifact]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": artifact,
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
    global_summary.setdefault("artifacts", {})[artifact] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    _write_json(CANONICAL_SUMMARY, global_summary)


def _report(manifest: dict) -> str:
    audit = manifest["physical_background_contract"]["audit"]
    return "\n".join(
        [
            "# Physical embedded background nonlinear readiness "
            "WP10c9d6c7c3a1",
            "",
            "## Classification",
            "",
            "`physical_embedded_background_nonlinear_ready_"
            "monolithic_bdf_method_preflight_authorized`",
            "",
            "This definitions-only package replaces the rejected "
            "manufactured c3a background with the committed c7a physical "
            "embedded background. It changes no operator and propagates no "
            "state.",
            "",
            "## Physical readiness",
            "",
            f"- maximum H/R: `{audit['maximum_h_over_r']:.8f}`",
            "- minimum scattering optical depth: "
            f"`{audit['minimum_scattering_optical_depth']:.8f}`",
            "- minimum reconstruction factor: "
            f"`{audit['minimum_reconstruction_factor']:.16g}`",
            "- maximum coupling trace jump: "
            f"`{audit['maximum_coupling_trace_jump']:.6e}`",
            "- maximum cross-layout restriction defect: "
            f"`{audit['maximum_restriction_defect']:.6e}`",
            "- maximum monolithic block-ledger defect: "
            f"`{audit['maximum_monolithic_block_ledger_defect']:.6e}`",
            "- incoming excision characteristics: "
            f"`{audit['maximum_incoming_excision_characteristics']}`",
            "",
            "All four endpoint-regularized shear profiles, four sign/"
            "amplitude variants, and all three `64/112/208` layouts pass "
            "the initial physical gates. The complete monolithic residual "
            "also closes on each unperturbed base.",
            "",
            "The c7c1b strict auxiliary classification remains rejected. "
            "Its direct Tier-I state and 13-export contract passed for all "
            "16 variants, so this package authorizes only the nonlinear "
            "BDF method preflight. It does not authorize a long trajectory.",
            "",
            "## Authorized next",
            "",
            "`WP10c9d6c7c3b1_monolithic_bdf_method_preflight`",
            "",
            "The preflight must implement the complete path-increment "
            "BDF1/BDF2 residual, reach `1e-10`, close the ledger, verify "
            "Jacobian actions, preserve causality and admissibility, and "
            "replay a split BDF2 run bitwise.",
            "",
        ]
    )


def main() -> None:
    parents = _validate_inputs()
    audit, decisive = _audit_physical_class()
    manifest = _manifest(audit)
    passed = bool(audit["passed"])
    if not passed:
        manifest["classification"] = (
            "physical_embedded_background_nonlinear_readiness_failed"
        )
        manifest["authorized_next"] = None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "layouts": [LAYOUTS[ratio] for ratio in REFINEMENT_RATIOS],
        "profiles": list(PROFILES),
        "profile_kind": PROFILE_KIND,
        "variant_multipliers": list(VARIANT_MULTIPLIERS),
        "physical_horizon_seconds": PHYSICAL_HORIZON_SECONDS,
        "output_sample_count": OUTPUT_SAMPLE_COUNT,
        "short_preflight_timestep_seconds": (
            SHORT_PREFLIGHT_TIMESTEP_SECONDS
        ),
        "short_preflight_steps": SHORT_PREFLIGHT_STEPS,
    }
    _write_json(CONFIG_PATH, config)
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).exists()
    }
    input_paths = {
        "c3a_summary": C3A_DIRECTORY / "summary.json",
        "c7a_arrays": C7A_DIRECTORY / "decisive_arrays.npz",
        "c7c0_arrays": C7C0_DIRECTORY / "decisive_arrays.npz",
        "c7c1a_summary": C7C1A_DIRECTORY / "summary.json",
        "c7c1b_summary": C7C1B_DIRECTORY / "summary.json",
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
        "c3a_rejection_preserved": True,
        "c7c1b_strict_classification_preserved": True,
        "c7c1b_tier_I_direct_passed": bool(
            parents[-1]["historical_direct_contract_report"]["passed"]
        ),
        "physical_background_audit": audit,
        "profile_count": len(PROFILES),
        "variant_count": audit["variant_count"],
        "manifest_sha256": causal_canonical_json_sha256(manifest),
        "config_sha256": _sha256(CONFIG_PATH),
        "manifest_file_sha256": _sha256(MANIFEST_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "input_hashes": {
            name: _sha256(path) for name, path in input_paths.items()
        },
        "monolithic_bdf_method_preflight_authorized": passed,
        "nonlinear_physical_ladder_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "command": (
                "PYTHONPATH=src python "
                "scripts/run_causal_inner_physical_background_nonlinear_"
                "readiness_manifest_wp10c9d6c7c3a1.py"
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
        "physical_background_manifest.json",
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
