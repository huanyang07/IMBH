#!/usr/bin/env python3
"""Audit the entropy-complete projected seven-field local architecture.

This runner performs no spatial step and advances no trajectory.  It reuses
the frozen Stage-2 state envelope, embeds every base chart at hydrostatic
height, evaluates every prospective off-equilibrium stencil, and binds the
complete reduced 7x7 radial quasilinear pencil at every point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_projected_architecture_correction_manifest_wp10c9d6c7c3b5c4f25fized1 as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    audit_generalized_maxwell_cattaneo_source_ledger,
    audit_specialized_nonlinear_causality,
    generalized_maxwell_cattaneo_principal,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "entropy_complete_projected_local_structural_audit_passed"
)
CAUSALITY_FAILURE = (
    "entropy_complete_projected_reference_or_light_cone_causality_failed"
)
HYPERBOLICITY_FAILURE = (
    "entropy_complete_projected_strong_hyperbolicity_failed"
)
LEDGER_FAILURE = "entropy_complete_projected_energy_entropy_ledger_failed"
DERIVATION_FAILURE = "entropy_complete_projected_derivation_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizef_"
    "entropy_complete_path_conservative_spatial_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_projected_local_structural_audit_"
    "wp10c9d6c7c3b5c4f25fizee"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_PROJECTED_"
    "LOCAL_STRUCTURAL_AUDIT_WP10C9D6C7C3B5C4F25FIZEE_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_projected_local_structural_"
    "audit_wp10c9d6c7c3b5c4f25fizee.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_projected_local_structural_"
    "audit_wp10c9d6c7c3b5c4f25fizee.py"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
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
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {
            "real": float(np.real(number)),
            "imaginary": float(np.imag(number)),
        }
    return value


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _validate_parent(*, require_clean: bool) -> dict:
    hashes = parent.parent._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = parent.parent._read_json(
        parent.CANONICAL_DIRECTORY / "summary.json"
    )
    contract = parent.parent._read_json(
        parent.CANONICAL_DIRECTORY / "architecture_contract.json"
    )
    provenance = parent.parent._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["local_structural_audit_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
        or contract["causality_and_hyperbolicity_standard"]["binding_object"]
        != "complete_reduced_7_by_7_radial_quasilinear_pencil"
    ):
        raise RuntimeError("corrected architecture authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if parent.parent._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"corrected architecture source changed: {relative}")
    stage2 = parent.parent.parent.parent
    envelope_hashes = stage2._validate_checksums(stage2.CANONICAL_DIRECTORY)
    if require_clean and parent.parent._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("local structural audit requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
        "stage2": stage2,
        "envelope_hashes": envelope_hashes,
    }


def _base_entries(
    arrays: dict[str, np.ndarray],
    *,
    centers: np.ndarray,
    failed_face_radius: float,
) -> list[tuple[str, str, int, float, np.ndarray]]:
    entries: list[tuple[str, str, int, float, np.ndarray]] = []

    def append_profile(label: str, profile: np.ndarray) -> None:
        for cell, chart in enumerate(np.asarray(profile, dtype=float)):
            entries.append(
                (f"{label}_cell_{cell:03d}", label, cell, float(centers[cell]), chart)
            )

    append_profile("primary_20ms", arrays["primary_20ms_base_charts5"])
    append_profile("heldout_16ms", arrays["heldout_16ms_base_charts5"])
    accepted = np.asarray(arrays["accepted_trajectory_base_charts5"], dtype=float)
    for endpoint, profile in enumerate(accepted):
        append_profile(f"accepted_{endpoint:02d}", profile)
    append_profile(
        "rejected_full_step", arrays["rejected_full_step_base_charts5"]
    )
    entries.append(
        (
            "old_failed_face_003",
            "old_failed_face",
            3,
            float(failed_face_radius),
            np.asarray(arrays["failed_face_chart5"], dtype=float),
        )
    )
    if len(entries) != 8401:
        raise RuntimeError("frozen base-envelope count changed")
    return entries


def _chart7_at_equilibrium(context, radius: float, chart5: np.ndarray):
    radial = parent.parent.parent.boundary_diagnostic.radial
    old_state = radial._cell_state(context, float(radius), chart5)
    chart7 = np.concatenate(
        (
            np.asarray(chart5, dtype=float),
            [np.log(old_state.thermodynamics.proper_half_thickness), 0.0],
        )
    )
    return old_state, chart7


def _principal_at(context, radius: float, chart7: np.ndarray, geometry):
    return generalized_maxwell_cattaneo_principal(
        geometry,
        chart7,
        proper_vertical_frequency=float(
            context.vertical_frequency.frequency(float(radius))
        ),
        alpha=float(context.alpha),
        stress_factor=float(context.stress_factor),
    )


def _advective_basis(principal) -> tuple[np.ndarray, float]:
    state = principal.local_state
    transport = float(
        state.conservative_flux6_over_c[4] / state.conservative_state6[4]
    )
    eigenvalues = np.real(principal.eigenvalues_over_c)
    selected = np.argsort(np.abs(eigenvalues - transport))[:3]
    maximum_gap = float(np.max(np.abs(eigenvalues[selected] - transport)))
    physical_vectors = (
        principal.primitive_column_scales[:, None]
        * principal.right_eigenvectors_scaled[:, selected]
    )
    common_scales = np.asarray(
        [1.0, 0.1, 0.1, 1.0, 1.0e-4, 1.0, 0.03], dtype=float
    )
    dimensionless = physical_vectors / common_scales[:, None]
    basis, _ = np.linalg.qr(dimensionless)
    return basis[:, :3], maximum_gap


def _point_metrics(principal, *, alpha: float) -> tuple[dict, tuple[str, ...]]:
    state = principal.local_state
    reference = audit_specialized_nonlinear_causality(state)
    ledger = audit_generalized_maxwell_cattaneo_source_ledger(
        state, alpha=alpha
    )
    dominant_margin = float(
        state.specific_enthalpy_over_c2 - abs(float(state.chart[4]))
    )
    metrics = {
        "maximum_imaginary_speed_over_c": principal.maximum_imaginary_speed_over_c,
        "maximum_light_cone_excess_over_c": principal.maximum_light_cone_excess_over_c,
        "maximum_eigenpair_relative_defect": principal.maximum_eigenpair_relative_defect,
        "eigenvector_condition_number": principal.eigenvector_condition_number,
        "maximum_biorthogonality_defect": principal.maximum_biorthogonality_defect,
        "maximum_projector_idempotence_defect": principal.maximum_projector_idempotence_defect,
        "scaled_temporal_condition_number": principal.scaled_temporal_condition_number,
        "four_velocity_normalization_relative_defect": state.four_velocity_normalization_relative_defect,
        "shear_tensor_trace_relative_defect": state.shear_tensor_trace_relative_defect,
        "shear_tensor_orthogonality_relative_defect": state.shear_tensor_orthogonality_relative_defect,
        "shear_radial_work_relative_defect": state.shear_radial_work_relative_defect,
        "reference_causality_minimum_margin": reference.minimum_margin,
        "dominant_energy_margin": dominant_margin,
        "vertical_total_energy_relative_defect": ledger.vertical_total_energy_relative_defect,
        "vertical_reversible_exchange_relative_defect": ledger.vertical_reversible_exchange_relative_defect,
        "minimum_entropy_production_rate": ledger.minimum_entropy_production_rate,
    }
    reasons = []
    if metrics["maximum_imaginary_speed_over_c"] > 1.0e-10:
        reasons.append("strong_hyperbolicity:complex_speed")
    if metrics["maximum_light_cone_excess_over_c"] > 1.0e-10:
        reasons.append("causality:light_cone")
    if metrics["maximum_eigenpair_relative_defect"] > 1.0e-8:
        reasons.append("strong_hyperbolicity:eigenpair")
    if metrics["eigenvector_condition_number"] > 1.0e8:
        reasons.append("strong_hyperbolicity:eigenbasis_condition")
    if metrics["scaled_temporal_condition_number"] > 1.0e8:
        reasons.append("derivation:temporal_condition")
    if max(
        metrics["maximum_biorthogonality_defect"],
        metrics["maximum_projector_idempotence_defect"],
    ) > 1.0e-8:
        reasons.append("strong_hyperbolicity:projector")
    if max(
        metrics["four_velocity_normalization_relative_defect"],
        metrics["shear_tensor_trace_relative_defect"],
        metrics["shear_tensor_orthogonality_relative_defect"],
        metrics["shear_radial_work_relative_defect"],
    ) > 1.0e-10:
        reasons.append("derivation:physical_tensor_constraint")
    if metrics["reference_causality_minimum_margin"] < 1.0e-8:
        reasons.append("causality:full_tensor_reference")
    if metrics["dominant_energy_margin"] < 1.0e-8:
        reasons.append("causality:dominant_energy")
    if max(
        metrics["vertical_total_energy_relative_defect"],
        metrics["vertical_reversible_exchange_relative_defect"],
    ) > 1.0e-10:
        reasons.append("ledger:vertical_energy")
    if metrics["minimum_entropy_production_rate"] < 0.0:
        reasons.append("ledger:entropy_sign")
    return metrics, tuple(reasons)


def _update_extreme(
    extremes: dict[str, dict],
    metrics: dict,
    label: str,
) -> None:
    minima = {
        "reference_causality_minimum_margin",
        "dominant_energy_margin",
        "minimum_entropy_production_rate",
    }
    for key, raw in metrics.items():
        value = float(raw)
        if key not in extremes or (
            value < extremes[key]["value"]
            if key in minima
            else value > extremes[key]["value"]
        ):
            extremes[key] = {"value": value, "label": label}


def _derivative_ladder(context, radius: float, chart5: np.ndarray) -> tuple[dict, dict]:
    old_state, chart7 = _chart7_at_equilibrium(context, radius, chart5)
    factors = np.asarray([2.0, 1.0, 0.5], dtype=float)
    pencils = [
        generalized_maxwell_cattaneo_principal(
            old_state.geometry,
            chart7,
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(radius)
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            derivative_step_factor=float(factor),
        )
        for factor in factors
    ]
    temporal = np.asarray([item.temporal_matrix for item in pencils])
    radial = np.asarray([item.radial_matrix for item in pencils])
    eigenvalues = np.asarray([item.eigenvalues_over_c for item in pencils])
    defects = {}
    for name, matrices in (("temporal", temporal), ("radial", radial)):
        scale = max(float(np.linalg.norm(matrices[-1], ord=np.inf)), 1.0)
        defects[f"{name}_coarse_middle_relative_defect"] = float(
            np.linalg.norm(matrices[0] - matrices[1], ord=np.inf) / scale
        )
        defects[f"{name}_middle_fine_relative_defect"] = float(
            np.linalg.norm(matrices[1] - matrices[2], ord=np.inf) / scale
        )
    return defects, {
        "factors": factors,
        "chart7": chart7,
        "temporal": temporal,
        "radial": radial,
        "eigenvalues": eigenvalues,
    }


def _classification(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return PASS_CLASSIFICATION
    if any(reason.startswith("ledger:") for reason in reasons):
        return LEDGER_FAILURE
    if any(reason.startswith("causality:") for reason in reasons):
        return CAUSALITY_FAILURE
    if any(reason.startswith("strong_hyperbolicity:") for reason in reasons):
        return HYPERBOLICITY_FAILURE
    return DERIVATION_FAILURE


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    stage2 = validated["stage2"]
    envelope_meta = parent.parent._read_json(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.json"
    )
    with np.load(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.npz",
        allow_pickle=False,
    ) as archive:
        envelope = {name: np.array(archive[name], copy=True) for name in archive.files}

    source = parent.parent.parent.boundary_diagnostic.manifest.parent.engine.execution.source
    start = time.perf_counter()
    inputs = source._initial_inputs()
    context = inputs["base"]["configuration"]["context"]
    context_seconds = time.perf_counter() - start
    centers = np.asarray(context.grid.centers, dtype=float)
    entries = _base_entries(
        envelope,
        centers=centers,
        failed_face_radius=float(envelope_meta["failed_face_radius_cm"]),
    )

    extremes: dict[str, dict] = {}
    base_eigenvalues = np.empty((len(entries), 7), dtype=complex)
    base_radii = np.empty(len(entries), dtype=float)
    base_conditions = np.empty(len(entries), dtype=float)
    radius_by_chart: dict[bytes, float] = {}
    minimum_neighbor_cosine = 1.0
    minimum_neighbor_pair = ("", "")
    previous_segment = None
    previous_label = None
    previous_basis = None
    maximum_advective_gap = 0.0
    first_failure: dict | None = None
    audited_base = 0

    for index, (label, segment, _cell, radius, chart5) in enumerate(entries):
        old_state, chart7 = _chart7_at_equilibrium(context, radius, chart5)
        principal = _principal_at(context, radius, chart7, old_state.geometry)
        metrics, reasons = _point_metrics(principal, alpha=float(context.alpha))
        _update_extreme(extremes, metrics, label)
        basis, advective_gap = _advective_basis(principal)
        maximum_advective_gap = max(maximum_advective_gap, advective_gap)
        if previous_segment == segment and previous_basis is not None:
            singular = np.linalg.svd(
                previous_basis.conj().T @ basis,
                compute_uv=False,
            )
            cosine = float(np.min(np.clip(np.real(singular), 0.0, 1.0)))
            if cosine < minimum_neighbor_cosine:
                minimum_neighbor_cosine = cosine
                minimum_neighbor_pair = (str(previous_label), label)
            if cosine < 0.90:
                reasons = tuple(reasons) + (
                    "strong_hyperbolicity:advective_subspace_continuity",
                )
        previous_segment = segment
        previous_label = label
        previous_basis = basis
        base_eigenvalues[index] = principal.eigenvalues_over_c
        base_radii[index] = radius
        base_conditions[index] = principal.eigenvector_condition_number
        radius_by_chart.setdefault(np.asarray(chart5, dtype=float).tobytes(), radius)
        audited_base += 1
        if reasons:
            first_failure = {
                "label": label,
                "scope": "base",
                "radius_cm": radius,
                "chart7": chart7,
                "reasons": reasons,
                "metrics": metrics,
                "eigenvalues_over_c": principal.eigenvalues_over_c,
            }
            break
        if (index + 1) % 500 == 0:
            print(f"audited base points: {index + 1}/{len(entries)}", flush=True)

    witness_charts: list[np.ndarray] = []
    witness_eigenvalues: list[np.ndarray] = []
    witness_radii: list[float] = []
    witness_labels_out: list[str] = []
    audited_witness = 0
    seen_witnesses: set[tuple[float, bytes]] = set()
    if first_failure is None:
        axis_steps = np.asarray(envelope["axis_perturbation_steps5"], dtype=float)
        witnesses = np.asarray(envelope["witness_charts5"], dtype=float)
        labels = np.asarray(envelope["witness_labels"])
        height_stencil = np.asarray(envelope["height_departure_stencil"], dtype=float)
        vertical_stencil = np.asarray(
            envelope["vertical_velocity_over_c_stencil"], dtype=float
        )
        amplitude_stencil = np.asarray(envelope["stress_amplitude_factors"], dtype=float)
        stress_signs = np.asarray(envelope["stress_signs"], dtype=float)

        def audit_witness(label: str, radius: float, chart7: np.ndarray, geometry) -> bool:
            nonlocal audited_witness, first_failure
            key = (float(radius), np.asarray(chart7, dtype=float).tobytes())
            if key in seen_witnesses:
                return True
            seen_witnesses.add(key)
            principal = _principal_at(context, radius, chart7, geometry)
            metrics, reasons = _point_metrics(principal, alpha=float(context.alpha))
            _update_extreme(extremes, metrics, label)
            witness_charts.append(np.asarray(chart7, dtype=float))
            witness_eigenvalues.append(np.asarray(principal.eigenvalues_over_c))
            witness_radii.append(float(radius))
            witness_labels_out.append(label)
            audited_witness += 1
            if reasons:
                first_failure = {
                    "label": label,
                    "scope": "witness",
                    "radius_cm": radius,
                    "chart7": chart7,
                    "reasons": reasons,
                    "metrics": metrics,
                    "eigenvalues_over_c": principal.eigenvalues_over_c,
                }
                return False
            if audited_witness % 250 == 0:
                print(f"audited witness points: {audited_witness}", flush=True)
            return True

        for witness_index, (raw_label, chart5) in enumerate(zip(labels, witnesses, strict=True)):
            label0 = str(raw_label)
            radius = radius_by_chart.get(np.asarray(chart5, dtype=float).tobytes())
            if radius is None and label0 == "failed_face_003":
                radius = float(envelope_meta["failed_face_radius_cm"])
            if radius is None:
                raise RuntimeError(f"cannot recover frozen witness radius: {label0}")
            old_state, equilibrium_chart7 = _chart7_at_equilibrium(context, radius, chart5)
            if not audit_witness(
                f"witness_{witness_index:02d}_{label0}_equilibrium",
                radius,
                equilibrium_chart7,
                old_state.geometry,
            ):
                break
            for field in range(5):
                for sign in (-1.0, 1.0):
                    perturbed5 = np.array(chart5, copy=True)
                    perturbed5[field] += sign * axis_steps[field]
                    perturbed_state, perturbed7 = _chart7_at_equilibrium(
                        context, radius, perturbed5
                    )
                    if not audit_witness(
                        f"witness_{witness_index:02d}_{label0}_axis_{field}_{sign:+.0f}",
                        radius,
                        perturbed7,
                        perturbed_state.geometry,
                    ):
                        break
                if first_failure is not None:
                    break
            if first_failure is not None:
                break
            equilibrium_principal = _principal_at(
                context, radius, equilibrium_chart7, old_state.geometry
            )
            equilibrium_stress = (
                equilibrium_principal.local_state.equilibrium_specific_stress
            )
            stress_values = [0.0]
            for amplitude in amplitude_stencil:
                if amplitude == 0.0:
                    continue
                for sign in stress_signs:
                    stress_values.append(float(sign * amplitude * equilibrium_stress))
            for height_departure in height_stencil:
                for vertical_velocity in vertical_stencil:
                    for stress in stress_values:
                        chart7 = np.array(equilibrium_chart7, copy=True)
                        chart7[4] = stress
                        chart7[5] += height_departure
                        chart7[6] = vertical_velocity
                        if not audit_witness(
                            (
                                f"witness_{witness_index:02d}_{label0}_"
                                f"H{height_departure:+.3f}_w{vertical_velocity:+.3f}_"
                                f"chi{stress:+.9e}"
                            ),
                            radius,
                            chart7,
                            old_state.geometry,
                        ):
                            break
                    if first_failure is not None:
                        break
                if first_failure is not None:
                    break
            if first_failure is not None:
                break

    derivative_metrics = {}
    derivative_arrays: dict[str, np.ndarray] = {}
    if first_failure is None:
        representative5 = np.asarray(
            envelope["primary_20ms_base_charts5"][36], dtype=float
        )
        failed5 = np.asarray(envelope["failed_face_chart5"], dtype=float)
        for name, radius, chart5 in (
            ("representative", float(centers[36]), representative5),
            (
                "old_failed_face",
                float(envelope_meta["failed_face_radius_cm"]),
                failed5,
            ),
        ):
            metrics, arrays = _derivative_ladder(context, radius, chart5)
            derivative_metrics[name] = metrics
            for key, value in arrays.items():
                derivative_arrays[f"{name}_{key}"] = np.asarray(value)
            maximum = max(metrics.values())
            if maximum > 1.0e-7:
                first_failure = {
                    "label": name,
                    "scope": "derivative_ladder",
                    "radius_cm": radius,
                    "reasons": ("derivation:derivative_ladder",),
                    "metrics": metrics,
                }
                break

    failure_reasons = (
        tuple(first_failure["reasons"]) if first_failure is not None else ()
    )
    classification = _classification(failure_reasons)
    passed = classification == PASS_CLASSIFICATION
    old_failed_index = len(entries) - 1
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "fail_fast": True,
        "base_points_planned": len(entries),
        "base_points_audited": audited_base,
        "witness_points_audited": audited_witness,
        "unique_witness_points": len(seen_witnesses),
        "context_construction_wall_seconds": context_seconds,
        "audit_wall_seconds": time.perf_counter() - start,
        "extremes": extremes,
        "minimum_advective_neighbor_subspace_cosine": minimum_neighbor_cosine,
        "minimum_advective_neighbor_subspace_pair": minimum_neighbor_pair,
        "maximum_advective_eigenvalue_gap": maximum_advective_gap,
        "derivative_ladders": derivative_metrics,
        "old_failed_face": {
            "old_model_maximum_imaginary_speed_over_c": envelope_meta[
                "old_failed_face_maximum_imaginary_speed_over_c"
            ],
            "new_model_eigenvalues_over_c": (
                base_eigenvalues[old_failed_index]
                if audited_base == len(entries)
                else np.full(7, np.nan)
            ),
            "new_model_maximum_imaginary_speed_over_c": (
                float(np.max(np.abs(np.imag(base_eigenvalues[old_failed_index]))))
                if audited_base == len(entries)
                else None
            ),
        },
        "first_failure": first_failure,
        "new_trajectory_steps": 0,
    }
    arrays = {
        "base_radii_cm": base_radii[:audited_base],
        "base_eigenvalues_over_c": base_eigenvalues[:audited_base],
        "base_eigenvector_condition_numbers": base_conditions[:audited_base],
        "witness_radii_cm": np.asarray(witness_radii, dtype=float),
        "witness_charts7": np.asarray(witness_charts, dtype=float).reshape(-1, 7),
        "witness_eigenvalues_over_c": np.asarray(
            witness_eigenvalues, dtype=complex
        ).reshape(-1, 7),
        "witness_labels": np.asarray(witness_labels_out, dtype="U180"),
        **derivative_arrays,
    }
    return _plain(metrics), arrays


def _report(metrics: dict) -> str:
    passed = bool(metrics["passed"])
    extremes = metrics["extremes"]
    failure = metrics["first_failure"]
    decision = (
        f"Authorized next: `{AUTHORIZED_NEXT_ON_PASS}` only."
        if passed
        else "No later package is authorized; the first binding failure must be diagnosed prospectively."
    )
    return "\n".join(
        (
            "# Entropy-complete projected seven-field local structural audit",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The fail-closed audit evaluated {metrics['base_points_audited']} of {metrics['base_points_planned']} frozen base points and {metrics['witness_points_audited']} unique prospective off-equilibrium witness points. No spatial step, nonlinear time root, or trajectory state was constructed.",
            "",
            "The binding object was the complete entropy-current-corrected 7x7 Kerr--Schild radial pencil. The audit retained both temporal and radial shear-rate derivatives, exact physical stress-energy projections, finite-inertia height/vertical momentum, tensor constraints, source energy exchange, and nonnegative extended-entropy production.",
            "",
            f"Worst imaginary speed: `{extremes.get('maximum_imaginary_speed_over_c', {}).get('value')}`. Worst light-cone excess: `{extremes.get('maximum_light_cone_excess_over_c', {}).get('value')}`. Worst eigenvector condition number: `{extremes.get('eigenvector_condition_number', {}).get('value')}`. Minimum reference causality margin: `{extremes.get('reference_causality_minimum_margin', {}).get('value')}`.",
            "",
            f"Minimum neighboring advective-subspace cosine: `{metrics['minimum_advective_neighbor_subspace_cosine']}`. The old five-field failed face now has maximum imaginary speed `{metrics['old_failed_face']['new_model_maximum_imaginary_speed_over_c']}`; it remained nonpropagating.",
            "",
            f"First failure: `{failure}`.",
            "",
            decision,
            "No seven-field trajectory, fixed-Q invariant object, slow-flux atlas, reduced cycle, or complete-cycle execution is authorized by this local result.",
            "",
        )
    )


def _update_catalog(summary: dict, status: str) -> None:
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
                    "sha256": parent.parent._sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = parent.parent._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": parent.parent._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    parent.parent._write_json(CANONICAL_SUMMARY, catalog)


def _execute() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("local structural audit result already exists")
    validated = _validate_parent(require_clean=True)
    metrics, arrays = _audit()
    passed = bool(metrics["passed"])
    authorized_next = AUTHORIZED_NEXT_ON_PASS if passed else None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "audit_completed": True,
        "base_points_audited": metrics["base_points_audited"],
        "witness_points_audited": metrics["witness_points_audited"],
        "complete_reduced_principal_certified": passed,
        "physical_tensor_constraints_certified": passed,
        "source_energy_entropy_ledger_certified": passed,
        "old_failed_face_repaired_by_new_architecture": passed,
        "new_trajectory_steps": 0,
        "spatial_discretization_authorized": False,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True)
    parent.parent._write_json(CANONICAL_DIRECTORY / "audit_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "audit_arrays.npz", arrays)
    parent.parent._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_hashes": validated["hashes"],
            "stage2_artifact": validated["stage2"].ARTIFACT,
            "stage2_envelope_hashes": validated["envelope_hashes"],
            "canonical_sources_only": True,
            "mutable_scratch_used": False,
        },
    )
    parent.parent._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    source_paths = (
        THIS_RUNNER,
        THIS_TEST,
        PHYSICAL_SOURCE,
        PHYSICAL_TEST,
        REPORT_RELATIVE,
    )
    parent.parent._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PASS" if passed else "FAIL",
            "implementation_commit": parent.parent._git("rev-parse", "HEAD"),
            "implementation_tree": parent.parent._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: parent.parent._sha256(ROOT / path) for path in source_paths
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{parent.parent._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary, "PASS" if passed else "FAIL")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        parser.error("choose --execute")
    summary = _execute()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
