#!/usr/bin/env python3
"""Run the invariant-cluster corrected seven-field local audit.

Every frozen base and witness state is checked from scratch.  The complete
pointwise principal, tensor/source gates, and derivative ladders are retained.
The three advective characteristics must remain attached to the exact
material transport speed and uniformly separated from the complementary
cluster.  Coarse neighboring subspace overlap is recorded but nonbinding.
"""

from __future__ import annotations

import argparse
import csv
import json
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

import run_causal_inner_advective_path_continuity_audit_correction_manifest_wp10c9d6c7c3b5c4f25fizee6 as parent  # noqa: E402
import run_causal_inner_entropy_complete_projected_local_structural_audit_wp10c9d6c7c3b5c4f25fizee as frozen_audit  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "invariant_cluster_local_structural_audit_passed"
CAUSALITY_FAILURE = "invariant_cluster_local_structural_audit_causality_failed"
HYPERBOLICITY_FAILURE = (
    "invariant_cluster_local_structural_audit_hyperbolicity_failed"
)
LEDGER_FAILURE = "invariant_cluster_local_structural_audit_ledger_failed"
DERIVATION_FAILURE = "invariant_cluster_local_structural_audit_derivation_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizef_"
    "entropy_complete_path_conservative_spatial_manifest"
)
ARTIFACT = (
    "causal_inner_invariant_cluster_local_structural_audit_"
    "wp10c9d6c7c3b5c4f25fizee7"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_INVARIANT_CLUSTER_LOCAL_"
    "STRUCTURAL_AUDIT_WP10C9D6C7C3B5C4F25FIZEE7_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_invariant_cluster_local_structural_audit_"
    "wp10c9d6c7c3b5c4f25fizee7.py"
)
THIS_TEST = (
    "tests/test_causal_inner_invariant_cluster_local_structural_audit_"
    "wp10c9d6c7c3b5c4f25fizee7.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "c2201f277cad6c03a35d9bde145162fa62ccbe6452a489a8c6b011b5f0a85420"
)
PHYSICAL_SOURCE = parent.PHYSICAL_SOURCE
PHYSICAL_TEST = parent.PHYSICAL_TEST
PHYSICAL_SOURCE_SHA256 = parent.PHYSICAL_SOURCE_SHA256
PHYSICAL_TEST_SHA256 = parent.PHYSICAL_TEST_SHA256
FROZEN_AUDIT_RUNNER = frozen_audit.THIS_RUNNER
FROZEN_AUDIT_RUNNER_SHA256 = (
    "4143bc70e73a673926b192c5f727692c47d9f765e10de91449a436aa3e5354b8"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("invariant audit correction checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "correction_contract.json"
    )
    provenance = utils._read_json(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["all_parent_results_preserved"]
        or not summary["corrected_full_local_audit_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
        or contract["corrected_mathematical_standard"]
        ["coarse_neighbor_subspace_cosine"]
        != "diagnostic_only"
    ):
        raise RuntimeError("invariant audit correction authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"audit correction source changed: {relative}")
    if utils._sha256(ROOT / PHYSICAL_SOURCE) != PHYSICAL_SOURCE_SHA256:
        raise RuntimeError("physical source changed")
    if utils._sha256(ROOT / PHYSICAL_TEST) != PHYSICAL_TEST_SHA256:
        raise RuntimeError("physical test changed")
    if utils._sha256(ROOT / FROZEN_AUDIT_RUNNER) != FROZEN_AUDIT_RUNNER_SHA256:
        raise RuntimeError("frozen audit helper changed")
    stage2 = frozen_audit.parent.parent.parent.parent
    envelope_hashes = utils._validate_checksums(stage2.CANONICAL_DIRECTORY)
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("invariant cluster audit requires clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "stage2": stage2,
        "envelope_hashes": envelope_hashes,
    }


def _cluster_metrics(principal) -> tuple[dict, tuple[str, ...], np.ndarray]:
    eigenvalues = np.asarray(principal.eigenvalues_over_c)
    transport = float(principal.local_state.transport_velocity_over_c)
    real_values = np.real(eigenvalues)
    selected = np.argsort(np.abs(real_values - transport))[:3]
    selected_set = set(int(index) for index in selected)
    complement = np.asarray(
        [index for index in range(7) if index not in selected_set], dtype=int
    )
    maximum_offset = float(np.max(np.abs(real_values[selected] - transport)))
    minimum_gap = float(
        np.min(
            np.abs(
                real_values[selected, None] - real_values[complement][None, :]
            )
        )
    )
    reasons = []
    if maximum_offset > 1.0e-6:
        reasons.append("strong_hyperbolicity:advective_transport_offset")
    if minimum_gap < 1.0e-4:
        reasons.append("strong_hyperbolicity:advective_cluster_gap")
    return (
        {
            "advective_transport_speed_over_c": transport,
            "maximum_advective_cluster_transport_offset_over_c": maximum_offset,
            "minimum_advective_cluster_complement_gap_over_c": minimum_gap,
        },
        tuple(reasons),
        selected,
    )


def _classify(reasons: tuple[str, ...]) -> str:
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
    envelope_meta = _utils()._read_json(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.json"
    )
    with np.load(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.npz", allow_pickle=False
    ) as archive:
        envelope = {name: np.array(archive[name], copy=True) for name in archive.files}

    source = (
        frozen_audit.parent.parent.parent.boundary_diagnostic.manifest.parent.engine.execution.source
    )
    start = time.perf_counter()
    context = source._initial_inputs()["base"]["configuration"]["context"]
    context_seconds = time.perf_counter() - start
    centers = np.asarray(context.grid.centers, dtype=float)
    entries = frozen_audit._base_entries(
        envelope,
        centers=centers,
        failed_face_radius=float(envelope_meta["failed_face_radius_cm"]),
    )

    extremes: dict[str, dict] = {}
    base_eigenvalues = np.empty((len(entries), 7), dtype=complex)
    base_radii = np.empty(len(entries), dtype=float)
    base_conditions = np.empty(len(entries), dtype=float)
    base_cluster_gaps = np.empty(len(entries), dtype=float)
    base_transport_offsets = np.empty(len(entries), dtype=float)
    radius_by_chart: dict[bytes, float] = {}
    minimum_neighbor_cosine = 1.0
    minimum_neighbor_pair = ("", "")
    previous_segment = None
    previous_label = None
    previous_basis = None
    first_failure: dict | None = None
    audited_base = 0

    for index, (label, segment, _cell, radius, chart5) in enumerate(entries):
        old_state, chart7 = frozen_audit._chart7_at_equilibrium(
            context, radius, chart5
        )
        principal = frozen_audit._principal_at(
            context, radius, chart7, old_state.geometry
        )
        metrics, reasons = frozen_audit._point_metrics(
            principal, alpha=float(context.alpha)
        )
        cluster, cluster_reasons, _selected = _cluster_metrics(principal)
        metrics.update(cluster)
        reasons = tuple(reasons) + cluster_reasons
        frozen_audit._update_extreme(extremes, metrics, label)
        basis, _advective_gap = frozen_audit._advective_basis(principal)
        if previous_segment == segment and previous_basis is not None:
            singular = np.linalg.svd(
                previous_basis.conj().T @ basis, compute_uv=False
            )
            cosine = float(np.min(np.clip(np.real(singular), 0.0, 1.0)))
            if cosine < minimum_neighbor_cosine:
                minimum_neighbor_cosine = cosine
                minimum_neighbor_pair = (str(previous_label), label)
        previous_segment = segment
        previous_label = label
        previous_basis = basis
        base_eigenvalues[index] = principal.eigenvalues_over_c
        base_radii[index] = radius
        base_conditions[index] = principal.eigenvector_condition_number
        base_cluster_gaps[index] = cluster[
            "minimum_advective_cluster_complement_gap_over_c"
        ]
        base_transport_offsets[index] = cluster[
            "maximum_advective_cluster_transport_offset_over_c"
        ]
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
    witness_cluster_gaps: list[float] = []
    witness_transport_offsets: list[float] = []
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
        amplitude_stencil = np.asarray(
            envelope["stress_amplitude_factors"], dtype=float
        )
        stress_signs = np.asarray(envelope["stress_signs"], dtype=float)

        def audit_witness(label: str, radius: float, chart7: np.ndarray, geometry) -> bool:
            nonlocal audited_witness, first_failure
            key = (float(radius), np.asarray(chart7, dtype=float).tobytes())
            if key in seen_witnesses:
                return True
            seen_witnesses.add(key)
            principal = frozen_audit._principal_at(
                context, radius, chart7, geometry
            )
            metrics, reasons = frozen_audit._point_metrics(
                principal, alpha=float(context.alpha)
            )
            cluster, cluster_reasons, _selected = _cluster_metrics(principal)
            metrics.update(cluster)
            reasons = tuple(reasons) + cluster_reasons
            frozen_audit._update_extreme(extremes, metrics, label)
            witness_charts.append(np.asarray(chart7, dtype=float))
            witness_eigenvalues.append(np.asarray(principal.eigenvalues_over_c))
            witness_radii.append(float(radius))
            witness_labels_out.append(label)
            witness_cluster_gaps.append(
                cluster["minimum_advective_cluster_complement_gap_over_c"]
            )
            witness_transport_offsets.append(
                cluster["maximum_advective_cluster_transport_offset_over_c"]
            )
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

        for witness_index, (raw_label, chart5) in enumerate(
            zip(labels, witnesses, strict=True)
        ):
            label0 = str(raw_label)
            radius = radius_by_chart.get(np.asarray(chart5, dtype=float).tobytes())
            if radius is None and label0 == "failed_face_003":
                radius = float(envelope_meta["failed_face_radius_cm"])
            if radius is None:
                raise RuntimeError(f"cannot recover frozen witness radius: {label0}")
            old_state, equilibrium_chart7 = frozen_audit._chart7_at_equilibrium(
                context, radius, chart5
            )
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
                    perturbed_state, perturbed7 = frozen_audit._chart7_at_equilibrium(
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
            equilibrium_principal = frozen_audit._principal_at(
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
                    stress_values.append(
                        float(sign * amplitude * equilibrium_stress)
                    )
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
            metrics, arrays = frozen_audit._derivative_ladder(
                context, radius, chart5
            )
            derivative_metrics[name] = metrics
            for key, value in arrays.items():
                derivative_arrays[f"{name}_{key}"] = np.asarray(value)
            if max(metrics.values()) > 1.0e-7:
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
    classification = _classify(failure_reasons)
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
        "minimum_coarse_neighbor_subspace_cosine_diagnostic": minimum_neighbor_cosine,
        "minimum_coarse_neighbor_subspace_pair_diagnostic": minimum_neighbor_pair,
        "minimum_base_cluster_complement_gap_over_c": float(
            np.min(base_cluster_gaps[:audited_base])
        ),
        "maximum_base_cluster_transport_offset_over_c": float(
            np.max(base_transport_offsets[:audited_base])
        ),
        "minimum_witness_cluster_complement_gap_over_c": (
            float(np.min(witness_cluster_gaps)) if witness_cluster_gaps else None
        ),
        "maximum_witness_cluster_transport_offset_over_c": (
            float(np.max(witness_transport_offsets))
            if witness_transport_offsets
            else None
        ),
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
                float(
                    np.max(
                        np.abs(np.imag(base_eigenvalues[old_failed_index]))
                    )
                )
                if audited_base == len(entries)
                else None
            ),
        },
        "first_failure": first_failure,
        "all_parent_results_preserved": True,
        "coarse_neighbor_subspace_cosine_binding": False,
        "new_trajectory_steps": 0,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if passed else None,
    }
    arrays = {
        "base_radii_cm": base_radii[:audited_base],
        "base_eigenvalues_over_c": base_eigenvalues[:audited_base],
        "base_eigenvector_condition_numbers": base_conditions[:audited_base],
        "base_cluster_complement_gaps_over_c": base_cluster_gaps[:audited_base],
        "base_cluster_transport_offsets_over_c": base_transport_offsets[:audited_base],
        "witness_radii_cm": np.asarray(witness_radii, dtype=float),
        "witness_charts7": np.asarray(witness_charts, dtype=float).reshape(-1, 7),
        "witness_eigenvalues_over_c": np.asarray(
            witness_eigenvalues, dtype=complex
        ).reshape(-1, 7),
        "witness_cluster_complement_gaps_over_c": np.asarray(
            witness_cluster_gaps, dtype=float
        ),
        "witness_cluster_transport_offsets_over_c": np.asarray(
            witness_transport_offsets, dtype=float
        ),
        "witness_labels": np.asarray(witness_labels_out, dtype="U180"),
        **derivative_arrays,
    }
    return frozen_audit._plain(metrics), arrays


def _report(metrics: dict) -> str:
    decision = (
        f"Authorized next: `{AUTHORIZED_NEXT_ON_PASS}` only."
        if metrics["passed"]
        else "No later package is authorized; the first invariant audit failure must be diagnosed prospectively."
    )
    return "\n".join(
        (
            "# Invariant-cluster seven-field local structural audit",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"The audit evaluated {metrics['base_points_audited']} of {metrics['base_points_planned']} base states and {metrics['witness_points_audited']} unique off-equilibrium witnesses from the frozen envelope. All parent positive and negative results remain preserved.",
            "",
            f"Minimum base cluster gap: `{metrics['minimum_base_cluster_complement_gap_over_c']}`. Maximum base transport offset: `{metrics['maximum_base_cluster_transport_offset_over_c']}`. Minimum witness cluster gap: `{metrics['minimum_witness_cluster_complement_gap_over_c']}`. Maximum witness transport offset: `{metrics['maximum_witness_cluster_transport_offset_over_c']}`.",
            "",
            f"The minimum coarse neighboring subspace cosine `{metrics['minimum_coarse_neighbor_subspace_cosine_diagnostic']}` at `{metrics['minimum_coarse_neighbor_subspace_pair_diagnostic']}` is recorded as diagnostic only under the prospectively corrected invariant standard. First binding failure: `{metrics['first_failure']}`.",
            "",
            decision,
            "No spatial step, seven-field trajectory, fixed-Q invariant object, slow atlas, reduced cycle, or complete-cycle execution is authorized by this local audit.",
            "",
        )
    )


def _update_catalog(summary: dict, status: str) -> None:
    utils = _utils()
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
                    "sha256": utils._sha256(path),
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
    catalog = utils._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("invariant local audit result already exists")
    utils = _utils()
    parent_data = _validate_parent(require_clean=True)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    frozen_audit._save_npz(CANONICAL_DIRECTORY / "audit_arrays.npz", arrays)
    utils._write_json(CANONICAL_DIRECTORY / "audit_metrics.json", metrics)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": parent_data["hashes"],
            "frozen_envelope_artifact": parent_data["stage2"].ARTIFACT,
            "frozen_envelope_hashes": parent_data["envelope_hashes"],
            "physical_source_sha256": PHYSICAL_SOURCE_SHA256,
            "physical_test_sha256": PHYSICAL_TEST_SHA256,
            "frozen_audit_helper_sha256": FROZEN_AUDIT_RUNNER_SHA256,
        },
    )
    passed = bool(metrics["passed"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": passed,
        "all_parent_results_preserved": True,
        "invariant_cluster_local_audit_completed": True,
        "base_points_audited": metrics["base_points_audited"],
        "witness_points_audited": metrics["witness_points_audited"],
        "complete_reduced_principal_certified": passed,
        "physical_tensor_constraints_certified": passed,
        "source_energy_entropy_ledger_certified": passed,
        "advective_cluster_certified": passed,
        "new_trajectory_steps": 0,
        "spatial_manifest_authorized": passed,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if passed else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    source_paths = (
        THIS_RUNNER,
        THIS_TEST,
        FROZEN_AUDIT_RUNNER,
        PHYSICAL_SOURCE,
        PHYSICAL_TEST,
        REPORT_RELATIVE,
    )
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "PASS" if passed else "FAIL",
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in source_paths
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
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
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
    metrics, arrays = _audit()
    summary = _canonicalize(metrics, arrays)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
