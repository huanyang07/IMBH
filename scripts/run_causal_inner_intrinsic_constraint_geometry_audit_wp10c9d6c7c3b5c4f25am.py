#!/usr/bin/env python3
"""Audit an intrinsic minimum-norm fixed-Q state chart."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import null_space


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_intrinsic_constraint_geometry_manifest_wp10c9d6c7c3b5c4f25al as manifest  # noqa: E402
import run_causal_inner_nonlinear_bundle_screen_wp10c9d6c7c3b5c4f25ak as screen_tools  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    evaluate_causal_five_field_monolithic_backward_euler,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25am"
MANIFEST_COMMIT = "894922c42b4c6835102401b1ab16b662c23ee058"
MANIFEST_PARENT = "b0c4db89143785990997e0908aa85d76a911a434"
MANIFEST_TREE = "394f23047c969bcb4b45d2a59ae64ecf585c29c2"

PASS_CLASSIFICATION = (
    "intrinsic_constraint_geometry_passed_"
    "equilibrium_centered_slow_fast_hybrid_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "intrinsic_constraint_geometry_failed_reduced_architecture_blocked"
)

ARTIFACT = (
    "causal_inner_intrinsic_constraint_geometry_audit_"
    "wp10c9d6c7c3b5c4f25am"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_intrinsic_constraint_geometry_audit_"
    "wp10c9d6c7c3b5c4f25am.py"
)
THIS_TEST = (
    "tests/test_causal_inner_intrinsic_constraint_geometry_audit_"
    "wp10c9d6c7c3b5c4f25am.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_INTRINSIC_CONSTRAINT_GEOMETRY_"
    "AUDIT_WP10C9D6C7C3B5C4F25AM_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("intrinsic-geometry manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("intrinsic-geometry manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("intrinsic-geometry manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["coordinate_geometry"][
            "physical_reaction_lift_used_for_state_retraction"
        ]
        or contract["saved_generator_diagnostics"][
            "instantaneous_eigenvalues_are_normal_hyperbolicity_certificate"
        ]
    ):
        raise RuntimeError("intrinsic-geometry execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("intrinsic-geometry audit requires a clean tracked tree")
    for name, expected in manifest.parent.manifest.parent.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _orthogonal_geometry(constraint: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(constraint, dtype=float)
    gram = rows @ rows.T
    normal = rows.T @ np.linalg.solve(gram, np.eye(rows.shape[0]))
    projector = np.eye(rows.shape[1]) - normal @ rows
    tangent = null_space(rows)
    return {
        "constraint": rows,
        "gram": gram,
        "normal": normal,
        "projector": projector,
        "tangent": tangent,
    }


def _retract(
    data: dict,
    base_reaction,
    normal: np.ndarray,
    scaled_trial: np.ndarray,
    component_bound: float,
) -> tuple[np.ndarray, dict]:
    delta = np.asarray(scaled_trial, dtype=float).ravel().copy()
    q0 = np.asarray(base_reaction.q3_value, dtype=float)
    qscale = np.asarray(base_reaction.q3_derivative_norms, dtype=float)
    face = 36 * int(data["layout"].refinement_ratio)
    factors = None
    iterations = 0
    for _outer in range(8):
        for _inner in range(16):
            state = data["state"] + (data["columns"].ravel() * delta).reshape(
                data["state"].shape
            )
            q3, factors = causal_five_field_exterior_q3(
                data["context"], state, exterior_face_index=face
            )
            error = (np.asarray(q3) - q0) / qscale
            iterations += 1
            if float(np.max(np.abs(error))) <= 1.0e-12:
                break
            delta -= normal @ error
        maximum = float(np.max(np.abs(delta)))
        if maximum <= component_bound * (1.0 + 1.0e-12):
            break
        delta *= component_bound / maximum
    state = data["state"] + (data["columns"].ravel() * delta).reshape(
        data["state"].shape
    )
    q3, factors = causal_five_field_exterior_q3(
        data["context"], state, exterior_face_index=face
    )
    defect = float(np.max(np.abs((np.asarray(q3) - q0) / qscale)))
    return state, {
        "normalized_Q3_defect": defect,
        "iterations": iterations,
        "scaled_delta": delta,
        "minimum_Q3_reconstruction_factor": float(np.min(factors)),
        "maximum_Q3_reconstruction_factor": float(np.max(factors)),
    }


def _anchor_generator(anchor: str) -> np.ndarray:
    if anchor == "primary":
        path = screen_tools.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz"
    else:
        path = screen_tools.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz"
    with np.load(path) as source:
        return np.asarray(source["complete_fixed_Q_generator"], dtype=float)


def _anchor_audit(anchor: str, fiber: dict, gates: dict) -> tuple[dict, dict]:
    began = time.perf_counter()
    data = screen_tools._anchor_data(anchor)
    base_rate, reaction, evaluation, physical, timing = (
        screen_tools._continuous_fixed_q_rate(data, data["state"])
    )
    generator = _anchor_generator(anchor)
    geometry = _orthogonal_geometry(reaction.q3_scaled_derivative)
    constraint = geometry["constraint"]
    normal = geometry["normal"]
    projector = geometry["projector"]
    tangent = geometry["tangent"]
    intrinsic = tangent.T @ generator @ tangent
    intrinsic_poles = np.linalg.eigvals(intrinsic)
    old = np.asarray(fiber[f"{anchor}_right_basis"], dtype=float)
    projected_old = projector @ old
    projected_rank = int(np.linalg.matrix_rank(projected_old))
    projected_condition = float(np.linalg.cond(projected_old))
    projected_basis = np.linalg.qr(projected_old, mode="reduced")[0]
    projected_operator = projected_basis.T @ generator @ projected_basis
    directions, energy_growth = screen_tools._energy_directions(
        projected_operator, manifest.PROJECTED_FIBER_DIRECTIONS
    )
    q_defects = []
    component_changes = []
    minimum_factors = []
    maximum_factors = []
    h_over_r = []
    optical_depth = []
    incoming = []
    iterations = []
    saved_deltas = []
    for direction_index in range(directions.shape[1]):
        lifted = projected_basis @ directions[:, direction_index]
        radius = manifest.MAXIMUM_COMPONENT_AMPLITUDE / float(
            np.max(np.abs(lifted))
        )
        for sign in (-1.0, 1.0):
            state, retraction = _retract(
                data,
                reaction,
                normal,
                sign * radius * lifted,
                manifest.MAXIMUM_COMPONENT_AMPLITUDE,
            )
            audit = screen_tools._state_audit(data["context"], state)
            endpoint = evaluate_causal_five_field_monolithic_backward_euler(
                state, state, 1.0, data["context"], path_quadrature_order=6
            )
            q_defects.append(retraction["normalized_Q3_defect"])
            component_changes.append(
                float(np.max(np.abs(retraction["scaled_delta"])))
            )
            minimum_factors.append(
                min(
                    retraction["minimum_Q3_reconstruction_factor"],
                    audit["minimum_reconstruction_factor"],
                )
            )
            maximum_factors.append(retraction["maximum_Q3_reconstruction_factor"])
            h_over_r.append(audit["maximum_h_over_r"])
            optical_depth.append(audit["minimum_scattering_optical_depth"])
            incoming.append(endpoint.incoming_excision_characteristics)
            iterations.append(retraction["iterations"])
            saved_deltas.append(retraction["scaled_delta"])
    identity = np.eye(projector.shape[0])
    metrics = {
        "constraint_rank": int(np.linalg.matrix_rank(constraint)),
        "constraint_gram_condition_number": float(np.linalg.cond(geometry["gram"])),
        "minimum_norm_normal_spectral_norm": float(np.linalg.norm(normal, 2)),
        "physical_reaction_lift_spectral_norm": float(
            np.linalg.norm(reaction.reaction_lift, 2)
        ),
        "reaction_to_minimum_norm_normal_amplification": float(
            np.linalg.norm(reaction.reaction_lift, 2) / np.linalg.norm(normal, 2)
        ),
        "projector_idempotence_defect": float(
            np.max(np.abs(projector @ projector - projector))
        ),
        "projector_symmetry_defect": float(np.max(np.abs(projector.T - projector))),
        "projector_complement_defect": float(
            np.max(np.abs(projector + normal @ constraint - identity))
        ),
        "tangent_annihilation_defect": float(np.max(np.abs(constraint @ tangent))),
        "tangent_basis_orthogonality_defect": float(
            np.max(np.abs(tangent.T @ tangent - np.eye(tangent.shape[1])))
        ),
        "intrinsic_tangent_dimension": int(tangent.shape[1]),
        "intrinsic_instantaneous_positive_real_part_count_diagnostic": int(
            np.count_nonzero(np.real(intrinsic_poles) > 0.0)
        ),
        "intrinsic_instantaneous_spectral_abscissa_per_second_diagnostic": float(
            np.max(np.real(intrinsic_poles))
        ),
        "intrinsic_instantaneous_minimum_real_part_per_second_diagnostic": float(
            np.min(np.real(intrinsic_poles))
        ),
        "projected_old_fiber_rank": projected_rank,
        "projected_old_fiber_condition_number": projected_condition,
        "old_fiber_tangent_projection_relative_change": float(
            np.linalg.norm(projected_old - old)
            / max(float(np.linalg.norm(old)), np.finfo(float).tiny)
        ),
        "projected_old_fiber_positive_real_part_count_diagnostic": int(
            np.count_nonzero(np.real(np.linalg.eigvals(projected_operator)) > 0.0)
        ),
        "projected_old_fiber_spectral_abscissa_per_second_diagnostic": float(
            np.max(np.real(np.linalg.eigvals(projected_operator)))
        ),
        "minimum_projected_energy_growth_per_second_diagnostic": float(
            np.min(energy_growth)
        ),
        "maximum_projected_energy_growth_per_second_diagnostic": float(
            np.max(energy_growth)
        ),
        "base_rate_norm_per_second": float(np.linalg.norm(base_rate)),
        "base_rate_tangency_relative_defect": float(
            np.linalg.norm(constraint @ base_rate)
            / max(float(np.linalg.norm(base_rate)), np.finfo(float).tiny)
        ),
        "base_incoming_excision_characteristics": int(
            evaluation.incoming_excision_characteristics
        ),
        "maximum_normalized_Q3_retraction_defect": float(np.max(q_defects)),
        "maximum_scaled_component_perturbation": float(np.max(component_changes)),
        "minimum_reconstruction_factor": float(np.min(minimum_factors)),
        "maximum_reconstruction_factor": float(np.max(maximum_factors)),
        "maximum_H_over_R": float(np.max(h_over_r)),
        "minimum_scattering_optical_depth": float(np.min(optical_depth)),
        "maximum_incoming_excision_characteristics": int(np.max(incoming)),
        "maximum_retraction_iterations": int(np.max(iterations)),
        "base_continuous_rate_wall_seconds": timing[
            "total_continuous_rate_wall_seconds"
        ],
        "audit_wall_seconds": time.perf_counter() - began,
    }
    metrics["passed"] = bool(
        metrics["constraint_rank"] == gates["constraint_rank_equal"]
        and metrics["constraint_gram_condition_number"]
        <= gates["constraint_gram_condition_number_max"]
        and metrics["minimum_norm_normal_spectral_norm"]
        <= gates["normal_spectral_norm_max"]
        and metrics["projector_idempotence_defect"]
        <= gates["projector_idempotence_defect_max"]
        and metrics["projector_symmetry_defect"]
        <= gates["projector_symmetry_defect_max"]
        and metrics["tangent_annihilation_defect"]
        <= gates["tangent_annihilation_defect_max"]
        and metrics["tangent_basis_orthogonality_defect"]
        <= gates["tangent_basis_orthogonality_defect_max"]
        and metrics["projected_old_fiber_rank"]
        == gates["projected_old_fiber_rank_equal"]
        and metrics["projected_old_fiber_condition_number"]
        <= gates["projected_old_fiber_condition_number_max"]
        and metrics["maximum_normalized_Q3_retraction_defect"]
        <= gates["maximum_normalized_Q3_retraction_defect"]
        and metrics["maximum_scaled_component_perturbation"]
        <= gates["maximum_scaled_component_perturbation"] * (1.0 + 1.0e-9)
        and metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"]
        and metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"]
        and metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"]
        and metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"]
        and metrics["base_incoming_excision_characteristics"]
        == gates["incoming_excision_characteristics_equal"]
        and metrics["maximum_incoming_excision_characteristics"]
        == gates["incoming_excision_characteristics_equal"]
    )
    arrays = {
        "constraint_rows": constraint,
        "minimum_norm_normal": normal,
        "projected_old_fiber_basis": projected_basis,
        "projected_old_fiber_operator": projected_operator,
        "intrinsic_instantaneous_poles": intrinsic_poles,
        "projected_energy_directions": directions,
        "projected_energy_growth_per_second": energy_growth,
        "retracted_scaled_deltas": np.asarray(saved_deltas),
        "retraction_Q3_defects": np.asarray(q_defects),
    }
    return metrics, arrays


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
                    "sha256": _sha(path),
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
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("intrinsic-geometry audit is already canonicalized")
    began = time.perf_counter()
    with np.load(manifest.parent.manifest.FIBER_DIRECTORY / "decisive_fibers.npz") as source:
        fiber = {name: np.asarray(source[name]) for name in source.files}
    gates = frozen["contract"]["binding_gates"]
    anchors = {}
    arrays = {}
    for anchor in manifest.ANCHORS:
        anchors[anchor], local = _anchor_audit(anchor, fiber, gates)
        for name, value in local.items():
            arrays[f"{anchor}_{name}"] = value
    passed = all(anchors[name]["passed"] for name in manifest.ANCHORS)
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_constrained_equilibrium_branch_and_fast_transition_collocation_manifest"
        if passed
        else None
    )
    metrics = {
        "anchors": anchors,
        "all_binding_geometry_gates_passed": passed,
        "instantaneous_spectrum_is_diagnostic_only": True,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "nonlinear_rate_samples_away_from_base": 0,
        "propagated_states": 0,
        "total_wall_seconds": time.perf_counter() - began,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "intrinsic_constraint_geometry_passed": passed,
        "reaction_lift_state_chart_rejection_preserved": True,
        "instantaneous_spectrum_is_normal_hyperbolicity_certificate": False,
        "selected_architecture": (
            "equilibrium_centered_conservative_slow_fast_hybrid"
            if passed
            else None
        ),
        "authorized_next": authorized_next,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "intrinsic_geometry.npz", **arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "reaction_chart_rejection_hashes": _checksums(manifest.PARENT_DIRECTORY),
            "fiber_package_hashes": _checksums(
                manifest.parent.manifest.FIBER_DIRECTORY
            ),
        },
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                manifest.THIS_RUNNER: _sha(ROOT / manifest.THIS_RUNNER),
                manifest.THIS_TEST: _sha(ROOT / manifest.THIS_TEST),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": manifest.parent.manifest.parent.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    lines = [
        "# Intrinsic constraint-geometry audit WP10c9d6c7c3b5c4f25am",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        "The physical reaction operator remains certified for rate enforcement but rejected as a finite-amplitude state chart. The minimum-norm orthogonal DQ3 chart was audited independently.",
        "",
    ]
    for anchor in manifest.ANCHORS:
        item = anchors[anchor]
        lines.extend(
            (
                f"## {anchor}",
                "",
                f"Normal norm `{item['minimum_norm_normal_spectral_norm']:.6e}` versus reaction-lift norm `{item['physical_reaction_lift_spectral_norm']:.6e}`; amplification `{item['reaction_to_minimum_norm_normal_amplification']:.6e}`.",
                "",
                f"The maximum Q3 retraction defect is `{item['maximum_normalized_Q3_retraction_defect']:.6e}`. The old 28-space changes by `{item['old_fiber_tangent_projection_relative_change']:.6e}` under tangent projection and retains rank `{item['projected_old_fiber_rank']}`.",
                "",
                f"The intrinsic 557-dimensional instantaneous operator has `{item['intrinsic_instantaneous_positive_real_part_count_diagnostic']}` positive-real-part eigenvalues and spectral abscissa `{item['intrinsic_instantaneous_spectral_abscissa_per_second_diagnostic']:.6e} s^-1`; these are diagnostic because the anchor rate norm is `{item['base_rate_norm_per_second']:.6e} s^-1`, not an equilibrium.",
                "",
            )
        )
    lines.extend(
        (
            "## Decision",
            "",
            "A reduced cycle must be centered on constrained fast equilibria (or invariant branch states), not on instantaneous eigenvalues of moving checkpoints. Normal hyperbolicity, memory reduction, event surfaces, and transition maps must be rebuilt at those branch anchors in intrinsic coordinates.",
            "",
            f"Authorized next artifact: `{authorized_next}`. No online solver or predictive cycle is authorized.",
            "",
        )
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
