#!/usr/bin/env python3
"""Audit the equilibrium-centered conservative slow-fast hybrid architecture."""

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
from scipy.linalg import eigvalsh, expm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_equilibrium_centered_hybrid_manifest_wp10c9d6c7c3b5c4f25an as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ao"
MANIFEST_COMMIT = "7c38c35c930666a3cce2937092072ae145fc01ec"
MANIFEST_PARENT = "0a710475eb1d37d074783aeaa33d0c2e51f461af"
MANIFEST_TREE = "57345e45059f76736e0867cde9c0e555f779fffa"

PASS_CLASSIFICATION = (
    "equilibrium_centered_conservative_slow_fast_hybrid_architecture_"
    "certified_offline_branch_seed_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "hybrid_architecture_structural_audit_failed_reduced_slow_evolution_blocked"
)

ARTIFACT = (
    "causal_inner_equilibrium_centered_hybrid_architecture_audit_"
    "wp10c9d6c7c3b5c4f25ao"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_equilibrium_centered_hybrid_architecture_audit_"
    "wp10c9d6c7c3b5c4f25ao.py"
)
THIS_TEST = (
    "tests/test_causal_inner_equilibrium_centered_hybrid_architecture_audit_"
    "wp10c9d6c7c3b5c4f25ao.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_CENTERED_HYBRID_"
    "ARCHITECTURE_AUDIT_WP10C9D6C7C3B5C4F25AO_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

HIGH_ORDER_DIRECTORY = manifest.stable.manifest.PARENT_DIRECTORY
STABLE_DIRECTORY = manifest.stable.CANONICAL_DIRECTORY
GEOMETRY_DIRECTORY = manifest.geometry.CANONICAL_DIRECTORY


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
        raise RuntimeError("hybrid architecture manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("hybrid architecture manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("hybrid architecture manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["selected_architecture"]
        != "equilibrium_centered_conservative_slow_fast_hybrid"
        or contract["conditional_fast_branch"]["square_equation_count"]
        != manifest.FULL_STATE_DIMENSION
        or contract["claim_boundary"]["predictive_cycle_authorized"]
    ):
        raise RuntimeError("hybrid architecture execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    _checksums(GEOMETRY_DIRECTORY)
    _checksums(STABLE_DIRECTORY)
    _checksums(HIGH_ORDER_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("hybrid architecture audit requires a clean tracked tree")
    for name, expected in manifest.stable.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _incidence_matrix(cells: int) -> np.ndarray:
    incidence = np.zeros((cells, cells + 1), dtype=float)
    indices = np.arange(cells)
    incidence[indices, indices] = -1.0
    incidence[indices, indices + 1] = 1.0
    return incidence


def _finite_volume_audit() -> tuple[dict, dict[str, np.ndarray]]:
    incidence = _incidence_matrix(manifest.COARSE_CELLS)
    rng = np.random.default_rng(20260818)
    face_flux = rng.normal(
        size=(manifest.CONSERVATIVE_COMPONENTS, manifest.COARSE_CELLS + 1)
    )
    divergence = face_flux @ incidence.T
    boundary_difference = face_flux[:, -1] - face_flux[:, 0]
    telescoped = np.sum(divergence, axis=1)
    absolute_defect = float(np.max(np.abs(telescoped - boundary_difference)))
    scale = max(1.0, float(np.max(np.abs(boundary_difference))))
    return (
        {
            "cell_count": manifest.COARSE_CELLS,
            "face_count": manifest.COARSE_CELLS + 1,
            "conservative_component_count": manifest.CONSERVATIVE_COMPONENTS,
            "incidence_rank": int(np.linalg.matrix_rank(incidence)),
            "global_telescoping_absolute_defect": absolute_defect,
            "global_telescoping_relative_defect": absolute_defect / scale,
        },
        {
            "incidence": incidence,
            "face_flux": face_flux,
            "cell_divergence": divergence,
            "boundary_difference": boundary_difference,
        },
    )


def _reset_geometry_audit() -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(GEOMETRY_DIRECTORY / "intrinsic_geometry.npz") as source:
        geometry_arrays = {name: np.asarray(source[name]) for name in source.files}
    rng = np.random.default_rng(250)
    metrics = {}
    saved = {}
    for anchor in ("primary", "heldout"):
        constraint = geometry_arrays[f"{anchor}_constraint_rows"]
        normal = geometry_arrays[f"{anchor}_minimum_norm_normal"]
        projector = np.eye(manifest.FULL_STATE_DIMENSION) - normal @ constraint
        raw_jump = rng.normal(size=manifest.FULL_STATE_DIMENSION)
        invariant_impulse = 1.0e-4 * rng.normal(
            size=manifest.CONSERVATIVE_COMPONENTS
        )
        reset_jump = projector @ raw_jump + normal @ invariant_impulse
        constraint_defect = constraint @ reset_jump - invariant_impulse
        tangent_augmentation = projector @ raw_jump
        minimum_norm_jump = normal @ invariant_impulse
        metrics[anchor] = {
            "constraint_normal_identity_defect": float(
                np.max(
                    np.abs(
                        constraint @ normal
                        - np.eye(manifest.CONSERVATIVE_COMPONENTS)
                    )
                )
            ),
            "projector_constraint_defect": float(
                np.max(np.abs(constraint @ projector))
            ),
            "reset_constraint_absolute_defect": float(
                np.max(np.abs(constraint_defect))
            ),
            "reset_constraint_relative_defect": float(
                np.max(np.abs(constraint_defect))
                / max(1.0, float(np.max(np.abs(invariant_impulse))))
            ),
            "minimum_norm_jump_norm": float(np.linalg.norm(minimum_norm_jump)),
            "augmented_jump_norm": float(
                np.linalg.norm(minimum_norm_jump + tangent_augmentation)
            ),
        }
        saved[f"{anchor}_invariant_impulse"] = invariant_impulse
        saved[f"{anchor}_reset_jump"] = reset_jump
        saved[f"{anchor}_constraint_defect"] = constraint_defect
    return metrics, saved


def _stable_family() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(HIGH_ORDER_DIRECTORY / "decisive_model.npz") as source:
        model = {name: np.asarray(source[name]) for name in source.files}
    with np.load(STABLE_DIRECTORY / "parametric_diagnostics.npz") as source:
        diagnostics = {name: np.asarray(source[name]) for name in source.files}
    transform = diagnostics["stable_coordinate_transform"]
    operator_0 = model["primary_stable_reduced_operator"]
    metric_0 = model["primary_metric"]
    operator_1 = transform.T @ model["heldout_stable_reduced_operator"] @ transform
    metric_1 = transform.T @ model["heldout_metric"] @ transform
    return (
        {
            "operator_0": operator_0,
            "metric_0": metric_0,
            "generator_0": metric_0 @ operator_0,
            "operator_1": operator_1,
            "metric_1": metric_1,
            "generator_1": metric_1 @ operator_1,
        },
        diagnostics,
    )


def _descriptor_energy_audit() -> tuple[dict, dict[str, np.ndarray]]:
    family, diagnostics = _stable_family()
    timestep = manifest.MINIMUM_AVERAGE_MACROSTEP_SECONDS
    parameters = np.asarray((0.0, 0.5, 1.0))
    amplifications = []
    spectral_abscissae = []
    for parameter in parameters:
        metric = (
            (1.0 - parameter) * family["metric_0"]
            + parameter * family["metric_1"]
        )
        generator = (
            (1.0 - parameter) * family["generator_0"]
            + parameter * family["generator_1"]
        )
        operator = np.linalg.solve(metric, generator)
        propagator = expm(timestep * operator)
        energy_ratio = eigvalsh(
            propagator.T @ metric @ propagator,
            metric,
            subset_by_index=[manifest.ONLINE_CONTINUOUS_DIMENSION - 1] * 2,
        )[0]
        amplifications.append(float(max(0.0, energy_ratio)))
        spectral_abscissae.append(
            float(np.max(np.real(np.linalg.eigvals(operator))))
        )
    stable_metrics = _read(STABLE_DIRECTORY / "metrics.json")
    return (
        {
            "parameters": parameters,
            "macrostep_seconds": timestep,
            "energy_amplification_factors": amplifications,
            "maximum_energy_amplification_factor": max(amplifications),
            "spectral_abscissae_per_second": spectral_abscissae,
            "maximum_spectral_abscissa_per_second": max(spectral_abscissae),
            "inherited_projected_stable_kernel_cycle_wall_seconds": (
                stable_metrics["worst_benchmark"][
                    "projected_exponential_cycle_wall_seconds"
                ]
            ),
            "legacy_coefficients_are_promoted_to_branch_closure": False,
            "role": "method_stability_and_cost_witness_only",
        },
        {
            "parameters": parameters,
            "energy_amplification_factors": np.asarray(amplifications),
            "spectral_abscissae_per_second": np.asarray(spectral_abscissae),
            "stable_parameter_grid": diagnostics["stable_parameter"],
        },
    )


def _timed_median(function, repetitions: int) -> float:
    function()
    samples = []
    for _ in range(repetitions):
        began = time.perf_counter()
        function()
        samples.append(time.perf_counter() - began)
    return float(np.median(samples))


def _online_algebra_cost_audit() -> dict:
    family, _ = _stable_family()
    incidence = _incidence_matrix(manifest.COARSE_CELLS)
    rng = np.random.default_rng(442)
    state = rng.normal(size=manifest.ONLINE_CONTINUOUS_DIMENSION)
    flux_map = rng.normal(
        size=(
            manifest.CONSERVATIVE_COMPONENTS * (manifest.COARSE_CELLS + 1),
            manifest.ONLINE_CONTINUOUS_DIMENSION,
        )
    )
    event_map = rng.normal(size=(4, manifest.ONLINE_CONTINUOUS_DIMENSION))

    def algebra_kernel():
        parameter = 0.371
        operator = (
            (1.0 - parameter) * family["operator_0"]
            + parameter * family["operator_1"]
        )
        rate = operator @ state
        faces = (flux_map @ state).reshape(
            manifest.CONSERVATIVE_COMPONENTS, manifest.COARSE_CELLS + 1
        )
        divergence = faces @ incidence.T
        events = event_map @ state
        return float(rate[0] + divergence[0, 0] + events[0])

    median_seconds = _timed_median(algebra_kernel, 101)
    stable_metrics = _read(STABLE_DIRECTORY / "metrics.json")
    inherited = stable_metrics["worst_benchmark"][
        "projected_exponential_cycle_wall_seconds"
    ]
    overhead = median_seconds * manifest.MAXIMUM_MACROSTEPS
    projected = inherited + overhead
    return {
        "online_continuous_dimension": manifest.ONLINE_CONTINUOUS_DIMENSION,
        "macrostep_count": manifest.MAXIMUM_MACROSTEPS,
        "minimum_average_macrostep_seconds": (
            manifest.MINIMUM_AVERAGE_MACROSTEP_SECONDS
        ),
        "dense_interpolation_flux_and_event_median_wall_seconds_per_step": (
            median_seconds
        ),
        "projected_additional_algebra_cycle_wall_seconds": overhead,
        "inherited_stable_exponential_cycle_wall_seconds": inherited,
        "projected_total_online_algebra_cycle_wall_seconds": projected,
        "projected_total_online_algebra_cycle_wall_days": projected / 86_400.0,
        "projected_fraction_of_three_day_wall_budget": (
            projected / manifest.WALL_BUDGET_SECONDS
        ),
        "online_truth_calls_per_macrostep": 0,
        "offline_database_generation_cost_included": False,
    }


def _dimension_and_coordinate_audit() -> dict:
    high_order = _read(HIGH_ORDER_DIRECTORY / "metrics.json")
    map_metrics = {
        anchor: {
            "resolved_coordinate_map_rank": high_order["base_metrics"][anchor][
                "conservative_map_rank"
            ],
            "resolved_coordinate_map_smallest_singular_value": high_order[
                "base_metrics"
            ][anchor]["conservative_map_smallest_singular_value"],
            "hidden_coordinate_annihilation_defect": high_order["best"][anchor][
                "hidden_conservative_annihilation_defect"
            ],
        }
        for anchor in ("primary", "heldout")
    }
    identities = {
        "full_equals_resolved_plus_hidden": (
            manifest.FULL_STATE_DIMENSION
            == manifest.RESOLVED_DIMENSION + manifest.HIDDEN_DIMENSION
        ),
        "resolved_partition": (
            manifest.RESOLVED_DIMENSION
            == manifest.TRUE_CONSERVATIVE_DIMENSION
            + manifest.CONSTITUTIVE_STORAGE_DIMENSION
            + manifest.EXPLICIT_STABLE_DIMENSION
        ),
        "hidden_partition": (
            manifest.HIDDEN_DIMENSION
            == manifest.STABLE_MEMORY_DIMENSION
            + manifest.ELIMINATED_EVENT_DIMENSION
            + manifest.TRUNCATED_STABLE_DIMENSION
        ),
        "online_partition": (
            manifest.ONLINE_CONTINUOUS_DIMENSION
            == manifest.RESOLVED_DIMENSION + manifest.STABLE_MEMORY_DIMENSION
        ),
        "conditional_branch_square": (
            manifest.RESOLVED_DIMENSION + manifest.HIDDEN_DIMENSION
            == manifest.FULL_STATE_DIMENSION
        ),
    }
    return {
        "identities": identities,
        "all_dimension_identities_exact": all(identities.values()),
        "conditional_branch_unknown_count": manifest.FULL_STATE_DIMENSION,
        "conditional_branch_equation_count": (
            manifest.RESOLVED_DIMENSION + manifest.HIDDEN_DIMENSION
        ),
        "resolved_coordinate_maps": map_metrics,
        "both_resolved_coordinate_maps_full_rank": all(
            item["resolved_coordinate_map_rank"] == manifest.RESOLVED_DIMENSION
            for item in map_metrics.values()
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
                    "sha256": _sha(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
        raise RuntimeError("hybrid architecture audit is already canonicalized")
    dimension = _dimension_and_coordinate_audit()
    finite_volume, finite_volume_arrays = _finite_volume_audit()
    reset, reset_arrays = _reset_geometry_audit()
    descriptor, descriptor_arrays = _descriptor_energy_audit()
    cost = _online_algebra_cost_audit()
    gates = frozen["contract"]["binding_architecture_audit_gates"]
    maximum_reset_defect = max(
        item["reset_constraint_relative_defect"] for item in reset.values()
    )
    passed = bool(
        dimension["all_dimension_identities_exact"]
        and dimension["both_resolved_coordinate_maps_full_rank"]
        and dimension["conditional_branch_unknown_count"]
        == gates["conditional_branch_unknown_count_equal"]
        and dimension["conditional_branch_equation_count"]
        == gates["conditional_branch_equation_count_equal"]
        and finite_volume["global_telescoping_relative_defect"]
        <= gates["finite_volume_global_telescoping_defect_max"]
        and maximum_reset_defect
        <= gates["minimum_norm_reset_constraint_defect_max"]
        and descriptor["maximum_energy_amplification_factor"]
        <= gates["legacy_descriptor_energy_amplification_max"]
        and descriptor["inherited_projected_stable_kernel_cycle_wall_seconds"]
        <= gates["legacy_stable_kernel_projected_cycle_wall_seconds_max"]
        and cost["online_continuous_dimension"]
        <= gates["online_continuous_dimension_max"]
        and cost["online_truth_calls_per_macrostep"]
        == gates["online_truth_calls_per_macrostep_equal"]
        and cost["projected_total_online_algebra_cycle_wall_seconds"]
        <= frozen["contract"]["runtime_contract"][
            "all_online_algebra_budget_seconds"
        ]
    )
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_first_conditional_fast_branch_seed_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    metrics = {
        "dimension_and_coordinate_map": dimension,
        "finite_volume_conservation": finite_volume,
        "minimum_norm_hybrid_reset": reset,
        "legacy_descriptor_method_witness": descriptor,
        "online_algebra_cost": cost,
        "maximum_reset_constraint_relative_defect": maximum_reset_defect,
        "all_binding_architecture_gates_passed": passed,
    }
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "architecture_diagnostics.npz",
        **finite_volume_arrays,
        **reset_arrays,
        **descriptor_arrays,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "selected_architecture": (
            "equilibrium_centered_conservative_slow_fast_hybrid"
        ),
        "conditional_branch_problem_is_square": dimension[
            "all_dimension_identities_exact"
        ],
        "exact_finite_volume_conservation_structure_passed": (
            finite_volume["global_telescoping_relative_defect"]
            <= gates["finite_volume_global_telescoping_defect_max"]
        ),
        "constraint_preserving_reset_structure_passed": (
            maximum_reset_defect
            <= gates["minimum_norm_reset_constraint_defect_max"]
        ),
        "stable_macro_kernel_method_and_cost_witness_passed": (
            descriptor["maximum_energy_amplification_factor"] <= 1.0
            and cost["projected_total_online_algebra_cycle_wall_seconds"]
            <= frozen["contract"]["runtime_contract"][
                "all_online_algebra_budget_seconds"
            ]
        ),
        "online_continuous_dimension_upper_bound": (
            manifest.ONLINE_CONTINUOUS_DIMENSION
        ),
        "projected_online_algebra_cycle_wall_days": cost[
            "projected_total_online_algebra_cycle_wall_days"
        ],
        "physical_conditional_branch_found": False,
        "physical_transition_orbit_found": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "intrinsic_geometry_package_hashes": _checksums(GEOMETRY_DIRECTORY),
            "stable_online_package_hashes": _checksums(STABLE_DIRECTORY),
            "high_order_model_package_hashes": _checksums(HIGH_ORDER_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
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
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": manifest.stable.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Equilibrium-centered conservative slow-fast hybrid architecture audit WP10c9d6c7c3b5c4f25ao",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Selected mathematical architecture",
                "",
                "Let `y=(c96, eta66)` be 162 resolved physical observables and let the 398-column basis `H(x)` span `ker D C_phys(x)`. At fixed `y` and branch label, the offline branch state solves",
                "",
                "`C_phys(x) - y = 0`,",
                "",
                "`H(x)^T W(x) F_Q(x) = 0`.",
                "",
                "This is a square 560-equation problem. It does not force the full rate to zero: the branch may move along the resolved slow coordinates. Normal hyperbolicity is evaluated only from the hidden Jacobian at a converged conditional branch root.",
                "",
                "Online, the 96 M/J/E coordinates evolve through a conservative finite-volume flux divergence. The remaining 66 resolved coordinates and at most 280 memory states use an equilibrium-centered dissipative descriptor update. A discrete branch label replaces linear macro-propagation of the 28 positive-growth directions.",
                "",
                "Fast switches are offline intrinsic fixed-Q orthogonal-collocation boundary-value solves. Their reset maps integrate the conservative flux/source impulse and preserve global Q3 in the absence of an external impulse.",
                "",
                "## Structural results",
                "",
                f"- Both inherited resolved coordinate maps have rank 162. The hidden dimension is 398 and the online continuous upper bound is 442.",
                f"- The finite-volume global telescoping defect is `{finite_volume['global_telescoping_relative_defect']:.6e}`.",
                f"- The maximum actual minimum-norm reset constraint defect is `{maximum_reset_defect:.6e}`.",
                f"- The largest 5.7888 s stable-descriptor energy amplification factor across primary/midpoint/held-out is `{descriptor['maximum_energy_amplification_factor']:.6e}`.",
                f"- A deliberately dense per-step algebra witness plus recomputed stable exponential projects to `{cost['projected_total_online_algebra_cycle_wall_seconds']:.6f}` wall seconds (`{cost['projected_total_online_algebra_cycle_wall_days']:.6e}` days) for 100,000 macrosteps.",
                "",
                "The runtime result establishes feasibility of the online algebraic architecture, not the cost of building its offline database.",
                "",
                "## Claim boundary and next gate",
                "",
                f"Authorized next artifact: `{authorized_next}`.",
                "",
                "No physical conditional branch or transition has yet been found. The old 442-state coefficients remain a transfer/stability/cost witness and are not promoted to an equilibrium-centered physical closure. No predictive cycle or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
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
