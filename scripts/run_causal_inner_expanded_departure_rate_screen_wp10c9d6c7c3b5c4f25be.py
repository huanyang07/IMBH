#!/usr/bin/env python3
"""Execute the amplitude-0.01 primary departure-rate screen."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_expanded_departure_rate_screen_manifest_wp10c9d6c7c3b5c4f25bd as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25be"
MANIFEST_COMMIT = "f28915be7a6ec85288273dea51a00c1ef5a7ef23"
MANIFEST_PARENT = "6e8069036443d8986466ec775f2a1d1b99fc9977"
MANIFEST_TREE = "f941dc38690f171b62297f2d55e3a6585f782fef"

NONLINEAR_CLASSIFICATION = (
    "expanded_primary_departure_rate_screen_amplitude_0p01_passed_"
    "nonlinear_signal_resolved_mixed_direction_database_manifest_authorized"
)
UNRESOLVED_CLASSIFICATION = (
    "expanded_primary_departure_rate_screen_amplitude_0p01_passed_"
    "nonlinear_signal_not_resolved_amplitude_0p02_chart_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "expanded_primary_departure_rate_screen_amplitude_0p01_failed_"
    "nonlinear_architecture_identification_blocked"
)

ARTIFACT = (
    "causal_inner_expanded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25be"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_expanded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25be.py"
)
THIS_TEST = (
    "tests/test_causal_inner_expanded_departure_rate_screen_"
    "wp10c9d6c7c3b5c4f25be.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXPANDED_DEPARTURE_RATE_"
    "SCREEN_WP10C9D6C7C3B5C4F25BE_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

FROZEN_SCIENTIFIC_SOURCES = (
    "scripts/run_causal_inner_guarded_departure_rate_screen_wp10c9d6c7c3b5c4f25ba.py",
    "scripts/run_causal_inner_exact_geometric_departure_chart_preflight_wp10c9d6c7c3b5c4f25ay.py",
    "scripts/run_causal_inner_explicit_nonlinear_470_architecture_audit_wp10c9d6c7c3b5c4f25aw.py",
    "scripts/run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
)


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
        raise RuntimeError("expanded rate-screen manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("expanded rate-screen manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("expanded rate-screen manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    lock = _read(manifest.ARTIFACT_DIRECTORY / "parent_lock.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_nonbase_continuous_rate_evaluations"]
        != manifest.CANDIDATE_COUNT
        or summary["component_bound"] != manifest.COMPONENT_BOUND
        or summary["full_closure_database_claimed"]
        or contract["claim_boundary"][
            "sixteen_axial_samples_are_a_full_28D_closure_database"
        ]
    ):
        raise RuntimeError("expanded rate-screen authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("expanded_departure_chart", manifest.CHART_PATH),
        ("online_470_geometry", manifest.GEOMETRY_PATH),
        ("complete_primary_generator", manifest.GENERATOR_PATH),
        ("prior_departure_rate_screen", manifest.PRIOR_SCREEN_PATH),
    ):
        if _sha(path) != lock["decisive_input_hashes"][name]:
            raise RuntimeError(f"expanded rate-screen input changed: {path}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    _checksums(manifest.prior_screen.CANONICAL_DIRECTORY)
    for relative in FROZEN_SCIENTIFIC_SOURCES:
        if _git("hash-object", relative) != _git(
            "rev-parse", f"{MANIFEST_COMMIT}:{relative}"
        ):
            raise RuntimeError(f"scientific source changed after manifest: {relative}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("expanded rate-screen requires a clean tracked tree")
    for name, expected in (
        manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items()
    ):
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name], dtype=float) for name in source.files}


def _load_inputs() -> dict:
    chart_metrics = _read(manifest.parent.CANONICAL_DIRECTORY / "metrics.json")
    chart = _load_npz(manifest.CHART_PATH)
    geometry = _load_npz(manifest.GEOMETRY_PATH)
    generator_data = _load_npz(manifest.GENERATOR_PATH)
    prior = _load_npz(manifest.PRIOR_SCREEN_PATH)
    states = chart["candidate_primitive_states"]
    deltas = chart["candidate_scaled_deltas"]
    coordinates = chart["candidate_departure_coordinates"]
    candidates = chart_metrics["candidates"]
    generator = generator_data["complete_fixed_Q_generator"]
    base_rate = generator_data["fixed_Q_rate"]
    if (
        states.shape != (manifest.CANDIDATE_COUNT, 112, 5)
        or deltas.shape != (manifest.CANDIDATE_COUNT, 560)
        or coordinates.shape != (manifest.CANDIDATE_COUNT, 28)
        or len(candidates) != manifest.CANDIDATE_COUNT
        or generator.shape != (560, 560)
        or base_rate.shape != (560,)
        or prior["central_departure_nonlinear_relative_defects"].shape != (8, 3)
    ):
        raise RuntimeError("expanded rate-screen input dimensions changed")
    if [item["candidate_index"] for item in candidates] != list(
        range(manifest.CANDIDATE_COUNT)
    ):
        raise RuntimeError("expanded chart candidate ordering changed")
    if any(
        item["component_bound"] != manifest.COMPONENT_BOUND for item in candidates
    ):
        raise RuntimeError("expanded chart component bound changed")
    return {
        "chart": chart,
        "states": states,
        "deltas": deltas,
        "coordinates": coordinates,
        "candidates": candidates,
        "memory_basis": geometry["stable_memory_coordinate_basis"],
        "departure_basis": geometry["departure_coordinate_basis"],
        "generator": generator,
        "base_rate": base_rate,
        "prior": prior,
    }


def _pair_analysis(
    candidates: list[dict],
    departure_coordinates: np.ndarray,
    departure_rate_increments: np.ndarray,
    departure_linear_references: np.ndarray,
    directions: np.ndarray,
    prior: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    count = directions.shape[1]
    radii = np.empty(count, dtype=float)
    growth = np.empty(count, dtype=float)
    linear_growth = np.empty(count, dtype=float)
    nonlinear = np.empty(count, dtype=float)
    amplification = np.empty(count, dtype=float)
    exponent = np.empty(count, dtype=float)
    secant_cubic = np.empty(count, dtype=float)
    prior_radii = prior["effective_departure_radii"][:, -1]
    prior_growth = prior["central_radial_growth_per_second"][:, -1]
    prior_nonlinear = prior[
        "central_departure_nonlinear_relative_defects"
    ][:, -1]
    for direction_index in range(count):
        indices = [
            index
            for index, item in enumerate(candidates)
            if item["direction_index"] == direction_index
        ]
        if len(indices) != 2:
            raise RuntimeError("expanded signed radial pair is incomplete")
        negative, positive = sorted(indices, key=lambda index: candidates[index]["sign"])
        direction = directions[:, direction_index]
        coordinate_odd = 0.5 * (
            departure_coordinates[positive] - departure_coordinates[negative]
        )
        rate_odd = 0.5 * (
            departure_rate_increments[positive]
            - departure_rate_increments[negative]
        )
        linear_odd = 0.5 * (
            departure_linear_references[positive]
            - departure_linear_references[negative]
        )
        radius = float(direction @ coordinate_odd)
        if radius <= np.finfo(float).tiny:
            raise RuntimeError("expanded radial coordinate lost its signed direction")
        radii[direction_index] = radius
        growth[direction_index] = float(direction @ rate_odd / radius)
        linear_growth[direction_index] = float(direction @ linear_odd / radius)
        nonlinear[direction_index] = _relative_error(rate_odd, linear_odd)
        amplification[direction_index] = float(
            nonlinear[direction_index]
            / max(prior_nonlinear[direction_index], np.finfo(float).tiny)
        )
        radius_ratio = float(radius / prior_radii[direction_index])
        exponent[direction_index] = float(
            math.log(amplification[direction_index]) / math.log(radius_ratio)
        )
        secant_cubic[direction_index] = float(
            (growth[direction_index] - prior_growth[direction_index])
            / (radius**2 - prior_radii[direction_index] ** 2)
        )
    metrics = {
        "median_current_departure_nonlinear_relative_defect": float(
            np.median(nonlinear)
        ),
        "minimum_current_departure_nonlinear_relative_defect": float(
            np.min(nonlinear)
        ),
        "maximum_current_departure_nonlinear_relative_defect": float(
            np.max(nonlinear)
        ),
        "median_prior_departure_nonlinear_relative_defect": float(
            np.median(prior_nonlinear)
        ),
        "median_nonlinear_amplification": float(np.median(amplification)),
        "median_effective_amplitude_exponent": float(np.median(exponent)),
        "minimum_current_radial_growth_per_second": float(np.min(growth)),
        "maximum_current_radial_growth_per_second": float(np.max(growth)),
        "nonpositive_current_radial_growth_count": int(
            np.count_nonzero(growth <= 0.0)
        ),
        "negative_secant_cubic_growth_count": int(
            np.count_nonzero(secant_cubic < 0.0)
        ),
    }
    arrays = {
        "effective_departure_radii": radii,
        "central_radial_growth_per_second": growth,
        "central_linear_radial_growth_per_second": linear_growth,
        "central_departure_nonlinear_relative_defects": nonlinear,
        "prior_effective_departure_radii": prior_radii,
        "prior_central_radial_growth_per_second": prior_growth,
        "prior_departure_nonlinear_relative_defects": prior_nonlinear,
        "nonlinear_amplification_from_0p005": amplification,
        "effective_amplitude_exponents": exponent,
        "secant_cubic_growth_coefficients": secant_cubic,
    }
    return metrics, arrays


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    inputs = _load_inputs()
    data = manifest.prior_screen.manifest.parent.manifest.failed_screen._anchor_data(
        "primary"
    )
    components = manifest.parent.chart_tools.coordinate_tools._coordinate_components()
    total_rates = []
    free_rates = []
    actions = []
    multipliers = []
    online_rates = []
    departure_rates = []
    linear_references = []
    departure_linear_references = []
    evaluation_metrics = []
    failures = []
    began = time.perf_counter()
    for index, state in enumerate(inputs["states"]):
        try:
            item, arrays = manifest.prior_screen._continuous_rate(data, state)
            coordinate_jacobian, coordinate_metrics = (
                manifest.parent.chart_tools._coordinate_jacobian(state, components)
            )
            linear = inputs["generator"] @ inputs["deltas"][index]
            increment = arrays["total_rate"] - inputs["base_rate"]
            departure_increment = inputs["departure_basis"].T @ increment
            departure_linear = inputs["departure_basis"].T @ linear
            online_rate = np.concatenate(
                (
                    coordinate_jacobian @ arrays["total_rate"],
                    inputs["memory_basis"].T @ arrays["total_rate"],
                    inputs["departure_basis"].T @ arrays["total_rate"],
                )
            )
            candidate = inputs["candidates"][index]
            item.update(
                {
                    "candidate_index": index,
                    "direction_index": candidate["direction_index"],
                    "component_bound": candidate["component_bound"],
                    "sign": candidate["sign"],
                    "state_rate_linear_relative_defect": _relative_error(
                        increment, linear
                    ),
                    "departure_rate_linear_relative_defect": _relative_error(
                        departure_increment, departure_linear
                    ),
                    "coordinate_Jacobian_rank": coordinate_metrics["rank"],
                    "coordinate_Jacobian_condition_number": coordinate_metrics[
                        "condition_number"
                    ],
                }
            )
            total_rates.append(arrays["total_rate"])
            free_rates.append(arrays["free_rate"])
            actions.append(arrays["reaction_action"])
            multipliers.append(arrays["multiplier"])
            online_rates.append(online_rate)
            departure_rates.append(departure_increment)
            linear_references.append(linear)
            departure_linear_references.append(departure_linear)
            evaluation_metrics.append(item)
            status = "accepted"
        except Exception as error:  # canonicalize first fail-closed truth error
            failures.append(
                {
                    "candidate_index": index,
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            status = "failed"
        print(
            json.dumps(
                {
                    "candidate": index + 1,
                    "total": manifest.CANDIDATE_COUNT,
                    "direction": inputs["candidates"][index]["direction_index"],
                    "component_bound": manifest.COMPONENT_BOUND,
                    "sign": inputs["candidates"][index]["sign"],
                    "status": status,
                    "elapsed_seconds": time.perf_counter() - began,
                }
            ),
            flush=True,
        )
        if failures:
            break

    departure_increments = np.asarray(departure_rates, dtype=float)
    departure_linear_array = np.asarray(departure_linear_references, dtype=float)
    pair_metrics = {}
    pair_arrays = {}
    if len(evaluation_metrics) == manifest.CANDIDATE_COUNT:
        pair_metrics, pair_arrays = _pair_analysis(
            inputs["candidates"],
            inputs["coordinates"],
            departure_increments,
            departure_linear_array,
            inputs["chart"]["energy_directions"],
            inputs["prior"],
        )

    def maximum(name: str, default=math.inf) -> float:
        values = [item[name] for item in evaluation_metrics]
        return float(max(values)) if values else float(default)

    def minimum(name: str, default=-math.inf) -> float:
        values = [item[name] for item in evaluation_metrics]
        return float(min(values)) if values else float(default)

    metrics = {
        "planned_nonbase_rate_evaluations": manifest.CANDIDATE_COUNT,
        "completed_nonbase_rate_evaluations": len(evaluation_metrics),
        "failed_rate_evaluations": len(failures),
        "failures": failures,
        "maximum_state_rate_linear_relative_defect": maximum(
            "state_rate_linear_relative_defect"
        ),
        "maximum_departure_rate_linear_relative_defect": maximum(
            "departure_rate_linear_relative_defect"
        ),
        "minimum_reconstruction_factor": minimum(
            "minimum_reconstruction_factor", math.inf
        ),
        "maximum_reconstruction_factor": maximum("maximum_reconstruction_factor"),
        "maximum_raw_Schur_condition_number": maximum(
            "raw_Schur_condition_number"
        ),
        "maximum_reaction_identity_defect": maximum("reaction_identity_defect"),
        "maximum_rate_tangency_relative_defect": maximum(
            "rate_tangency_relative_defect"
        ),
        "maximum_coordinate_Jacobian_condition_number": maximum(
            "coordinate_Jacobian_condition_number"
        ),
        "maximum_H_over_R": maximum("maximum_H_over_R"),
        "minimum_scattering_optical_depth": minimum(
            "minimum_scattering_optical_depth"
        ),
        "maximum_incoming_excision_characteristics": maximum(
            "incoming_excision_characteristics"
        ),
        "total_truth_wall_seconds": time.perf_counter() - began,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "pair_nonlinearity": pair_metrics,
        "evaluations": evaluation_metrics,
    }
    arrays = {
        "candidate_primitive_states": inputs["states"][: len(evaluation_metrics)],
        "candidate_scaled_deltas": inputs["deltas"][: len(evaluation_metrics)],
        "candidate_departure_coordinates": inputs["coordinates"][
            : len(evaluation_metrics)
        ],
        "total_rates_per_second": np.asarray(total_rates, dtype=float),
        "free_rates_per_second": np.asarray(free_rates, dtype=float),
        "physical_reaction_actions_per_second": np.asarray(actions, dtype=float),
        "multiplier_coordinates_per_second": np.asarray(multipliers, dtype=float),
        "online_470_coordinate_rates_per_second": np.asarray(
            online_rates, dtype=float
        ),
        "departure_rate_increments_per_second": departure_increments,
        "linear_rate_references_per_second": np.asarray(
            linear_references, dtype=float
        ),
        "departure_linear_references_per_second": departure_linear_array,
        "base_fixed_Q_rate_per_second": inputs["base_rate"],
        **pair_arrays,
    }
    return metrics, arrays


def _gate_checks(metrics: dict, gates: dict) -> dict:
    return {
        "completed": metrics["completed_nonbase_rate_evaluations"]
        == gates["completed_nonbase_rate_evaluations_equal"],
        "failed": metrics["failed_rate_evaluations"]
        == gates["failed_rate_evaluations_equal"],
        "reconstruction_minimum": metrics["minimum_reconstruction_factor"]
        >= gates["minimum_reconstruction_factor"],
        "reconstruction_maximum": metrics["maximum_reconstruction_factor"]
        <= gates["maximum_reconstruction_factor"],
        "Schur_condition": metrics["maximum_raw_Schur_condition_number"]
        <= gates["maximum_raw_Schur_condition_number"],
        "reaction_identity": metrics["maximum_reaction_identity_defect"]
        <= gates["maximum_reaction_identity_defect"],
        "rate_tangency": metrics["maximum_rate_tangency_relative_defect"]
        <= gates["maximum_rate_tangency_relative_defect"],
        "coordinate_condition": metrics[
            "maximum_coordinate_Jacobian_condition_number"
        ]
        <= gates["maximum_coordinate_Jacobian_condition_number"],
        "height": metrics["maximum_H_over_R"] <= gates["maximum_H_over_R"],
        "optical_depth": metrics["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth"],
        "incoming_excision": metrics[
            "maximum_incoming_excision_characteristics"
        ]
        == gates["maximum_incoming_excision_characteristics_equal"],
        "generator_budget": metrics["new_complete_generator_assemblies"]
        == gates["new_complete_generator_assemblies_equal"],
        "root_budget": metrics["new_nonlinear_roots"]
        == gates["new_nonlinear_roots_equal"],
        "propagation_budget": metrics["propagated_states"]
        == gates["propagated_states_equal"],
    }


def _classify(evaluator_passed: bool, median_signal: float) -> tuple[str, str | None]:
    if not evaluator_passed:
        return FAIL_CLASSIFICATION, None
    if median_signal >= manifest.NONLINEAR_SIGNAL_THRESHOLD:
        return (
            NONLINEAR_CLASSIFICATION,
            "definitions_only_mixed_direction_adaptive_28D_database_manifest",
        )
    return (
        UNRESOLVED_CLASSIFICATION,
        "definitions_only_exact_departure_chart_amplitude_0p02_manifest",
    )


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
        raise RuntimeError("expanded departure-rate screen is already canonicalized")
    metrics, arrays = _evaluate()
    checks = _gate_checks(metrics, frozen["contract"]["binding_evaluator_gates"])
    evaluator_passed = all(checks.values())
    median_signal = metrics["pair_nonlinearity"].get(
        "median_current_departure_nonlinear_relative_defect", -math.inf
    )
    classification, authorized_next = _classify(evaluator_passed, median_signal)
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics})
    np.savez_compressed(CANONICAL_DIRECTORY / "departure_rate_screen.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": evaluator_passed,
        "component_bound": manifest.COMPONENT_BOUND,
        "completed_nonbase_rate_evaluations": metrics[
            "completed_nonbase_rate_evaluations"
        ],
        "failed_rate_evaluations": metrics["failed_rate_evaluations"],
        "median_current_departure_nonlinear_relative_defect": median_signal,
        "median_prior_departure_nonlinear_relative_defect": metrics[
            "pair_nonlinearity"
        ].get("median_prior_departure_nonlinear_relative_defect"),
        "median_nonlinear_amplification": metrics["pair_nonlinearity"].get(
            "median_nonlinear_amplification"
        ),
        "nonlinear_signal_resolved": bool(
            evaluator_passed and median_signal >= manifest.NONLINEAR_SIGNAL_THRESHOLD
        ),
        "full_28D_closure_identified": False,
        "heldout_state_validated": False,
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
            "manifest_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "chart_hashes": _checksums(manifest.parent.CANONICAL_DIRECTORY),
            "prior_screen_hashes": _checksums(
                manifest.prior_screen.CANONICAL_DIRECTORY
            ),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        *FROZEN_SCIENTIFIC_SOURCES,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if evaluator_passed else "REJECTED",
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
            "thread_environment": (
                manifest.parent.chart_tools.coordinate_tools.THREAD_ENVIRONMENT
            ),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    pair = metrics["pair_nonlinearity"]
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Expanded departure-rate screen WP10c9d6c7c3b5c4f25be",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Result",
                "",
                f"Completed `{metrics['completed_nonbase_rate_evaluations']}` of `{manifest.CANDIDATE_COUNT}` exact amplitude-0.01 rate evaluations with `{metrics['failed_rate_evaluations']}` failures.",
                "",
                f"The median central 28D nonlinear fraction is `{median_signal:.6e}`, compared with `{pair.get('median_prior_departure_nonlinear_relative_defect', math.nan):.6e}` at amplitude 0.005. Median amplification is `{pair.get('median_nonlinear_amplification', math.nan):.6e}`.",
                "",
                f"Current radial growth is nonpositive in `{pair.get('nonpositive_current_radial_growth_count', 0)}` of 8 directions and the 0.005-to-0.01 secant cubic coefficient is negative in `{pair.get('negative_secant_cubic_growth_count', 0)}` directions. These are diagnostic, not branch-selection gates.",
                "",
                f"Authorized next artifact: `{authorized_next}`. No full 28D closure, held-out validation, online trajectory, or predictive cycle is claimed.",
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
