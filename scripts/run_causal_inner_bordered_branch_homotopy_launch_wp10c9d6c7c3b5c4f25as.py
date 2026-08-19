#!/usr/bin/env python3
"""Launch the bordered conditional-branch homotopy to tau=1/64."""

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

import run_causal_inner_bordered_branch_homotopy_launch_manifest_wp10c9d6c7c3b5c4f25ar as manifest  # noqa: E402
import run_causal_inner_first_conditional_branch_seed_preflight_wp10c9d6c7c3b5c4f25aq as preflight  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    _integrated_mapped_storage,
    _spatial_nodes,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25as"
MANIFEST_COMMIT = "420190213acc0a6ecf6c228742439a242089c176"
MANIFEST_PARENT = "b66cac5ff7e99f9346cbeebc3aa6de4c105f4fea"
MANIFEST_TREE = "fa7421d9026abaffa74e9b3528861f7341286776"

PASS_CLASSIFICATION = (
    "bordered_homotopy_launch_tau_1_over_64_passed_"
    "adaptive_homotopy_continuation_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "bordered_homotopy_launch_failed_"
    "conditional_branch_path_requires_diagnosis"
)

ARTIFACT = (
    "causal_inner_bordered_branch_homotopy_launch_"
    "wp10c9d6c7c3b5c4f25as"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_bordered_branch_homotopy_launch_"
    "wp10c9d6c7c3b5c4f25as.py"
)
THIS_TEST = (
    "tests/test_causal_inner_bordered_branch_homotopy_launch_"
    "wp10c9d6c7c3b5c4f25as.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_BORDERED_BRANCH_HOMOTOPY_"
    "LAUNCH_WP10C9D6C7C3B5C4F25AS_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
PREFLIGHT_ARRAYS = preflight.CANONICAL_DIRECTORY / "preflight_diagnostics.npz"


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
        raise RuntimeError("homotopy-launch manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("homotopy-launch manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("homotopy-launch manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["tau_target"] != manifest.TAU_TARGET
        or contract["claim_boundary"]["tau_one_reached"]
    ):
        raise RuntimeError("homotopy-launch authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, path in (
        ("preflight_diagnostics", PREFLIGHT_ARRAYS),
        ("preflight_metrics", preflight.CANONICAL_DIRECTORY / "metrics.json"),
    ):
        if _sha(path) != contract["decisive_input_hashes"][name]:
            raise RuntimeError(f"decisive input changed: {path}")
    _checksums(preflight.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("homotopy launch requires a clean tracked tree")
    for name, expected in preflight.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _ruiz_equilibrate(matrix: np.ndarray, passes: int = 8):
    equilibrated = np.asarray(matrix, dtype=float).copy()
    row_scale = np.ones(equilibrated.shape[0], dtype=float)
    column_scale = np.ones(equilibrated.shape[1], dtype=float)
    tiny = np.finfo(float).tiny
    for _ in range(int(passes)):
        norms = np.maximum(np.linalg.norm(equilibrated, axis=1), tiny)
        factor = 1.0 / np.sqrt(norms)
        equilibrated *= factor[:, None]
        row_scale *= factor
        norms = np.maximum(np.linalg.norm(equilibrated, axis=0), tiny)
        factor = 1.0 / np.sqrt(norms)
        equilibrated *= factor[None, :]
        column_scale *= factor
    return equilibrated, row_scale, column_scale


def _equilibrated_solve(matrix: np.ndarray, right_hand_side: np.ndarray):
    equilibrated, row_scale, column_scale = _ruiz_equilibrate(matrix)
    solution = column_scale * np.linalg.solve(
        equilibrated, row_scale * np.asarray(right_hand_side, dtype=float)
    )
    relative_residual = float(
        np.linalg.norm(matrix @ solution - right_hand_side)
        / max(float(np.linalg.norm(right_hand_side)), np.finfo(float).tiny)
    )
    return solution, {
        "equilibrated_condition_number": float(np.linalg.cond(equilibrated)),
        "relative_linear_residual": relative_residual,
        "row_scale_spread": float(np.max(row_scale) / np.min(row_scale)),
        "column_scale_spread": float(
            np.max(column_scale) / np.min(column_scale)
        ),
    }


def _anchor_system() -> dict:
    components = preflight._coordinate_components()
    with np.load(PREFLIGHT_ARRAYS, allow_pickle=False) as source:
        saved = {name: np.asarray(source[name]) for name in source.files}
    for name, actual in (
        ("primitive_anchor", components["state"]),
        ("primitive_column_scales", components["columns"]),
        ("conservation_row_scales", components["rows"]),
        ("coordinate_jacobian", components["jacobian"]),
    ):
        if not np.array_equal(saved[name], actual):
            raise RuntimeError(f"homotopy anchor input changed: {name}")
    jacobian = saved["coordinate_jacobian"]
    rate = saved["fixed_Q_rate"]
    generator = saved["complete_fixed_Q_generator"]
    omega = float(np.linalg.norm(rate) / np.sqrt(rate.size))
    scaled_rate = rate / omega
    multiplier = np.linalg.solve(jacobian @ jacobian.T, jacobian @ scaled_rate)
    hidden_residual = scaled_rate - jacobian.T @ multiplier
    matrix = np.block(
        [
            [
                jacobian,
                np.zeros((manifest.parent.manifest.RESOLVED_DIMENSION,) * 2),
            ],
            [generator / omega, -jacobian.T],
        ]
    )
    base_target_residual = np.concatenate(
        (
            np.zeros(manifest.parent.manifest.RESOLVED_DIMENSION),
            manifest.TAU_TARGET * hidden_residual,
        )
    )
    predictor_correction, linear = _equilibrated_solve(
        matrix, -base_target_residual
    )
    tau_zero_residual = np.concatenate(
        (
            np.zeros(manifest.parent.manifest.RESOLVED_DIMENSION),
            scaled_rate - jacobian.T @ multiplier - hidden_residual,
        )
    )
    coordinate_anchor = preflight._coordinate_value(components["state"], components)
    face = 36 * int(components["data"]["layout"].refinement_ratio)
    q3_anchor, q3_factors = causal_five_field_exterior_q3(
        components["context"], components["state"], exterior_face_index=face
    )
    return {
        "components": components,
        "saved": saved,
        "coordinate_anchor": coordinate_anchor,
        "jacobian_anchor": jacobian,
        "rate_scale": omega,
        "multiplier_anchor": multiplier,
        "hidden_residual_anchor": hidden_residual,
        "initial_matrix": matrix,
        "base_target_residual": base_target_residual,
        "predictor_correction": predictor_correction,
        "linear_metrics": linear,
        "tau_zero_residual": tau_zero_residual,
        "q3_anchor": np.asarray(q3_anchor, dtype=float),
        "q3_anchor_minimum_reconstruction_factor": float(np.min(q3_factors)),
    }


def _coordinate_jacobian(state: np.ndarray, system: dict) -> tuple[np.ndarray, dict]:
    components = system["components"]
    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = _node_reconstruction_weights(components["context"], state)
    mapped, _height = _descriptor_matrices(
        components["context"],
        state,
        components["columns"],
        components["rows"],
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    physical = C * components["rows"].ravel()[:, None] * mapped
    top = np.zeros((manifest.parent.manifest.MAPPED_COORDINATES, 560))
    for coarse_cell, (start, stop) in enumerate(components["groups"]):
        for field in range(manifest.parent.manifest.FIELDS_PER_CELL):
            target = manifest.parent.manifest.FIELDS_PER_CELL * coarse_cell + field
            source_rows = (
                manifest.parent.manifest.FIELDS_PER_CELL * np.arange(start, stop)
                + field
            )
            top[target] = np.sum(physical[source_rows], axis=0)
    top /= components["mapped_row_scales"][:, None]
    return np.vstack((top, components["stable_dual"])), {
        "mapped_reconstruction_relative_defect": float(reconstruction_defect),
        "reconstruction_partition_defect": float(partition_defect),
    }


def _coordinate_value_with_factor(state: np.ndarray, system: dict):
    components = system["components"]
    integrated, factors, _node_values = _integrated_mapped_storage(
        components["context"], state, _spatial_nodes(components["context"])
    )
    mapped = np.zeros(manifest.parent.manifest.MAPPED_COORDINATES)
    for coarse_cell, (start, stop) in enumerate(components["groups"]):
        mapped[
            manifest.parent.manifest.FIELDS_PER_CELL * coarse_cell :
            manifest.parent.manifest.FIELDS_PER_CELL * (coarse_cell + 1)
        ] = np.sum(integrated[start:stop], axis=0)
    mapped /= components["mapped_row_scales"]
    scaled_delta = (
        (np.asarray(state) - components["state"]) / components["columns"]
    ).ravel()
    stable = components["stable_dual"] @ scaled_delta
    return np.concatenate((mapped, stable)), float(np.min(factors))


class _Evaluator:
    def __init__(self, system: dict, tau: float, maximum_calls: int):
        self.system = system
        self.tau = float(tau)
        self.maximum_calls = int(maximum_calls)
        self.calls = 0
        self.events = []

    def __call__(self, unknown: np.ndarray, label: str):
        if self.calls >= self.maximum_calls:
            raise RuntimeError("homotopy rate-evaluation budget exhausted")
        delta = np.asarray(unknown[:560], dtype=float)
        multiplier = np.asarray(unknown[560:], dtype=float)
        maximum_departure = float(np.max(np.abs(delta)))
        if maximum_departure > 5.0e-3 * (1.0 + 1.0e-12):
            raise ValueError("homotopy candidate left the frozen trust region")
        components = self.system["components"]
        state = components["state"] + (
            components["columns"].ravel() * delta
        ).reshape(components["state"].shape)
        began = time.perf_counter()
        rate, reaction, evaluation, physical, timings = (
            preflight.nonlinear._continuous_fixed_q_rate(
                components["data"], state
            )
        )
        coordinate, minimum_factor = _coordinate_value_with_factor(
            state, self.system
        )
        jacobian, coordinate_diagnostics = _coordinate_jacobian(
            state, self.system
        )
        coordinate_residual = coordinate - self.system["coordinate_anchor"]
        stationarity = (
            rate / self.system["rate_scale"]
            - jacobian.T @ multiplier
            - (1.0 - self.tau) * self.system["hidden_residual_anchor"]
        )
        residual = np.concatenate((coordinate_residual, stationarity))
        self.calls += 1
        meta = {
            "state": state,
            "rate": rate,
            "reaction": reaction,
            "evaluation": evaluation,
            "physical": physical,
            "jacobian": jacobian,
            "coordinate": coordinate,
            "coordinate_residual": coordinate_residual,
            "stationarity_residual": stationarity,
            "minimum_coordinate_reconstruction_factor": minimum_factor,
            "coordinate_diagnostics": coordinate_diagnostics,
            "maximum_scaled_anchor_departure": maximum_departure,
            "rate_timings": timings,
        }
        event = {
            "call": self.calls,
            "label": label,
            "residual_infinity": float(np.max(np.abs(residual))),
            "residual_two_norm": float(np.linalg.norm(residual)),
            "coordinate_residual_infinity": float(
                np.max(np.abs(coordinate_residual))
            ),
            "stationarity_residual_infinity": float(
                np.max(np.abs(stationarity))
            ),
            "maximum_scaled_anchor_departure": maximum_departure,
            "wall_seconds": time.perf_counter() - began,
        }
        self.events.append(event)
        print(
            f"f25as: call={self.calls} label={label} "
            f"inf={event['residual_infinity']:.6e} "
            f"two={event['residual_two_norm']:.6e}",
            flush=True,
        )
        return residual, meta


def _broyden_launch(system: dict, contract: dict):
    policy = contract["nonlinear_policy"]
    gates = contract["binding_gates"]
    unknown_base = np.concatenate(
        (np.zeros(560), system["multiplier_anchor"])
    )
    unknown = unknown_base + system["predictor_correction"]
    matrix = np.asarray(system["initial_matrix"], dtype=float).copy()
    evaluator = _Evaluator(
        system, manifest.TAU_TARGET, gates["maximum_new_rate_evaluations"]
    )
    residual, meta = evaluator(unknown, "linear_predictor")
    secant = unknown - unknown_base
    response = residual - system["base_target_residual"]
    denominator = float(secant @ secant)
    if denominator > np.finfo(float).tiny:
        matrix += np.outer(response - matrix @ secant, secant) / denominator
    iterations = []
    tolerance = gates["complete_target_residual_infinity_max"]
    for iteration in range(int(policy["maximum_iterations"])):
        if float(np.max(np.abs(residual))) <= tolerance:
            break
        correction, linear = _equilibrated_solve(matrix, -residual)
        accepted = False
        current_merit = float(np.linalg.norm(residual))
        for factor in policy["line_search_relative_factors"]:
            trial = unknown + float(factor) * correction
            if float(np.max(np.abs(trial[:560]))) > 5.0e-3 * (1.0 + 1.0e-12):
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "factor": float(factor),
                        "accepted": False,
                        "reason": "trust_region",
                    }
                )
                continue
            trial_residual, trial_meta = evaluator(
                trial, f"iteration_{iteration + 1}_factor_{factor:g}"
            )
            trial_merit = float(np.linalg.norm(trial_residual))
            accepted = trial_merit < current_merit
            iterations.append(
                {
                    "iteration": iteration + 1,
                    "factor": float(factor),
                    "accepted": accepted,
                    "reason": "merit_decrease" if accepted else "no_decrease",
                    "linear": linear,
                    "trial_residual_infinity": float(
                        np.max(np.abs(trial_residual))
                    ),
                    "trial_residual_two_norm": trial_merit,
                }
            )
            if accepted:
                step = trial - unknown
                change = trial_residual - residual
                denominator = float(step @ step)
                if denominator > np.finfo(float).tiny:
                    matrix += np.outer(change - matrix @ step, step) / denominator
                unknown = trial
                residual = trial_residual
                meta = trial_meta
                break
        if not accepted:
            break
    return {
        "unknown": unknown,
        "residual": residual,
        "meta": meta,
        "matrix": matrix,
        "iterations": iterations,
        "events": evaluator.events,
        "rate_evaluations": evaluator.calls,
    }


def _physical_audit(solution: dict, system: dict, gates: dict) -> dict:
    meta = solution["meta"]
    reaction = meta["reaction"]
    face = 36 * int(system["components"]["data"]["layout"].refinement_ratio)
    q3, q3_factors = causal_five_field_exterior_q3(
        system["components"]["context"],
        meta["state"],
        exterior_face_index=face,
    )
    q3_defect = float(
        np.max(
            np.abs(np.asarray(q3) - system["q3_anchor"])
            / np.maximum(
                np.asarray(reaction.q3_derivative_norms), np.finfo(float).tiny
            )
        )
    )
    minimum_reconstruction = min(
        meta["minimum_coordinate_reconstruction_factor"],
        float(np.min(q3_factors)),
        float(reaction.minimum_q3_reconstruction_factor),
        float(meta["physical"]["minimum_reconstruction_factor"]),
    )
    complete_inf = float(np.max(np.abs(solution["residual"])))
    coordinate_inf = float(np.max(np.abs(meta["coordinate_residual"])))
    stationarity_inf = float(np.max(np.abs(meta["stationarity_residual"])))
    checks = {
        "complete_residual": complete_inf
        <= gates["complete_target_residual_infinity_max"],
        "coordinate_residual": coordinate_inf
        <= gates["coordinate_residual_infinity_max"],
        "stationarity_residual": stationarity_inf
        <= gates["stationarity_residual_infinity_max"],
        "Q3": q3_defect <= gates["normalized_Q3_defect_max"],
        "reconstruction": minimum_reconstruction
        >= gates["minimum_reconstruction_factor_min"],
        "raw_schur_rank": reaction.raw_schur_numerical_rank
        == gates["raw_schur_rank_equal"],
        "raw_schur_condition": reaction.raw_schur_condition_number
        <= gates["raw_schur_condition_number_max"],
        "raw_schur_solve": reaction.maximum_raw_schur_solve_relative_defect
        <= gates["raw_schur_solve_relative_defect_max"],
        "height": meta["physical"]["maximum_h_over_r"]
        <= gates["maximum_h_over_r_max"],
        "optical_depth": meta["physical"]["minimum_scattering_optical_depth"]
        >= gates["minimum_scattering_optical_depth_min"],
        "trust_region": meta["maximum_scaled_anchor_departure"]
        <= contract_value(gates, "linear_predictor_maximum_scaled_component_max"),
        "rate_evaluation_budget": solution["rate_evaluations"]
        <= gates["maximum_new_rate_evaluations"],
        "incoming_excision_diagnostic": (
            meta["evaluation"].incoming_excision_characteristics == 0
        ),
        "reaction_identity_diagnostic": reaction.maximum_identity_defect <= 1.0e-10,
        "reaction_ledger_diagnostic": (
            reaction.maximum_reaction_ledger_relative_defect <= 1.0e-10
        ),
    }
    return {
        "checks": checks,
        "all_binding_checks_passed": all(checks.values()),
        "complete_residual_infinity": complete_inf,
        "coordinate_residual_infinity": coordinate_inf,
        "stationarity_residual_infinity": stationarity_inf,
        "normalized_Q3_defect": q3_defect,
        "minimum_reconstruction_factor": minimum_reconstruction,
        "raw_schur_numerical_rank": int(reaction.raw_schur_numerical_rank),
        "raw_schur_condition_number": float(reaction.raw_schur_condition_number),
        "raw_schur_solve_relative_defect": float(
            reaction.maximum_raw_schur_solve_relative_defect
        ),
        "maximum_h_over_r": meta["physical"]["maximum_h_over_r"],
        "minimum_scattering_optical_depth": meta["physical"][
            "minimum_scattering_optical_depth"
        ],
        "incoming_excision_characteristics": int(
            meta["evaluation"].incoming_excision_characteristics
        ),
        "maximum_scaled_anchor_departure": meta[
            "maximum_scaled_anchor_departure"
        ],
    }


def contract_value(gates: dict, name: str) -> float:
    return float(gates[name])


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
        raise RuntimeError("homotopy launch is already canonicalized")
    began = time.perf_counter()
    system = _anchor_system()
    gates = frozen["contract"]["binding_gates"]
    preflight_checks = {
        "tau_zero_residual": float(np.max(np.abs(system["tau_zero_residual"])))
        <= gates["tau_zero_complete_residual_infinity_max"],
        "equilibrated_condition": system["linear_metrics"][
            "equilibrated_condition_number"
        ]
        <= gates["equilibrated_initial_matrix_condition_number_max"],
        "linear_predictor_trust": float(
            np.max(np.abs(system["predictor_correction"][:560]))
        )
        <= gates["linear_predictor_maximum_scaled_component_max"],
        "linear_solve": system["linear_metrics"]["relative_linear_residual"]
        <= 1.0e-10,
    }
    if not all(preflight_checks.values()):
        raise RuntimeError("homotopy launch algebraic preflight failed closed")
    solution = _broyden_launch(system, frozen["contract"])
    physical = _physical_audit(solution, system, gates)
    passed = physical["all_binding_checks_passed"]
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_adaptive_bordered_homotopy_continuation_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    checkpoint_path = CANONICAL_DIRECTORY / "homotopy_tau_1_over_64.npz"
    np.savez_compressed(
        checkpoint_path,
        tau=np.asarray(manifest.TAU_TARGET),
        primitive_anchor=system["components"]["state"],
        primitive_column_scales=system["components"]["columns"],
        coordinate_anchor=system["coordinate_anchor"],
        rate_scale=np.asarray(system["rate_scale"]),
        hidden_residual_anchor=system["hidden_residual_anchor"],
        scaled_primitive_delta=solution["unknown"][:560],
        coordinate_multiplier=solution["unknown"][560:],
        primitive_state=solution["meta"]["state"],
        fixed_Q_rate=solution["meta"]["rate"],
        coordinate_jacobian=solution["meta"]["jacobian"],
        complete_residual=solution["residual"],
        carried_broyden_matrix=solution["matrix"],
    )
    with np.load(checkpoint_path, allow_pickle=False) as replay:
        checkpoint_roundtrip = bool(
            np.array_equal(replay["primitive_state"], solution["meta"]["state"])
            and np.array_equal(replay["complete_residual"], solution["residual"])
            and np.array_equal(replay["carried_broyden_matrix"], solution["matrix"])
        )
    passed = bool(passed and checkpoint_roundtrip)
    if not checkpoint_roundtrip:
        classification = FAIL_CLASSIFICATION
        authorized_next = None
    metrics = {
        "algebraic_preflight": {
            "checks": preflight_checks,
            "tau_zero_complete_residual_infinity": float(
                np.max(np.abs(system["tau_zero_residual"]))
            ),
            "rate_scale_per_second": system["rate_scale"],
            "anchor_hidden_residual_norm": float(
                np.linalg.norm(system["hidden_residual_anchor"])
            ),
            "linear_predictor_maximum_scaled_component": float(
                np.max(np.abs(system["predictor_correction"][:560]))
            ),
            "linear_predictor_scaled_norm": float(
                np.linalg.norm(system["predictor_correction"][:560])
            ),
            **system["linear_metrics"],
        },
        "nonlinear": {
            "tau": manifest.TAU_TARGET,
            "rate_evaluations": solution["rate_evaluations"],
            "events": solution["events"],
            "iterations": solution["iterations"],
            "checkpoint_roundtrip_bitwise": checkpoint_roundtrip,
        },
        "physical_and_binding_audit": physical,
        "total_wall_seconds": time.perf_counter() - began,
    }
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "tau_reached": manifest.TAU_TARGET if passed else 0.0,
        "tau_one_reached": False,
        "rate_evaluations": solution["rate_evaluations"],
        "complete_residual_infinity": physical["complete_residual_infinity"],
        "coordinate_residual_infinity": physical["coordinate_residual_infinity"],
        "stationarity_residual_infinity": physical[
            "stationarity_residual_infinity"
        ],
        "maximum_scaled_anchor_departure": physical[
            "maximum_scaled_anchor_departure"
        ],
        "checkpoint_roundtrip_bitwise": checkpoint_roundtrip,
        "physical_conditional_branch_found": False,
        "normal_hyperbolicity_certified": False,
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
            "preflight_package_hashes": _checksums(preflight.CANONICAL_DIRECTORY),
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
            "thread_environment": preflight.THREAD_ENVIRONMENT,
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
                "# Bordered branch homotopy launch WP10c9d6c7c3b5c4f25as",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Result",
                "",
                f"The exact tau=0 anchor was advanced to tau=`{manifest.TAU_TARGET:.6f}` using `{solution['rate_evaluations']}` new complete fixed-Q rate evaluations. The final complete residual infinity norm is `{physical['complete_residual_infinity']:.6e}`, with coordinate and stationarity parts `{physical['coordinate_residual_infinity']:.6e}` and `{physical['stationarity_residual_infinity']:.6e}`.",
                "",
                f"The accepted scaled state departure is `{physical['maximum_scaled_anchor_departure']:.6e}`. The normalized Q3 defect is `{physical['normalized_Q3_defect']:.6e}` and the raw Schur condition number is `{physical['raw_schur_condition_number']:.6e}`.",
                "",
                "This is a homotopy launch point, not a conditional branch root. Tau=1, normal hyperbolicity, transitions, a predictive cycle, and reduced slow evolution remain unestablished.",
                "",
                f"Authorized next artifact: `{authorized_next}`.",
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
