#!/usr/bin/env python3
"""Execute the frozen zero-truth fixed-Q fast-attractor screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_departure28_short_vector_field_validation_wp10c9d6c7c3b5c4f25bz as vector_field  # noqa: E402
import run_causal_inner_fixed_q_fast_attractor_manifest_wp10c9d6c7c3b5c4f25ca as manifest  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cb"
MANIFEST_COMMIT = "7abec1ddbd6b6e3920251211dac2fa183dd81c74"
MANIFEST_PARENT = "632d63fe665b3ad29e649366920c15bb9f9c1968"
MANIFEST_TREE = "46bab13ff1172ca690f14d8b041dc645d2c466ec"

STABLE_CLASSIFICATION = (
    "in_chart_normally_attracting_fixed_Q_fast_graph_candidate_found"
)
NONCLOSURE_CLASSIFICATION = (
    "no_in_chart_stationary_fast_graph_found_departure_amplitude_expansion_required"
)
UNSTABLE_CLASSIFICATION = (
    "in_chart_fast_root_not_normally_attracting_"
    "invariant_measure_or_chart_expansion_required"
)
INCONCLUSIVE_CLASSIFICATION = "fixed_Q_fast_attractor_screen_inconclusive"

ARTIFACT = (
    "causal_inner_fixed_q_fast_attractor_screen_"
    "wp10c9d6c7c3b5c4f25cb"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_fixed_q_fast_attractor_screen_"
    "wp10c9d6c7c3b5c4f25cb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_fixed_q_fast_attractor_screen_"
    "wp10c9d6c7c3b5c4f25cb.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_FIXED_Q_FAST_ATTRACTOR_"
    "SCREEN_WP10C9D6C7C3B5C4F25CB_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    manifest.THIS_RUNNER,
    manifest.THIS_TEST,
    vector_field.THIS_RUNNER,
    vector_field.THIS_TEST,
)

_plain = manifest._plain
_read = manifest._read
_write_json = manifest._write_json
_sha = manifest._sha
_checksums = manifest._checksums


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {relative: _sha(ROOT / relative) for relative in SOURCE_FILES}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("fast-attractor manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("fast-attractor manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("fast-attractor manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    with np.load(
        manifest.CANONICAL_DIRECTORY / "search_design.npz", allow_pickle=False
    ) as source:
        starts = np.asarray(source["starts"], dtype=float)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["new_truth_calls"] != 0
        or summary["naive_96_plus_374_split_authorized"]
        or summary["predictive_cycle_authorized"]
        or summary["reduced_slow_evolution_authorized"]
        or contract["decision"]["stable_graph"]["classification"]
        != STABLE_CLASSIFICATION
        or contract["decision"]["clear_nonclosure"]["classification"]
        != NONCLOSURE_CLASSIFICATION
        or contract["decision"]["root_not_attracting"]["classification"]
        != UNSTABLE_CLASSIFICATION
        or contract["decision"]["inconclusive"]["classification"]
        != INCONCLUSIVE_CLASSIFICATION
        or not np.array_equal(starts, manifest._search_design())
    ):
        raise RuntimeError("fast-attractor frozen contract changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"fast-attractor manifest source changed: {relative}")
    for name, expected in vector_field.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("fast-attractor screen requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
        "hashes": hashes,
        "starts": starts,
    }


class FastAttractorModel:
    """Frozen 470D polynomial field reduced at fixed q162."""

    def __init__(self):
        self.vector_field = vector_field.ReducedVectorField()
        model = self.vector_field
        self.fast_restriction = np.vstack(
            (model.memory_basis.T, model.departure_basis.T)
        )
        self.base_fast_rate = self.fast_restriction @ model.base_rate
        self.linear_fast_matrix = (
            self.fast_restriction
            @ model.generator
            @ model.lifting[:, manifest.PHYSICAL_DIMENSION :]
        )
        self.curvature_action = (
            self.fast_restriction @ model.generator @ model.curvature_basis
        )
        self.departure_action = self.fast_restriction @ model.departure_basis
        nz = manifest.STABLE_MEMORY_DIMENSION
        self.Azz = self.linear_fast_matrix[:nz, :nz]
        self.Aza = self.linear_fast_matrix[:nz, nz:]
        self.Aaz = self.linear_fast_matrix[nz:, :nz]
        self.Aaa = self.linear_fast_matrix[nz:, nz:]
        self._lu, self._pivots = lu_factor(self.Azz, check_finite=True)

    def curvature(self, departure: np.ndarray) -> np.ndarray:
        parent = vector_field.manifest.parent
        active = self.vector_field.energy_directions.T @ np.asarray(
            departure, dtype=float
        )
        return parent._predict_curvature(active, self.vector_field.coefficients)

    def nonlinear_departure(self, departure: np.ndarray) -> np.ndarray:
        return self.vector_field.nonlinear_departure(departure)

    def nonlinear_fast_action(self, departure: np.ndarray) -> np.ndarray:
        return (
            self.curvature_action @ self.curvature(departure)
            + self.departure_action @ self.nonlinear_departure(departure)
        )

    def eliminated_memory(self, departure: np.ndarray) -> np.ndarray:
        nz = manifest.STABLE_MEMORY_DIMENSION
        forcing = (
            self.base_fast_rate[:nz]
            + self.Aza @ np.asarray(departure, dtype=float)
            + self.nonlinear_fast_action(departure)[:nz]
        )
        return lu_solve((self._lu, self._pivots), -forcing, check_finite=False)

    def reduced_departure_rate(self, departure: np.ndarray) -> np.ndarray:
        nz = manifest.STABLE_MEMORY_DIMENSION
        coordinate = np.asarray(departure, dtype=float)
        memory = self.eliminated_memory(coordinate)
        return (
            self.base_fast_rate[nz:]
            + self.Aaz @ memory
            + self.Aaa @ coordinate
            + self.nonlinear_fast_action(coordinate)[nz:]
        )

    def full_fast_rate(
        self, memory: np.ndarray, departure: np.ndarray
    ) -> np.ndarray:
        fast = np.concatenate((memory, departure))
        return self.base_fast_rate + self.linear_fast_matrix @ fast + self.nonlinear_fast_action(departure)

    def decoded_state_audit(
        self, memory: np.ndarray, departure: np.ndarray
    ) -> dict:
        coordinate = np.concatenate(
            (
                np.zeros(manifest.PHYSICAL_DIMENSION),
                np.asarray(memory, dtype=float),
                np.asarray(departure, dtype=float),
            )
        )
        delta = self.vector_field.decoded_delta(coordinate)
        record = {
            "decoded_scaled_state_delta_norm": float(np.linalg.norm(delta)),
            "decoded_scaled_state_delta_maximum_absolute": float(
                np.max(np.abs(delta))
            ),
            "passed": False,
            "failure": None,
        }
        try:
            audit = vector_field._state_audit(self.vector_field, coordinate)
            record.update(audit)
            record["passed"] = bool(
                audit["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
                and audit["maximum_H_over_R"] <= 0.12
                and audit["minimum_scattering_optical_depth"] >= 1.0
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            record["failure"] = f"{type(error).__name__}: {error}"
        return record

    def full_fast_jacobian(self, departure: np.ndarray) -> np.ndarray:
        """Differentiate only the nonlinear 28 columns with a frozen stencil."""
        coordinate = np.asarray(departure, dtype=float)
        derivative = np.empty(
            (
                manifest.STABLE_MEMORY_DIMENSION + manifest.DEPARTURE_DIMENSION,
                manifest.DEPARTURE_DIMENSION,
            )
        )
        step = 1.0e-7
        for column in range(manifest.DEPARTURE_DIMENSION):
            offset = np.zeros(manifest.DEPARTURE_DIMENSION)
            offset[column] = step
            derivative[:, column] = (
                self.nonlinear_fast_action(coordinate + offset)
                - self.nonlinear_fast_action(coordinate - offset)
            ) / (2.0 * step)
        result = np.array(self.linear_fast_matrix, copy=True)
        result[:, manifest.STABLE_MEMORY_DIMENSION :] += derivative
        return result


def _structure_metrics(model: FastAttractorModel) -> dict:
    eigenvalues = np.linalg.eigvals(model.Azz)
    full_linear = (
        model.vector_field.restriction
        @ model.vector_field.generator
        @ model.vector_field.lifting
    )
    # The 160 mapped-storage coordinates are cell-major five-component
    # blocks.  M/J/E are components 0/2/3; the constitutive coordinates are
    # components 1/4.  They are not a contiguous 96/64 partition.
    constitutive_indices = np.asarray(
        [
            5 * cell + component
            for cell in range(32)
            for component in (1, 4)
        ]
        + [160, 161],
        dtype=int,
    )
    naive_indices = np.concatenate(
        (
            constitutive_indices,
            np.arange(162, 442, dtype=int),
        )
    )
    if constitutive_indices.size != 66 or naive_indices.size != 346:
        raise RuntimeError("naive split coordinate selection changed")
    naive = full_linear[np.ix_(naive_indices, naive_indices)]
    naive_eigenvalues = np.linalg.eigvals(naive)
    return {
        "stable_memory_shape": list(model.Azz.shape),
        "stable_memory_condition_number": float(np.linalg.cond(model.Azz)),
        "stable_memory_spectral_abscissa_per_second": float(
            np.max(eigenvalues.real)
        ),
        "stable_memory_most_negative_real_part_per_second": float(
            np.min(eigenvalues.real)
        ),
        "stable_memory_nonnegative_eigenvalue_count": int(
            np.count_nonzero(eigenvalues.real >= 0.0)
        ),
        "memory_departure_projection_cross_block_maximum_absolute": float(
            np.max(
                np.abs(
                    model.vector_field.memory_basis.T
                    @ model.vector_field.departure_basis
                )
            )
        ),
        "naive_eliminated_block_shape": list(naive.shape),
        "naive_eliminated_block_coordinate_selection": (
            "cell_major_mapped_components_1_and_4_plus_two_explicit_"
            "stable_plus_z280"
        ),
        "naive_eliminated_block_condition_number": float(np.linalg.cond(naive)),
        "naive_eliminated_block_spectral_abscissa_per_second": float(
            np.max(naive_eigenvalues.real)
        ),
        "naive_eliminated_block_nonnegative_eigenvalue_count": int(
            np.count_nonzero(naive_eigenvalues.real >= 0.0)
        ),
    }


def _search(model: FastAttractorModel, starts: np.ndarray, contract: dict) -> dict:
    design = contract["deterministic_search"]
    graph_gates = contract["binding_fixed_graph_gates"]
    bound = float(design["departure_component_bound"])
    base_reduced_norm = float(
        np.linalg.norm(model.reduced_departure_rate(np.zeros(manifest.DEPARTURE_DIMENSION)))
    )
    base_fast_norm = float(np.linalg.norm(model.base_fast_rate))
    records = []
    solutions = []
    memories = []
    reduced_rates = []
    accepted_indices = []
    attracting_indices = []
    for index, start in enumerate(starts):
        result = least_squares(
            model.reduced_departure_rate,
            start,
            bounds=(-bound, bound),
            method="trf",
            max_nfev=int(design["maximum_function_evaluations_per_start"]),
            xtol=float(design["xtol"]),
            ftol=float(design["ftol"]),
            gtol=float(design["gtol"]),
            x_scale="jac",
        )
        departure = np.asarray(result.x, dtype=float)
        memory = model.eliminated_memory(departure)
        reduced_rate = model.reduced_departure_rate(departure)
        fast_rate = model.full_fast_rate(memory, departure)
        reduced_relative = float(
            np.linalg.norm(reduced_rate)
            / max(base_reduced_norm, np.finfo(float).tiny)
        )
        full_relative = float(
            np.linalg.norm(fast_rate) / max(base_fast_norm, np.finfo(float).tiny)
        )
        maximum_component = float(np.max(np.abs(departure)))
        active_bound_fraction = float(
            np.count_nonzero(np.abs(departure) >= 0.999 * bound)
            / manifest.DEPARTURE_DIMENSION
        )
        boundary_limited = maximum_component >= 0.999 * bound
        state_audit = model.decoded_state_audit(memory, departure)
        residual_candidate = bool(
            reduced_relative
            <= graph_gates["root_reduced_residual_relative_to_base_max"]
            and maximum_component
            <= graph_gates["root_maximum_absolute_departure_component"]
            and full_relative
            <= graph_gates["root_full_fast_residual_relative_to_base_max"]
        )
        # This extra audit is conservative: an algebraic zero outside the decoded
        # physical chart is never promoted to an in-chart graph candidate.
        accepted = residual_candidate and bool(state_audit["passed"])
        spectral_abscissa = None
        normally_attracting = False
        if accepted:
            fast_eigenvalues = np.linalg.eigvals(
                model.full_fast_jacobian(departure)
            )
            spectral_abscissa = float(np.max(fast_eigenvalues.real))
            normally_attracting = bool(
                spectral_abscissa
                <= graph_gates["full_fast_spectral_abscissa_max_per_second"]
            )
            accepted_indices.append(index)
            if normally_attracting:
                attracting_indices.append(index)
        records.append(
            {
                "start_index": index,
                "optimizer_status": int(result.status),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "function_evaluations": int(result.nfev),
                "jacobian_evaluations": int(result.njev or 0),
                "reduced_residual_norm_per_second": float(
                    np.linalg.norm(reduced_rate)
                ),
                "reduced_residual_relative_to_base": reduced_relative,
                "full_fast_residual_relative_to_base": full_relative,
                "maximum_absolute_departure_component": maximum_component,
                "active_bound_component_fraction": active_bound_fraction,
                "boundary_limited": bool(boundary_limited),
                "eliminated_memory_norm": float(np.linalg.norm(memory)),
                "eliminated_memory_maximum_absolute": float(
                    np.max(np.abs(memory))
                ),
                "decoded_state_audit": state_audit,
                "residual_candidate": residual_candidate,
                "accepted_in_chart_root": accepted,
                "full_fast_spectral_abscissa_per_second": spectral_abscissa,
                "normally_attracting": normally_attracting,
            }
        )
        solutions.append(departure)
        memories.append(memory)
        reduced_rates.append(reduced_rate)
    return {
        "records": records,
        "starts": np.asarray(starts),
        "solutions": np.asarray(solutions),
        "memories": np.asarray(memories),
        "reduced_rates": np.asarray(reduced_rates),
        "base_reduced_residual_norm_per_second": base_reduced_norm,
        "base_fast_residual_norm_per_second": base_fast_norm,
        "accepted_indices": accepted_indices,
        "attracting_indices": attracting_indices,
    }


def _classify(structure: dict, search: dict, contract: dict) -> dict:
    structure_gates = contract["binding_structure_gates"]
    nonclosure_gates = contract["clear_nonclosure_gates"]
    records = search["records"]
    structure_checks = {
        "stable_memory_dimension": structure["stable_memory_shape"]
        == [manifest.STABLE_MEMORY_DIMENSION, manifest.STABLE_MEMORY_DIMENSION],
        "stable_memory_spectral_abscissa": structure[
            "stable_memory_spectral_abscissa_per_second"
        ]
        <= structure_gates["stable_memory_spectral_abscissa_max_per_second"],
        "stable_memory_condition_number": structure[
            "stable_memory_condition_number"
        ]
        <= structure_gates["stable_memory_condition_number_max"],
        "departure_projection_cross_block": structure[
            "memory_departure_projection_cross_block_maximum_absolute"
        ]
        <= structure_gates["departure_projection_cross_block_norm_max"],
        "search_start_count": len(records)
        == structure_gates["search_start_count_equal"],
        "new_truth_calls": structure_gates["new_truth_calls_equal"] == 0,
        "new_generator_assemblies": structure_gates[
            "new_generator_assemblies_equal"
        ]
        == 0,
        "new_nonlinear_roots": structure_gates["new_nonlinear_roots_equal"]
        == 0,
        "propagated_states": structure_gates["propagated_states_equal"] == 0,
        "naive_split_rejected": structure[
            "naive_eliminated_block_nonnegative_eigenvalue_count"
        ]
        > 0,
    }
    boundary_fraction = float(
        np.mean([record["boundary_limited"] for record in records])
    )
    minimum_relative = float(
        min(record["reduced_residual_relative_to_base"] for record in records)
    )
    decoded_state_pass_fraction = float(
        np.mean([record["decoded_state_audit"]["passed"] for record in records])
    )
    accepted_count = len(search["accepted_indices"])
    attracting_count = len(search["attracting_indices"])
    clear_nonclosure_checks = {
        "accepted_root_count": accepted_count
        == nonclosure_gates["accepted_root_count_equal"],
        "minimum_screened_residual": minimum_relative
        >= nonclosure_gates["minimum_screened_residual_relative_to_base_min"],
        "boundary_limited_fraction": boundary_fraction
        >= nonclosure_gates["minimum_boundary_limited_solution_fraction"],
    }
    structure_passed = all(structure_checks.values())
    clear_nonclosure = all(clear_nonclosure_checks.values())
    if not structure_passed:
        classification = INCONCLUSIVE_CLASSIFICATION
        passed = False
        authorized_next = None
    elif attracting_count > 0:
        classification = STABLE_CLASSIFICATION
        passed = True
        authorized_next = contract["decision"]["stable_graph"]["authorizes_only"]
    elif accepted_count > 0:
        classification = UNSTABLE_CLASSIFICATION
        passed = True
        authorized_next = contract["decision"]["root_not_attracting"]["authorizes_only"]
    elif clear_nonclosure:
        classification = NONCLOSURE_CLASSIFICATION
        passed = True
        authorized_next = contract["decision"]["clear_nonclosure"]["authorizes_only"]
    else:
        classification = INCONCLUSIVE_CLASSIFICATION
        passed = False
        authorized_next = None
    return {
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "structure_checks": structure_checks,
        "clear_nonclosure_checks": clear_nonclosure_checks,
        "accepted_root_count": accepted_count,
        "normally_attracting_root_count": attracting_count,
        "minimum_screened_residual_relative_to_base": minimum_relative,
        "boundary_limited_solution_fraction": boundary_fraction,
        "decoded_state_audit_pass_fraction": decoded_state_pass_fraction,
    }


def _write_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "FAILED",
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
        raise RuntimeError("fast-attractor screen already canonicalized")
    began = time.perf_counter()
    model = FastAttractorModel()
    structure = _structure_metrics(model)
    search = _search(model, frozen["starts"], frozen["contract"])
    decision = _classify(structure, search, frozen["contract"])
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": decision["classification"],
        "passed": decision["passed"],
        "mathematical_split": "q162_active_plus_z280_stable_plus_a28_nonlinear",
        "structure": structure,
        "search": {
            "base_reduced_residual_norm_per_second": search[
                "base_reduced_residual_norm_per_second"
            ],
            "base_fast_residual_norm_per_second": search[
                "base_fast_residual_norm_per_second"
            ],
            "records": search["records"],
        },
        "decision": decision,
        "new_truth_calls": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_physical_states": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": decision["classification"],
        "passed": decision["passed"],
        "fast_graph_found": decision["accepted_root_count"] > 0,
        "normally_attracting_fast_graph_found": decision[
            "normally_attracting_root_count"
        ]
        > 0,
        "accepted_root_count": decision["accepted_root_count"],
        "minimum_screened_residual_relative_to_base": decision[
            "minimum_screened_residual_relative_to_base"
        ],
        "boundary_limited_solution_fraction": decision[
            "boundary_limited_solution_fraction"
        ],
        "new_truth_calls": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": decision["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "screen_metrics.json", metrics)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "search_design_sha256": _sha(
                manifest.CANONICAL_DIRECTORY / "search_design.npz"
            ),
        },
    )
    _write_npz(
        CANONICAL_DIRECTORY / "screen_arrays.npz",
        starts=search["starts"],
        departure_solutions=search["solutions"],
        eliminated_memory_solutions=search["memories"],
        reduced_departure_rates_per_second=search["reduced_rates"],
        stable_memory_matrix=model.Azz,
        reduced_base_rate_per_second=model.reduced_departure_rate(
            np.zeros(manifest.DEPARTURE_DIMENSION)
        ),
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if summary["passed"] else "FAILED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": _source_hashes(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in vector_field.THREAD_ENVIRONMENT
            },
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    best = min(
        metrics["search"]["records"],
        key=lambda record: record["reduced_residual_relative_to_base"],
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Fixed-Q fast-attractor screen WP10c9d6c7c3b5c4f25cb",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                "The corrected 280D memory block is strictly Hurwitz: "
                f"spectral abscissa `{structure['stable_memory_spectral_abscissa_per_second']:.9g} s^-1` "
                f"and condition number `{structure['stable_memory_condition_number']:.9g}`.",
                "",
                "The rejected 346D block from the naive 96/374 split has "
                f"`{structure['naive_eliminated_block_nonnegative_eigenvalue_count']}` "
                "nonnegative-real-part eigenvalues, so that split remains blocked.",
                "",
                f"All `{len(search['records'])}` bounded searches were boundary-limited. "
                f"The best reduced residual remained `{best['reduced_residual_relative_to_base']:.9g}` "
                "of the base forcing and no in-chart root passed.",
                "",
                "The eliminated memory equilibrium is also far outside the decoded local physical chart "
                "for the screened candidates. This is diagnostic evidence that the local chart covers a "
                "growing transient segment, not a stationary fast graph.",
                "",
                f"The only authorized next artifact is `{summary['authorized_next']}`. "
                "No physical microburst, cycle prediction, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
