#!/usr/bin/env python3
"""Freeze and execute the hardened adaptive-refresh refined fixed-Q ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_fixed_q_adaptive_refresh_primary_revalidation_wp10c9d6c7c3b5c4f24e11 as e11  # noqa: E402
import run_causal_inner_face36_fixed_q_authentic_history_ladder_wp10c9d6c7c3b5c4f24e1 as e1  # noqa: E402
import run_causal_inner_face36_fixed_q_exact_refresh_diagnostic_wp10c9d6c7c3b5c4f24e2 as e2  # noqa: E402


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e13a"
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_hardened_"
    "manifest_wp10c9d6c7c3b5c4f24e13a"
)
RESULT_ARTIFACT = (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_hardened_"
    "wp10c9d6c7c3b5c4f24e13a"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
RESULT_DIRECTORY = ROOT / "results/canonical" / RESULT_ARTIFACT
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / RESULT_ARTIFACT
PRIMARY_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
HELDOUT_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_adaptive_refresh_refined_"
    "ladder_wp10c9d6c7c3b5c4f24e13.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_adaptive_refresh_refined_"
    "ladder_wp10c9d6c7c3b5c4f24e13.py"
)
SUPERSEDED_MANIFEST_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_manifest_"
    "wp10c9d6c7c3b5c4f24e13"
)
REFERENCE_ARRAYS_NAME = "continuous_references.npz"
REUSED_CASES = ("primary_coarse", "heldout_coarse")
REFINED_CASES = (
    "primary_middle",
    "heldout_middle",
    "primary_fine",
    "heldout_fine",
)
CONTRACT = {
    "schema_version": 1,
    "supersedes_work_package": "WP10c9d6c7c3b5c4f24e13",
    "reused_cases": list(REUSED_CASES),
    "refined_cases": list(REFINED_CASES),
    "case_order": list(e1.CASE_ORDER),
    "timesteps_seconds": list(e1.TIMESTEPS),
    "binding_temporal_form": "exact_increment_primary",
    "direct_rate_form": "post_root_parity_audit_only",
    "exact_jacobian_refresh_policy": "on_line_search_failure",
    "maximum_exact_jacobian_assemblies_per_root": 2,
    "required_schur_solve_method": "row_column_equilibrated_LU_refined_1",
    "require_all_existing_step_acceptance_gates": True,
    "require_bitwise_restart_roundtrip": True,
    "require_bitwise_BDF2_replay": True,
    "minimum_state_rate_convergence_order": 0.9,
    "minimum_reaction_action_convergence_order": 0.9,
    "binding_order_error": "absolute_l2_error_against_frozen_continuous_reference",
    "reported_relative_error_denominator": "frozen_continuous_reference_l2_norm",
    "varying_max_norm_relative_error": "diagnostic_only",
    "deterministic_BDF1_predictor": "timestep_times_continuous_constrained_rate",
    "deterministic_multiplier_predictor": "continuous_constrained_multiplier",
    "optional_untracked_predictors_forbidden": True,
    "stage_local_scratch_required": True,
    "canonical_prior_stage_validation_required": True,
    "available_order_gate_applied_after_each_stage": True,
    "numerical_floor_may_rescue_failed_order": False,
    "fail_fast": True,
    "may_change_physical_equations": False,
    "may_relax_any_gate": False,
    "one_Q_execution_manifest_authorized": False,
    "fixed_Q_micro_solver_authorized": False,
    "reduced_slow_evolution_authorized": False,
}


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


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_tree_is_clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT
        ).returncode
        == 0
    )


def _source_files() -> tuple[Path, ...]:
    layer = ROOT / "src/imri_qpe/layer3_minidisk_1d"
    explicit = (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT
        / "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
        "wp10c9d6c7c3b5c4f24e1.py",
        ROOT
        / "scripts/run_causal_inner_face36_fixed_q_adaptive_refresh_primary_"
        "revalidation_wp10c9d6c7c3b5c4f24e11.py",
        ROOT
        / "scripts/run_causal_inner_face36_fixed_q_exact_refresh_diagnostic_"
        "wp10c9d6c7c3b5c4f24e2.py",
    )
    return tuple(sorted({*explicit, *layer.glob("*.py")}))


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha(path) for path in _source_files()
    }


def _checksum_entries(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    return entries


def _validate_checksums(directory: Path, required: set[str]) -> None:
    entries = _checksum_entries(directory)
    if set(entries) != required:
        raise RuntimeError(f"canonical checksum inventory changed: {directory}")
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum failed: {directory / name}")


def _references() -> dict[str, np.ndarray]:
    with np.load(
        MANIFEST_DIRECTORY / REFERENCE_ARRAYS_NAME,
        allow_pickle=False,
    ) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _continuous_reference_arrays() -> dict[str, np.ndarray]:
    arrays = {}
    for state_label in ("primary_20ms", "heldout_16ms"):
        data = e1._state_data(state_label)
        arrays[f"{state_label}_state_rate"] = np.asarray(
            data["continuous_rate"]
        )
        arrays[f"{state_label}_reaction_action"] = np.asarray(
            data["continuous_reaction_action"]
        )
    return arrays


def _deterministic_seed(label: str, timestep_index: int, data: dict):
    del label
    timestep = float(e1.TIMESTEPS[timestep_index])
    return (
        timestep * np.asarray(data["continuous_rate"]),
        np.asarray(data["continuous_multiplier"]),
    )


def _parents() -> tuple[dict, dict, dict]:
    primary = _read(PRIMARY_DIRECTORY / "summary.json")
    heldout = _read(HELDOUT_DIRECTORY / "summary.json")
    superseded = _read(SUPERSEDED_MANIFEST_DIRECTORY / "summary.json")
    if (
        not primary["passed"]
        or not primary["heldout_retry_manifest_authorized"]
        or not heldout["passed"]
        or not heldout["refined_ladder_manifest_authorized"]
        or not superseded["passed"]
        or not superseded["definitions_only"]
    ):
        raise RuntimeError("refined adaptive-refresh ladder is not authorized")
    return primary, heldout, superseded


def _stage_directory(case: str) -> Path:
    return ROOT / "results/canonical" / (
        "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_stage_"
        f"{case}_wp10c9d6c7c3b5c4f24e13a"
    )


def _stage_scratch_directory(case: str) -> Path:
    return CHECKPOINT_DIRECTORY / case / _git("rev-parse", "HEAD")


def _required_decisive_names() -> set[str]:
    fields = {
        "primitive_charts",
        "primitive_increment",
        "scaled_rate_per_s",
        "scaled_interval_rate_per_s",
        "multipliers",
        "scaled_reaction_rate_action_per_s",
        "augmented_scaled_residual",
    }
    return {f"{stage}_{name}" for stage in ("bdf1", "bdf2") for name in fields}


def _validate_stage(case: str) -> tuple[dict, dict]:
    directory = _stage_directory(case)
    _validate_checksums(
        directory,
        {
            "contract.json",
            "decisive_arrays.npz",
            "metrics.json",
            "provenance.json",
            "summary.json",
        },
    )
    summary = _read(directory / "summary.json")
    metrics = _read(directory / "metrics.json")
    provenance = _read(directory / "provenance.json")
    if (
        not summary["passed"]
        or summary["case"] != case
        or summary["classification"]
        != f"adaptive_refresh_refined_ladder_stage_{case}_passed"
        or not metrics["passed"]
        or metrics["case"] != case
        or not metrics["restart_roundtrip_bitwise"]
        or not metrics["BDF2_replay_bitwise"]
        or _read(directory / "contract.json") != CONTRACT
        or provenance["refined_ladder_manifest_summary_sha256"]
        != _sha(MANIFEST_DIRECTORY / "summary.json")
    ):
        raise RuntimeError(f"prior refined stage changed: {case}")
    state_label, timestep_index = e1.CASE_DEFINITIONS[case]
    if (
        metrics["state_label"] != state_label
        or metrics["timestep_seconds"] != e1.TIMESTEPS[timestep_index]
    ):
        raise RuntimeError(f"prior refined stage identity changed: {case}")
    with np.load(directory / "decisive_arrays.npz", allow_pickle=False) as arrays:
        if not _required_decisive_names().issubset(arrays.files):
            raise RuntimeError(f"prior refined stage arrays are incomplete: {case}")
    return summary, metrics


def _validate_frozen_contract(*, prior_to: str | None = None) -> None:
    required = {
        "execution_manifest.json",
        "provenance.json",
        REFERENCE_ARRAYS_NAME,
        "summary.json",
    }
    _validate_checksums(MANIFEST_DIRECTORY, required)
    if _read(MANIFEST_DIRECTORY / "execution_manifest.json") != CONTRACT:
        raise RuntimeError("refined-ladder execution contract changed")
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["refined_ladder_execution_authorized"]
    ):
        raise RuntimeError("refined-ladder execution authorization changed")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    if provenance["source_hashes"] != _source_hashes():
        raise RuntimeError("frozen refined-ladder source bundle changed")
    frozen = {
        "primary_summary_sha256": _sha(PRIMARY_DIRECTORY / "summary.json"),
        "primary_metrics_sha256": _sha(PRIMARY_DIRECTORY / "metrics.json"),
        "primary_arrays_sha256": _sha(PRIMARY_DIRECTORY / "decisive_arrays.npz"),
        "heldout_summary_sha256": _sha(HELDOUT_DIRECTORY / "summary.json"),
        "heldout_metrics_sha256": _sha(HELDOUT_DIRECTORY / "metrics.json"),
        "heldout_arrays_sha256": _sha(HELDOUT_DIRECTORY / "decisive_arrays.npz"),
    }
    if any(provenance[name] != digest for name, digest in frozen.items()):
        raise RuntimeError("certified coarse evidence changed")
    if provenance["continuous_references_sha256"] != _sha(
        MANIFEST_DIRECTORY / REFERENCE_ARRAYS_NAME
    ):
        raise RuntimeError("frozen continuous references changed")
    _parents()
    if prior_to is not None:
        position = e1.CASE_ORDER.index(prior_to)
        for prior in e1.CASE_ORDER[:position]:
            if prior in REFINED_CASES:
                _validate_stage(prior)


def _configure(case: str) -> Path:
    e11._configure()
    scratch = _stage_scratch_directory(case)
    e1.WORK_PACKAGE = WORK_PACKAGE
    e1.CHECKPOINT_DIRECTORY = scratch
    e1.THIS_RUNNER = THIS_RUNNER
    e1.GATES = {**e1.GATES, "maximum_complete_Jacobian_assemblies": 2}
    original_identity = e1._identity
    original_metrics = e1._result_metrics

    def identity() -> dict:
        payload = original_identity()
        payload.update(
            {
                "refined_ladder_runner_sha256": _sha(ROOT / THIS_RUNNER),
                "refined_ladder_test_sha256": _sha(ROOT / THIS_TEST),
                "refined_ladder_manifest_summary_sha256": _sha(
                    MANIFEST_DIRECTORY / "summary.json"
                ),
                "primary_coarse_summary_sha256": _sha(
                    PRIMARY_DIRECTORY / "summary.json"
                ),
                "heldout_coarse_summary_sha256": _sha(
                    HELDOUT_DIRECTORY / "summary.json"
                ),
            }
        )
        return payload

    def result_metrics(result, data) -> dict:
        payload = original_metrics(result, data)
        payload["maximum_exact_Jacobian_assemblies_allowed"] = 2
        return payload

    e1._identity = identity
    e1._result_metrics = result_metrics
    e1._old_direct_seed = _deterministic_seed
    return scratch


def _coarse_metrics(case: str) -> dict:
    directory = PRIMARY_DIRECTORY if case == "primary_coarse" else HELDOUT_DIRECTORY
    _validate_checksums(
        directory,
        {
            "contract.json",
            "decisive_arrays.npz",
            "metrics.json",
            "provenance.json",
            "summary.json",
        },
    )
    metrics = _read(directory / "metrics.json")
    if not metrics["passed"] or not metrics["BDF2_replay_bitwise"]:
        raise RuntimeError(f"certified {case} evidence changed")
    return metrics


def _seed_prior_cases(case: str, scratch: Path) -> None:
    limit = e1.CASE_ORDER.index(case)
    scratch.mkdir(parents=True, exist_ok=True)
    for prior in e1.CASE_ORDER[:limit]:
        if prior in REUSED_CASES:
            payload = _coarse_metrics(prior)
        else:
            _, payload = _validate_stage(prior)
        _write(scratch / f"{prior}.json", payload)


def _freeze() -> dict:
    primary, heldout, superseded = _parents()
    if not _tracked_tree_is_clean():
        raise RuntimeError("refined-ladder manifest requires a clean tree")
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_refined_ladder_hardened_manifest_frozen_"
            "fail_fast_execution_authorized"
        ),
        "passed": True,
        "definitions_only": True,
        "supersedes_work_package": "WP10c9d6c7c3b5c4f24e13",
        "next_case": "primary_middle",
        "refined_ladder_execution_authorized": True,
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    MANIFEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(MANIFEST_DIRECTORY / "execution_manifest.json", CONTRACT)
    _write_npz(
        MANIFEST_DIRECTORY / REFERENCE_ARRAYS_NAME,
        **_continuous_reference_arrays(),
    )
    _write(MANIFEST_DIRECTORY / "summary.json", summary)
    _write(
        MANIFEST_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "definition_commit": _git("rev-parse", "HEAD"),
            "definition_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": _source_hashes(),
            "primary_summary_sha256": _sha(PRIMARY_DIRECTORY / "summary.json"),
            "primary_metrics_sha256": _sha(PRIMARY_DIRECTORY / "metrics.json"),
            "primary_arrays_sha256": _sha(
                PRIMARY_DIRECTORY / "decisive_arrays.npz"
            ),
            "heldout_summary_sha256": _sha(HELDOUT_DIRECTORY / "summary.json"),
            "heldout_metrics_sha256": _sha(HELDOUT_DIRECTORY / "metrics.json"),
            "heldout_arrays_sha256": _sha(
                HELDOUT_DIRECTORY / "decisive_arrays.npz"
            ),
            "continuous_references_sha256": _sha(
                MANIFEST_DIRECTORY / REFERENCE_ARRAYS_NAME
            ),
            "primary_classification": primary["classification"],
            "heldout_classification": heldout["classification"],
            "superseded_classification": superseded["classification"],
        },
    )
    names = (
        "execution_manifest.json",
        "provenance.json",
        REFERENCE_ARRAYS_NAME,
        "summary.json",
    )
    (MANIFEST_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(MANIFEST_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    e2._catalog(MANIFEST_DIRECTORY, MANIFEST_ARTIFACT, summary, "PROSPECTIVE")
    return summary


def _bdf2_arrays(case: str, *, scratch: Path | None = None) -> dict[str, np.ndarray]:
    if scratch is not None:
        path = scratch / f"{case}_bdf2.npz"
        names = {
            "state_rate": "scaled_rate_per_s",
            "reaction_action": "scaled_reaction_rate_action_per_s",
        }
    else:
        if case == "primary_coarse":
            path = PRIMARY_DIRECTORY / "decisive_arrays.npz"
        elif case == "heldout_coarse":
            path = HELDOUT_DIRECTORY / "decisive_arrays.npz"
        else:
            _validate_stage(case)
            path = _stage_directory(case) / "decisive_arrays.npz"
        names = {
            "state_rate": "bdf2_scaled_rate_per_s",
            "reaction_action": "bdf2_scaled_reaction_rate_action_per_s",
        }
    if not path.exists():
        raise RuntimeError(f"BDF2 decisive arrays are missing: {path}")
    with np.load(path, allow_pickle=False) as source:
        if not set(names.values()).issubset(source.files):
            raise RuntimeError(f"BDF2 decisive arrays are incomplete: {path}")
        return {key: np.asarray(source[name]) for key, name in names.items()}


def _fixed_reference_errors(
    case: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    state_label, _ = e1.CASE_DEFINITIONS[case]
    references = _references()
    result = {}
    for quantity in ("state_rate", "reaction_action"):
        reference = references[f"{state_label}_{quantity}"]
        absolute = float(np.linalg.norm(arrays[quantity] - reference))
        scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        result[f"{quantity}_absolute_error"] = absolute
        result[f"{quantity}_fixed_reference_relative_error"] = absolute / scale
    return result


def _order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine < 0.0:
        return float("nan")
    if fine == 0.0:
        return float("inf")
    return float(math.log(coarse / fine) / math.log(2.0))


def _available_convergence(case: str, scratch: Path) -> dict:
    previous = {
        "primary_middle": "primary_coarse",
        "heldout_middle": "heldout_coarse",
        "primary_fine": "primary_middle",
        "heldout_fine": "heldout_middle",
    }[case]
    prior_errors = _fixed_reference_errors(previous, _bdf2_arrays(previous))
    current_errors = _fixed_reference_errors(
        case,
        _bdf2_arrays(case, scratch=scratch),
    )
    state_order = _order(
        prior_errors["state_rate_absolute_error"],
        current_errors["state_rate_absolute_error"],
    )
    action_order = _order(
        prior_errors["reaction_action_absolute_error"],
        current_errors["reaction_action_absolute_error"],
    )
    passed = bool(
        math.isfinite(state_order)
        and math.isfinite(action_order)
        and state_order >= CONTRACT["minimum_state_rate_convergence_order"]
        and action_order
        >= CONTRACT["minimum_reaction_action_convergence_order"]
    )
    return {
        "previous_case": previous,
        "current_case": case,
        "prior": prior_errors,
        "current": current_errors,
        "state_rate_order": state_order,
        "reaction_action_order": action_order,
        "passed": passed,
    }


def _canonicalize_stage(case: str, metrics: dict, scratch: Path) -> dict:
    directory = _stage_directory(case)
    solver_passed = bool(metrics["passed"])
    convergence = None
    if solver_passed:
        for stage in ("bdf1", "bdf2"):
            if not (scratch / f"{case}_{stage}.npz").exists():
                raise RuntimeError(f"supported {case} is missing {stage} arrays")
        convergence = _available_convergence(case, scratch)
    passed = bool(solver_passed and convergence is not None and convergence["passed"])
    metrics = {
        **metrics,
        "solver_and_replay_passed": solver_passed,
        "available_convergence": convergence,
        "passed": passed,
        "failed_stage": (
            metrics.get("failed_stage")
            if not solver_passed
            else None if passed else "available_convergence_order"
        ),
    }
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            f"adaptive_refresh_refined_ladder_stage_{case}_passed"
            if passed
            else f"adaptive_refresh_refined_ladder_stage_{case}_failed"
        ),
        "passed": passed,
        "solver_and_replay_passed": solver_passed,
        "available_convergence": convergence,
        "case": case,
        "next_case": (
            e1.CASE_ORDER[e1.CASE_ORDER.index(case) + 1]
            if passed and case != e1.CASE_ORDER[-1]
            else None
        ),
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    arrays = {}
    for stage in ("bdf1", "bdf2"):
        path = scratch / f"{case}_{stage}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as source:
                for name in source.files:
                    if name != "metrics_json":
                        arrays[f"{stage}_{name}"] = np.asarray(source[name])
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "contract.json", CONTRACT)
    _write(directory / "metrics.json", metrics)
    _write(directory / "summary.json", summary)
    _write_npz(directory / "decisive_arrays.npz", **arrays)
    _write(
        directory / "provenance.json",
        {
            "schema_version": 1,
            **e1._identity(),
            "case": case,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = (
        "contract.json",
        "decisive_arrays.npz",
        "metrics.json",
        "provenance.json",
        "summary.json",
    )
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(directory / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    e2._catalog(
        directory,
        directory.name,
        summary,
        "SUPPORTED" if passed else "REJECTED",
    )
    return summary


def _execute_case(case: str) -> dict:
    if case not in REFINED_CASES:
        raise ValueError("case is not in the refined adaptive-refresh ladder")
    _validate_frozen_contract(prior_to=case)
    if not _tracked_tree_is_clean():
        raise RuntimeError("refined-ladder execution requires a clean tree")
    scratch = _configure(case)
    if scratch.exists() and any(scratch.iterdir()):
        raise RuntimeError(f"stage scratch is not empty: {scratch}")
    _seed_prior_cases(case, scratch)
    metrics = e1._solve_case(case)
    summary = _canonicalize_stage(case, metrics, scratch)
    canonical_metrics = _read(_stage_directory(case) / "metrics.json")
    return {
        "passed": bool(summary["passed"]),
        "summary": summary,
        "metrics": canonical_metrics,
    }


def _finalize() -> dict:
    _validate_frozen_contract()
    if not _tracked_tree_is_clean():
        raise RuntimeError("refined-ladder finalization requires a clean tree")
    stages = {}
    for case in REFINED_CASES:
        stage, _ = _validate_stage(case)
        stages[case] = stage
    convergence = {
        "primary_20ms": {
            "coarse_to_middle": stages["primary_middle"][
                "available_convergence"
            ],
            "middle_to_fine": stages["primary_fine"][
                "available_convergence"
            ],
        },
        "heldout_16ms": {
            "coarse_to_middle": stages["heldout_middle"][
                "available_convergence"
            ],
            "middle_to_fine": stages["heldout_fine"][
                "available_convergence"
            ],
        },
    }
    passed = bool(
        all(stage["passed"] for stage in stages.values())
        and all(
            pair["passed"]
            for state in convergence.values()
            for pair in state.values()
        )
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "adaptive_refresh_refined_fixed_Q_history_ladder_certified_"
            "one_Q_manifest_authorized"
            if passed
            else "adaptive_refresh_refined_fixed_Q_history_ladder_failed"
        ),
        "passed": passed,
        "stages": {
            case: {
                "classification": stage["classification"],
                "passed": stage["passed"],
            }
            for case, stage in stages.items()
        },
        "convergence": convergence,
        "one_Q_execution_manifest_authorized": passed,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(RESULT_DIRECTORY / "contract.json", CONTRACT)
    _write(RESULT_DIRECTORY / "summary.json", summary)
    _write(
        RESULT_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": _source_hashes(),
            "manifest_summary_sha256": _sha(
                MANIFEST_DIRECTORY / "summary.json"
            ),
            "stage_summary_hashes": {
                case: _sha(_stage_directory(case) / "summary.json")
                for case in REFINED_CASES
            },
        },
    )
    names = ("contract.json", "provenance.json", "summary.json")
    (RESULT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(RESULT_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    e2._catalog(
        RESULT_DIRECTORY,
        RESULT_ARTIFACT,
        summary,
        "SUPPORTED" if summary["passed"] else "REJECTED",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--case", choices=REFINED_CASES)
    parser.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args()
    selected = int(arguments.freeze) + int(arguments.case is not None) + int(
        arguments.finalize
    )
    if selected != 1:
        raise SystemExit("select exactly one --freeze, --case, or --finalize")
    if arguments.freeze:
        payload = _freeze()
    elif arguments.finalize:
        payload = _finalize()
    else:
        payload = _execute_case(arguments.case)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
