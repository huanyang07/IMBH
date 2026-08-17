#!/usr/bin/env python3
"""Execute the frozen primary fixed-Q doubled-timestep rung."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

e14d = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
e14l = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    load_causal_five_field_fixed_q_continuation_state,
    save_causal_five_field_fixed_q_continuation_state,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14v"
ARTIFACT = (
    "causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v"
)
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
SEED_PATH = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_continuation_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e14b/canonical_seed_continuation.npz"
)
FINE_REFERENCE_DIRECTORY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)
FINE_TIMESTEP_SECONDS = 1.0e-7
COARSE_TIMESTEP_SECONDS = 2.0e-7


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(e14d._plain(payload), indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _clean() -> bool:
    return bool(
        subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode == 0
        and subprocess.run(("git", "diff", "--cached", "--quiet"), cwd=ROOT).returncode == 0
    )


def _checksums(directory: Path) -> dict[str, str]:
    entries = {}
    for line in (directory / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        if _sha(directory / name) != digest:
            raise RuntimeError(f"canonical checksum changed: {directory / name}")
    return entries


def _policy(_label: str) -> dict:
    return {
        "cold": True,
        "initial_exact_jacobian_required": True,
        "maximum_exact_jacobian_refreshes": 2,
        "use_carried_solver_state": False,
        "exact_jacobian_refresh_policy": "on_line_search_failure",
    }


@contextmanager
def _runtime():
    d_values = {
        "WORK_PACKAGE": WORK_PACKAGE,
        "ARTIFACT": ARTIFACT,
        "MANIFEST_ARTIFACT": MANIFEST_ARTIFACT,
        "MANIFEST_DIRECTORY": MANIFEST_DIRECTORY,
        "SEED_PATH": SEED_PATH,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "SOURCE_FILES": SOURCE_FILES,
        "_root_policy": _policy,
        "_solve_root": e14l._solve_root,
    }
    l_values = {
        "WORK_PACKAGE": WORK_PACKAGE,
        "ARTIFACT": ARTIFACT,
        "MANIFEST_ARTIFACT": MANIFEST_ARTIFACT,
        "MANIFEST_DIRECTORY": MANIFEST_DIRECTORY,
        "SEED_PATH": SEED_PATH,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "SOURCE_FILES": SOURCE_FILES,
        "_root_policy": _policy,
    }
    d_original = {name: getattr(e14d, name) for name in d_values}
    l_original = {name: getattr(e14l, name) for name in l_values}
    try:
        for name, value in d_values.items():
            setattr(e14d, name, value)
        for name, value in l_values.items():
            setattr(e14l, name, value)
        yield
    finally:
        for name, value in d_original.items():
            setattr(e14d, name, value)
        for name, value in l_original.items():
            setattr(e14l, name, value)


def _validate_manifest() -> dict:
    _checksums(MANIFEST_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_manifest.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    reference = _read(MANIFEST_DIRECTORY / "reference_lock.json")
    if (
        not summary["passed"]
        or not summary["operational_timestep_rung_2e7_execution_authorized"]
        or contract["matched_endpoint"]["coarse_timestep_seconds"] != COARSE_TIMESTEP_SECONDS
        or contract["matched_endpoint"]["fine_timestep_seconds"] != FINE_TIMESTEP_SECONDS
        or contract["decision"]["pass"] != "operational_timestep_rung_2e7_certified"
    ):
        raise RuntimeError("operational-timestep contract changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen operational-timestep source changed: {relative}")
    if _sha(SEED_PATH) != reference["seed_sha256"]:
        raise RuntimeError("operational-timestep seed changed")
    if _sha(FINE_REFERENCE_DIRECTORY / "result_warm_1.npz") != reference["fine_endpoint_result_sha256"]:
        raise RuntimeError("operational-timestep fine reference changed")
    current_threads = {name: os.environ.get(name) for name in e14d.THREAD_ENVIRONMENT}
    if current_threads != provenance["thread_environment"]:
        raise RuntimeError("operational-timestep thread environment changed")
    return {"summary": summary, "contract": contract, "provenance": provenance}


def _identity() -> dict:
    return {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "test_sha256": _sha(ROOT / THIS_TEST),
        "manifest_summary_sha256": _sha(MANIFEST_DIRECTORY / "summary.json"),
        "manifest_contract_sha256": _sha(MANIFEST_DIRECTORY / "execution_manifest.json"),
        "seed_sha256": _sha(SEED_PATH),
    }


def _load_seed(data: dict, identity: dict):
    seed = load_causal_five_field_fixed_q_continuation_state(SEED_PATH, data["context"])
    if (
        seed.current_order != 2
        or seed.next_order != 2
        or seed.history.previous_timestep_seconds != FINE_TIMESTEP_SECONDS
        or seed.nonlinear_solver_state is not None
    ):
        raise RuntimeError("operational-timestep seed semantics changed")
    path = SCRATCH_DIRECTORY / "seed_roundtrip.npz"
    save_causal_five_field_fixed_q_continuation_state(path, data["context"], seed)
    loaded = load_causal_five_field_fixed_q_continuation_state(path, data["context"])
    if not causal_five_field_fixed_q_continuation_states_equal(seed, loaded):
        raise RuntimeError("operational-timestep seed roundtrip changed")
    _write(SCRATCH_DIRECTORY / "seed_validation.json", {
        "passed": True,
        "roundtrip_bitwise": True,
        "seed_sha256": _sha(SEED_PATH),
        "identity": identity,
    })
    return loaded


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
            })
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog["artifacts"][ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "latest_work_package": WORK_PACKAGE,
    })
    _write(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, coarse_result, fine_state: np.ndarray, fine_action: np.ndarray) -> None:
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("operational-timestep canonical package already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    for path in sorted(SCRATCH_DIRECTORY.iterdir()):
        if path.is_file():
            shutil.copy2(path, CANONICAL_DIRECTORY / path.name)
    passed = metrics["scientific_passed"]
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "operational_timestep_rung_2e7_certified"
            if passed else "operational_timestep_rung_2e7_failed"
        ),
        "passed": passed,
        "scientific_passed": passed,
        "trajectory_executed": True,
        "accepted_BDF2_roots": int(coarse_result is not None and coarse_result.accepted),
        "accepted_horizon_seconds": COARSE_TIMESTEP_SECONDS if coarse_result is not None and coarse_result.accepted else 0.0,
        "operational_timestep_rung_4e7_manifest_authorized": passed,
        "operational_timestep_rung_4e7_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "metrics.json", metrics)
    _write(CANONICAL_DIRECTORY / "provenance.json", {
        "schema_version": 1,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "source_hashes": {relative: _sha(ROOT / relative) for relative in SOURCE_FILES},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "thread_environment": {name: os.environ.get(name) for name in e14d.THREAD_ENVIRONMENT},
    })
    if coarse_result is not None:
        e14d._write_npz(
            CANONICAL_DIRECTORY / "matched_endpoint_arrays.npz",
            coarse_primitive_charts=coarse_result.primitive_charts,
            fine_primitive_charts=fine_state,
            coarse_reaction_action_per_s=coarse_result.scaled_reaction_rate_action_per_s,
            fine_reaction_action_per_s=fine_action,
        )
    files = [path for path in sorted(CANONICAL_DIRECTORY.iterdir()) if path.is_file()]
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    _catalog(summary)


def _run() -> dict:
    frozen = _validate_manifest()
    if not _clean():
        raise RuntimeError("operational-timestep execution requires a clean tree")
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("operational-timestep scratch directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    with _runtime():
        identity = _identity()
        _write(SCRATCH_DIRECTORY / "execution_identity.json", identity)
        data = e14d.e1._state_data("primary_20ms")
        start = _load_seed(data, identity)
        coarse_result = None
        coarse_continuation = None
        root_metrics = None
        failure_stage = None
        scientific = True
        try:
            coarse_result, coarse_continuation, root_metrics = e14d._advance(
                "coarse_2e7", data, start, COARSE_TIMESTEP_SECONDS, identity
            )
        except e14d.BindingRootFailure as error:
            coarse_result = error.result
            root_metrics = error.metrics
            scientific = False
            failure_stage = str(error)

        replay = {"executed": False, "passed": False}
        if scientific:
            replay_result, replay_metrics = e14l._solve_root(
                "coarse_2e7", data, start, COARSE_TIMESTEP_SECONDS, identity,
                artifact_label="replay_coarse_2e7",
            )
            replay_ok = bool(replay_result.accepted and replay_result.exact_jacobian_assemblies <= 2)
            if replay_ok:
                replay_continuation = causal_five_field_fixed_q_continuation_state(
                    replay_result,
                    data["context"],
                    start.current_primitive_charts,
                    primitive_column_scales=data["columns"],
                    conservation_row_scales=data["rows"],
                    parent_cell_indices=data["layout"].parent_cell_indices,
                    refinement_ratio=data["layout"].refinement_ratio,
                    elapsed_time_seconds=start.elapsed_time_seconds + COARSE_TIMESTEP_SECONDS,
                    completed_steps=start.completed_steps + 1,
                    provenance=identity,
                )
                result_equal = e14d._bitwise_results_equal(
                    coarse_result, replay_result, root_metrics, replay_metrics
                )
                continuation_equal = causal_five_field_fixed_q_continuation_states_equal(
                    coarse_continuation, replay_continuation
                )
            else:
                result_equal = False
                continuation_equal = False
            replay = {
                "executed": True,
                "passed": bool(replay_ok and result_equal and continuation_equal),
                "accepted": replay_ok,
                "result_bitwise": result_equal,
                "continuation_bitwise": continuation_equal,
                "metrics": replay_metrics,
            }
            scientific = bool(scientific and replay["passed"])
            if not replay["passed"]:
                failure_stage = "bitwise_coarse_replay"

        with np.load(FINE_REFERENCE_DIRECTORY / "result_warm_1.npz", allow_pickle=False) as fine:
            fine_state = np.asarray(fine["primitive_charts"])
            fine_action = np.asarray(fine["scaled_reaction_rate_action_per_s"])
        matched = {"executed": False, "passed": False}
        if coarse_result is not None and coarse_result.accepted:
            state_defect = e14d._scaled_endpoint_difference(
                fine_state,
                coarse_result.primitive_charts,
                start.current_primitive_charts,
                data["columns"],
            )
            action_defect = e14d._relative(
                fine_action, coarse_result.scaled_reaction_rate_action_per_s
            )
            matched = {
                "executed": True,
                "passed": bool(state_defect <= 0.1 and action_defect <= 0.1),
                "state_difference_relative_to_coarse_change": state_defect,
                "reaction_action_relative_difference": action_defect,
                "state_gate": 0.1,
                "reaction_action_gate": 0.1,
            }
            scientific = bool(scientific and matched["passed"])
            if not matched["passed"] and failure_stage is None:
                failure_stage = "matched_endpoint"
        ledger = (
            max(
                root_metrics["maximum_reaction_ledger_relative_defect"],
                root_metrics["maximum_constraint_action_ledger_relative_defect"],
            )
            if root_metrics is not None and root_metrics.get("accepted") else float("inf")
        )
        ledger_passed = bool(ledger <= 1.0e-12)
        scientific = bool(scientific and ledger_passed)
        if not ledger_passed and failure_stage is None:
            failure_stage = "ledger"
        metrics = {
            "schema_version": 1,
            "work_package": WORK_PACKAGE,
            "classification": (
                "operational_timestep_rung_2e7_certified"
                if scientific else "operational_timestep_rung_2e7_failed"
            ),
            "scientific_passed": scientific,
            "failure_stage": failure_stage,
            "coarse_root": root_metrics,
            "matched_endpoint": matched,
            "replay": replay,
            "cumulative_absolute_ledger_defect": ledger,
            "cumulative_ledger_passed": ledger_passed,
            "frozen_manifest": frozen["summary"],
            "identity": identity,
        }
        _write(SCRATCH_DIRECTORY / "execution_metrics.json", metrics)
        _canonicalize(metrics, coarse_result, fine_state, fine_action)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("select --run")
    print(json.dumps(e14d._plain(_run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
