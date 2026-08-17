#!/usr/bin/env python3
"""Execute the frozen short held-out fixed-Q continuation."""

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
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14t"
ARTIFACT = (
    "causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t"
)
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_heldout_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14s"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
SEED_PATH = MANIFEST_DIRECTORY / "heldout_seed_continuation.npz"
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "scripts/run_causal_inner_face36_fixed_q_heldout_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14s.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)
TIMESTEP_SECONDS = 1.0e-7


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


def _validate_manifest() -> dict:
    _checksums(MANIFEST_DIRECTORY)
    summary = _read(MANIFEST_DIRECTORY / "summary.json")
    contract = _read(MANIFEST_DIRECTORY / "execution_manifest.json")
    provenance = _read(MANIFEST_DIRECTORY / "provenance.json")
    seed_metrics = _read(MANIFEST_DIRECTORY / "seed_reconstruction.json")
    if (
        not summary["passed"]
        or not summary["heldout_continuation_execution_authorized"]
        or contract["state"] != "heldout_16ms"
        or contract["trajectory"]["root_order"] != ["cold_1", "warm_1", "warm_2"]
        or contract["solver_contract"]["warm_iteration_reserve_trigger"] != 6
        or contract["solver_contract"]["warm_failed_relative_backtrack_trigger"] != 4
        or not seed_metrics["bdf1_residual_bitwise"]
        or not seed_metrics["bdf2_residual_bitwise"]
        or not seed_metrics["continuation_roundtrip_bitwise"]
    ):
        raise RuntimeError("held-out continuation contract changed")
    for relative, digest in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen held-out source changed: {relative}")
    if _sha(SEED_PATH) != provenance["seed_sha256"]:
        raise RuntimeError("held-out continuation seed changed")
    return {"summary": summary, "contract": contract, "provenance": provenance}


@contextmanager
def _legacy_runtime():
    d_replacements = {
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
        "TIMESTEP_SECONDS": TIMESTEP_SECONDS,
        "_root_policy": e14l._root_policy,
        "_solve_root": e14l._solve_root,
    }
    l_replacements = {
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
    }
    d_original = {name: getattr(e14d, name) for name in d_replacements}
    l_original = {name: getattr(e14l, name) for name in l_replacements}
    try:
        for name, value in d_replacements.items():
            setattr(e14d, name, value)
        for name, value in l_replacements.items():
            setattr(e14l, name, value)
        yield
    finally:
        for name, value in d_original.items():
            setattr(e14d, name, value)
        for name, value in l_original.items():
            setattr(e14l, name, value)


def _root_accounting(main_metrics: dict) -> dict:
    labels = [label for label in ("cold_1", "warm_1", "warm_2") if label in main_metrics]
    accepted = [label for label in labels if main_metrics[label]["accepted"]]
    rejected = [label for label in labels if not main_metrics[label]["accepted"]]
    cumulative = sum(
        max(
            main_metrics[label]["maximum_reaction_ledger_relative_defect"],
            main_metrics[label]["maximum_constraint_action_ledger_relative_defect"],
        )
        for label in accepted
    )
    return {
        "attempted_roots": labels,
        "accepted_roots": accepted,
        "rejected_roots": rejected,
        "accepted_trajectory_horizon_seconds": len(accepted) * TIMESTEP_SECONDS,
        "accepted_trajectory_cumulative_ledger": cumulative,
        "planned_ladder_complete": len(labels) == 3 and not rejected,
    }


def _catalog(summary: dict) -> None:
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["case"] != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
                }
            )
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
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, data: dict, main_results: dict) -> None:
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("held-out canonical package already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    for path in sorted(SCRATCH_DIRECTORY.iterdir()):
        if path.is_file():
            shutil.copy2(path, CANONICAL_DIRECTORY / path.name)
    passed = metrics["scientific_passed"]
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "heldout_bounded_continuation_certified"
            if passed
            else "heldout_bounded_continuation_failed"
        ),
        "passed": passed,
        "scientific_passed": passed,
        "trajectory_executed": True,
        "attempted_BDF2_roots": len(metrics["root_accounting"]["attempted_roots"]),
        "accepted_BDF2_roots": len(metrics["root_accounting"]["accepted_roots"]),
        "rejected_BDF2_roots": len(metrics["root_accounting"]["rejected_roots"]),
        "accepted_horizon_seconds": metrics["root_accounting"][
            "accepted_trajectory_horizon_seconds"
        ],
        "operational_timestep_manifest_authorized": passed,
        "operational_timestep_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write(CANONICAL_DIRECTORY / "summary.json", summary)
    _write(CANONICAL_DIRECTORY / "metrics.json", metrics)
    _write(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "source_hashes": {relative: _sha(ROOT / relative) for relative in SOURCE_FILES},
            "manifest_sha256": _sha(MANIFEST_DIRECTORY / "execution_manifest.json"),
            "seed_sha256": _sha(SEED_PATH),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name) for name in e14d.THREAD_ENVIRONMENT
            },
        },
    )
    if main_results:
        labels = [label for label in ("cold_1", "warm_1", "warm_2") if label in main_results]
        e14d._write_npz(
            CANONICAL_DIRECTORY / "decisive_arrays.npz",
            labels=np.asarray(labels),
            primitive_charts=np.stack([main_results[label].primitive_charts for label in labels]),
            scaled_rates_per_s=np.stack([main_results[label].scaled_rate_per_s for label in labels]),
            reaction_actions_per_s=np.stack(
                [main_results[label].scaled_reaction_rate_action_per_s for label in labels]
            ),
            q3_target=np.asarray(data["reaction"].q3_value),
        )
    files = [path for path in sorted(CANONICAL_DIRECTORY.iterdir()) if path.is_file()]
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    _catalog(summary)


def _run() -> dict:
    frozen = _validate_manifest()
    if not _clean():
        raise RuntimeError("held-out continuation requires a clean tree")
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("held-out continuation scratch directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    with _legacy_runtime():
        identity = e14d._execution_identity()
        e14d._write_json(SCRATCH_DIRECTORY / "execution_identity.json", identity)
        data = e14d.e1._state_data("heldout_16ms")
        continuation = e14d._load_seed(data, identity)
        main_results = {}
        main_continuations = {}
        main_metrics = {}
        scientific = True
        failure_stage = None
        try:
            for label in ("cold_1", "warm_1", "warm_2"):
                result, continuation, metrics = e14d._advance(
                    label, data, continuation, TIMESTEP_SECONDS, identity
                )
                main_results[label] = result
                main_continuations[label] = continuation
                main_metrics[label] = metrics
        except e14d.BindingRootFailure as error:
            main_results[error.label] = error.result
            main_metrics[error.label] = error.metrics
            scientific = False
            failure_stage = str(error)
        replay = {"executed": False, "passed": False}
        if scientific:
            replay_start = load_causal_five_field_fixed_q_continuation_state(
                SCRATCH_DIRECTORY / "checkpoint_warm_1.npz",
                data["context"],
                expected_provenance=identity,
            )
            replay_result, replay_metrics = e14l._solve_root(
                "warm_2",
                data,
                replay_start,
                TIMESTEP_SECONDS,
                identity,
                artifact_label="replay_warm_2",
            )
            replay_accepted = bool(
                replay_result.accepted and replay_result.exact_jacobian_assemblies <= 1
            )
            if replay_accepted:
                replay_continuation = causal_five_field_fixed_q_continuation_state(
                    replay_result,
                    data["context"],
                    replay_start.current_primitive_charts,
                    primitive_column_scales=data["columns"],
                    conservation_row_scales=data["rows"],
                    parent_cell_indices=data["layout"].parent_cell_indices,
                    refinement_ratio=data["layout"].refinement_ratio,
                    elapsed_time_seconds=replay_start.elapsed_time_seconds + TIMESTEP_SECONDS,
                    completed_steps=replay_start.completed_steps + 1,
                    provenance=identity,
                )
                root_equal = e14d._bitwise_results_equal(
                    main_results["warm_2"],
                    replay_result,
                    main_metrics["warm_2"],
                    replay_metrics,
                )
                continuation_equal = causal_five_field_fixed_q_continuation_states_equal(
                    main_continuations["warm_2"], replay_continuation
                )
            else:
                root_equal = False
                continuation_equal = False
            replay = {
                "executed": True,
                "passed": bool(replay_accepted and root_equal and continuation_equal),
                "accepted": replay_accepted,
                "result_bitwise": root_equal,
                "continuation_bitwise": continuation_equal,
                "metrics": replay_metrics,
            }
            scientific = bool(scientific and replay["passed"])
            if not replay["passed"]:
                failure_stage = "bitwise_final_warm_replay"
        accounting = _root_accounting(main_metrics)
        ledger_passed = bool(
            accounting["planned_ladder_complete"]
            and accounting["accepted_trajectory_cumulative_ledger"] <= 3.0e-12
        )
        scientific = bool(scientific and ledger_passed)
        if not ledger_passed and failure_stage is None:
            failure_stage = "cumulative_ledger"
        metrics = {
            "schema_version": 1,
            "work_package": WORK_PACKAGE,
            "classification": (
                "heldout_bounded_continuation_certified"
                if scientific
                else "heldout_bounded_continuation_failed"
            ),
            "scientific_passed": scientific,
            "failure_stage": failure_stage,
            "main_roots": main_metrics,
            "replay": replay,
            "root_accounting": accounting,
            "cumulative_absolute_ledger_defect": accounting[
                "accepted_trajectory_cumulative_ledger"
            ],
            "cumulative_ledger_passed": ledger_passed,
            "warm_residual_evaluations": [
                main_metrics[label]["function_evaluations"]
                for label in ("warm_1", "warm_2")
                if label in main_metrics
            ],
            "warm_exact_assembly_count": sum(
                main_metrics[label]["exact_Jacobian_assemblies"]
                for label in ("warm_1", "warm_2")
                if label in main_metrics
            ),
            "identity": identity,
            "frozen_manifest": frozen["summary"],
        }
        e14d._write_json(SCRATCH_DIRECTORY / "execution_metrics.json", metrics)
        _canonicalize(metrics, data, main_results)
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
