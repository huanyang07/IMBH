#!/usr/bin/env python3
"""Supersede the doubled-step manifest with an admissible predictor."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

base = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u"
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14w"
ARTIFACT = (
    "causal_inner_face36_fixed_q_operational_timestep_predictor_manifest_"
    "wp10c9d6c7c3b5c4f24e14w"
)
ARTIFACT_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SUPERSEDED_DIRECTORY = base.ARTIFACT_DIRECTORY
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_predictor_manifest_"
    "wp10c9d6c7c3b5c4f24e14w.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_operational_timestep_predictor_manifest_"
    "wp10c9d6c7c3b5c4f24e14w.py"
)
EXECUTION_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_predictor_rung_"
    "wp10c9d6c7c3b5c4f24e14x.py"
)
EXECUTION_TEST = (
    "tests/test_causal_inner_face36_fixed_q_operational_timestep_predictor_rung_"
    "wp10c9d6c7c3b5c4f24e14x.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    EXECUTION_RUNNER,
    EXECUTION_TEST,
    base.THIS_RUNNER,
    base.EXECUTION_RUNNER,
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)

CONTRACT = deepcopy(base.CONTRACT)
CONTRACT.update({
    "work_package": WORK_PACKAGE,
    "supersedes_before_nonlinear_root": "WP10c9d6c7c3b5c4f24e14u",
})
CONTRACT["solver_contract"].update({
    "initial_predictor": "previous_accepted_scaled_primitive_increment",
    "last_rate_extrapolation_forbidden": True,
    "previous_accepted_scaled_increment_maximum": 0.0045488441553660965,
    "maximum_scaled_primitive_change": 0.005,
})


@contextmanager
def _base_globals():
    values = {
        "WORK_PACKAGE": WORK_PACKAGE,
        "ARTIFACT": ARTIFACT,
        "ARTIFACT_DIRECTORY": ARTIFACT_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "EXECUTION_RUNNER": EXECUTION_RUNNER,
        "EXECUTION_TEST": EXECUTION_TEST,
        "SOURCE_FILES": SOURCE_FILES,
        "CONTRACT": CONTRACT,
    }
    original = {name: getattr(base, name) for name in values}
    try:
        for name, value in values.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def _validate_parents() -> dict:
    parents = base._validate_parents()
    superseded_hashes = base._checksums(SUPERSEDED_DIRECTORY)
    superseded = base._read(SUPERSEDED_DIRECTORY / "summary.json")
    if (
        superseded["classification"]
        != "operational_timestep_rung_2e7_manifest_frozen_execution_authorized"
        or not superseded["passed"]
    ):
        raise RuntimeError("superseded doubled-step manifest changed")
    parents["superseded_manifest"] = superseded
    parents["package_hashes"]["superseded_manifest"] = superseded_hashes
    parents["pre_root_execution_failure"] = {
        "classification": "operational_timestep_predictor_preflight_failed",
        "nonlinear_root_solved": False,
        "trajectory_horizon_seconds_added": 0.0,
        "reason": "last-rate variable-step BDF predictor exceeded the unchanged primitive-change bound",
        "unbounded_predictor_maximum": 0.009097688310732193,
        "bound": 0.005,
    }
    return parents


def _freeze() -> dict:
    parents = _validate_parents()
    if not base._clean():
        raise RuntimeError("predictor-repair manifest requires a clean tree")
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": "operational_timestep_predictor_repair_manifest_frozen_execution_authorized",
        "passed": True,
        "definitions_only": True,
        "trajectory_executed": False,
        "operational_timestep_rung_2e7_execution_authorized": True,
        "operational_timestep_rung_4e7_manifest_authorized": False,
        "operational_timestep_rung_4e7_execution_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "physical_microburst_authorized": False,
        "fast_averaging_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    base._write(ARTIFACT_DIRECTORY / "execution_manifest.json", CONTRACT)
    base._write(ARTIFACT_DIRECTORY / "parent_lock.json", parents)
    base._write(ARTIFACT_DIRECTORY / "reference_lock.json", {
        "seed_path": str(base.SEED_PATH.relative_to(ROOT)),
        "seed_sha256": base._sha(base.SEED_PATH),
        "fine_cold_result_sha256": base._sha(base.FINE_REFERENCE_DIRECTORY / "result_cold_1.npz"),
        "fine_endpoint_result_sha256": base._sha(base.FINE_REFERENCE_DIRECTORY / "result_warm_1.npz"),
    })
    base._write(ARTIFACT_DIRECTORY / "summary.json", summary)
    base._write(ARTIFACT_DIRECTORY / "provenance.json", {
        "schema_version": 1,
        "definition_commit": base._git("rev-parse", "HEAD"),
        "definition_tree": base._git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "source_hashes": {relative: base._sha(ROOT / relative) for relative in SOURCE_FILES},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    })
    files = ("execution_manifest.json", "parent_lock.json", "provenance.json", "reference_lock.json", "summary.json")
    (ARTIFACT_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{base._sha(ARTIFACT_DIRECTORY / name)}  {name}\n" for name in files),
        encoding="utf-8",
    )
    with _base_globals():
        base._catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("select --freeze")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
