#!/usr/bin/env python3
"""Execute the doubled-step rung with the frozen admissible predictor."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

base = importlib.import_module(  # noqa: E402
    "run_causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    causal_bdf_coefficients,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14x"
ARTIFACT = (
    "causal_inner_face36_fixed_q_operational_timestep_predictor_rung_"
    "wp10c9d6c7c3b5c4f24e14x"
)
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_operational_timestep_predictor_manifest_"
    "wp10c9d6c7c3b5c4f24e14w"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_predictor_rung_"
    "wp10c9d6c7c3b5c4f24e14x.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_operational_timestep_predictor_rung_"
    "wp10c9d6c7c3b5c4f24e14x.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "scripts/run_causal_inner_face36_fixed_q_operational_timestep_predictor_manifest_"
    "wp10c9d6c7c3b5c4f24e14w.py",
    base.THIS_RUNNER,
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
)


def _bounded_predictors(continuation, columns: np.ndarray):
    previous = (
        continuation.history.previous_primitive_increment / columns
    ).ravel()
    desired = np.array(previous, copy=True)
    coefficients = causal_bdf_coefficients(
        2,
        base.COARSE_TIMESTEP_SECONDS,
        continuation.history.previous_timestep_seconds,
    )
    rate = (
        coefficients.current_increment_coefficient * desired
        + coefficients.previous_increment_coefficient * previous
    ) / base.COARSE_TIMESTEP_SECONDS
    multiplier = np.linalg.solve(
        continuation.next_reaction_channel_transform,
        continuation.raw_multiplier_predictor,
    )
    if np.max(np.abs(desired)) > 5.0e-3:
        raise RuntimeError("frozen operational-timestep predictor exceeds bound")
    return rate, multiplier


@contextmanager
def _patched_runtime():
    values = {
        "WORK_PACKAGE": WORK_PACKAGE,
        "ARTIFACT": ARTIFACT,
        "MANIFEST_ARTIFACT": MANIFEST_ARTIFACT,
        "MANIFEST_DIRECTORY": MANIFEST_DIRECTORY,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "SOURCE_FILES": SOURCE_FILES,
    }
    original = {name: getattr(base, name) for name in values}
    old_predictors = base.e14d._predictors
    try:
        for name, value in values.items():
            setattr(base, name, value)
        base.e14d._predictors = _bounded_predictors
        yield
    finally:
        base.e14d._predictors = old_predictors
        for name, value in original.items():
            setattr(base, name, value)


def _run():
    with _patched_runtime():
        return base._run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("select --run")
    print(json.dumps(base.e14d._plain(_run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
