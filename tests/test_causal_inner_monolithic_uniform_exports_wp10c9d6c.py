from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_causal_inner_monolithic_uniform_exports_wp10c9d6c.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_uniform_exports_wp10c9d6c"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location("wp10c9d6c_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def test_wp10c9d6c_contract_is_uniform_only_and_production_neutral() -> None:
    assert RUNNER.LABELS == (
        "uniform_N64",
        "uniform_N128",
        "uniform_N256",
    )
    assert RUNNER.PERTURBATIONS == (
        "common_mode",
        "heldout_near_excision",
        "heldout_mid_inner",
    )
    assert RUNNER.MINIMUM_EXPORT_ORDER == 0.75
    assert RUNNER.MAXIMUM_FINE_PHYSICAL_DIFFERENCE == 0.05
    assert RUNNER.MINIMUM_HISTORY_COSINE == 0.90
    assert RUNNER.MINIMUM_ERROR_COSINE == 0.90


def test_wp10c9d6c_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c"
    assert summary["method_passed"]
    assert not summary["uses_production_generator"]
    assert not summary["uses_production_anchor_storage_derivative"]
    assert summary["candidate_base_rate_is_self_consistent"]
    assert summary["center_broken_within_cell_paths"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    if summary["passed"]:
        assert summary["classification"] == (
            "monolithic_uniform_physical_exports_passed_"
            "embedded_export_discrimination_authorized"
        )
        assert summary["common_mode_passed"]
        assert summary["held_out_ladders_passed"]
        assert summary["embedded_export_discrimination_authorized"]
    else:
        assert not summary["embedded_export_discrimination_authorized"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
