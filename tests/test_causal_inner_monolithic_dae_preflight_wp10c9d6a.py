from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts/run_causal_inner_monolithic_dae_preflight_wp10c9d6a.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/causal_inner_monolithic_dae_preflight_wp10c9d6a"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location("wp10c9d6a_runner", RUNNER_PATH)
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


def test_wp10c9d6a_contract_is_production_neutral_and_path_explicit() -> None:
    assert RUNNER.ANALYZED_BASE_COMMIT == (
        "e836df2e2c0e1d180f3a8c56383498578434762e"
    )
    assert RUNNER.MAXIMUM_BLOCK_LEDGER_DEFECT == 1.0e-12
    assert RUNNER.MAXIMUM_JVP_ORDER_DIFFERENCE == 1.0e-8
    assert RUNNER.MINIMUM_VERTICAL_EXTERIOR_DERIVATIVE > 0.0
    assert RUNNER.MINIMUM_TEMPORAL_LOOP_DEFECT > 0.0


def test_wp10c9d6a_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6a"
    assert summary["passed"]
    assert summary["assembly_passed"]
    assert summary["temporal_product_nonexactness_detected"]
    assert summary["classification"] == (
        "monolithic_descriptor_path_assembly_certified_"
        "manufactured_preflight_authorized"
    )
    assert not summary["strict_endpoint_storage_potential_authorized"]
    assert summary["declared_temporal_path_product_required"]
    assert summary["manufactured_equilibrium_and_wave_preflight_authorized"]
    assert not summary["physical_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["incoming_excision_characteristics"] == 0
    assert not summary["uses_production_generator"]
    assert not summary["uses_production_anchor_storage_derivative"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
