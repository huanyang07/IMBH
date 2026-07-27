from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/run_causal_inner_micro_export_preflight_wp10c9d0.py"
)
OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_preflight_wp10c9d0.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d0_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_cumulative_trapezoid_integrates_linear_history() -> None:
    module = _module()
    times = np.linspace(0.0, 2.0, 9)
    values = np.column_stack((times, 2.0 * times))
    integrated = module._cumulative_trapezoid(times, values)
    expected = np.column_stack((0.5 * times**2, times**2))
    assert np.allclose(integrated, expected, rtol=0.0, atol=2.0e-15)


def test_ladder_metrics_recovers_second_order() -> None:
    module = _module()
    times = np.linspace(0.0, 1.0, 21)
    base = np.column_stack(
        [
            1.0 + 0.2 * np.sin((index + 1) * times)
            for index in range(len(module.OBSERVABLE_NAMES))
        ]
    )
    shape = np.column_stack(
        [
            0.001 * (index + 1) * (1.0 + times)
            for index in range(len(module.OBSERVABLE_NAMES))
        ]
    )
    histories = {
        "coarse": base + 16.0 * shape,
        "medium": base + 4.0 * shape,
        "fine": base + shape,
    }
    baselines = {
        label: np.zeros(len(module.OBSERVABLE_NAMES), dtype=float)
        for label in histories
    }
    result = module._ladder_metrics(
        ("coarse", "medium", "fine"),
        histories,
        baselines,
        indices=module.GROUPS["exported"],
    )
    assert result["observed_order_rms"] == pytest.approx(2.0)
    assert result["observed_order_maximum"] == pytest.approx(2.0)
    assert result["passed"] is True


def test_committed_evidence_hash_and_decision_are_self_consistent() -> None:
    if not OUTPUT.exists():
        pytest.skip("WP10c9d0 evidence has not been generated")
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / payload["arrays_path"]
    assert payload["work_package"] == "WP10c9d0"
    assert payload["arrays_sha256"] == _sha256(arrays)
    assert payload["fixed_q_micro_solver_authorized"] == bool(
        payload["method_contract_passed"]
        and payload["cumulative_export_passed"]
    )
    if payload["fixed_q_micro_solver_authorized"]:
        assert payload["classification"] == (
            "conservative_cumulative_exports_authorize_constrained_"
            "micro_solver_feasibility"
        )
    else:
        assert payload["classification"] == (
            "conservative_micro_exports_fail_spatial_gate"
        )
