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
    / "scripts/"
    "run_causal_inner_cell_fluctuation_ledger_wp10c9d3.py"
)
OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_cell_fluctuation_ledger_wp10c9d3.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d3_runner", RUNNER)
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


def test_orders_recovers_second_order() -> None:
    module = _module()
    np.testing.assert_allclose(
        module._orders(np.asarray([16.0, 4.0, 1.0])),
        np.full(2, 2.0),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_committed_evidence_hash_and_decision_are_self_consistent() -> None:
    if not OUTPUT.exists():
        pytest.skip("WP10c9d3 evidence has not been generated")
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / payload["arrays_path"]
    assert payload["work_package"] == "WP10c9d3"
    assert payload["arrays_sha256"] == _sha256(arrays)
    expected = all(
        case["passed"] for case in payload["cases"].values()
    )
    assert payload["method_contract_passed"] is expected
    assert payload["radial_well_balance_audit_authorized"] is expected
    assert payload["production_operator_authorized"] is False
    assert payload["fixed_q_micro_solver_authorized"] is False
