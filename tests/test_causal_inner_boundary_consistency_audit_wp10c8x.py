from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_boundary_consistency_audit_wp10c8x as wp10c8x


def test_observed_order_reports_nested_contraction() -> None:
    assert wp10c8x._observed_order(4.0, 1.0) == pytest.approx(2.0)
    assert wp10c8x._observed_order(0.0, 1.0) is None
    assert wp10c8x._observed_order(1.0, 0.0) is None


def test_machine_evidence_keeps_fine_history_blocked() -> None:
    if not wp10c8x.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c8x.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    assert payload["work_package"] == "WP10c8x"
    assert payload["classification"] == (
        "static_pass_but_common_initial_mode_unresolved"
    )
    assert payload["passed_static_candidates"] == [
        "production",
        "flux_linear",
        "storage_linear",
        "both_linear",
    ]
    assert payload["passed_history_candidates"] == []
    assert payload["initially_matched_candidates"] == []
    assert not payload["candidates"]["flux_cell_centered"]["passed"]
    assert not payload["candidates"]["storage_cell_centered"]["passed"]

    for name in payload["passed_static_candidates"]:
        result = payload["history_candidates"][name]
        assert result["available"]
        assert not result["passed"]
        assert not result["initial_match_passed"]
        for row in result["rows"].values():
            operator = row["operator"]
            assert operator["passed"]
            assert (
                operator[
                    "maximum_scaled_generator_factorization_defect"
                ]
                <= wp10c8x.MAXIMUM_GENERATOR_FACTORIZATION_DEFECT
            )
            assert (
                operator["maximum_relative_storage_action_defect"]
                <= wp10c8x.MAXIMUM_STORAGE_ACTION_DEFECT
            )

    assert payload["history_operator_equivalence"][
        "production_vs_storage_linear"
    ]["all_bitwise_equal"]
    assert payload["history_operator_equivalence"][
        "flux_linear_vs_both_linear"
    ]["all_bitwise_equal"]
    assert payload["decision"] == {
        "bounded_history_candidate_passed": False,
        "bounded_history_completed": True,
        "fixed_q_averaging_authorized": False,
        "n512_history_authorized": False,
        "production_boundary_replacement_authorized": False,
        "static_boundary_candidate_available": True,
    }
    assert payload["scope"]["formal_fast_average_certified"] is False
    assert payload["scope"]["reduced_architecture_selected"] is False
