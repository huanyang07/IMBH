from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_terminal_prognosis_manifest_wp10c9d6c7c3b5c4f25dv as manifest


def test_extension_requires_trend_distance_and_cost_gates() -> None:
    prognosis = manifest._contract()["prognosis"]
    assert prognosis["extension_requires_every_tail_fraction_to_decrease"]
    assert prognosis["extension_requires_linear_crossing_within_budget"]
    assert prognosis["extension_requires_projected_wall_time_within_budget"]
    assert prognosis["maximum_additional_half_step_roots"] == 24


def test_branch_first_equations_are_explicit() -> None:
    architecture = manifest._contract()["selected_mathematical_architecture"]
    assert architecture["branch_equation"] == "G_b(q,h)=Q*F(Lq+Zh)=0"
    assert architecture["fold_continuation"] == "pseudo_arclength_bordered_G_equals_zero"
    assert architecture["online_full_truth_calls"] == 0
    assert architecture["online_y470_roots"] == 0


def test_cold_branch_does_not_require_pretending_hot_truth_exists() -> None:
    policy = manifest._contract()["decision_policy"]
    assert policy["cold_branch_can_be_certified_without_hot_branch"]
    assert policy["hot_branch_and_complete_impulse_remain_blocked"]
    assert policy["reduced_cycle_remains_blocked"]
