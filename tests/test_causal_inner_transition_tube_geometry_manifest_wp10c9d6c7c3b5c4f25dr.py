from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_transition_tube_geometry_manifest_wp10c9d6c7c3b5c4f25dr as manifest


def test_split_is_prospective_and_covers_every_state_once() -> None:
    train = set(manifest.TRAIN_STATE_INDICES)
    holdout = set(manifest.HOLDOUT_STATE_INDICES)
    assert train.isdisjoint(holdout)
    assert train | holdout == set(range(manifest.STATE_COUNT))
    assert {0, manifest.STATE_COUNT - 1} <= train


def test_scalar_dynamics_does_not_force_rank_one_embedding() -> None:
    contract = manifest._contract()
    architecture = contract["mathematical_architecture"]
    assert architecture["transition_dynamic_coordinate"] == "one_scalar_progress_s"
    assert architecture["embedding_rank_adaptive"]
    assert architecture["maximum_hidden_embedding_rank"] == 16
    assert architecture["hot_exit_required_before_impulse_collapse"]


def test_rejected_full_step_is_never_part_of_the_tube() -> None:
    contract = manifest._contract()
    trajectory = contract["trajectory"]
    assert trajectory["rejected_full_step_06_excluded"]
    assert trajectory["accepted_history_only"]
    assert trajectory["new_truth_calls"] == 0
    assert len(manifest._accepted_stage_directories()) == 17


def test_no_impulse_or_reduced_cycle_is_authorized() -> None:
    policy = manifest._contract()["decision_policy"]
    assert not policy["hot_branch_truth_authorized"]
    assert not policy["transition_impulse_fit_authorized"]
    assert not policy["reduced_slow_evolution_authorized"]
