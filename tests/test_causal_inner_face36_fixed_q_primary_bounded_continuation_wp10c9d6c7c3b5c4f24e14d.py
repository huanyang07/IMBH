import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_c4f24e14d_frozen_contract_authorizes_only_primary(monkeypatch) -> None:
    for name, value in RUNNER.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    summary = _read(RUNNER.MANIFEST_DIRECTORY / "summary.json")
    assert summary["primary_bounded_continuation_execution_authorized"]
    assert not summary["heldout_continuation_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    with pytest.raises(RuntimeError, match="frozen source changed"):
        RUNNER._validate_frozen_contract(require_clean=False)


def test_c4f24e14d_root_policy_is_cold_then_warm() -> None:
    cold = RUNNER._root_policy("cold_1")
    assert cold["initial_exact_jacobian_required"]
    assert cold["maximum_exact_jacobian_refreshes"] == 2
    assert not cold["use_carried_solver_state"]
    for label in ("warm_1", "warm_2", "warm_3"):
        warm = RUNNER._root_policy(label)
        assert not warm["initial_exact_jacobian_required"]
        assert warm["maximum_exact_jacobian_refreshes"] == 1
        assert warm["use_carried_solver_state"]


def test_c4f24e14d_nonpropagating_controls_are_cold() -> None:
    for label in ("cold_shadow", "half_1", "half_2"):
        policy = RUNNER._root_policy(label)
        assert policy["cold"]
        assert policy["maximum_exact_jacobian_refreshes"] == 2


def test_c4f24e14d_predictor_uses_checkpoint_interval() -> None:
    class History:
        previous_primitive_increment = np.full((2, 5), 2.0)
        previous_timestep_seconds = 4.0

    class Continuation:
        history = History()
        next_reaction_channel_transform = np.diag([2.0, 4.0, 5.0])
        raw_multiplier_predictor = np.asarray([2.0, 8.0, 15.0])

    rate, multiplier = RUNNER._predictors(
        Continuation(),
        np.full((2, 5), 0.5),
    )
    assert np.array_equal(rate, np.ones(10))
    assert np.array_equal(multiplier, np.asarray([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    ("scientific", "cost", "expected"),
    [
        (True, True, "bounded_continuation_and_reuse_passed"),
        (True, False, "bounded_continuation_valid_cost_failed"),
        (False, True, "bounded_continuation_failed"),
        (False, False, "bounded_continuation_failed"),
    ],
)
def test_c4f24e14d_classification_is_diagonal(
    scientific,
    cost,
    expected,
) -> None:
    assert RUNNER._classification(scientific, cost) == expected


def test_c4f24e14d_scaled_comparisons() -> None:
    columns = np.full((2, 5), 2.0)
    start = np.zeros((2, 5))
    full = np.full((2, 5), 2.0)
    refined = np.full((2, 5), 2.2)
    assert np.isclose(
        RUNNER._scaled_state_absolute(refined, full, columns),
        0.1,
    )
    assert np.isclose(
        RUNNER._scaled_endpoint_difference(
            refined,
            full,
            start,
            columns,
        ),
        0.1,
    )


def test_c4f24e14d_failure_aware_accounting_excludes_rejected_root() -> None:
    accepted = {
        "accepted": True,
        "maximum_reaction_ledger_relative_defect": 2.0e-13,
        "maximum_constraint_action_ledger_relative_defect": 3.0e-13,
    }
    rejected = {
        "accepted": False,
        "maximum_reaction_ledger_relative_defect": 5.0e-13,
        "maximum_constraint_action_ledger_relative_defect": 7.0e-13,
    }
    accounting = RUNNER._failure_aware_root_accounting(
        {"cold_1": accepted, "warm_1": rejected}
    )
    assert accounting["attempted_roots"] == ["cold_1", "warm_1"]
    assert accounting["accepted_roots"] == ["cold_1"]
    assert accounting["rejected_roots"] == ["warm_1"]
    assert accounting["accepted_trajectory_horizon_seconds"] == 1.0e-7
    assert accounting["accepted_trajectory_cumulative_ledger"] == 3.0e-13
    assert accounting["rejected_candidate_diagnostic_ledgers"] == {
        "warm_1": 7.0e-13
    }
    assert not accounting["planned_ladder_complete"]


def test_c4f24e14d_legacy_checkpoint_marks_counter_untrusted() -> None:
    checkpoint = CANONICAL / "checkpoint_cold_1.npz"
    if not checkpoint.exists():
        pytest.skip("canonical e14d checkpoint is recorded after execution")
    data = RUNNER.e1._state_data("primary_20ms")
    loaded = RUNNER.load_causal_five_field_fixed_q_continuation_state(
        checkpoint,
        data["context"],
    )
    solver = loaded.nonlinear_solver_state
    assert solver is not None
    assert solver.schema_version == 1
    assert solver.counter_semantics == "legacy_untrusted_aggregate"
    assert solver.total_broyden_updates == solver.broyden_updates_since_exact


def test_c4f24e14d_canonical_package_closes_when_present() -> None:
    if not CANONICAL.exists():
        pytest.skip("canonical e14d result is recorded after execution")
    summary = _read(CANONICAL / "summary.json")
    assert summary["trajectory_executed"]
    entries = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == (
            digest
        )


def test_c4f24e14d_cli_validate_only(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_validate_frozen_contract",
        lambda require_clean: {"passed": True},
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--validate"])
    RUNNER.main()
    assert '"passed": true' in capsys.readouterr().out


def test_c4f24e14d_cli_requires_one_mode(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(SystemExit):
        RUNNER.main()
