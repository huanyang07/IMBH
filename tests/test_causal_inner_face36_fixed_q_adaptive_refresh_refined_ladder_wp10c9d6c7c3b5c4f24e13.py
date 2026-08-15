import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_refined_ladder_hardened_"
    "manifest_wp10c9d6c7c3b5c4f24e13a"
)
PRIMARY = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_primary_"
    "wp10c9d6c7c3b5c4f24e11"
)
HELDOUT = ROOT / "results/canonical" / (
    "causal_inner_face36_fixed_q_adaptive_refresh_heldout_"
    "wp10c9d6c7c3b5c4f24e12"
)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
RUNNER = importlib.import_module(
    "run_causal_inner_face36_fixed_q_adaptive_refresh_refined_"
    "ladder_wp10c9d6c7c3b5c4f24e13"
)


def _read(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_c4f24e13a_manifest_freezes_only_refined_cases() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["supersedes_work_package"] == "WP10c9d6c7c3b5c4f24e13"
    assert summary["next_case"] == "primary_middle"
    assert summary["refined_ladder_execution_authorized"]
    assert contract["reused_cases"] == ["primary_coarse", "heldout_coarse"]
    assert contract["refined_cases"] == [
        "primary_middle",
        "heldout_middle",
        "primary_fine",
        "heldout_fine",
    ]
    assert contract["timesteps_seconds"] == [1.0e-7, 5.0e-8, 2.5e-8]


def test_c4f24e13a_preserves_adaptive_and_scientific_gates() -> None:
    summary = _read(MANIFEST, "summary.json")
    contract = _read(MANIFEST, "execution_manifest.json")
    assert contract["binding_temporal_form"] == "exact_increment_primary"
    assert contract["direct_rate_form"] == "post_root_parity_audit_only"
    assert contract["exact_jacobian_refresh_policy"] == "on_line_search_failure"
    assert contract["maximum_exact_jacobian_assemblies_per_root"] == 2
    assert contract["minimum_state_rate_convergence_order"] == 0.9
    assert contract["minimum_reaction_action_convergence_order"] == 0.9
    assert contract["binding_order_error"].startswith("absolute_l2_error")
    assert contract["optional_untracked_predictors_forbidden"]
    assert contract["stage_local_scratch_required"]
    assert contract["canonical_prior_stage_validation_required"]
    assert contract["available_order_gate_applied_after_each_stage"]
    assert not contract["numerical_floor_may_rescue_failed_order"]
    assert contract["require_bitwise_restart_roundtrip"]
    assert contract["require_bitwise_BDF2_replay"]
    assert contract["fail_fast"]
    assert not contract["may_relax_any_gate"]
    assert not summary["one_Q_execution_manifest_authorized"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c4f24e13a_parent_authorizations_are_positive() -> None:
    primary = _read(PRIMARY, "summary.json")
    heldout = _read(HELDOUT, "summary.json")
    assert primary["passed"]
    assert primary["heldout_retry_manifest_authorized"]
    assert heldout["passed"]
    assert heldout["refined_ladder_manifest_authorized"]


def test_c4f24e13a_manifest_checksums_close() -> None:
    entries = {}
    for line in (MANIFEST / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    assert set(entries) == {
        "continuous_references.npz",
        "execution_manifest.json",
        "provenance.json",
        "summary.json",
    }
    for name, digest in entries.items():
        assert hashlib.sha256((MANIFEST / name).read_bytes()).hexdigest() == digest


def test_c4f24e13a_frozen_source_and_parent_contract_closes() -> None:
    RUNNER._validate_frozen_contract(prior_to="primary_middle")


def test_c4f24e13a_predictor_is_deterministic_and_self_contained() -> None:
    rate = np.asarray([1.0, -2.0, 3.0])
    multiplier = np.asarray([4.0, 5.0, 6.0])
    increment, actual_multiplier = RUNNER._deterministic_seed(
        "primary_20ms",
        1,
        {
            "continuous_rate": rate,
            "continuous_multiplier": multiplier,
        },
    )
    assert np.array_equal(increment, 5.0e-8 * rate)
    assert np.array_equal(actual_multiplier, multiplier)


def test_c4f24e13a_order_gate_uses_absolute_errors() -> None:
    assert RUNNER._order(0.2, 0.1) == pytest.approx(1.0)
    assert RUNNER._order(0.2, 0.11) < 0.9


@pytest.mark.parametrize(
    ("arguments", "function_name", "payload"),
    [
        (
            ["runner", "--freeze"],
            "_freeze",
            {"passed": True, "classification": "mock_freeze"},
        ),
        (
            ["runner", "--case", "primary_middle"],
            "_execute_case",
            {"passed": True, "summary": {}, "metrics": {}},
        ),
        (
            ["runner", "--finalize"],
            "_finalize",
            {"passed": True, "classification": "mock_finalize"},
        ),
    ],
)
def test_c4f24e13a_cli_modes_have_one_top_level_passed_contract(
    monkeypatch,
    arguments,
    function_name,
    payload,
) -> None:
    if function_name == "_execute_case":
        monkeypatch.setattr(RUNNER, function_name, lambda case: payload)
    else:
        monkeypatch.setattr(RUNNER, function_name, lambda: payload)
    monkeypatch.setattr(sys, "argv", arguments)
    RUNNER.main()


def test_c4f24e13a_cli_fails_closed_on_rejected_case(monkeypatch) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_execute_case",
        lambda case: {"passed": False, "summary": {}, "metrics": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--case", "primary_middle"],
    )
    with pytest.raises(SystemExit) as raised:
        RUNNER.main()
    assert raised.value.code == 1
