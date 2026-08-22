from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_adaptive_complete_cycle_manifest_wp10c9d6c7c3b5c4f25fd as target


def test_architecture_uses_original_free_field_and_supersedes_fixed_q_time() -> None:
    architecture = target._architecture()
    assert architecture["dynamics"]["truth_field"].startswith("original unconstrained")
    assert architecture["fixed_Q_scope"]["physical_clock"] == "forbidden"
    assert architecture["fixed_Q_scope"]["reaction_in_cycle_dynamics"] == "forbidden"
    assert architecture["mode_policy"]["known_retained_mode"].startswith("accepted full-model cold")


def test_execution_contract_is_fail_closed_and_slow_model_remains_gated() -> None:
    cost = target._cost_projection()
    contract = target._execution_contract(cost)
    assert cost["offline_cost_gate_passed"]
    assert cost["online_cost_gate_passed"]
    assert contract["adaptive_acquisition"]["maximum_exact_free_field_witnesses"] == 192
    assert contract["adaptive_acquisition"]["rule"].startswith("reject before propagation")
    assert contract["cycle_section"]["slow_macro_drift_allowed"]
    assert "reduced slow evolution remains separately gated" in contract["post_pass_authorization"]


def test_inputs_preserve_cold_but_not_fixed_q_transition() -> None:
    locked = target._validate_inputs(require_clean=False)
    assert all(locked["cold_retained_gates"].values())
    assert locked["classifications"]["reaction_diagnosis"].startswith(
        "conservative_free_field_hidden_amplitude_rom_selected"
    )


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    architecture = helper._read(
        target.CANONICAL_DIRECTORY / "mathematical_architecture.json"
    )
    assert summary["passed"]
    assert summary["mathematical_architecture_verified"]
    assert summary["complete_cycle_execution_authorized"]
    assert not summary["complete_cycle_executed"]
    assert not summary["fixed_Q_physical_phase_authorized"]
    assert architecture["fixed_Q_scope"]["physical_clock"] == "forbidden"
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
