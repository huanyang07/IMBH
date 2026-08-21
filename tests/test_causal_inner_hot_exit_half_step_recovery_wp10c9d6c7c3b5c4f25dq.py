from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hot_exit_half_step_recovery_wp10c9d6c7c3b5c4f25dq as execution


def test_execution_contract_keeps_the_trust_bound() -> None:
    contract = execution._static_execution_contract()
    assert contract["timestep_seconds"] == 5.0e-8
    assert contract["maximum_scaled_primitive_change"] == 5.0e-3
    assert not contract["trust_bound_relaxed"]
    assert not contract["rejected_full_step_propagated"]
    assert contract["rejected_root_never_propagates"]


def test_first_two_roots_are_cold_then_warm_reuse_begins() -> None:
    assert execution._root_policy("step_01")["cold"]
    assert execution._root_policy("step_02")["cold"]
    assert not execution._root_policy("step_03")["cold"]
    assert execution._root_policy("step_01")["maximum_exact_jacobian_refreshes"] == 2
    assert execution._root_policy("step_03")["maximum_exact_jacobian_refreshes"] == 1


def test_stage_paths_and_seed_are_isolated_from_failed_full_step() -> None:
    assert execution.base._input_checkpoint(1) == execution.manifest.SEED_CHECKPOINT
    assert "step_05" in str(execution.manifest.SEED_CHECKPOINT)
    assert not (execution.manifest.PARENT_STEP_06 / "checkpoint_step_06.npz").exists()
    assert execution.base._stage_directory(1).name.endswith("step_01")


def test_execution_identity_hashes_the_half_step_contract(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"half-step-test-checkpoint")
    lock_directory = tmp_path / "lock"
    lock_directory.mkdir()
    (lock_directory / "execution_lock.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(execution, "LOCK_DIRECTORY", lock_directory)
    identity = execution._execution_identity(1, checkpoint)
    assert identity["work_package"] == execution.WORK_PACKAGE
    assert identity["manifest_contract_sha256"] == execution._sha(
        execution.manifest.CANONICAL_DIRECTORY
        / "half_step_recovery_contract.json"
    )
    assert identity["input_checkpoint_sha256"] == execution._sha(checkpoint)
    assert identity["execution_lock_sha256"] == execution._sha(
        lock_directory / "execution_lock.json"
    )
