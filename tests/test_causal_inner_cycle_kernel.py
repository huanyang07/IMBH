import tempfile
from pathlib import Path

import numpy as np
import pytest

import run_causal_inner_reduced_hybrid_cycle_kernel_certificate_wp10c9d6c7c3b5c4f25fizzx1 as runner
from imri_qpe.layer3_minidisk_1d.causal_inner_cycle_kernel import (
    CycleKernelCheckpoint,
    integrate_cycle_kernel,
    load_cycle_kernel_checkpoint,
    require_production_cycle_metadata,
    save_cycle_kernel_checkpoint,
)


def test_synthetic_kernel_advances_two_events_and_audits_every_endpoint():
    kernel, initial = runner._fixture()
    result = integrate_cycle_kernel(
        kernel,
        initial,
        end_time_seconds=2.5,
        absolute_tolerance=np.full(5, 1.0e-10),
        relative_tolerance=1.0e-9,
        maximum_accepted_steps=128,
    )
    assert [value.name for value in result.reduced.events] == [
        "cold_to_hot",
        "hot_to_cold",
    ]
    assert len(result.endpoint_audits) == len(result.reduced.accepted_checkpoints)
    assert len(result.endpoint_audits) >= 16
    assert result.reduced_ledger_relative_defect <= 2.0e-12
    assert all(value.inner_incoming_count == 0 for value in result.endpoint_audits)
    assert all(value.outer_incoming_count == 11 for value in result.endpoint_audits)


def test_production_gate_rejects_synthetic_and_missing_holdouts():
    kernel, _ = runner._fixture()
    with pytest.raises(ValueError, match="incomplete"):
        require_production_cycle_metadata(kernel.metadata)
    metadata = dict(kernel.metadata)
    metadata.update(
        {
            "physical_model_complete": True,
            "physical_payload_hashes_complete": True,
            "heldout_physical_validation_complete": True,
            "independent_spatial_holdout_complete": True,
            "independent_sequence_or_cycle_holdout_complete": True,
        }
    )
    with pytest.raises(ValueError, match="synthetic"):
        require_production_cycle_metadata(metadata)


def test_cycle_checkpoint_is_bitwise_and_hash_locked():
    kernel, initial = runner._fixture()
    checkpoint = CycleKernelCheckpoint(initial, kernel.physical_bundle_sha256, "f" * 64)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cycle.npz"
        save_cycle_kernel_checkpoint(checkpoint, path)
        loaded = load_cycle_kernel_checkpoint(
            path,
            expected_physical_bundle_sha256=kernel.physical_bundle_sha256,
            expected_kernel_contract_sha256="f" * 64,
        )
        assert runner._checkpoint_equal(loaded.reduced, initial)
        with pytest.raises(ValueError, match="bundle hash"):
            load_cycle_kernel_checkpoint(
                path,
                expected_physical_bundle_sha256="wrong",
                expected_kernel_contract_sha256="f" * 64,
            )


def test_outside_atlas_and_event_sheet_fail_closed():
    kernel, initial = runner._fixture()
    outside = initial.state5.copy()
    outside[0] += 10.0
    with pytest.raises(ValueError, match="outside"):
        kernel.rhs(0.0, outside, 0)
    with pytest.raises(ValueError, match="outside"):
        kernel.guard_value(outside, 0)
