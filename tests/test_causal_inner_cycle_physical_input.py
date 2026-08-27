import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_cycle_physical_input import (
    load_cycle_physical_input_bundle,
    save_cycle_physical_input_bundle,
    validate_cycle_physical_input_bundle,
)
import run_causal_inner_cycle_physical_input_bundle_schema_and_validator_certificate_wp10c9d6c7c3b5c4f25fizzt1 as fixture


def _bundle():
    return fixture._synthetic_bundle()


def test_synthetic_fixture_passes_structure_and_fails_physical_claim():
    metadata, driver, branch, events, heldout, conservation = _bundle()
    audit = validate_cycle_physical_input_bundle(
        metadata,
        driver,
        branch,
        events,
        heldout,
        conservation_map=conservation,
        require_physical=False,
    )
    assert audit.structurally_passed
    assert not audit.physically_usable
    with pytest.raises(ValueError, match="structural/synthetic"):
        validate_cycle_physical_input_bundle(
            metadata,
            driver,
            branch,
            events,
            heldout,
            conservation_map=conservation,
            require_physical=True,
        )


def test_nonpositive_phase_rate_and_periodic_mismatch_fail_closed():
    metadata, driver, branch, events, heldout, conservation = _bundle()
    broken = dict(driver)
    broken["phase_rate_per_second"] = driver["phase_rate_per_second"].copy()
    broken["phase_rate_per_second"][3] = 0.0
    with pytest.raises(ValueError, match="phase rate"):
        validate_cycle_physical_input_bundle(
            metadata, broken, branch, events, heldout,
            conservation_map=conservation, require_physical=False,
        )
    broken = dict(driver)
    broken["outer_incoming_characteristics11"] = driver[
        "outer_incoming_characteristics11"
    ].copy()
    broken["outer_incoming_characteristics11"][-1, 0, 0, 0] += 0.2
    with pytest.raises(ValueError, match="periodically"):
        validate_cycle_physical_input_bundle(
            metadata, broken, branch, events, heldout,
            conservation_map=conservation, require_physical=False,
        )


def test_ledger_mismatch_and_split_leakage_fail_closed():
    metadata, driver, branch, events, heldout, conservation = _bundle()
    broken = dict(driver)
    broken["distributed_source_ledger_rate4"] = driver[
        "distributed_source_ledger_rate4"
    ].copy()
    broken["distributed_source_ledger_rate4"][1, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="physical ledger"):
        validate_cycle_physical_input_bundle(
            metadata, broken, branch, events, heldout,
            conservation_map=conservation, require_physical=False,
        )
    leaked = copy.deepcopy(metadata)
    leaked["heldout_event_indices"] = [5, 7]
    with pytest.raises(ValueError, match="event training and heldout"):
        validate_cycle_physical_input_bundle(
            leaked, driver, branch, events, heldout,
            conservation_map=conservation, require_physical=False,
        )


def test_bundle_roundtrip_is_bitwise():
    metadata, driver, branch, events, heldout, _conservation = _bundle()
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "bundle"
        save_cycle_physical_input_bundle(target, metadata, driver, branch, events, heldout)
        loaded = load_cycle_physical_input_bundle(target)
    assert loaded[0] == metadata
    for original, replay in zip((driver, branch, events, heldout), loaded[1:], strict=True):
        assert original.keys() == replay.keys()
        for name in original:
            assert np.array_equal(original[name], replay[name])
