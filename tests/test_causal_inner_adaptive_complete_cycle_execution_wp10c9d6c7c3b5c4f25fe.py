from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import (  # noqa: E402
    ConservativeCoordinateSplit,
    HiddenAmplitudeState,
)
import run_causal_inner_adaptive_complete_cycle_execution_wp10c9d6c7c3b5c4f25fe as target  # noqa: E402


def test_frozen_parent_authorizes_only_this_execution() -> None:
    locked = target._validate_parent(require_clean=False)
    contract = locked["contract"]
    assert contract["authorized_execution"] == (
        "WP10c9d6c7c3b5c4f25fe_complete_cycle_execution"
    )
    assert contract["adaptive_acquisition"]["maximum_patches"] == 64
    assert contract["adaptive_acquisition"]["maximum_exact_free_field_witnesses"] == 192
    assert contract["adaptive_acquisition"]["maximum_witnesses_per_patch"] == 3
    assert contract["initialization"]["discard_as_physical_time"].startswith(
        "all fixed-Q"
    )


def test_affine_fit_obeys_withheld_pattern() -> None:
    eta = np.asarray((0.0, 1.0, 0.5))
    exact = np.stack((
        np.asarray((1.0, -2.0)),
        np.asarray((3.0, 2.0)),
        np.asarray((2.0, 0.0)),
    ))
    for training in (
        np.asarray((False, True, True)),
        np.asarray((True, False, True)),
        np.asarray((True, True, False)),
    ):
        intercept, slope = target._fit_affine_axis(eta, exact, training)
        prediction = np.stack(
            [target._axis_prediction(intercept, slope, value) for value in eta]
        )
        np.testing.assert_allclose(prediction, exact, rtol=0.0, atol=3.0e-15)


def test_section_crossing_is_same_orientation_only() -> None:
    assert target._section_crossing_fraction(0.2, -0.1) is None
    assert target._section_crossing_fraction(-0.25, 0.75) == 0.25
    assert target._section_crossing_fraction(-0.25, -0.1) is None


def test_restart_roundtrip_is_bitwise() -> None:
    state = HiddenAmplitudeState(
        macro=np.asarray((1.0, -2.0)),
        amplitudes=np.asarray((0.25,)),
        forcing_phase=0.125,
        mode="cold_recovery",
        elapsed_seconds=0.75,
    )
    restored = target._state_roundtrip(state)
    assert target._bitwise_state(state, restored)
    coordinate = np.asarray((1.0, np.nextafter(2.0, 3.0), -4.0))
    np.testing.assert_array_equal(target._coordinate_roundtrip(coordinate), coordinate)


def test_truth_free_patch_replay_preserves_exact_macro_ledger() -> None:
    split = ConservativeCoordinateSplit(
        macro_restriction=np.asarray(((1.0, 0.0, 0.0),)),
        macro_lift=np.asarray(((1.0,), (0.0,), (0.0,))),
        hidden_dual=np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        hidden_lift=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
    )
    patch_metrics = {
        "patch_index": 0,
        "mode_after": "cold_recovery",
    }
    anchor = np.asarray((1.0, 2.0, 3.0))
    patch_arrays = {
        "hidden_basis388xr": np.eye(2),
        "anchor_coordinate470": anchor,
        "affine_intercept470_per_s": np.asarray((4.0, 1.0, -2.0)),
        "affine_slope470_per_s": np.zeros(3),
    }
    coordinate, _state = target._replay_patch(
        split, patch_metrics, patch_arrays, anchor, half_step=False
    )
    expected = anchor + target.manifest.MACRO_STEP_SECONDS * np.asarray(
        (4.0, 1.0, -2.0)
    )
    np.testing.assert_allclose(coordinate, expected, rtol=0.0, atol=2.0e-16)
    half_path, _state = target._replay_patch_path(
        split, patch_metrics, patch_arrays, anchor, half_step=True
    )
    assert len(half_path) == 2
    np.testing.assert_allclose(half_path[-1], expected, rtol=0.0, atol=2.0e-16)


def test_hidden_basis_selection_binds_training_and_holdout() -> None:
    rates = np.asarray((
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ))
    basis, rank, attempts = target._select_mode_basis(
        rates,
        np.asarray((True, True, False)),
        np.asarray((False, False, True)),
    )
    assert rank == 2
    assert basis.shape == (3, 2)
    assert attempts["2"]["maximum_training_defect"] < 1.0e-14
    assert attempts["2"]["maximum_holdout_defect"] < 1.0e-14


def test_execution_forbids_fixed_q_physical_work() -> None:
    source = (ROOT / target.THIS_RUNNER).read_text(encoding="utf-8")
    assert '"fixed_Q_physical_rate_calls": 0' in source
    assert '"fixed_Q_reaction_calls": 0' in source
    assert '"nonlinear_roots": 0' in source
    assert '"BDF_microsteps": 0' in source
