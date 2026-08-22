from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import (
    ConservativeHeunEngine,
    ConservativeCoordinateSplit,
    ConservativeHiddenAmplitudeModel,
    HiddenAmplitudeState,
    HystereticModeSelector,
    LocalAffineReducedPatch,
    canonical_rate_basis,
    polynomial_holdout,
    relative_projection_defects,
)


def _split() -> ConservativeCoordinateSplit:
    return ConservativeCoordinateSplit(
        macro_restriction=np.asarray(((1.0, 0.0, 0.0),)),
        macro_lift=np.asarray(((1.0,), (0.0,), (0.0,))),
        hidden_dual=np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        hidden_lift=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
    )


def test_conservative_split_roundtrips_state_and_rate() -> None:
    split = _split()
    coordinate = np.asarray((2.0, -3.0, 4.0))
    macro, hidden = split.split(coordinate)
    np.testing.assert_array_equal(split.compose(macro, hidden), coordinate)
    assert max(split.identity_defects.values()) == 0.0


def test_hidden_amplitude_model_never_projects_macro_ledger() -> None:
    model = ConservativeHiddenAmplitudeModel(
        split=_split(),
        hidden_origin=np.asarray((1.0, -1.0)),
        hidden_basis=np.asarray(((1.0,), (0.0,))),
    )
    state = HiddenAmplitudeState(
        macro=np.asarray((2.0,)),
        amplitudes=np.asarray((3.0,)),
        forcing_phase=0.25,
        mode="cold",
    )
    np.testing.assert_array_equal(model.decode(state), np.asarray((2.0, 4.0, -1.0)))
    macro_rate, amplitude_rate, unresolved = model.project_rate(
        np.asarray((5.0, 7.0, 0.0))
    )
    np.testing.assert_array_equal(macro_rate, np.asarray((5.0,)))
    np.testing.assert_array_equal(amplitude_rate, np.asarray((7.0,)))
    assert unresolved == 0.0


def test_rate_basis_and_polynomial_holdout() -> None:
    samples = np.asarray(((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
    basis, _singular, energy = canonical_rate_basis(samples, rank=1)
    assert np.max(relative_projection_defects(samples, basis)) < 1.0e-14
    assert energy[0] == 1.0
    nodes = np.linspace(0.0, 1.0, 5)
    values = np.stack((1.0 + nodes + nodes * nodes, 2.0 - nodes), axis=1)
    heldout, predictions, relative = polynomial_holdout(
        nodes, values, np.asarray((0, 2, 4))
    )
    np.testing.assert_array_equal(heldout, np.asarray((1, 3)))
    np.testing.assert_allclose(predictions, values[heldout], atol=1.0e-14)
    assert np.max(relative) < 1.0e-14


def test_local_affine_heun_engine_is_conservative_and_fail_closed() -> None:
    split = _split()
    model = ConservativeHiddenAmplitudeModel(
        split=split,
        hidden_origin=np.asarray((0.2, -0.1)),
        hidden_basis=np.eye(2),
    )
    patch = LocalAffineReducedPatch(
        anchor_macro=np.asarray((1.0,)),
        anchor_amplitudes=np.zeros(2),
        anchor_reduced_rate=np.asarray((2.0, 0.5, 0.25)),
        physical_rate_delta=np.asarray((0.2, 0.05, -0.02)),
        macro_step_seconds=0.1,
        mode="hot",
        anchor_id="hot-0",
        maximum_absolute_eta=1.25,
    )
    engine = ConservativeHeunEngine(
        model=model,
        patch=patch,
        forcing_angular_frequency=2.0,
        maximum_embedded_error_fraction=0.1,
    )
    state = HiddenAmplitudeState(
        macro=patch.anchor_macro,
        amplitudes=np.zeros(2),
        forcing_phase=0.0,
        mode="hot",
    )
    accepted = engine.step(state, 0.1)
    assert accepted.accepted
    assert accepted.macro_ledger_defect < 5.0e-15
    np.testing.assert_allclose(accepted.predictor_eta, 1.0, atol=5.0e-15)
    np.testing.assert_allclose(
        model.split.macro_restriction @ model.decode(accepted.candidate),
        accepted.candidate.macro,
        atol=5.0e-15,
    )
    rejected = engine.step(state, 0.2)
    assert not rejected.accepted
    assert "predictor_outside_patch" in rejected.failure_reasons


def test_hysteretic_mode_selector_requires_margin_and_persistence() -> None:
    selector = HystereticModeSelector(
        relative_switch_margin=0.1, persistence_steps=2
    )
    first = selector.update(
        current_mode="cold", normalized_distances={"cold": 1.0, "hot": 0.5}
    )
    assert first.mode == "cold"
    assert first.pending_mode == "hot"
    second = selector.update(
        current_mode=first.mode,
        normalized_distances={"cold": 1.0, "hot": 0.5},
        pending_mode=first.pending_mode,
        pending_count=first.pending_count,
    )
    assert second.mode == "hot"
    assert second.switched
    no_switch = selector.update(
        current_mode="hot", normalized_distances={"cold": 0.95, "hot": 1.0}
    )
    assert no_switch.mode == "hot"
